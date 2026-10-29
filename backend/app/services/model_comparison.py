from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.services.market_calendar import future_session_closes, resolve_exchange
from app.services.features import TECHNICAL_FACTOR_NAMES, create_technical_factors

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
except ImportError as exc:
    Prophet = None
    PROPHET_IMPORT_ERROR = str(exc)
else:
    PROPHET_IMPORT_ERROR = ""

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = str(exc)
else:
    XGBOOST_IMPORT_ERROR = ""

try:
    import torch
    from torch import nn
except ImportError as exc:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = str(exc)
else:
    TORCH_IMPORT_ERROR = ""


HORIZON_SPECS: dict[str, tuple[str, int]] = {
    "3d": ("3 trading days", 3),
    "7d": ("7 trading days", 7),
    "15d": ("15 trading days", 15),
    "30d": ("30 trading days (about 1 month)", 30),
    "3m": ("3 months", 3),
    "6m": ("6 months", 6),
}
MAX_VALIDATION_FOLDS = 8
MAX_TEST_FOLDS = 8
MIN_CALIBRATION_SAMPLES = 4
TEST_FRACTION = 0.2
MIN_LAG = 50
PRICE_FEATURE_NAMES = (
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "return_lag_10",
    "return_lag_20",
    "return_mean_5",
    "return_mean_20",
    "return_volatility_5",
    "return_volatility_20",
    "price_deviation_5",
    "price_deviation_20",
    "price_deviation_50",
)
MARKET_FACTOR_NAMES = (
    "benchmark_return_1",
    "benchmark_return_5",
    "benchmark_volatility_20",
)
BASE_FEATURE_NAMES = (*PRICE_FEATURE_NAMES, *TECHNICAL_FACTOR_NAMES)
ALL_FEATURE_NAMES = (*BASE_FEATURE_NAMES, *MARKET_FACTOR_NAMES)


def _feature_names(history: pd.DataFrame) -> tuple[str, ...]:
    if set(MARKET_FACTOR_NAMES).issubset(history.columns):
        return ALL_FEATURE_NAMES
    return BASE_FEATURE_NAMES


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    minimum_rows: int
    factory: Callable[[], object] | None = None


MODEL_SPECS = (
    ModelSpec(
        "ridge",
        "Machine learning",
        60,
        lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    ),
    ModelSpec(
        "random_forest",
        "Machine learning",
        70,
        lambda: RandomForestRegressor(
            n_estimators=120,
            max_depth=9,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    ModelSpec(
        "extra_trees",
        "Machine learning",
        70,
        lambda: ExtraTreesRegressor(
            n_estimators=120,
            max_depth=9,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    ModelSpec(
        "hist_gradient_boosting",
        "Machine learning",
        70,
        lambda: HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        ),
    ),
    ModelSpec(
        "xgboost",
        "Machine learning",
        70,
        lambda: XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.04,
            objective="reg:squarederror",
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=1,
        ),
    ),
    ModelSpec("lstm", "Deep learning", 100),
    ModelSpec("arima", "Statistical", 60),
    ModelSpec("sarima", "Statistical", 100),
    ModelSpec("exponential_smoothing", "Statistical", 60),
    ModelSpec("prophet", "Statistical", 60),
)

BASELINE_NAMES = ("naive", "seasonal_naive")
ALLOWED_MODELS = set(BASELINE_NAMES) | {spec.name for spec in MODEL_SPECS}


def _resolve_horizon(
    horizon: str | int,
    last_timestamp: pd.Timestamp,
    exchange: str,
) -> tuple[int, str, str, list[pd.Timestamp]]:
    numeric_horizon = (
        horizon
        if isinstance(horizon, int)
        else int(horizon)
        if horizon.strip().isdigit()
        else None
    )
    if numeric_horizon is not None:
        if numeric_horizon < 1 or numeric_horizon > 132:
            raise ValueError("Trading-session horizons must be between 1 and 132")
        end_date = last_timestamp.date() + timedelta(days=numeric_horizon * 3 + 10)
        horizon_label = (
            "1 trading day"
            if numeric_horizon == 1
            else f"{numeric_horizon} trading days"
        )
        horizon_key = str(numeric_horizon)
    else:
        horizon_key = horizon.strip().lower()
        if horizon_key not in HORIZON_SPECS:
            options = ", ".join(HORIZON_SPECS)
            raise ValueError(f"Unsupported forecast horizon. Choose one of: {options}")
        horizon_label, amount = HORIZON_SPECS[horizon_key]
        if horizon_key.endswith("m"):
            end_date = (last_timestamp + pd.DateOffset(months=amount)).date()
        else:
            end_date = (
                last_timestamp + pd.DateOffset(days=amount * 3 + 10)
            ).date()

    session_closes = future_session_closes(last_timestamp, exchange, end_date)
    requested_sessions = (
        numeric_horizon
        if numeric_horizon is not None
        else amount
        if not horizon_key.endswith("m")
        else None
    )
    if requested_sessions is not None:
        if len(session_closes) < requested_sessions:
            raise ValueError("Could not locate enough future exchange sessions")
        session_closes = session_closes[:requested_sessions]
    if not session_closes:
        raise ValueError("No future exchange sessions exist within this horizon")
    return len(session_closes), horizon_label, horizon_key, session_closes


def _price_features(
    log_prices: list[float] | np.ndarray,
    technical_factors: np.ndarray | None = None,
    market_factors: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(log_prices, dtype=float)
    if len(values) <= MIN_LAG:
        raise ValueError(f"At least {MIN_LAG + 1} prices are required for lag features")

    returns = np.diff(values)
    price_features = np.asarray(
        [
            returns[-1],
            returns[-2],
            returns[-3],
            returns[-5],
            returns[-10],
            returns[-20],
            float(np.mean(returns[-5:])),
            float(np.mean(returns[-20:])),
            float(np.std(returns[-5:])),
            float(np.std(returns[-20:])),
            values[-1] - float(np.mean(values[-5:])),
            values[-1] - float(np.mean(values[-20:])),
            values[-1] - float(np.mean(values[-50:])),
        ],
        dtype=float,
    )
    output = price_features
    if technical_factors is not None:
        output = np.concatenate([output, technical_factors.astype(float)])
    if market_factors is not None:
        output = np.concatenate([output, market_factors.astype(float)])
    return output


def _supervised_returns(
    log_prices: np.ndarray,
    history: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    factors = create_technical_factors(history)
    has_market_factors = set(MARKET_FACTOR_NAMES).issubset(history.columns)
    for target_index in range(MIN_LAG + 1, len(log_prices)):
        price_history = log_prices[:target_index]
        observed_factors = factors.iloc[target_index - 1].to_numpy(dtype=float)
        market_factors = (
            history.iloc[target_index - 1][list(MARKET_FACTOR_NAMES)].to_numpy(
                dtype=float
            )
            if has_market_factors
            else None
        )
        if not np.isfinite(observed_factors).all() or (
            market_factors is not None and not np.isfinite(market_factors).all()
        ):
            continue
        features.append(
            _price_features(price_history, observed_factors, market_factors)
        )
        targets.append(float(log_prices[target_index] - log_prices[target_index - 1]))
    if not features:
        return (
            np.empty((0, len(_feature_names(history)))),
            np.empty(0),
            np.empty(0, dtype=int),
        )

    feature_array = np.vstack(features)
    target_array = np.asarray(targets)
    selected_indices = _select_training_features(feature_array, target_array)
    return feature_array[:, selected_indices], target_array, selected_indices


def _select_training_features(
    features: np.ndarray,
    targets: np.ndarray,
    correlation_limit: float = 0.98,
) -> np.ndarray:
    """Keep nonconstant, target-relevant representatives of redundant inputs."""
    if features.ndim != 2 or features.shape[1] == 0:
        return np.empty(0, dtype=int)
    spread = np.std(features, axis=0)
    candidate_indices = np.flatnonzero(np.isfinite(spread) & (spread > 1e-10))
    if len(candidate_indices) == 0:
        return np.asarray([int(np.argmax(np.nan_to_num(spread)))], dtype=int)
    if len(candidate_indices) <= 1:
        return candidate_indices

    candidates = features[:, candidate_indices]
    correlations = np.corrcoef(candidates, rowvar=False)
    if np.std(targets) <= 1e-10:
        target_correlations = np.zeros(candidates.shape[1], dtype=float)
    else:
        target_correlations = np.asarray(
            [
                np.corrcoef(candidates[:, column], targets)[0, 1]
                for column in range(candidates.shape[1])
            ],
            dtype=float,
        )
    target_correlations = np.nan_to_num(
        target_correlations, nan=0.0, posinf=0.0, neginf=0.0
    )
    priority = np.argsort(-np.abs(target_correlations), kind="stable")
    retained: list[int] = []
    for candidate in priority:
        if all(
            abs(float(correlations[candidate, other])) < correlation_limit
            for other in retained
        ):
            retained.append(int(candidate))
    return np.sort(candidate_indices[np.asarray(retained, dtype=int)])


def _availability(name: str) -> str | None:
    if name == "prophet" and Prophet is None:
        return f"Prophet is unavailable: {PROPHET_IMPORT_ERROR}"
    if name == "xgboost" and XGBRegressor is None:
        return f"XGBoost is unavailable: {XGBOOST_IMPORT_ERROR}"
    if name == "lstm" and torch is None:
        return f"PyTorch is unavailable; install torch to enable the LSTM model: {TORCH_IMPORT_ERROR}"
    return None


def _lstm_recursive_forecast(log_prices: np.ndarray, steps: int) -> list[float]:
    if torch is None or nn is None:
        raise RuntimeError(_availability("lstm") or "PyTorch is unavailable")

    torch.manual_seed(42)
    torch.set_num_threads(1)
    sequence_length = 20
    returns = np.diff(log_prices)
    sample_count = len(returns) - sequence_length
    split = max(1, int(sample_count * 0.85))
    training_returns = returns[: sequence_length + split]
    mean = float(np.mean(training_returns))
    scale = float(np.std(training_returns))
    if not math.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    standardized = (returns - mean) / scale
    sequences = np.asarray(
        [
            standardized[index - sequence_length : index]
            for index in range(sequence_length, len(standardized))
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        standardized[sequence_length:],
        dtype=np.float32,
    )
    if len(sequences) < 30:
        raise ValueError("Insufficient sequences for LSTM training")

    train_x = torch.from_numpy(sequences[:split, :, None])
    train_y = torch.from_numpy(targets[:split, None])
    validation_x = torch.from_numpy(sequences[split:, :, None])
    validation_y = torch.from_numpy(targets[split:, None])

    class PriceLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
            self.output = nn.Linear(16, 1)

        def forward(self, sequence):
            output, _ = self.lstm(sequence)
            return self.output(output[:, -1, :])

    model = PriceLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    patience = 5
    stale_epochs = 0

    for _ in range(40):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = (
                float(loss_function(model(validation_x), validation_y).item())
                if len(validation_x)
                else float(loss.item())
            )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    generated_returns: list[float] = []
    rolling_returns = list(standardized)
    current_log_price = float(log_prices[-1])
    model.eval()
    for _ in range(steps):
        sequence = torch.tensor(
            rolling_returns[-sequence_length:], dtype=torch.float32
        )[None, :, None]
        with torch.no_grad():
            predicted_return = float(model(sequence).item()) * scale + mean
        current_log_price += predicted_return
        generated_returns.append(float(np.exp(current_log_price)))
        rolling_returns.append((predicted_return - mean) / scale)
    return generated_returns


def _recursive_ml_forecast(
    name: str,
    log_prices: np.ndarray,
    steps: int,
    observed_history: pd.DataFrame,
) -> list[float]:
    if name == "lstm":
        return _lstm_recursive_forecast(log_prices, steps)

    features, targets, selected_indices = _supervised_returns(
        log_prices, observed_history
    )
    minimum_rows = next(spec.minimum_rows for spec in MODEL_SPECS if spec.name == name)
    if len(features) < minimum_rows:
        raise ValueError(f"Requires at least {minimum_rows} supervised training rows")
    spec = next(spec for spec in MODEL_SPECS if spec.name == name)
    model = spec.factory()
    model.fit(features, targets)

    history = list(float(value) for value in log_prices)
    latest_factors = create_technical_factors(observed_history).iloc[-1].to_numpy(
        dtype=float
    )
    if not np.isfinite(latest_factors).all():
        raise ValueError("Insufficient valid historical OHLCV for technical factors")
    market_factors = (
        observed_history[list(MARKET_FACTOR_NAMES)].iloc[-1].to_numpy(dtype=float)
        if set(MARKET_FACTOR_NAMES).issubset(observed_history.columns)
        else None
    )
    if market_factors is not None and not np.isfinite(market_factors).all():
        raise ValueError("Insufficient aligned benchmark history for forecast")
    forecasts: list[float] = []
    for _ in range(steps):
        predicted_return = float(
            model.predict(
                _price_features(history, latest_factors, market_factors)[
                    selected_indices
                ][None, :]
            )[0]
        )
        if not math.isfinite(predicted_return):
            raise ValueError("Model produced a non-finite return")
        history.append(history[-1] + predicted_return)
        forecasts.append(float(np.exp(history[-1])))
    return forecasts


def _prophet_forecast(
    history: pd.DataFrame,
    future_dates: list[pd.Timestamp],
) -> list[float]:
    if Prophet is None:
        raise RuntimeError(_availability("prophet") or "Prophet is unavailable")
    training = pd.DataFrame(
        {
            "ds": pd.to_datetime(history["timestamp"], utc=True).dt.tz_localize(None).dt.normalize(),
            "y": np.log(history["close"].to_numpy(dtype=float)),
        }
    )
    future = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                pd.Series(future_dates), utc=True
            ).dt.tz_localize(None).dt.normalize()
        }
    )
    coverage = training["ds"].max() - training["ds"].min()
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=coverage >= pd.Timedelta(days=21),
        yearly_seasonality=coverage >= pd.Timedelta(days=730),
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,
    )
    model.fit(training)
    return np.exp(model.predict(future)["yhat"].to_numpy(dtype=float)).tolist()


def _model_forecast(
    name: str,
    history: pd.DataFrame,
    steps: int,
    future_dates: list[pd.Timestamp],
) -> list[float]:
    closes = history["close"].to_numpy(dtype=float)
    if name == "naive":
        return [float(closes[-1])] * steps
    if name == "seasonal_naive":
        period = 5
        if len(closes) <= period:
            raise ValueError("Seasonal naive requires at least six observations")
        generated = list(float(value) for value in closes)
        output: list[float] = []
        for _ in range(steps):
            forecast = generated[-period]
            generated.append(forecast)
            output.append(forecast)
        return output
    if name == "prophet":
        return _prophet_forecast(history, future_dates)
    if name in {spec.name for spec in MODEL_SPECS if spec.family == "Machine learning"} | {"lstm"}:
        return _recursive_ml_forecast(name, np.log(closes), steps, history)

    log_close = np.log(closes)
    if name == "arima":
        forecast = ARIMA(log_close, order=(1, 1, 1)).fit().forecast(steps=steps)
    elif name == "sarima":
        forecast = SARIMAX(
            log_close,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 5),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=80).forecast(steps=steps)
    elif name == "exponential_smoothing":
        model = ExponentialSmoothing(
            log_close,
            trend="add",
            seasonal="add" if len(log_close) >= 30 else None,
            seasonal_periods=5 if len(log_close) >= 30 else None,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast = model.forecast(steps)
    else:
        raise ValueError(f"Unsupported model: {name}")

    values = np.exp(np.asarray(forecast, dtype=float))
    if len(values) != steps:
        raise ValueError(f"{name} returned an incomplete forecast path")
    return values.tolist()


def _feature_importance(
    name: str,
    history: pd.DataFrame,
) -> list[dict[str, float | str]]:
    spec = next((candidate for candidate in MODEL_SPECS if candidate.name == name), None)
    if spec is None or spec.family != "Machine learning" or name == "lstm":
        return []
    closes = history["close"].to_numpy(dtype=float)
    features, targets, selected_indices = _supervised_returns(
        np.log(closes), history
    )
    if len(features) < spec.minimum_rows:
        return []
    model = spec.factory()
    model.fit(features, targets)
    estimator = model[-1] if hasattr(model, "steps") else model
    importances = getattr(estimator, "feature_importances_", None)
    importance_kind = "tree split importance"
    if importances is None:
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            return []
        importances = np.abs(np.asarray(coefficients, dtype=float).reshape(-1))
        importance_kind = "absolute standardized coefficient"
    values = np.asarray(importances, dtype=float)
    total = float(values.sum())
    if not np.isfinite(values).all() or total <= 0:
        return []
    return [
        {
            "feature": ALL_FEATURE_NAMES[selected_indices[index]],
            "importance": float(values[index] / total),
            "method": importance_kind,
        }
        for index in np.argsort(values)[::-1][:10]
    ]


def _selected_feature_names(history: pd.DataFrame) -> list[str]:
    closes = history["close"].to_numpy(dtype=float)
    _, _, selected_indices = _supervised_returns(np.log(closes), history)
    return [ALL_FEATURE_NAMES[index] for index in selected_indices]


def _metrics(actual: list[float], predicted: list[float], reference: list[float]) -> dict:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    denominator = np.abs(actual_values) + np.abs(predicted_values)
    smape = np.mean(
        np.divide(
            2 * np.abs(actual_values - predicted_values),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0,
        )
    )
    return {
        "observations": int(len(actual_values)),
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "smape": float(smape * 100),
        "bias": float(np.mean(predicted_values - actual_values)),
        "directional_accuracy": float(
            np.mean(
                np.sign(predicted_values - reference_values)
                == np.sign(actual_values - reference_values)
            )
        ),
    }


def _align_benchmark_features(
    bars: pd.DataFrame,
    benchmark_history: pd.DataFrame | None,
) -> tuple[pd.DataFrame, bool]:
    if benchmark_history is None or not {"timestamp", "close"}.issubset(
        benchmark_history.columns
    ):
        return bars, False

    benchmark = benchmark_history[["timestamp", "close"]].copy()
    benchmark["timestamp"] = pd.to_datetime(
        benchmark["timestamp"], utc=True, errors="coerce"
    )
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = (
        benchmark.dropna(subset=["timestamp", "close"])
        .loc[lambda frame: frame["close"] > 0]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    if len(benchmark) < 21:
        return bars, False

    returns = benchmark["close"].pct_change()
    benchmark["benchmark_return_1"] = returns
    benchmark["benchmark_return_5"] = benchmark["close"].pct_change(5)
    benchmark["benchmark_volatility_20"] = returns.rolling(20).std()
    aligned = pd.merge_asof(
        bars[["timestamp"]].sort_values("timestamp"),
        benchmark[["timestamp", *MARKET_FACTOR_NAMES]],
        on="timestamp",
        direction="backward",
    )
    if aligned.empty or not np.isfinite(
        aligned[list(MARKET_FACTOR_NAMES)].iloc[-1].to_numpy(dtype=float)
    ).all():
        return bars, False
    result = bars.copy()
    for name in MARKET_FACTOR_NAMES:
        result[name] = aligned[name].to_numpy(dtype=float)
    return result, True


def _prediction_intervals(
    predicted_path: list[float],
    validation_residuals: list[list[float]],
) -> list[dict]:
    output: list[dict] = []
    for predicted, residuals in zip(predicted_path, validation_residuals, strict=True):
        lower = upper = None
        if len(residuals) >= MIN_CALIBRATION_SAMPLES:
            errors = np.abs(np.asarray(residuals, dtype=float))
            rank = math.ceil((len(errors) + 1) * 0.8)
            if rank <= len(errors):
                radius = float(np.sort(errors)[rank - 1])
                lower = max(0.0, predicted - radius)
                upper = predicted + radius
        output.append(
            {
                "lower_80": lower,
                "upper_80": upper,
                "calibration_samples": len(residuals),
            }
        )
    return output


def compare_models(
    raw: pd.DataFrame,
    symbol: str,
    interval: str,
    horizon: str | int = "3d",
    selected_model: str = "auto",
    exchange: str = "auto",
    benchmark_history: pd.DataFrame | None = None,
) -> dict:
    if interval != "1d":
        raise ValueError(
            "Calendar-based forecasts currently support daily bars only. "
            "Select the daily interval and retry."
        )
    if selected_model != "auto" and selected_model not in ALLOWED_MODELS:
        raise ValueError(f"Unknown model: {selected_model}")
    if not {"timestamp", "close"}.issubset(raw.columns):
        raise ValueError("Market data must contain timestamp and close columns")

    exchange_name = resolve_exchange(symbol, exchange)
    market_columns = [
        column
        for column in ("timestamp", "open", "high", "low", "close", "volume")
        if column in raw.columns
    ]
    bars = raw[market_columns].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    for column in market_columns[1:]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    has_full_ohlcv = {"open", "high", "low", "volume"}.issubset(bars.columns)
    if has_full_ohlcv:
        valid_ohlcv = (
            (bars["open"] > 0)
            & (bars["high"] > 0)
            & (bars["low"] > 0)
            & (bars["close"] > 0)
            & (bars["volume"] >= 0)
            & (bars["low"] <= bars["high"])
            & (bars["low"] <= bars["open"])
            & (bars["low"] <= bars["close"])
            & (bars["high"] >= bars["open"])
            & (bars["high"] >= bars["close"])
        )
        bars = bars.loc[valid_ohlcv]
    bars = (
        bars.dropna(subset=["timestamp", "close"])
        .loc[lambda frame: frame["close"] > 0]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    bars, benchmark_available = _align_benchmark_features(
        bars, benchmark_history
    )
    if len(bars) < MIN_LAG + 80:
        raise ValueError(
            f"At least {MIN_LAG + 80} valid daily observations are required; "
            f"received {len(bars)}."
        )

    steps, horizon_label, horizon_key, future_dates = _resolve_horizon(
        horizon, pd.Timestamp(bars["timestamp"].iloc[-1]), exchange_name
    )
    if steps > 132:
        raise ValueError("The selected exchange-date horizon is longer than supported")
    closes = bars["close"].to_numpy(dtype=float)
    count = len(bars)
    earliest_origin = max(MIN_LAG + 80, int(count * 0.55))
    latest_origin = count - steps - 1
    test_start = int(count * (1 - TEST_FRACTION))
    latest_validation_origin = min(latest_origin, test_start - steps - 1)
    if latest_validation_origin < earliest_origin:
        raise ValueError(
            f"Not enough observations for leakage-free validation and a final "
            f"chronological test period at {horizon_label}. Load more daily history."
        )
    origins = np.unique(
        np.linspace(
            earliest_origin,
            latest_validation_origin,
            num=min(
                MAX_VALIDATION_FOLDS,
                latest_validation_origin - earliest_origin + 1,
            ),
            dtype=int,
        )
    )
    if len(origins) < 3:
        raise ValueError(
            "Not enough distinct chronological validation windows for this horizon"
        )
    test_origins = np.unique(
        np.linspace(
            test_start - 1,
            latest_origin,
            num=min(MAX_TEST_FOLDS, latest_origin - test_start + 2),
            dtype=int,
        )
    )
    if len(test_origins) < 2:
        raise ValueError(
            "Not enough distinct chronological test windows for this horizon"
        )

    available_names = ["naive", "seasonal_naive"]
    results: dict[str, dict] = {
        "naive": {"family": "Baseline", "eligible": True, "status": "evaluated"},
        "seasonal_naive": {"family": "Baseline", "eligible": True, "status": "evaluated"},
    }
    predictions: dict[str, list[list[float]]] = {
        "naive": [],
        "seasonal_naive": [],
    }
    for spec in MODEL_SPECS:
        unavailable = _availability(spec.name)
        if (
            not has_full_ohlcv
            and spec.family == "Machine learning"
            and spec.name != "lstm"
        ):
            unavailable = (
                "Supervised OHLCV models require open, high, low, and volume "
                "history; those fields are missing."
            )
        if unavailable:
            results[spec.name] = {
                "family": spec.family,
                "eligible": False,
                "status": "unavailable",
                "reason": unavailable[:240],
            }
        else:
            available_names.append(spec.name)
            predictions[spec.name] = []
            results[spec.name] = {
                "family": spec.family,
                "eligible": True,
                "status": "evaluating",
            }

    residuals: dict[str, list[list[float]]] = {
        name: [[] for _ in range(steps)] for name in available_names
    }
    validation_actuals: list[float] = []
    validation_references: list[float] = []
    validation_timestamps: list[str] = []
    validation_actual_paths: list[list[float]] = []
    validation_reference_paths: list[list[float]] = []
    failures: dict[str, str] = {}

    for origin in origins:
        history = bars.iloc[: origin + 1].copy()
        target_slice = bars.iloc[origin + 1 : origin + steps + 1]
        if len(target_slice) != steps:
            continue
        target_dates = [pd.Timestamp(value) for value in target_slice["timestamp"]]
        actual_path = target_slice["close"].to_numpy(dtype=float)
        validation_actuals.append(float(actual_path[-1]))
        validation_references.append(float(closes[origin]))
        validation_timestamps.append(target_dates[-1].isoformat())
        validation_actual_paths.append(actual_path.tolist())
        validation_reference_paths.append([float(closes[origin])] * steps)

        for name in available_names:
            if name in failures:
                continue
            try:
                path = _model_forecast(name, history, steps, target_dates)
                if (
                    len(path) != steps
                    or not np.isfinite(path).all()
                    or np.any(np.asarray(path) <= 0)
                ):
                    raise ValueError("Model returned invalid price forecasts")
                predictions[name].append(path)
                for step, (actual, predicted) in enumerate(
                    zip(actual_path, path, strict=True)
                ):
                    residuals[name][step].append(float(actual - predicted))
            except Exception as exc:
                logger.warning(
                    "Model %s failed at validation origin %s: %s", name, origin, exc
                )
                failures[name] = str(exc)[:240]

    if not validation_actuals:
        raise ValueError("Could not construct chronological validation folds")

    for name in available_names:
        if name in failures or len(predictions[name]) != len(validation_actuals):
            results[name] = {
                **results[name],
                "eligible": False,
                "status": "failed",
                "reason": failures.get(
                    name, "Model missed one or more chronological validation folds"
                ),
            }
            continue
        endpoint_predictions = [path[-1] for path in predictions[name]]
        results[name]["metrics"] = _metrics(
            validation_actuals,
            endpoint_predictions,
            validation_references,
        )
        results[name]["horizon_metrics"] = _metrics(
            [
                value
                for path in validation_actual_paths
                for value in path
            ],
            [
                value
                for path in predictions[name]
                for value in path
            ],
            [
                value
                for path in validation_reference_paths
                for value in path
            ],
        )
        results[name]["validation_predictions"] = [
            {
                "timestamp": timestamp,
                "actual": actual,
                "predicted": predicted,
            }
            for timestamp, actual, predicted in zip(
                validation_timestamps,
                validation_actuals,
                endpoint_predictions,
                strict=True,
            )
        ]

    naive_mae = results["naive"].get("horizon_metrics", {}).get("mae")
    if naive_mae is None:
        raise ValueError("Naive baseline could not be evaluated on validation folds")

    ranked = sorted(
        (
            name
            for name, result in results.items()
            if result.get("eligible") and "horizon_metrics" in result
        ),
        key=lambda name: results[name]["horizon_metrics"]["mae"],
    )
    winners = [
        name
        for name in ranked
        if name == "naive"
        or results[name]["horizon_metrics"]["mae"] < naive_mae
    ]
    recommended = winners[0] if winners else "naive"
    model_name = recommended if selected_model == "auto" else selected_model
    if model_name not in results or not results[model_name].get("eligible"):
        reason = results.get(model_name, {}).get("reason")
        raise ValueError(reason or f"Model {model_name} is unavailable for this data")

    test_actual_paths: list[list[float]] = []
    test_reference_paths: list[list[float]] = []
    test_predictions: dict[str, list[list[float]]] = {
        name: [] for name in available_names if results[name].get("eligible")
    }
    test_failures: dict[str, str] = {}
    for origin in test_origins:
        history = bars.iloc[: origin + 1].copy()
        target_slice = bars.iloc[origin + 1 : origin + steps + 1]
        if len(target_slice) != steps:
            continue
        target_dates = [pd.Timestamp(value) for value in target_slice["timestamp"]]
        actual_path = target_slice["close"].to_numpy(dtype=float)
        test_actual_paths.append(actual_path.tolist())
        test_reference_paths.append([float(closes[origin])] * steps)
        for name in test_predictions:
            if name in test_failures:
                continue
            try:
                path = _model_forecast(name, history, steps, target_dates)
                if (
                    len(path) != steps
                    or not np.isfinite(path).all()
                    or np.any(np.asarray(path) <= 0)
                ):
                    raise ValueError("Model returned invalid price forecasts")
                test_predictions[name].append(path)
            except Exception as exc:
                logger.warning(
                    "Model %s failed at final test origin %s: %s",
                    name,
                    origin,
                    exc,
                )
                test_failures[name] = str(exc)[:240]

    for name in test_predictions:
        if (
            name in test_failures
            or len(test_predictions[name]) != len(test_actual_paths)
        ):
            results[name]["test_status"] = "failed"
            results[name]["test_reason"] = test_failures.get(
                name, "Model missed one or more chronological test folds"
            )
            continue
        results[name]["test_status"] = "evaluated"
        results[name]["test_metrics"] = _metrics(
            [
                value
                for path in test_actual_paths
                for value in path
            ],
            [
                value
                for path in test_predictions[name]
                for value in path
            ],
            [
                value
                for path in test_reference_paths
                for value in path
            ],
        )
    test_naive_mae = results["naive"].get("test_metrics", {}).get("mae")
    if test_naive_mae is not None:
        for result in results.values():
            if "test_metrics" in result:
                result["test_baseline_mae"] = float(test_naive_mae)
                result["test_beats_naive"] = bool(
                    result["test_metrics"]["mae"] < test_naive_mae
                )

    final_history = bars.copy()
    forecast_path = _model_forecast(model_name, final_history, steps, future_dates)
    feature_importance = _feature_importance(model_name, final_history)
    selected_feature_names = (
        _selected_feature_names(final_history)
        if model_name in {
            spec.name for spec in MODEL_SPECS if spec.family == "Machine learning"
        }
        and model_name != "lstm"
        else []
    )
    selected_uses_technical_factors = model_name in {
        spec.name for spec in MODEL_SPECS if spec.family == "Machine learning"
    } and model_name != "lstm"
    intervals = _prediction_intervals(forecast_path, residuals[model_name])
    future_forecasts = [
        {
            "timestamp": timestamp.isoformat(),
            "predicted_close": float(predicted),
            **interval,
        }
        for timestamp, predicted, interval in zip(
            future_dates, forecast_path, intervals, strict=True
        )
    ]
    last_timestamp = pd.Timestamp(bars["timestamp"].iloc[-1])
    last_close = float(closes[-1])

    for name, result in results.items():
        result["baseline_mae"] = float(naive_mae)
        result["beats_naive"] = bool(
            result.get("horizon_metrics", {}).get("mae", float("inf")) < naive_mae
        )

    return {
        "symbol": symbol,
        "interval": interval,
        "exchange": exchange_name,
        "currency": raw.attrs.get("currency"),
        "horizon": int(horizon_key) if horizon_key.isdigit() else horizon_key,
        "horizon_label": horizon_label,
        "horizon_sessions": steps,
        "forecast_target": "daily_close",
        "selection_objective": (
            "Lowest full-path walk-forward validation MAE; non-naive recommendations "
            "must beat the naive baseline. Final test metrics are held out from selection."
        ),
        "selected_model": model_name,
        "recommended_model": recommended,
        "selection_mode": "automatic" if selected_model == "auto" else "manual",
        "selected_model_beats_naive": bool(
            results[model_name].get("horizon_metrics", {}).get("mae", float("inf"))
            < naive_mae
        ),
        "validation_folds": len(validation_actuals),
        "test_folds": len(test_actual_paths),
        "evaluation": {
            "validation_end": bars["timestamp"].iloc[test_start - 1].isoformat(),
            "test_start": bars["timestamp"].iloc[test_start].isoformat(),
            "test_end": bars["timestamp"].iloc[-1].isoformat(),
            "test_fraction": TEST_FRACTION,
            "test_metrics_used_for_selection": False,
            "evaluation_method": (
                "Expanding-window walk-forward validation on the earlier history "
                "and a later chronological rolling-origin test period."
            ),
        },
        "last_close": last_close,
        "last_bar_timestamp": last_timestamp.isoformat(),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "forecast": {
            "timestamp": future_forecasts[-1]["timestamp"],
            "predicted_close": future_forecasts[-1]["predicted_close"],
            "interval_method": (
                "80% empirical absolute-error range using walk-forward validation residuals; "
                "not a guarantee of future coverage"
            ),
            "future": future_forecasts,
        },
        "models": results,
        "ranking": ranked,
        "factor_analysis": {
            "price_and_technical": {
                "available": True,
                "source": "Yahoo Finance daily OHLCV",
                "features": list(TECHNICAL_FACTOR_NAMES),
                "used_by_selected_model": (
                    selected_feature_names
                    if selected_uses_technical_factors
                    else ["historical closing prices"]
                ),
                "feature_importance": feature_importance,
                "limitation": (
                    "Technical and volume features are computed from observations "
                    "available at each training origin. During recursive forecasts, "
                    "the last observed OHLCV-derived factors are held fixed; future "
                    "OHLCV is not fabricated. Daily VWAP is an OHLCV proxy, not "
                    "intraday transaction VWAP."
                ),
            },
            "volume_liquidity": {
                "available": has_full_ohlcv,
                "features": [
                    "volume_change",
                    "volume_zscore_20",
                    "obv_change_10",
                    "turnover_deviation_20",
                ]
                if has_full_ohlcv
                else [],
                "used_by_selected_model": (
                    [
                        feature
                        for feature in selected_feature_names
                        if feature
                        in {
                            "volume_change",
                            "volume_zscore_20",
                            "obv_change_10",
                            "turnover_deviation_20",
                        }
                    ]
                    if selected_uses_technical_factors
                    else []
                ),
                "limitation": (
                    "Volume, OBV, and approximate price-times-volume turnover are "
                    "available. Historical bid-ask spread and order-book depth are "
                    "not provided by the configured daily-bar source."
                    if has_full_ohlcv
                    else "Daily OHLCV inputs are incomplete."
                ),
            },
            "fundamentals": {
                "available": False,
                "used_by_selected_model": [],
                "limitation": (
                    "Current provider fundamentals are not historical point-in-time "
                    "snapshots and are excluded to avoid look-ahead leakage."
                ),
            },
            "news_sentiment": {
                "available": False,
                "used_by_selected_model": [],
                "limitation": (
                    "Historical point-in-time news and sentiment are not available "
                    "for every validation origin; current sentiment is not used as "
                    "a historical forecasting feature."
                ),
            },
            "market_and_macro": {
                "available": benchmark_available,
                "features": list(MARKET_FACTOR_NAMES)
                if benchmark_available
                else [],
                "used_by_selected_model": (
                    [
                        feature
                        for feature in selected_feature_names
                        if feature in MARKET_FACTOR_NAMES
                    ]
                    if selected_uses_technical_factors and benchmark_available
                    else []
                ),
                "limitation": (
                    "Lagged broad-market benchmark returns and volatility are aligned "
                    "to available closes and may be used by supervised models. "
                    "Point-in-time sector, FX, interest-rate, inflation, GDP, and "
                    "volatility-index inputs remain unavailable."
                    if benchmark_available
                    else "Historical aligned benchmark data is unavailable for this request. "
                    "Sector, FX, interest-rate, inflation, GDP, and volatility-index "
                    "inputs are not configured."
                ),
            },
            "corporate_events": {
                "available": False,
                "used_by_selected_model": [],
                "limitation": (
                    "Timestamped point-in-time event history is not integrated into "
                    "forecast features; current or retrospectively observed events "
                    "are not used as historical inputs."
                ),
            },
        },
        "data_note": (
            f"Future dates use the {exchange_name} exchange session calendar. "
            "The 80% ranges are empirically calibrated from same-horizon walk-forward "
            "forecast errors and are not guarantees; unavailable ranges indicate too "
            "few validation residuals."
        ),
    }
