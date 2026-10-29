from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from app.services.features import create_features

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
except ImportError as exc:
    Prophet = None
    PROPHET_IMPORT_ERROR = str(exc)
else:
    PROPHET_IMPORT_ERROR = ""

FEATURES = [
    "return_1",
    "return_5",
    "sma_5",
    "sma_20",
    "ema_12",
    "volatility_20",
    "rsi_14",
    "volume_change",
]


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
        40,
        lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    ),
    ModelSpec(
        "random_forest",
        "Machine learning",
        60,
        lambda: RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    ModelSpec(
        "extra_trees",
        "Machine learning",
        60,
        lambda: ExtraTreesRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    ),
    ModelSpec(
        "hist_gradient_boosting",
        "Machine learning",
        60,
        lambda: HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        ),
    ),
    ModelSpec("arima", "Statistical", 40),
    ModelSpec("prophet", "Statistical", 60),
)


def _future_timestamp(last_timestamp: pd.Timestamp, interval: str, horizon: int) -> pd.Timestamp:
    if interval == "1d":
        return last_timestamp + pd.offsets.BDay(horizon)

    minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60}
    if interval not in minutes:
        raise ValueError(f"Unsupported interval: {interval}")
    return last_timestamp + pd.Timedelta(minutes=minutes[interval] * horizon)


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


def _prophet_forecast(
    history: pd.DataFrame,
    future_timestamps: list[pd.Timestamp],
    interval: str,
) -> float:
    if Prophet is None:
        raise RuntimeError(f"Prophet is unavailable: {PROPHET_IMPORT_ERROR}")

    training = history[["timestamp", "close"]].rename(
        columns={"timestamp": "ds", "close": "y"}
    )
    training["ds"] = pd.to_datetime(training["ds"], utc=True).dt.tz_localize(None)
    future = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                pd.Series(future_timestamps), utc=True
            ).dt.tz_localize(None)
        }
    )
    coverage = training["ds"].max() - training["ds"].min()
    use_daily = interval != "1d" and coverage >= pd.Timedelta(days=2)
    use_weekly = coverage >= pd.Timedelta(days=21)
    use_yearly = coverage >= pd.Timedelta(days=730)
    model = Prophet(
        daily_seasonality=use_daily,
        weekly_seasonality=use_weekly,
        yearly_seasonality=use_yearly,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,
    )
    model.fit(training)
    return float(model.predict(future)["yhat"].iloc[-1])


def compare_models(
    raw: pd.DataFrame,
    symbol: str,
    interval: str,
    horizon: int = 1,
    selected_model: str = "auto",
) -> dict:
    if interval not in {"1d", "1m", "5m", "15m", "30m", "60m", "1h"}:
        raise ValueError(f"Unsupported interval: {interval}")
    if horizon < 1 or horizon > 20:
        raise ValueError("Horizon must be between 1 and 20 bars")
    if selected_model != "auto" and selected_model not in {
        "naive",
        "seasonal_naive",
        *(spec.name for spec in MODEL_SPECS),
    }:
        raise ValueError(f"Unknown model: {selected_model}")

    bars = raw.copy().sort_values("timestamp").reset_index(drop=True)
    if len(bars) < 80:
        raise ValueError("At least 80 market bars are required for comparison")

    frame = create_features(bars)
    target_column = f"target_{horizon}"
    frame[target_column] = frame["close"].shift(-horizon)
    usable = frame.dropna(subset=FEATURES + [target_column]).copy()
    if len(usable) < 50:
        raise ValueError("Insufficient usable observations after feature preparation")

    # Forecast origins are held out chronologically; labels are purged so no
    # training target reaches beyond the information available at each origin.
    earliest_origin = max(int(len(usable) * 0.6), 60)
    latest_origin = len(bars) - horizon - 1
    if latest_origin <= earliest_origin:
        raise ValueError("Insufficient data for the requested forecast horizon")
    origin_count = min(5, latest_origin - earliest_origin + 1)
    origins = np.unique(
        np.linspace(earliest_origin, latest_origin, num=origin_count, dtype=int)
    )

    results: dict[str, dict] = {}
    predictions: dict[str, list[float]] = {}
    actual_values: list[float] = []
    reference_values: list[float] = []
    target_timestamps: list[str] = []
    seasonal_period = {
        "1d": 5,
        "1m": 390,
        "5m": 78,
        "15m": 26,
        "30m": 13,
        "60m": 7,
        "1h": 7,
    }[interval]
    seasonal_eligible = bool(np.all(origins >= seasonal_period))
    if not seasonal_eligible:
        results["seasonal_naive"] = {
            "family": "Baseline",
            "eligible": False,
            "status": "insufficient_data",
            "reason": (
                f"Requires at least {seasonal_period} earlier observations "
                "at every validation origin"
            ),
        }

    for origin in origins:
        training = usable.loc[usable.index <= origin - horizon]
        if training.empty:
            continue

        validation_target = float(bars.loc[origin + horizon, "close"])
        reference = float(bars.loc[origin, "close"])
        actual_values.append(validation_target)
        reference_values.append(reference)
        target_timestamps.append(
            pd.Timestamp(bars.loc[origin + horizon, "timestamp"]).isoformat()
        )

        predictions.setdefault("naive", []).append(reference)
        if seasonal_eligible:
            seasonal_value = float(bars.loc[origin - seasonal_period, "close"])
            predictions.setdefault("seasonal_naive", []).append(seasonal_value)

        row = frame.loc[[origin], FEATURES]
        for spec in MODEL_SPECS:
            if len(training) < spec.minimum_rows:
                results.setdefault(
                    spec.name,
                    {
                        "family": spec.family,
                        "eligible": False,
                        "status": "insufficient_data",
                        "reason": f"Requires at least {spec.minimum_rows} training rows per fold",
                    },
                )
                continue

            if spec.name == "prophet" and Prophet is None:
                results[spec.name] = {
                    "family": spec.family,
                    "eligible": False,
                    "status": "unavailable",
                    "reason": "Install the Prophet package to enable this model",
                }
                continue

            try:
                if spec.name == "arima":
                    history = bars.loc[:origin, "close"].astype(float)
                    prediction = float(
                        ARIMA(history, order=(1, 1, 1)).fit().forecast(steps=horizon).iloc[-1]
                    )
                elif spec.name == "prophet":
                    future_timestamps = [
                        pd.Timestamp(value)
                        for value in bars.loc[origin + 1 : origin + horizon, "timestamp"]
                    ]
                    prediction = _prophet_forecast(
                        bars.loc[:origin], future_timestamps, interval
                    )
                else:
                    model = spec.factory()
                    model.fit(training[FEATURES], training[target_column])
                    prediction = float(model.predict(row)[0])
                if not np.isfinite(prediction) or prediction <= 0:
                    raise ValueError("Model produced a non-positive or non-finite price")
                predictions.setdefault(spec.name, []).append(prediction)
                results.setdefault(
                    spec.name,
                    {
                        "family": spec.family,
                        "eligible": True,
                        "status": "evaluated",
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Model %s failed at validation origin %s: %s",
                    spec.name,
                    origin,
                    exc,
                )
                results[spec.name] = {
                    "family": spec.family,
                    "eligible": False,
                    "status": "failed",
                    "reason": str(exc)[:240],
                }

    if not actual_values:
        raise ValueError("Could not construct chronological validation folds")

    results["naive"] = {
        "family": "Baseline",
        "eligible": True,
        "status": "evaluated",
    }
    if seasonal_eligible:
        results["seasonal_naive"] = {
            "family": "Baseline",
            "eligible": True,
            "status": "evaluated",
        }

    for name, values in predictions.items():
        if len(values) != len(actual_values):
            results[name] = {
                **results[name],
                "eligible": False,
                "status": "failed",
                "reason": "Model did not produce predictions for every validation fold",
            }
            continue
        results[name]["metrics"] = _metrics(actual_values, values, reference_values)
        results[name]["validation_predictions"] = [
            {"timestamp": timestamp, "actual": actual, "predicted": predicted}
            for timestamp, actual, predicted in zip(
                target_timestamps, actual_values, values, strict=True
            )
        ]

    naive_mae = results["naive"]["metrics"]["mae"]
    ranked = sorted(
        (
            name
            for name, result in results.items()
            if result["eligible"] and "metrics" in result
        ),
        key=lambda name: results[name]["metrics"]["mae"],
    )
    eligible_winners = [
        name
        for name in ranked
        if name == "naive" or results[name]["metrics"]["mae"] < naive_mae
    ]
    recommended = eligible_winners[0] if eligible_winners else "naive"
    model_name = recommended if selected_model == "auto" else selected_model

    if model_name not in results or not results[model_name].get("eligible"):
        raise ValueError(
            results.get(model_name, {}).get("reason", f"Model {model_name} is unavailable")
        )

    last_features = frame.iloc[[-1]][FEATURES]
    last_close = float(bars["close"].iloc[-1])
    if model_name == "naive":
        forecast_value = last_close
    elif model_name == "seasonal_naive":
        forecast_value = float(bars["close"].iloc[-1 - seasonal_period])
    elif model_name == "arima":
        fitted = ARIMA(bars["close"].astype(float), order=(1, 1, 1)).fit()
        forecast_value = float(fitted.forecast(steps=horizon).iloc[-1])
    elif model_name == "prophet":
        future_timestamp = _future_timestamp(
            pd.Timestamp(bars["timestamp"].iloc[-1]), interval, horizon
        )
        forecast_value = _prophet_forecast(
            bars, [future_timestamp], interval
        )
    else:
        final_training = usable
        final_model = next(spec for spec in MODEL_SPECS if spec.name == model_name).factory()
        final_model.fit(final_training[FEATURES], final_training[target_column])
        forecast_value = float(final_model.predict(last_features)[0])
    if not np.isfinite(forecast_value) or forecast_value <= 0:
        raise ValueError("Selected model produced a non-positive or non-finite price forecast")

    last_timestamp = pd.Timestamp(bars["timestamp"].iloc[-1])
    forecast_timestamp = _future_timestamp(last_timestamp, interval, horizon)
    for name, result in results.items():
        result["baseline_mae"] = naive_mae
        result["beats_naive"] = bool(
            result.get("metrics", {}).get("mae", float("inf")) < naive_mae
        )

    return {
        "symbol": symbol,
        "interval": interval,
        "horizon": horizon,
        "selection_objective": "Lowest chronological validation MAE; advanced models must beat naive",
        "selected_model": model_name,
        "recommended_model": recommended,
        "selection_mode": "automatic" if selected_model == "auto" else "manual",
        "selected_model_beats_naive": bool(
            results[model_name].get("metrics", {}).get("mae", float("inf")) < naive_mae
        ),
        "validation_folds": len(actual_values),
        "last_close": last_close,
        "last_bar_timestamp": last_timestamp.isoformat(),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "forecast": {
            "timestamp": forecast_timestamp.isoformat(),
            "predicted_close": forecast_value,
        },
        "models": results,
        "ranking": ranked,
        "data_note": (
            "Daily future timestamps use weekdays and do not account for exchange holidays. "
            "Forecasts are experimental and may use delayed provider data."
        ),
    }
