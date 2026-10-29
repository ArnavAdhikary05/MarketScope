
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.providers.yahoo import fetch_stock_data
from app.services.features import create_features

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

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


def train_model(symbol="AAPL", interval="1d"):
    period = "60d" if interval != "1d" else "5y"

    raw = fetch_stock_data(
        symbol, period=period, interval=interval
    )
    data = create_features(raw).dropna().copy()

    if len(data) < 100:
        raise ValueError("Not enough usable observations to train")

    # Chronological split. Never shuffle time-series rows.
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[FEATURES], train["target"])

    predictions = model.predict(test[FEATURES])
    actual = test["target"].to_numpy()

    mae = mean_absolute_error(actual, predictions)
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))

    # Baseline: predict that the next close equals the current close.
    baseline = test["close"].to_numpy()
    baseline_mae = mean_absolute_error(actual, baseline)

    model_path = ARTIFACTS / f"{symbol.replace('.', '_')}_{interval}.joblib"
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "symbol": symbol,
            "interval": interval,
        },
        model_path,
    )

    return {
        "symbol": symbol,
        "interval": interval,
        "train_rows": len(train),
        "test_rows": len(test),
        "mae": float(mae),
        "rmse": rmse,
        "baseline_mae": float(baseline_mae),
        "beats_baseline": bool(mae < baseline_mae),
        "model_path": str(model_path),
    }


if __name__ == "__main__":
    print(train_model("AAPL", "1d"))
