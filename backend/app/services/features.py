
import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "close", "volume"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Market data is missing required columns: {missing}")

    result = df.copy().sort_values("timestamp").reset_index(drop=True)

    close = result["close"].astype(float)
    volume = result["volume"].astype(float)

    # Historical returns
    result["return_1"] = close.pct_change(1)
    result["return_5"] = close.pct_change(5)

    # Moving averages
    result["sma_5"] = close.rolling(5).mean()
    result["sma_20"] = close.rolling(20).mean()
    result["ema_12"] = close.ewm(span=12, adjust=False).mean()

    # Volatility from past bar returns
    result["volatility_20"] = (
        result["return_1"].rolling(20).std()
    )

    # RSI (14-period)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    result["rsi_14"] = 100 - (100 / (1 + rs))
    result.loc[(loss == 0) & (gain > 0), "rsi_14"] = 100
    result.loc[(gain == 0) & (loss > 0), "rsi_14"] = 0
    result.loc[(gain == 0) & (loss == 0), "rsi_14"] = 50

    # Volume change
    result["volume_change"] = volume.pct_change()

    # Forecast next observed close, not a randomly shuffled target
    result["target"] = close.shift(-1)

    return result.replace([np.inf, -np.inf], np.nan)
