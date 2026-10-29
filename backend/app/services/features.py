
import numpy as np
import pandas as pd


TECHNICAL_FACTOR_NAMES = (
    "gap_return",
    "intraday_range",
    "volatility_5",
    "volatility_20",
    "sma_5_deviation",
    "sma_20_deviation",
    "ema_12_deviation",
    "rsi_14",
    "macd_histogram",
    "bollinger_position",
    "atr_14",
    "vwap_20_deviation",
    "volume_change",
    "volume_zscore_20",
    "obv_change_10",
    "momentum_10",
    "momentum_20",
    "support_distance_20",
    "resistance_distance_20",
    "turnover_deviation_20",
)


def create_technical_factors(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Market data is missing columns needed for technical factors: "
            f"{sorted(required - set(df.columns))}"
        )

    result = df.copy().sort_values("timestamp").reset_index(drop=True)
    open_price = pd.to_numeric(result["open"], errors="coerce").astype(float)
    high = pd.to_numeric(result["high"], errors="coerce").astype(float)
    low = pd.to_numeric(result["low"], errors="coerce").astype(float)
    close = pd.to_numeric(result["close"], errors="coerce").astype(float)
    volume = pd.to_numeric(result["volume"], errors="coerce").astype(float)
    returns = close.pct_change()
    typical_price = (high + low + close) / 3
    average_gain = close.diff().clip(lower=0).rolling(14, min_periods=14).mean()
    average_loss = -close.diff().clip(upper=0).rolling(14, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi.loc[(average_loss == 0) & (average_gain > 0)] = 100
    rsi.loc[(average_gain == 0) & (average_loss > 0)] = 0
    rsi.loc[(average_gain == 0) & (average_loss == 0)] = 50
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(
        span=26, adjust=False
    ).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    rolling_mean = close.rolling(20, min_periods=20).mean()
    rolling_std = close.rolling(20, min_periods=20).std(ddof=0)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std(ddof=0)
    turnover = close * volume
    turnover_mean = turnover.rolling(20, min_periods=20).mean()
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    vwap_volume = volume.rolling(20, min_periods=20).sum()
    vwap = (
        (typical_price * volume).rolling(20, min_periods=20).sum()
        / vwap_volume.replace(0, np.nan)
    )

    factors = pd.DataFrame(
        {
            "gap_return": open_price / previous_close - 1,
            "intraday_range": (high - low) / close,
            "volatility_5": returns.rolling(5, min_periods=5).std(),
            "volatility_20": returns.rolling(20, min_periods=20).std(),
            "sma_5_deviation": close / close.rolling(5, min_periods=5).mean() - 1,
            "sma_20_deviation": close / rolling_mean - 1,
            "ema_12_deviation": close / close.ewm(span=12, adjust=False).mean() - 1,
            "rsi_14": rsi / 100,
            "macd_histogram": (macd - macd_signal) / close,
            "bollinger_position": (close - rolling_mean) / (2 * rolling_std),
            "atr_14": true_range.rolling(14, min_periods=14).mean() / close,
            "vwap_20_deviation": close / vwap - 1,
            "volume_change": volume.pct_change(),
            "volume_zscore_20": (volume - volume_mean) / volume_std.replace(0, np.nan),
            "obv_change_10": (obv - obv.shift(10))
            / volume.rolling(10, min_periods=10).sum().replace(0, np.nan),
            "momentum_10": close.pct_change(10),
            "momentum_20": close.pct_change(20),
            "support_distance_20": close
            / low.rolling(20, min_periods=20).min()
            - 1,
            "resistance_distance_20": high.rolling(20, min_periods=20).max()
            / close
            - 1,
            "turnover_deviation_20": turnover
            / turnover_mean.replace(0, np.nan)
            - 1,
        }
    )
    return factors.replace([np.inf, -np.inf], np.nan)


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
