
import logging
import re

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


FALLBACK_CURRENCIES = {
    ".NS": "INR",
    ".BO": "INR",
    ".L": "GBp",
    ".TO": "CAD",
    ".V": "CAD",
    ".T": "JPY",
    ".AX": "AUD",
    ".HK": "HKD",
    ".SI": "SGD",
    ".KS": "KRW",
    ".KQ": "KRW",
    ".SS": "CNY",
    ".SZ": "CNY",
    ".SA": "BRL",
    ".PA": "EUR",
    ".DE": "EUR",
    ".AS": "EUR",
    ".MI": "EUR",
    ".MC": "EUR",
    ".BR": "EUR",
    ".SW": "CHF",
}


def _quote_currency(ticker: yf.Ticker, symbol: str) -> str | None:
    try:
        fast_info = ticker.fast_info
        currency = getattr(fast_info, "currency", None)
        if not currency and hasattr(fast_info, "get"):
            currency = fast_info.get("currency")
        if isinstance(currency, str) and re.fullmatch(r"[A-Za-z]{3}", currency):
            return currency.upper()
        if currency == "GBp":
            return "GBp"
    except Exception:
        logger.info("Yahoo Finance currency metadata unavailable for %s", symbol)

    normalized_symbol = symbol.upper()
    return next(
        (
            currency
            for suffix, currency in FALLBACK_CURRENCIES.items()
            if normalized_symbol.endswith(suffix)
        ),
        "USD" if "." not in normalized_symbol else None,
    )


def fetch_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data for research and educational use."""

    allowed_intervals = {
        "1m", "5m", "15m", "30m", "60m", "1h", "1d"
    }
    allowed_periods = {
        "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max", "60d"
    }

    if interval not in allowed_intervals:
        raise ValueError(f"Unsupported interval: {interval}")
    if period not in allowed_periods:
        raise ValueError(f"Unsupported history period: {period}")

    if not re.fullmatch(r"[A-Za-z0-9.^=-]{1,30}", symbol or ""):
        raise ValueError("Invalid stock symbol")

    ticker = yf.Ticker(symbol)
    df = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=True,
        actions=False,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {symbol}. "
            "Check the symbol, interval, or provider availability."
        )

    df = df.rename_axis("timestamp").reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError("Provider returned an unexpected data schema")

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"])
    df = df.dropna(subset=["open", "high", "low", "volume"])
    valid_ohlc = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["low"] <= df["high"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["volume"] >= 0)
    )
    df = df.loc[valid_ohlc]
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    if df.empty:
        raise ValueError(f"No valid OHLCV observations returned for {symbol}")

    df.attrs["currency"] = _quote_currency(ticker, symbol)
    df.attrs["source"] = "Yahoo Finance"
    return df
