from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Final

import pandas as pd
import pandas_market_calendars as market_calendars
import yfinance as yf

logger = logging.getLogger(__name__)

CALENDAR_ALIASES: Final[dict[str, str]] = {
    "NYSE": "NYSE",
    "NASDAQ": "NASDAQ",
    "NSE": "NSE",
    "BSE": "BSE",
    "LSE": "LSE",
    "TSX": "TSX",
    "TSXV": "TSXV",
    "ASX": "ASX",
    "HKEX": "HKEX",
    "JPX": "JPX",
    "XETR": "XETR",
}

BENCHMARK_SERIES: Final[dict[str, tuple[str, ...]]] = {
    "NYSE": ("^GSPC",),
    "NASDAQ": ("^IXIC", "^GSPC"),
    "NSE": ("^NSEI", "^BSESN"),
    "BSE": ("^BSESN", "^NSEI"),
    "LSE": ("^FTSE",),
    "TSX": ("^GSPTSE",),
    "TSXV": ("^GSPTSE",),
    "ASX": ("^AXJO",),
    "HKEX": ("^HSI",),
    "JPX": ("^N225",),
    "XETR": ("^GDAXI",),
}
BENCHMARK_SYMBOLS: Final[dict[str, str]] = {
    exchange: symbols[0] for exchange, symbols in BENCHMARK_SERIES.items()
}

YAHOO_EXCHANGES: Final[dict[str, str]] = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NasdaqGS": "NASDAQ",
    "NasdaqGM": "NASDAQ",
    "NasdaqCM": "NASDAQ",
    "NYQ": "NYSE",
    "NYSE": "NYSE",
    "ASE": "NYSE",
    "BSE": "BSE",
    "BOM": "BSE",
    "NSI": "NSE",
    "NSE": "NSE",
    "LSE": "LSE",
    "LON": "LSE",
    "TOR": "TSX",
    "TSX": "TSX",
    "TSXV": "TSXV",
    "ASX": "ASX",
    "HKG": "HKEX",
    "HKSE": "HKEX",
    "JPX": "JPX",
    "TYO": "JPX",
    "GER": "XETR",
    "XETR": "XETR",
}


def resolve_exchange(symbol: str, exchange: str = "auto") -> str:
    """Resolve an explicit exchange or use symbol suffix/provider metadata."""
    normalized_exchange = exchange.strip().upper()
    if normalized_exchange != "AUTO":
        if normalized_exchange not in CALENDAR_ALIASES:
            supported = ", ".join(CALENDAR_ALIASES)
            raise ValueError(f"Exchange must be auto or one of: {supported}")
        return CALENDAR_ALIASES[normalized_exchange]

    normalized_symbol = symbol.strip().upper()
    if normalized_symbol.endswith(".NS"):
        return "NSE"
    if normalized_symbol.endswith(".BO"):
        return "BSE"
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,30}", normalized_symbol):
        raise ValueError("Invalid stock symbol for exchange lookup")

    try:
        info = yf.Ticker(normalized_symbol).get_info()
    except Exception as exc:
        logger.info("Exchange metadata lookup failed for %s: %s", normalized_symbol, exc)
        raise ValueError(
            "Could not automatically identify this ticker's exchange. "
            "Select a supported exchange explicitly and retry."
        ) from exc

    exchange_code = str(info.get("exchange", "")).strip()
    resolved = YAHOO_EXCHANGES.get(exchange_code)
    if resolved is None:
        exchange_name = str(info.get("fullExchangeName", "")).strip().lower()
        if "nasdaq" in exchange_name:
            resolved = "NASDAQ"
        elif "new york stock" in exchange_name or exchange_name == "nyse":
            resolved = "NYSE"
        elif exchange_name == "bombay stock exchange":
            resolved = "BSE"
        elif exchange_name == "national stock exchange of india":
            resolved = "NSE"
        elif "london stock exchange" in exchange_name:
            resolved = "LSE"
        elif "toronto stock exchange" in exchange_name:
            resolved = "TSX"
        elif "tsx venture" in exchange_name:
            resolved = "TSXV"
        elif "australian securities exchange" in exchange_name:
            resolved = "ASX"
        elif "hong kong stock exchange" in exchange_name:
            resolved = "HKEX"
        elif "tokyo stock exchange" in exchange_name:
            resolved = "JPX"
        elif "xetra" in exchange_name or "frankfurt stock exchange" in exchange_name:
            resolved = "XETR"

    if resolved is None:
        raise ValueError(
            "The ticker's exchange is unsupported or could not be determined. "
            "Choose a supported exchange explicitly."
        )
    return resolved


def get_exchange_calendar(exchange: str):
    normalized = exchange.strip().upper()
    if normalized not in CALENDAR_ALIASES:
        raise ValueError(f"Unsupported exchange calendar: {exchange}")
    try:
        return market_calendars.get_calendar(CALENDAR_ALIASES[normalized])
    except RuntimeError as exc:
        raise ValueError(f"No trading calendar is available for {normalized}") from exc


def future_session_closes(
    last_observation: pd.Timestamp,
    exchange: str,
    horizon_end: date,
) -> list[pd.Timestamp]:
    """Return real scheduled session close timestamps through an inclusive date."""
    calendar = get_exchange_calendar(exchange)
    start_date = last_observation.date() + timedelta(days=1)
    if start_date > horizon_end:
        raise ValueError("Forecast horizon ends before the next exchange session")
    schedule = calendar.schedule(start_date=start_date, end_date=horizon_end)
    if schedule.empty:
        raise ValueError(
            f"No {exchange} trading sessions fall within the selected horizon"
        )
    return [
        pd.Timestamp(close)
        for close in pd.to_datetime(schedule["market_close"], utc=True).tolist()
    ]
