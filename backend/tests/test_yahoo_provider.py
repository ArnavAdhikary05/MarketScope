import unittest
from unittest.mock import Mock, patch

import pandas as pd

from app.providers.yahoo import _quote_currency, fetch_stock_data


class YahooProviderTests(unittest.TestCase):
    def test_currency_uses_provider_metadata_then_symbol_suffix_fallback(self):
        ticker = Mock()
        ticker.fast_info = {"currency": "JPY"}
        self.assertEqual(_quote_currency(ticker, "7203.T"), "JPY")

        ticker.fast_info = {}
        self.assertEqual(_quote_currency(ticker, "7203.T"), "JPY")
        self.assertEqual(_quote_currency(ticker, "RELIANCE.NS"), "INR")
        self.assertEqual(_quote_currency(ticker, "VOD.L"), "GBp")

    def test_history_returns_provider_currency_and_normalized_ohlcv(self):
        ticker = Mock()
        ticker.history.return_value = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2026-10-28"], name="Date"),
        )
        ticker.fast_info = {"currency": "CAD"}
        with patch("app.providers.yahoo.yf.Ticker", return_value=ticker):
            result = fetch_stock_data("SHOP.TO", period="1mo", interval="1d")

        self.assertEqual(result.attrs["currency"], "CAD")
        self.assertEqual(result.attrs["source"], "Yahoo Finance")
        self.assertEqual(
            list(result.columns),
            ["timestamp", "open", "high", "low", "close", "volume"],
        )
        self.assertEqual(float(result["close"].iloc[0]), 101.0)

    def test_invalid_ticker_is_rejected_before_provider_request(self):
        with patch("app.providers.yahoo.yf.Ticker") as ticker:
            with self.assertRaisesRegex(ValueError, "Invalid stock symbol"):
                fetch_stock_data("<invalid>")
        ticker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
