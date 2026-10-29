import json
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from app.services.features import (
    TECHNICAL_FACTOR_NAMES,
    create_technical_factors,
)
from app.services.news_sentiment import (
    _cache,
    _cache_lock,
    _aggregate_sentiment,
    _classify_news,
    _correlation_pairs,
    _correlation_result,
    _fetch_news,
    _utc_datetime,
    analyze_stock_sentiment,
)


def news_item(identifier: str, published_at: str) -> dict:
    return {
        "id": identifier,
        "title": f"Company update {identifier}",
        "summary": "",
        "url": f"https://example.test/{identifier}",
        "source": "Example News",
        "published_at": published_at,
    }


class SentimentAnalysisTests(unittest.TestCase):
    def setUp(self):
        with _cache_lock:
            _cache.clear()

    def test_ticker_news_falls_back_to_yahoo_when_finviz_is_empty(self):
        fallback_item = news_item("yahoo", "2026-10-28T15:00:00+00:00")
        with (
            patch("app.services.news_sentiment._finviz_news", return_value=[]),
            patch(
                "app.services.news_sentiment._yahoo_news",
                return_value=[fallback_item],
            ),
        ):
            result = _fetch_news("RELIANCE.NS")

        self.assertEqual(result["source"], "Yahoo Finance")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["attempted_sources"], ["Finviz", "Yahoo Finance"])
        self.assertEqual(len(result["items"]), 1)

    def test_sentiment_window_filters_news_before_classification(self):
        now = datetime.now(timezone.utc)
        recent = news_item("recent", (now - pd.Timedelta(days=3)).isoformat())
        older = news_item("older", (now - pd.Timedelta(days=45)).isoformat())
        raw_news = {
            "source": "Yahoo Finance",
            "attempted_sources": ["Finviz", "Yahoo Finance"],
            "fallback_used": True,
            "status": "available",
            "items": [recent, older],
        }
        with (
            patch("app.services.news_sentiment.resolve_exchange", return_value="NYSE"),
            patch("app.services.news_sentiment._fetch_news", return_value=raw_news),
            patch(
                "app.services.news_sentiment._classify_news",
                return_value={
                    "status": "available",
                    "provider": "Groq",
                    "model": "test",
                    "reason": None,
                    "items": [
                        {
                            "sentiment": "Positive",
                            "score": 0.5,
                            "theme": "earnings",
                            "rationale": "positive report",
                        }
                    ],
                },
            ) as classify,
            patch(
                "app.services.news_sentiment.fetch_stock_data",
                side_effect=RuntimeError("unavailable"),
            ),
            patch(
                "app.services.news_sentiment._cached",
                side_effect=lambda key, loader: {
                    "status": "unavailable",
                    "source": "Yahoo Finance",
                    "retrieved_at": now.isoformat(),
                    "values": {},
                }
                if key.startswith("fundamentals:")
                else [],
            ),
        ):
            result = analyze_stock_sentiment("aapl", "NYSE", days=30)

        self.assertEqual(result["news"]["window_days"], 30)
        self.assertEqual([item["id"] for item in result["news"]["items"]], ["n0"])
        classify.assert_called_once()
        self.assertEqual(classify.call_args.args[1][0]["title"], recent["title"])

    def test_groq_missing_key_is_explicit_and_does_not_invent_labels(self):
        with patch("app.services.news_sentiment.get_groq_api_key", return_value=None):
            result = _classify_news("AAPL", [news_item("n0", "2026-10-28T15:00:00+00:00")])

        self.assertEqual(result["status"], "unconfigured")
        self.assertEqual(result["items"], [])
        self.assertNotIn("api_key", result)

    def test_groq_classifications_are_validated_and_key_is_not_returned(self):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "items": [
                                    {
                                        "id": "n0",
                                        "sentiment": "Positive",
                                        "score": 0.8,
                                        "theme": "earnings",
                                        "rationale": "Revenue increased.",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        article = news_item("n0", "2026-10-28T15:00:00+00:00")
        with (
            patch("app.services.news_sentiment.get_groq_api_key", return_value="secret"),
            patch(
                "app.services.news_sentiment.requests.post",
                return_value=mock_response,
            ) as request,
        ):
            mock_response.raise_for_status.return_value = None
            result = _classify_news("AAPL", [article])

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["items"][0]["sentiment"], "Positive")
        self.assertEqual(result["items"][0]["score"], 0.8)
        self.assertNotIn("secret", json.dumps(result))
        self.assertEqual(
            request.call_args.kwargs["headers"]["Authorization"],
            "Bearer secret",
        )

    def test_groq_classifies_large_news_sets_in_bounded_batches(self):
        requests_seen = []

        def classify_request(*args, **kwargs):
            payload = kwargs["json"]
            requests_seen.append(payload)
            articles = json.loads(payload["messages"][1]["content"])["articles"]
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "id": article["id"],
                                            "sentiment": "Neutral",
                                            "score": 0,
                                            "theme": "test",
                                            "rationale": "No direction is stated.",
                                        }
                                        for article in articles
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
            return response

        articles = [
            news_item(f"n{index}", "2026-10-28T15:00:00+00:00")
            for index in range(23)
        ]
        with (
            patch("app.services.news_sentiment.get_groq_api_key", return_value="secret"),
            patch(
                "app.services.news_sentiment.requests.post",
                side_effect=classify_request,
            ),
        ):
            result = _classify_news("AAPL", articles)

        self.assertEqual(result["status"], "available")
        self.assertEqual(len(result["items"]), 23)
        self.assertEqual([len(json.loads(row["messages"][1]["content"])["articles"]) for row in requests_seen], [10, 10, 3])

    def test_invalid_groq_classification_is_rejected(self):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "items": [
                                    {
                                        "id": "n0",
                                        "sentiment": "Positive",
                                        "score": -0.5,
                                        "theme": "earnings",
                                        "rationale": "invalid sign",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        with (
            patch("app.services.news_sentiment.get_groq_api_key", return_value="secret"),
            patch(
                "app.services.news_sentiment.requests.post",
                return_value=mock_response,
            ),
        ):
            mock_response.raise_for_status.return_value = None
            result = _classify_news(
                "AAPL",
                [news_item("n0", "2026-10-28T15:00:00+00:00")],
            )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["items"], [])

    def test_unavailable_groq_model_returns_actionable_reason(self):
        response = Mock()
        response.status_code = 404
        response.json.return_value = {
            "error": {
                "code": "model_not_found",
                "message": "Model is unavailable.",
            }
        }
        http_error = requests.HTTPError(response=response)
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = http_error
        with (
            patch("app.services.news_sentiment.get_groq_api_key", return_value="secret"),
            patch(
                "app.services.news_sentiment.requests.post",
                return_value=mock_response,
            ),
        ):
            result = _classify_news(
                "AAPL",
                [news_item("n0", "2026-10-28T15:00:00+00:00")],
            )

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("configured Groq model is unavailable", result["reason"])
        self.assertNotIn("secret", json.dumps(result))

    def test_market_local_daily_aggregation_keeps_classification_counts(self):
        items = [
            news_item("n0", "2026-10-28T14:00:00+00:00"),
            news_item("n1", "2026-10-28T16:00:00+00:00"),
        ]
        labels = [
            {"sentiment": "Positive", "score": 0.7, "theme": "earnings", "rationale": ""},
            {"sentiment": "Neutral", "score": 0, "theme": "guidance", "rationale": ""},
        ]
        result = _aggregate_sentiment(items, labels, "NYSE")

        self.assertEqual(len(result["timeline"]), 1)
        self.assertEqual(result["timeline"][0]["date"], "2026-10-28")
        self.assertEqual(result["timeline"][0]["article_count"], 2)
        self.assertEqual(result["distribution"]["positive"], 1)
        self.assertEqual(result["distribution"]["neutral"], 1)

    def test_timestamp_alignment_uses_only_returns_after_news_is_available(self):
        history = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-02T21:00:00Z",
                        "2024-01-03T21:00:00Z",
                        "2024-01-04T21:00:00Z",
                    ],
                    utc=True,
                ),
                "close": [100.0, 110.0, 99.0],
            }
        )
        article = news_item("n0", "2024-01-02T20:00:00+00:00")
        timeline = [
            {
                "date": "2024-01-02",
                "score": 1.0,
                "article_count": 1,
            }
        ]

        pairs = _correlation_pairs(timeline, [article], history, "NYSE")

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["return_start_session"], "2024-01-02")
        self.assertEqual(pairs[0]["return_end_session"], "2024-01-03")
        self.assertAlmostEqual(pairs[0]["forward_return"], 0.1)

    def test_pearson_correlation_reports_sample_and_handles_constant_data(self):
        pairs = [
            {"sentiment_score": float(index), "forward_return": float(index * 2)}
            for index in range(8)
        ]
        result = _correlation_result(pairs)

        self.assertAlmostEqual(result["coefficient"], 1.0)
        self.assertLess(result["p_value"], 0.05)
        self.assertTrue(result["nominally_significant_0_05"])
        self.assertEqual(result["sample_size"], 8)

        constant = _correlation_result(
            [{"sentiment_score": 0.0, "forward_return": float(index)} for index in range(5)]
        )
        self.assertIsNone(constant["coefficient"])
        self.assertIsNone(constant["p_value"])

    def test_market_factor_features_are_point_in_time_and_finite(self):
        count = 100
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, count)))
        bars = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=count, freq="B"),
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": rng.integers(10_000, 100_000, count),
            }
        )
        factors = create_technical_factors(bars)
        changed_future = bars.copy()
        changed_future.loc[99, ["open", "high", "low", "close", "volume"]] *= 5
        factors_after_change = create_technical_factors(changed_future)

        self.assertEqual(set(factors.columns), set(TECHNICAL_FACTOR_NAMES))
        self.assertTrue(np.isfinite(factors.iloc[-1].to_numpy(dtype=float)).all())
        pd.testing.assert_series_equal(
            factors.iloc[80],
            factors_after_change.iloc[80],
            check_names=False,
        )

    def test_provider_timestamp_is_normalized_to_utc(self):
        timestamp = _utc_datetime("2026-10-28T10:00:00-04:00")
        self.assertEqual(timestamp, datetime(2026, 10, 28, 14, tzinfo=timezone.utc))

    def test_sentiment_api_normalizes_symbol_and_returns_analysis(self):
        from app.main import sentiment

        with patch(
            "app.main.analyze_stock_sentiment",
            return_value={"symbol": "AAPL", "sentiment": {"status": "unconfigured"}},
        ) as analyze:
            response = sentiment("aapl", "NASDAQ", 90)

        self.assertEqual(response["symbol"], "AAPL")
        analyze.assert_called_once_with("AAPL", "NASDAQ", 90)

    def test_sentiment_api_rejects_malformed_ticker(self):
        from fastapi import HTTPException

        from app.main import sentiment

        with self.assertRaises(HTTPException) as raised:
            sentiment("<bad>")
        self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
