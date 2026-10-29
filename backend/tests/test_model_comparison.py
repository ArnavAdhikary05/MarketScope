import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.services.market_calendar import future_session_closes, resolve_exchange
from app.services.model_comparison import (
    HORIZON_SPECS,
    MARKET_FACTOR_NAMES,
    _align_benchmark_features,
    _resolve_horizon,
    _select_training_features,
    compare_models,
)


def synthetic_bars(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0004, 0.012, rows)
    close = 100 * np.exp(np.cumsum(returns))
    timestamps = pd.date_range("2023-01-02", periods=rows, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close * (1 - 0.002),
            "high": close * (1 + 0.006),
            "low": close * (1 - 0.006),
            "close": close,
            "volume": rng.integers(10_000, 100_000, rows),
        }
    )


class ModelComparisonTests(unittest.TestCase):
    def test_comparison_uses_chronological_folds_and_returns_a_forecast(self):
        result = compare_models(
            synthetic_bars(),
            "TEST",
            "1d",
            horizon=5,
            selected_model="auto",
            exchange="NYSE",
        )

        self.assertEqual(result["horizon"], 5)
        self.assertEqual(result["selection_mode"], "automatic")
        self.assertEqual(result["exchange"], "NYSE")
        self.assertEqual(result["validation_folds"], 8)
        self.assertEqual(result["test_folds"], 8)
        self.assertFalse(result["evaluation"]["test_metrics_used_for_selection"])
        self.assertTrue(
            all(
                pd.Timestamp(fold["timestamp"])
                <= pd.Timestamp(result["evaluation"]["validation_end"])
                for fold in result["models"]["ridge"]["validation_predictions"]
            )
        )
        self.assertLess(
            pd.Timestamp(result["evaluation"]["validation_end"]),
            pd.Timestamp(result["evaluation"]["test_start"]),
        )
        self.assertEqual(result["models"]["naive"]["metrics"]["observations"], 8)
        self.assertEqual(
            result["models"]["naive"]["horizon_metrics"]["observations"], 40
        )
        self.assertEqual(
            result["models"]["naive"]["test_metrics"]["observations"], 40
        )
        self.assertTrue(result["models"]["ridge"]["eligible"])
        self.assertIn("horizon_metrics", result["models"]["ridge"])
        self.assertIn("test_metrics", result["models"]["ridge"])
        self.assertIn(
            "turnover_deviation_20",
            result["factor_analysis"]["volume_liquidity"]["features"],
        )
        self.assertFalse(result["factor_analysis"]["corporate_events"]["available"])
        self.assertIn(result["recommended_model"], result["ranking"])
        self.assertTrue(np.isfinite(result["forecast"]["predicted_close"]))
        self.assertTrue(result["forecast"]["timestamp"].startswith("2023-"))
        self.assertEqual(len(result["forecast"]["future"]), 5)
        self.assertEqual(result["forecast"]["future"][-1]["timestamp"], result["forecast"]["timestamp"])
        self.assertIn("walk-forward validation residuals", result["forecast"]["interval_method"])
        self.assertEqual(
            result["horizon_label"],
            f"{result['horizon_sessions']} trading days",
        )
        for model in result["models"].values():
            if model.get("eligible") and "metrics" in model:
                self.assertEqual(model["metrics"]["observations"], 8)

    def test_manual_naive_selection_returns_baseline_forecast(self):
        bars = synthetic_bars()
        result = compare_models(
            bars, "TEST", "1d", horizon=1, selected_model="naive", exchange="NYSE"
        )

        self.assertEqual(result["selected_model"], "naive")
        self.assertEqual(result["selection_mode"], "manual")
        self.assertEqual(result["forecast"]["predicted_close"], bars["close"].iloc[-1])
        self.assertTrue(
            all(
                point["predicted_close"] == bars["close"].iloc[-1]
                for point in result["forecast"]["future"]
            )
        )

    def test_named_horizon_returns_calendar_dates_and_serializable_intervals(self):
        bars = synthetic_bars()

        def deterministic_path(name, history, steps, future_dates):
            return [float(history["close"].iloc[-1])] * steps

        with patch(
            "app.services.model_comparison._model_forecast",
            side_effect=deterministic_path,
        ):
            result = compare_models(
                bars,
                "TEST",
                "1d",
                horizon="3d",
                selected_model="naive",
                exchange="NYSE",
            )

        last_timestamp = pd.Timestamp(bars["timestamp"].iloc[-1])
        steps, _, _, expected_dates = _resolve_horizon(
            "3d", last_timestamp, "NYSE"
        )
        future = result["forecast"]["future"]
        self.assertEqual(result["horizon"], "3d")
        self.assertEqual(result["horizon_sessions"], steps)
        self.assertEqual(
            [pd.Timestamp(point["timestamp"]) for point in future],
            expected_dates,
        )
        self.assertTrue(
            all(
                point["lower_80"] is not None
                and point["upper_80"] is not None
                and point["calibration_samples"] >= 4
                for point in future
            )
        )

    def test_numeric_string_horizon_returns_integer_session_count(self):
        def deterministic_path(name, history, steps, future_dates):
            return [float(history["close"].iloc[-1])] * steps

        with patch(
            "app.services.model_comparison._model_forecast",
            side_effect=deterministic_path,
        ):
            result = compare_models(
                synthetic_bars(),
                "TEST",
                "1d",
                horizon="1",
                selected_model="naive",
                exchange="NYSE",
            )

        self.assertEqual(result["horizon"], 1)
        self.assertEqual(result["horizon_label"], "1 trading day")
        self.assertEqual(result["horizon_sessions"], 1)
        self.assertEqual(len(result["forecast"]["future"]), 1)

    def test_invalid_manual_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            compare_models(
                synthetic_bars(),
                "TEST",
                "1d",
                selected_model="not-a-model",
                exchange="NYSE",
            )

    def test_named_day_horizon_uses_trading_sessions_not_calendar_days(self):
        last_observation = pd.Timestamp("2024-07-03 20:00:00", tz="UTC")
        steps, label, key, future = _resolve_horizon(
            "3d", last_observation, "NYSE"
        )

        self.assertEqual(key, "3d")
        self.assertEqual(label, HORIZON_SPECS["3d"][0])
        self.assertEqual(steps, 3)
        self.assertEqual(future[0].date(), date(2024, 7, 5))
        self.assertEqual(future[0].hour, 20)
        self.assertEqual(future[1].date(), date(2024, 7, 8))
        self.assertEqual(future[2].date(), date(2024, 7, 9))

    def test_one_and_custom_trading_day_horizons_use_exchange_sessions(self):
        last_observation = pd.Timestamp("2024-07-03 20:00:00", tz="UTC")

        one_day_steps, one_day_label, one_day_key, one_day_dates = _resolve_horizon(
            "1", last_observation, "NYSE"
        )
        custom_steps, custom_label, custom_key, custom_dates = _resolve_horizon(
            "12", last_observation, "NYSE"
        )

        self.assertEqual((one_day_steps, one_day_label, one_day_key), (1, "1 trading day", "1"))
        self.assertEqual(one_day_dates[0].date(), date(2024, 7, 5))
        self.assertEqual((custom_steps, custom_label, custom_key), (12, "12 trading days", "12"))
        self.assertEqual(len(custom_dates), 12)
        self.assertEqual(
            custom_dates,
            future_session_closes(
                last_observation,
                "NYSE",
                custom_dates[-1].date(),
            )[:12],
        )

    def test_custom_trading_day_horizon_is_capped(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 132"):
            _resolve_horizon("133", pd.Timestamp("2024-07-03", tz="UTC"), "NYSE")

    def test_training_feature_selection_removes_constants_and_redundant_features(self):
        x = np.arange(1, 21, dtype=float)
        features = np.column_stack([x, x * 2, np.sin(x), np.ones_like(x)])
        targets = x + np.cos(x)

        selected = _select_training_features(features, targets)

        self.assertNotIn(3, selected)
        self.assertFalse({0, 1}.issubset(set(selected)))
        self.assertEqual(len(selected), 2)

    def test_benchmark_features_align_backward_without_future_rows(self):
        bars = synthetic_bars()
        benchmark = pd.DataFrame(
            {
                "timestamp": bars["timestamp"],
                "close": np.linspace(100, 150, len(bars)),
            }
        )
        aligned, available = _align_benchmark_features(bars, benchmark)
        changed = benchmark.copy()
        changed.loc[200:, "close"] *= 4
        aligned_after_future_change, _ = _align_benchmark_features(bars, changed)

        self.assertTrue(available)
        self.assertTrue(set(MARKET_FACTOR_NAMES).issubset(aligned.columns))
        np.testing.assert_allclose(
            aligned.loc[150, list(MARKET_FACTOR_NAMES)].to_numpy(dtype=float),
            aligned_after_future_change.loc[
                150, list(MARKET_FACTOR_NAMES)
            ].to_numpy(dtype=float),
        )

    def test_benchmark_history_is_available_to_supervised_forecasts(self):
        bars = synthetic_bars()

        def deterministic_path(name, history, steps, future_dates):
            return [float(history["close"].iloc[-1])] * steps

        with patch(
            "app.services.model_comparison._model_forecast",
            side_effect=deterministic_path,
        ):
            result = compare_models(
                bars,
                "TEST",
                "1d",
                horizon=1,
                selected_model="ridge",
                exchange="NYSE",
                benchmark_history=bars[["timestamp", "close"]],
            )

        self.assertTrue(result["factor_analysis"]["market_and_macro"]["available"])
        self.assertEqual(
            result["factor_analysis"]["market_and_macro"]["features"],
            list(MARKET_FACTOR_NAMES),
        )
        self.assertTrue(result["models"]["ridge"]["eligible"])

    def test_compare_api_fetches_and_passes_exchange_benchmark(self):
        from app.main import compare as compare_route

        bars = synthetic_bars()
        benchmark = bars[["timestamp", "close"]].copy()
        with (
            patch("app.main.fetch_stock_data", side_effect=[bars, benchmark]) as fetch,
            patch("app.main.resolve_exchange", return_value="NASDAQ"),
            patch(
                "app.main.compare_models",
                return_value={"selected_model": "naive"},
            ) as compare_models_mock,
        ):
            response = compare_route("aapl", exchange="auto")

        self.assertEqual(response["selected_model"], "naive")
        self.assertEqual(fetch.call_count, 2)
        compare_models_mock.assert_called_once()
        self.assertEqual(
            compare_models_mock.call_args.kwargs["benchmark_history"].shape[1],
            2,
        )

    def test_six_month_horizon_contains_only_exchange_sessions(self):
        last_observation = pd.Timestamp("2024-01-31 21:00:00", tz="UTC")
        steps, _, _, future = _resolve_horizon(
            "6m", last_observation, "NYSE"
        )

        self.assertEqual(steps, len(future))
        self.assertGreater(steps, 100)
        self.assertEqual(future[-1].date(), date(2024, 7, 31))
        scheduled = future_session_closes(
            last_observation, "NYSE", date(2024, 7, 31)
        )
        self.assertEqual(future, scheduled)

    def test_indian_exchange_suffixes_resolve_without_metadata_lookup(self):
        self.assertEqual(resolve_exchange("RELIANCE.NS"), "NSE")
        self.assertEqual(resolve_exchange("500325.BO"), "BSE")
        self.assertEqual(resolve_exchange("7203.T", "JPX"), "JPX")
        self.assertEqual(resolve_exchange("VOD.L", "LSE"), "LSE")

    def test_calendar_based_horizon_rejects_intraday_bars(self):
        with self.assertRaisesRegex(ValueError, "daily bars only"):
            compare_models(
                synthetic_bars(),
                "TEST",
                "15m",
                horizon="3d",
                selected_model="auto",
                exchange="NYSE",
            )

    def test_long_horizon_fails_gracefully_with_insufficient_history(self):
        with self.assertRaisesRegex(ValueError, "Not enough observations"):
            compare_models(
                synthetic_bars(),
                "TEST",
                "1d",
                horizon="6m",
                selected_model="auto",
                exchange="NYSE",
            )


if __name__ == "__main__":
    unittest.main()
