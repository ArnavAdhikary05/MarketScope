import unittest

import numpy as np
import pandas as pd

from app.services.model_comparison import compare_models


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
            synthetic_bars(), "TEST", "1d", horizon=5, selected_model="auto"
        )

        self.assertEqual(result["horizon"], 5)
        self.assertEqual(result["selection_mode"], "automatic")
        self.assertEqual(result["validation_folds"], 5)
        self.assertEqual(result["models"]["naive"]["metrics"]["observations"], 5)
        self.assertIn(result["recommended_model"], result["ranking"])
        self.assertTrue(np.isfinite(result["forecast"]["predicted_close"]))
        self.assertTrue(result["forecast"]["timestamp"].startswith("2023-"))
        for model in result["models"].values():
            if model.get("eligible") and "metrics" in model:
                self.assertEqual(model["metrics"]["observations"], 5)

    def test_manual_naive_selection_returns_baseline_forecast(self):
        bars = synthetic_bars()
        result = compare_models(
            bars, "TEST", "1d", horizon=1, selected_model="naive"
        )

        self.assertEqual(result["selected_model"], "naive")
        self.assertEqual(result["selection_mode"], "manual")
        self.assertEqual(result["forecast"]["predicted_close"], bars["close"].iloc[-1])

    def test_invalid_manual_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            compare_models(synthetic_bars(), "TEST", "1d", selected_model="not-a-model")


if __name__ == "__main__":
    unittest.main()
