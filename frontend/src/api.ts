
import axios from "axios";

export type Bar = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type HistoryResponse = {
  symbol: string;
  interval: string;
  source: string;
  data_status: string;
  currency?: string | null;
  retrieved_at?: string;
  data: Bar[];
};

export type TrainingMetrics = {
  mae: number;
  rmse: number;
  baseline_mae: number;
  beats_baseline: boolean;
  train_rows: number;
  test_rows: number;
};

export type ForecastResponse = {
  last_close: number;
  predicted_next_close: number;
  last_bar_timestamp: string;
  generated_at: string;
  model: string;
  interpretation: string;
};

export type ModelMetrics = {
  observations: number;
  mae: number;
  rmse: number;
  smape: number;
  bias: number;
  directional_accuracy: number;
};

export type ModelResult = {
  family: string;
  eligible: boolean;
  status: string;
  reason?: string;
  metrics?: ModelMetrics;
  horizon_metrics?: ModelMetrics;
  test_metrics?: ModelMetrics;
  test_status?: string;
  test_reason?: string;
  test_baseline_mae?: number;
  test_beats_naive?: boolean;
  baseline_mae?: number;
  beats_naive?: boolean;
  validation_predictions?: Array<{
    timestamp: string;
    actual: number;
    predicted: number;
  }>;
};

export type ForecastPoint = {
  timestamp: string;
  predicted_close: number;
  lower_80: number | null;
  upper_80: number | null;
  calibration_samples: number;
};

export type ComparisonResult = {
  symbol: string;
  interval: string;
  exchange: string;
  currency?: string | null;
  horizon: string | number;
  horizon_label: string;
  horizon_sessions: number;
  selection_objective: string;
  selected_model: string;
  recommended_model: string;
  selection_mode: "automatic" | "manual";
  selected_model_beats_naive: boolean;
  validation_folds: number;
  test_folds: number;
  evaluation: {
    validation_end: string;
    test_start: string;
    test_end: string;
    test_fraction: number;
    test_metrics_used_for_selection: boolean;
    evaluation_method: string;
  };
  last_close: number;
  last_bar_timestamp: string;
  generated_at: string;
  forecast: {
    timestamp: string;
    predicted_close: number;
    interval_method: string;
    future: ForecastPoint[];
  };
  models: Record<string, ModelResult>;
  ranking: string[];
  data_note: string;
  forecast_target?: string;
  factor_analysis?: {
    price_and_technical: {
      available: boolean;
      source: string;
      features: string[];
      used_by_selected_model: string[];
      feature_importance: Array<{
        feature: string;
        importance: number;
        method: string;
      }>;
      limitation: string;
    };
    volume_liquidity: {
      available: boolean;
      features: string[];
      used_by_selected_model: string[];
      limitation: string;
    };
    fundamentals: {
      available: boolean;
      used_by_selected_model: string[];
      limitation: string;
    };
    news_sentiment: {
      available: boolean;
      used_by_selected_model: string[];
      limitation: string;
    };
    market_and_macro: {
      available: boolean;
      features: string[];
      used_by_selected_model: string[];
      limitation: string;
    };
    corporate_events: {
      available: boolean;
      used_by_selected_model: string[];
      limitation: string;
    };
  };
};

export type SentimentArticle = {
  id: string;
  title: string;
  summary: string;
  url: string;
  source: string;
  published_at: string;
  sentiment?: "Positive" | "Negative" | "Neutral";
  score?: number;
  theme?: string;
  rationale?: string;
};

export type SentimentResponse = {
  symbol: string;
  exchange: string;
  exchange_timezone: string;
  generated_at: string;
  news: {
    source: string | null;
    attempted_sources: string[];
    fallback_used: boolean;
    status: string;
    retrieved_at: string;
    window_days: number;
    items: SentimentArticle[];
  };
  sentiment: {
    provider: string;
    model: string | null;
    status: string;
    reason: string | null;
    timeline: Array<{
      date: string;
      score: number;
      article_count: number;
      positive: number;
      negative: number;
      neutral: number;
      latest_published_at: string;
      themes: Array<{ theme: string; count: number }>;
    }>;
    distribution: {
      positive: number;
      negative: number;
      neutral: number;
    };
    themes: Array<{ theme: string; count: number }>;
    significant_changes: Array<{
      date: string;
      change: number;
      from_score: number;
      to_score: number;
      threshold: number;
    }>;
    overall_score: number | null;
  };
  correlation: {
    method: string;
    coefficient: number | null;
    p_value: number | null;
    sample_size: number;
    nominally_significant_0_05: boolean;
    note: string;
    pairs: Array<{
      date: string;
      sentiment_score: number;
      forward_return: number;
      available_at: string;
      return_start_session: string;
      return_end_session: string;
    }>;
  };
  factors: {
    fundamentals: {
      status: string;
      source: string;
      retrieved_at: string;
      values: Record<string, number | string>;
      point_in_time_history?: boolean;
      forecast_use?: string;
    };
    market_indices: Array<{
      symbol: string;
      status: string;
      source: string;
      last_observed_at: string | null;
      change_20_sessions: number | null;
    }>;
    sector_performance: { status: string; reason: string };
    macroeconomic: { status: string; reason: string };
    corporate_events: { status: string; reason: string };
    forecast_usage: string;
  };
  limitations: string[];
};

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000",
  timeout: 30000,
});

export async function getHistory(
  symbol: string,
  interval: string,
  period: string,
) {
  const response = await api.get<HistoryResponse>(
    `/api/stocks/${encodeURIComponent(symbol)}/history`,
    { params: { interval, period } },
  );
  return response.data;
}

export async function trainModel(
  symbol: string,
  interval: string,
) {
  const response = await api.post<TrainingMetrics>(
    `/api/stocks/${encodeURIComponent(symbol)}/train`,
    null,
    { params: { interval } },
  );
  return response.data;
}

export async function predictStock(
  symbol: string,
  interval: string,
) {
  const response = await api.get<ForecastResponse>(
    `/api/stocks/${encodeURIComponent(symbol)}/predict`,
    { params: { interval } },
  );
  return response.data;
}

export async function compareStock(
  symbol: string,
  interval: string,
  period: string,
  horizon: string,
  model: string,
  exchange = "auto",
) {
  const response = await api.get<ComparisonResult>(
    `/api/stocks/${encodeURIComponent(symbol)}/compare`,
    {
      params: { interval, period, horizon, model, exchange },
      timeout: 300000,
    },
  );
  return response.data;
}

export async function getSentiment(
  symbol: string,
  exchange = "auto",
  days = 90,
) {
  const response = await api.get<SentimentResponse>(
    `/api/stocks/${encodeURIComponent(symbol)}/sentiment`,
    {
      params: { exchange, days },
      timeout: 90000,
    },
  );
  return response.data;
}

export function getApiError(error: unknown, fallback: string): string {
  if (axios.isAxiosError<{ detail?: string }>(error)) {
    return error.response?.data?.detail ?? error.message ?? fallback;
  }
  return error instanceof Error ? error.message : fallback;
}
