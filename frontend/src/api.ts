
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
  baseline_mae?: number;
  beats_naive?: boolean;
  validation_predictions?: Array<{
    timestamp: string;
    actual: number;
    predicted: number;
  }>;
};

export type ComparisonResult = {
  symbol: string;
  interval: string;
  horizon: number;
  selection_objective: string;
  selected_model: string;
  recommended_model: string;
  selection_mode: "automatic" | "manual";
  selected_model_beats_naive: boolean;
  validation_folds: number;
  last_close: number;
  last_bar_timestamp: string;
  generated_at: string;
  forecast: {
    timestamp: string;
    predicted_close: number;
  };
  models: Record<string, ModelResult>;
  ranking: string[];
  data_note: string;
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
  horizon: number,
  model: string,
) {
  const response = await api.get<ComparisonResult>(
    `/api/stocks/${encodeURIComponent(symbol)}/compare`,
    { params: { interval, period, horizon, model } },
  );
  return response.data;
}

export function getApiError(error: unknown, fallback: string): string {
  if (axios.isAxiosError<{ detail?: string }>(error)) {
    return error.response?.data?.detail ?? error.message ?? fallback;
  }
  return error instanceof Error ? error.message : fallback;
}
