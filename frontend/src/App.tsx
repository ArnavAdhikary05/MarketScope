import {
  Suspense,
  lazy,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { isAxiosError } from "axios";
import {
  compareStock,
  getHistory,
  getSentiment,
  predictStock,
  trainModel,
  type ComparisonResult,
  type ForecastResponse,
  type HistoryResponse,
  type SentimentResponse,
  type TrainingMetrics,
} from "./api";
import StockChart from "./components/StockChart";
import HelpPage from "./components/HelpPage";
import "./App.css";

const ForecastChart = lazy(() => import("./components/ForecastChart"));
const NewsSentimentPanel = lazy(() => import("./components/NewsSentimentPanel"));

const EMPTY_VALIDATION: NonNullable<
  ComparisonResult["models"][string]["validation_predictions"]
> = [];

const SYMBOL_CURRENCY_SUFFIXES: Record<string, string> = {
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
};

function Tooltip({ text, children }: { text: string; children: ReactNode }) {
  return (
    <span className="tooltip-wrap" data-tooltip={text} title={text}>
      {children}
    </span>
  );
}

function getErrorMessage(error: unknown, fallback: string) {
  if (isAxiosError<{ detail?: string }>(error)) {
    return error.response?.data?.detail ?? error.message ?? fallback;
  }
  return error instanceof Error ? error.message : fallback;
}

function formatPrice(value: number, ticker: string, currencyOverride?: string | null) {
  const suffix = Object.keys(SYMBOL_CURRENCY_SUFFIXES).find((key) =>
    ticker.toUpperCase().endsWith(key),
  );
  const currency =
    currencyOverride ??
    (suffix ? SYMBOL_CURRENCY_SUFFIXES[suffix] : "USD");
  if (currency === "GBp") {
    return `GBp ${new Intl.NumberFormat(undefined, {
      maximumFractionDigits: 2,
    }).format(value)}`;
  }
  if (!/^[A-Z]{3}$/.test(currency)) {
    return `${ticker} ${new Intl.NumberFormat(undefined, {
      maximumFractionDigits: 2,
    }).format(value)}`;
  }
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatTimestamp(value?: string) {
  if (!value) return "Unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
}

export default function App() {
  const [page, setPage] = useState<"overview" | "help">(() =>
    window.location.hash === "#help" ? "help" : "overview",
  );
  const [symbol, setSymbol] = useState("AAPL");
  const [exchange, setExchange] = useState("auto");
  const [interval, setInterval] = useState("1d");
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [sentiment, setSentiment] = useState<SentimentResponse | null>(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [sentimentError, setSentimentError] = useState("");
  const [sentimentDays, setSentimentDays] = useState(90);
  const [horizon, setHorizon] = useState("3d");
  const [customDays, setCustomDays] = useState("10");
  const [modelChoice, setModelChoice] = useState("auto");
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [activeAction, setActiveAction] = useState<
    "history" | "training" | "forecast" | "comparison" | null
  >(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const syncPageWithHash = () => {
      setPage(window.location.hash === "#help" ? "help" : "overview");
    };
    window.addEventListener("hashchange", syncPageWithHash);
    window.addEventListener("popstate", syncPageWithHash);
    return () => {
      window.removeEventListener("hashchange", syncPageWithHash);
      window.removeEventListener("popstate", syncPageWithHash);
    };
  }, []);

  const bars = history?.data ?? [];
  const marketPrice = (value: number) =>
    formatPrice(value, history?.symbol ?? symbol, history?.currency);
  const latestBar = bars.at(-1);
  const previousBar = bars.at(-2);
  const change = useMemo(() => {
    if (!latestBar || !previousBar) return null;
    const amount = latestBar.close - previousBar.close;
    return {
      amount,
      percent: previousBar.close === 0 ? 0 : (amount / previousBar.close) * 100,
    };
  }, [latestBar, previousBar]);

  async function loadHistory() {
    const normalizedSymbol = symbol.trim().toUpperCase();
    if (!normalizedSymbol) {
      setError("Enter a stock symbol to load market data.");
      return;
    }

    setSymbol(normalizedSymbol);
    setActiveAction("history");
    setError("");
    try {
      const result = await getHistory(
        normalizedSymbol,
        interval,
        interval === "1d" ? "1y" : "5d",
      );
      setHistory(result);
      setForecast(null);
      setComparison(null);
      setModelChoice("auto");
      setMetrics(null);
    } catch (requestError) {
      setError(getErrorMessage(requestError, "Could not fetch market data."));
    } finally {
      setActiveAction(null);
    }
  }

  async function train() {
    const normalizedSymbol = symbol.trim().toUpperCase();
    if (!normalizedSymbol) {
      setError("Enter a stock symbol before training.");
      return;
    }

    setActiveAction("training");
    setError("");
    setForecast(null);
    setComparison(null);
    setMetrics(null);
    try {
      setMetrics(await trainModel(normalizedSymbol, interval));
    } catch (requestError) {
      setError(getErrorMessage(requestError, "Model training failed."));
    } finally {
      setActiveAction(null);
    }
  }

  async function predict() {
    const normalizedSymbol = symbol.trim().toUpperCase();
    if (!normalizedSymbol) {
      setError("Enter a stock symbol before requesting a forecast.");
      return;
    }

    setActiveAction("forecast");
    setError("");
    setForecast(null);
    setComparison(null);
    setMetrics(null);
    try {
      setForecast(await predictStock(normalizedSymbol, interval));
    } catch (requestError) {
      setError(getErrorMessage(requestError, "Forecast generation failed."));
    } finally {
      setActiveAction(null);
    }
  }

  async function runComparison(
    requestedModel = modelChoice,
    requestedHorizon = horizon,
  ) {
    const normalizedSymbol = symbol.trim().toUpperCase();
    if (!normalizedSymbol) {
      setError("Enter a stock symbol before comparing models.");
      return;
    }
    if (interval !== "1d") {
      setError("Multi-day forecasts require the daily interval. Select Daily and retry.");
      return;
    }
    const horizonToRequest =
      requestedHorizon === "custom" ? customDays.trim() : requestedHorizon;
    if (
      /^\d+$/.test(horizonToRequest) &&
      (Number(horizonToRequest) < 1 || Number(horizonToRequest) > 132)
    ) {
      setError("Custom trading days must be between 1 and 132.");
      return;
    }
    if (requestedHorizon === "custom" && !/^\d+$/.test(horizonToRequest)) {
      setError("Enter a whole number of trading days between 1 and 132.");
      return;
    }

    setActiveAction("comparison");
    setError("");
    setComparison(null);
    setForecast(null);
    setMetrics(null);
    try {
      const historyPeriod = interval === "1d" ? "1y" : "5d";
      const comparisonPeriod = "10y";
      const [marketHistory, result] = await Promise.all([
        getHistory(normalizedSymbol, interval, historyPeriod),
        compareStock(
          normalizedSymbol,
          interval,
          comparisonPeriod,
          horizonToRequest,
          requestedModel,
          exchange,
        ),
      ]);
      setSymbol(normalizedSymbol);
      setHistory(marketHistory);
      setComparison(result);
      setForecast(null);
      setMetrics(null);
    } catch (requestError) {
      setError(getErrorMessage(requestError, "Model comparison failed."));
    } finally {
      setActiveAction(null);
    }
  }

  async function analyzeSentiment(requestedDays = sentimentDays) {
    setSentimentDays(requestedDays);
    const normalizedSymbol = symbol.trim().toUpperCase();
    if (!normalizedSymbol) {
      setSentimentError("Enter a stock symbol before analyzing news.");
      return;
    }

    setSentimentLoading(true);
    setSentimentError("");
    try {
      const result = await getSentiment(normalizedSymbol, exchange, requestedDays);
      setSymbol(normalizedSymbol);
      setSentiment(result);
    } catch (requestError) {
      setSentimentError(
        getErrorMessage(requestError, "News sentiment analysis failed."),
      );
    } finally {
      setSentimentLoading(false);
    }
  }

  const isBusy = activeAction !== null;
  const supportedHorizons = [
    ["1", "1 trading day"],
    ["3d", "3 trading days"],
    ["7d", "7 trading days"],
    ["15d", "15 trading days"],
    ["30d", "30 trading days (about 1 month)"],
    ["3m", "3 months"],
    ["6m", "6 months"],
    ["custom", "Custom trading days"],
  ] as const;
  const finalForecastPoint = comparison?.forecast.future.at(-1);
  const selectedModelResult = comparison?.models[comparison.selected_model];
  const selectedValidation = selectedModelResult?.validation_predictions ?? EMPTY_VALIDATION;
  const validationBaseline = comparison?.models.naive?.validation_predictions ?? EMPTY_VALIDATION;
  const validationPlot = useMemo(() => {
    if (selectedValidation.length === 0) return null;

    const values = selectedValidation.flatMap((point, index) => [
      point.actual,
      point.predicted,
      validationBaseline[index]?.predicted ?? point.actual,
    ]);
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const span = maximum - minimum || Math.max(Math.abs(minimum) * 0.01, 1);
    const x = (index: number) =>
      selectedValidation.length === 1
        ? 380
        : 40 + (index * 700) / (selectedValidation.length - 1);
    const y = (value: number) => 184 - ((value - minimum) / span) * 156;
    const polyline = (valueAt: (index: number) => number) =>
      selectedValidation
        .map((_, index) => `${x(index)},${y(valueAt(index))}`)
        .join(" ");

    return {
      minimum,
      maximum,
      actual: polyline((index) => selectedValidation[index].actual),
      predicted: polyline((index) => selectedValidation[index].predicted),
      naive: polyline(
        (index) => validationBaseline[index]?.predicted ?? selectedValidation[index].actual,
      ),
      x,
      y,
    };
  }, [selectedValidation, validationBaseline]);
  const chartForecast = useMemo(
    () =>
      comparison
        ? {
            last_bar_timestamp: comparison.last_bar_timestamp,
            last_close: comparison.last_close,
            future: comparison.forecast.future,
          }
        : null,
    [comparison],
  );

  function clearLoadedResults() {
    setHistory(null);
    setForecast(null);
    setComparison(null);
    setModelChoice("auto");
    setMetrics(null);
    setSentiment(null);
    setSentimentError("");
    setError("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="MarketScope overview">
          <span className="brand-mark" aria-hidden="true">M</span>
          <span>Market<span className="brand-accent">Scope</span></span>
        </a>

        <p className="nav-heading">WORKSPACE</p>
        <nav className="main-nav" aria-label="Main navigation">
          <Tooltip text="Return to the market overview dashboard.">
            <a
              className={`nav-link ${page === "overview" ? "active" : ""}`}
              href="#overview"
              onClick={() => setPage("overview")}
              aria-current={page === "overview" ? "page" : undefined}
            >
              <span aria-hidden="true">◫</span> Overview
            </a>
          </Tooltip>
          <Tooltip text="Jump to the price and volume chart.">
            <a className="nav-link" href="#market-data" onClick={() => setPage("overview")}>
              <span aria-hidden="true">⌁</span> Market data
            </a>
          </Tooltip>
          <Tooltip text="Jump to model comparison and forecasting controls.">
            <a className="nav-link" href="#forecast-lab" onClick={() => setPage("overview")}>
              <span aria-hidden="true">↗</span> Forecast lab
            </a>
          </Tooltip>
          <Tooltip text="Open the tutorial and feature explanations.">
            <a
              className={`nav-link ${page === "help" ? "active" : ""}`}
              href="#help"
              onClick={() => setPage("help")}
              aria-current={page === "help" ? "page" : undefined}
            >
              <span aria-hidden="true">?</span> Help & tutorial
            </a>
          </Tooltip>
        </nav>

        <div className="sidebar-note">
          <span className="status-dot" />
          <div>
            <strong>Research workspace</strong>
            <p>Experimental forecasts, not financial advice.</p>
          </div>
        </div>
      </aside>

      <main className="main-content" id={page === "help" ? "help" : "overview"}>
        <header className="topbar">
          <div>
            <p className="eyebrow">{page === "help" ? "GUIDE & DOCUMENTATION" : "MARKET ANALYTICS"}</p>
            <h1>{page === "help" ? "Help & tutorial" : "Overview"}</h1>
          </div>
          <div className="topbar-status">
            <span className="status-indicator" />
            <span>{history ? `${history.source} data` : "Provider not queried"}</span>
          </div>
        </header>

        {page === "help" ? (
          <HelpPage
            onBack={() => {
              window.location.hash = "overview";
              setPage("overview");
            }}
          />
        ) : (
          <>
        <section className="page-intro">
          <div>
            <p className="eyebrow">MARKET EXPLORER</p>
            <h2>Explore a market</h2>
            <p className="muted">
              Load provider-supplied price history and run an experimental model.
            </p>
          </div>
          <form
            className="market-controls"
            onSubmit={(event) => {
              event.preventDefault();
              void loadHistory();
            }}
          >
            <label className="field">
              <span>Symbol</span>
              <input
                aria-label="Stock ticker"
                autoComplete="off"
                disabled={isBusy}
                maxLength={30}
                value={symbol}
                onChange={(event) => {
                  setSymbol(event.target.value.toUpperCase());
                  clearLoadedResults();
                }}
                placeholder="AAPL"
                title="Enter a provider-supported ticker, for example AAPL or RELIANCE.NS."
              />
            </label>
            <label className="field">
              <span>Exchange</span>
              <select
                disabled={isBusy}
                value={exchange}
                onChange={(event) => {
                  setExchange(event.target.value);
                  clearLoadedResults();
                }}
                title="Choose the exchange calendar used to compute future trading sessions."
              >
                <option value="auto">Auto detect</option>
                <option value="NASDAQ">NASDAQ</option>
                <option value="NYSE">NYSE</option>
                <option value="NSE">NSE</option>
                <option value="BSE">BSE</option>
                <option value="LSE">London Stock Exchange</option>
                <option value="TSX">Toronto Stock Exchange</option>
                <option value="ASX">Australian Securities Exchange</option>
                <option value="HKEX">Hong Kong Exchange</option>
                <option value="JPX">Tokyo Stock Exchange</option>
                <option value="XETR">Xetra</option>
              </select>
            </label>
            <label className="field">
              <span>Interval</span>
              <select
                disabled={isBusy}
                value={interval}
                onChange={(event) => {
                  setInterval(event.target.value);
                  setHorizon("3d");
                  clearLoadedResults();
                }}
              >
                <option value="1d">Daily</option>
                <option value="15m">15 minutes</option>
                <option value="5m">5 minutes</option>
              </select>
            </label>
            <Tooltip text="Fetch OHLCV history for this symbol and interval from the configured market-data provider.">
              <button className="button button-primary" disabled={isBusy} type="submit">
                {activeAction === "history" ? "Loading…" : "Load data"}
              </button>
            </Tooltip>
          </form>
        </section>

        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <Tooltip text="Dismiss this error message.">
              <button type="button" onClick={() => setError("")} aria-label="Dismiss error">
                ×
              </button>
            </Tooltip>
          </div>
        )}

        {activeAction && (
          <p className="activity-message" role="status">
            {activeAction === "history" && "Fetching market history…"}
            {activeAction === "training" && "Training and evaluating the model…"}
            {activeAction === "forecast" && "Generating a forecast…"}
            {activeAction === "comparison" && "Comparing models on chronological validation folds…"}
          </p>
        )}

        <section className="metrics-grid" aria-label="Latest market summary">
          <article className="metric-card primary-metric">
            <div className="metric-heading">
              <span>Latest close</span>
              <span className="metric-icon">◈</span>
            </div>
            <strong>
              {latestBar
                ? marketPrice(latestBar.close)
                : "—"}
            </strong>
            <small>{latestBar ? formatTimestamp(latestBar.timestamp) : "Load data to begin"}</small>
          </article>
          <article className="metric-card">
            <div className="metric-heading"><span>Change</span><span className="metric-icon">↕</span></div>
            <strong className={change ? (change.amount >= 0 ? "positive" : "negative") : ""}>
              {change
                ? `${change.amount >= 0 ? "+" : ""}${marketPrice(change.amount)}`
                : "—"}
            </strong>
            <small>
              {change
                ? `${change.percent >= 0 ? "+" : ""}${change.percent.toFixed(2)}% vs previous bar`
                : "Requires at least two bars"}
            </small>
          </article>
          <article className="metric-card">
            <div className="metric-heading"><span>Observed bars</span><span className="metric-icon">▥</span></div>
            <strong>{history ? bars.length.toLocaleString() : "—"}</strong>
            <small>{history ? `${history.symbol} · ${history.interval}` : "No history loaded"}</small>
          </article>
          <article className="metric-card">
            <div className="metric-heading"><span>Last data timestamp</span><span className="metric-icon">◷</span></div>
            <strong className="timestamp-value">
              {latestBar ? formatTimestamp(latestBar.timestamp) : "—"}
            </strong>
            <small>{history?.data_status ?? "Provider status unavailable"}</small>
          </article>
        </section>

        <section className="panel chart-panel" id="market-data">
          {bars.length > 0 ? (
            <StockChart
              bars={bars}
              symbol={history?.symbol ?? symbol}
              forecast={chartForecast}
            />
          ) : (
            <div className="empty-chart">
              <span className="empty-chart-icon" aria-hidden="true">⌁</span>
              <h2>Your market chart will appear here</h2>
              <p>Enter a supported ticker and load historical data to view price action and volume.</p>
              <Tooltip text="Fetch historical bars for the ticker and interval selected above.">
                <button
                  className="button button-secondary"
                  disabled={isBusy}
                  onClick={() => void loadHistory()}
                  type="button"
                >
                  Load {symbol || "AAPL"} history
                </button>
              </Tooltip>
            </div>
          )}
        </section>

        <section className="panel forecast-panel" id="forecast-lab">
          <div className="section-heading">
            <div>
              <p className="eyebrow">CHRONOLOGICAL MODEL VALIDATION</p>
              <h2>Forecast lab</h2>
              <p className="muted">
                Walk-forward model selection and exchange-calendar forecasts. Long horizons require enough daily history.
              </p>
            </div>
            <div className="action-group">
              <div className="forecast-options">
                <label className="field forecast-horizon-field">
                  <span>Forecast horizon</span>
                  <select
                    aria-label="Forecast horizon"
                    disabled={isBusy}
                    value={horizon}
                    onChange={(event) => {
                      const value = event.target.value;
                      setHorizon(value);
                      if (value === "custom") {
                        setComparison(null);
                        setForecast(null);
                      } else if (comparison) {
                        void runComparison(modelChoice, value);
                      }
                    }}
                  >
                    {supportedHorizons.map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
                {horizon === "custom" && (
                  <label className="field custom-days-field">
                    <span>Trading days (1–132)</span>
                    <input
                      aria-label="Custom trading days"
                      disabled={isBusy}
                      inputMode="numeric"
                      max="132"
                      min="1"
                      onChange={(event) => setCustomDays(event.target.value)}
                      type="number"
                      value={customDays}
                    />
                  </label>
                )}
                <label className="field forecast-model-field">
                  <span>Forecast model</span>
                  <select
                    aria-label="Forecast model"
                    disabled={isBusy || interval !== "1d"}
                    value={modelChoice}
                    onChange={(event) => {
                      const value = event.target.value;
                      setModelChoice(value);
                      if (comparison) void runComparison(value);
                    }}
                  >
                    <option value="auto">Automatically Recommend Best Model</option>
                    {Object.entries(comparison?.models ?? {}).map(([name, result]) => (
                      <option key={name} value={name} disabled={!result.eligible}>
                        {name.replaceAll("_", " ")}
                        {!result.eligible ? " (unavailable)" : ""}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="forecast-actions">
                <Tooltip text="Evaluate eligible models on chronological folds and generate a forecast for the chosen horizon.">
                  <button
                    className="button button-primary"
                    disabled={isBusy || interval !== "1d"}
                    onClick={() => void runComparison()}
                    type="button"
                  >
                    {activeAction === "comparison" ? "Comparing…" : "Compare & forecast"}
                  </button>
                </Tooltip>
                <Tooltip text="Fit the Random Forest model and evaluate it on a chronological holdout period.">
                  <button
                    className="button button-secondary"
                    disabled={isBusy}
                    onClick={() => void train()}
                    type="button"
                  >
                    {activeAction === "training" ? "Training…" : "Train model"}
                  </button>
                </Tooltip>
                <Tooltip text="Generate the Random Forest next-bar estimate. Train the model first if no saved model exists.">
                  <button
                    className="button button-primary"
                    disabled={isBusy}
                    onClick={() => void predict()}
                    type="button"
                  >
                    {activeAction === "forecast" ? "Predicting…" : "Predict next bar"}
                  </button>
                </Tooltip>
              </div>
            </div>
          </div>
          {interval !== "1d" && (
            <p className="unsupported-forecast-note" role="status">
              Multi-day horizons use daily observations and exchange sessions. Select Daily to compare models and generate these forecasts.
            </p>
          )}

          {comparison && (
            <div className="comparison-block">
              <div className="evaluation-heading">
                <h3>Model comparison</h3>
                <span className={comparison.selected_model_beats_naive ? "evaluation-badge better" : "evaluation-badge"}>
                  {comparison.selection_mode === "automatic"
                    ? `Recommended: ${comparison.recommended_model.replaceAll("_", " ")}`
                    : `Manual: ${comparison.selected_model.replaceAll("_", " ")}`}
                </span>
              </div>
              <p className="evaluation-note">
                {comparison.selection_objective}. Model selection uses {comparison.validation_folds} expanding chronological validation folds;
                {` `}final test results use {comparison.test_folds} later rolling-origin folds and are not used for selection.
              </p>
              <div className="selected-model-metrics">
                <div>
                  <span>Selected model</span>
                  <strong>{comparison.selected_model.replaceAll("_", " ")}</strong>
                </div>
                <div>
                  <span>Validation path MAE</span>
                  <strong>{selectedModelResult?.horizon_metrics?.mae.toFixed(4) ?? "Unavailable"}</strong>
                </div>
                <div>
                  <span>Validation path RMSE</span>
                  <strong>{selectedModelResult?.horizon_metrics?.rmse.toFixed(4) ?? "Unavailable"}</strong>
                </div>
                <div>
                  <span>Directional accuracy</span>
                  <strong>
                    {selectedModelResult?.horizon_metrics
                      ? `${(selectedModelResult.horizon_metrics.directional_accuracy * 100).toFixed(1)}%`
                      : "Unavailable"}
                  </strong>
                </div>
                <div>
                  <span>Naive validation path MAE</span>
                  <strong>{comparison.models.naive?.horizon_metrics?.mae.toFixed(4) ?? "Unavailable"}</strong>
                </div>
                <div>
                  <span>Final test MAE</span>
                  <strong>{selectedModelResult?.test_metrics?.mae.toFixed(4) ?? "Unavailable"}</strong>
                </div>
                <div>
                  <span>Final test RMSE</span>
                  <strong>{selectedModelResult?.test_metrics?.rmse.toFixed(4) ?? "Unavailable"}</strong>
                </div>
                <div>
                  <span>Final test baseline</span>
                  <strong>
                    {selectedModelResult?.test_beats_naive === undefined
                      ? "Unavailable"
                      : selectedModelResult.test_beats_naive
                        ? "Beat naive"
                        : "Did not beat naive"}
                  </strong>
                </div>
                <div>
                  <span>Baseline comparison</span>
                  <strong>
                    {comparison.selected_model_beats_naive
                      ? "Beat naive"
                      : "Did not beat naive"}
                  </strong>
                </div>
              </div>
              <div className="comparison-table-wrap">
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Family / status</th>
                      <th>Validation path MAE</th>
                      <th>Validation path RMSE</th>
                      <th>Final test MAE</th>
                      <th>sMAPE</th>
                      <th>Direction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(comparison.models)
                      .sort(([a], [b]) => {
                        const aIndex = comparison.ranking.indexOf(a);
                        const bIndex = comparison.ranking.indexOf(b);
                        return (aIndex < 0 ? 99 : aIndex) - (bIndex < 0 ? 99 : bIndex);
                      })
                      .map(([name, result]) => (
                        <tr
                          key={name}
                          className={name === comparison.selected_model ? "selected-row" : ""}
                        >
                          <td>{name.replaceAll("_", " ")}</td>
                          <td>{result.family} · {result.status}</td>
                          <td>{result.horizon_metrics?.mae.toFixed(4) ?? result.reason ?? "—"}</td>
                          <td>{result.horizon_metrics?.rmse.toFixed(4) ?? "—"}</td>
                          <td>{result.test_metrics?.mae.toFixed(4) ?? "—"}</td>
                          <td>{result.horizon_metrics ? `${result.horizon_metrics.smape.toFixed(2)}%` : "—"}</td>
                          <td>
                            {result.horizon_metrics
                              ? `${(result.horizon_metrics.directional_accuracy * 100).toFixed(0)}%`
                              : "—"}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              {selectedValidation.length > 0 && (
                <div className="validation-chart">
                  <h3>Walk-forward validation actual vs forecast</h3>
                  {validationPlot && (
                    <div className="validation-plot-wrap">
                      <div className="validation-legend" aria-hidden="true">
                        <span><i className="legend-actual" /> Actual</span>
                        <span><i className="legend-predicted" /> Selected model</span>
                        <span><i className="legend-naive" /> Naive</span>
                      </div>
                      <svg
                        className="validation-plot"
                        viewBox="0 0 760 220"
                        preserveAspectRatio="none"
                        role="img"
                        aria-label="Walk-forward validation actual prices compared with selected model and naive forecasts"
                      >
                        <line x1="40" y1="28" x2="740" y2="28" className="plot-grid" />
                        <line x1="40" y1="106" x2="740" y2="106" className="plot-grid" />
                        <line x1="40" y1="184" x2="740" y2="184" className="plot-grid" />
                        <text x="4" y="31" className="plot-label">
                          {marketPrice(validationPlot.maximum)}
                        </text>
                        <text x="4" y="187" className="plot-label">
                          {marketPrice(validationPlot.minimum)}
                        </text>
                        <polyline points={validationPlot.naive} className="plot-naive" />
                        <polyline points={validationPlot.actual} className="plot-actual" />
                        <polyline points={validationPlot.predicted} className="plot-predicted" />
                        {selectedValidation.map((point, index) => (
                          <g key={point.timestamp}>
                            <circle
                              cx={validationPlot.x(index)}
                              cy={validationPlot.y(point.actual)}
                              r="3"
                              className="plot-dot-actual"
                            />
                            <circle
                              cx={validationPlot.x(index)}
                              cy={validationPlot.y(point.predicted)}
                              r="3"
                              className="plot-dot-predicted"
                            />
                          </g>
                        ))}
                      </svg>
                    </div>
                  )}
                  <div className="validation-points">
                    {selectedValidation.map((point, index) => (
                      <div className="validation-point" key={point.timestamp}>
                        <span>{formatTimestamp(point.timestamp)}</span>
                        <span>Actual {marketPrice(point.actual)}</span>
                        <span>Forecast {marketPrice(point.predicted)}</span>
                        <span>
                          Naive{" "}
                          {marketPrice(
                            validationBaseline[index]?.predicted ?? point.actual,
                          )}
                        </span>
                      </div>
                    ))}
                  </div>
                  <small>
                    These are held-out fold forecasts, not in-sample fitted values. The displayed future forecast is for{" "}
                    {formatTimestamp(comparison.forecast.timestamp)}.
                  </small>
                </div>
              )}
              <p className="evaluation-note">{comparison.data_note}</p>
              {comparison.factor_analysis && (
                <div className="forecast-factor-analysis">
                  <div className="evaluation-heading">
                    <h3>Forecast factor analysis</h3>
                    <span className="evaluation-badge">
                      Target: daily closing price only
                    </span>
                  </div>
                  <p className="evaluation-note">
                    Inputs are restricted to information available at each training origin. Future OHLC values are not generated.
                  </p>
                  {comparison.factor_analysis.price_and_technical.feature_importance.length > 0 ? (
                    <div className="feature-importance-list">
                      {comparison.factor_analysis.price_and_technical.feature_importance.map((feature) => (
                        <div className="feature-importance-row" key={feature.feature}>
                          <span>{feature.feature.replaceAll("_", " ")}</span>
                          <i>
                            <b style={{ width: `${feature.importance * 100}%` }} />
                          </i>
                          <strong>{(feature.importance * 100).toFixed(1)}%</strong>
                        </div>
                      ))}
                      <small>
                        Importance method: {comparison.factor_analysis.price_and_technical.feature_importance[0].method}.
                      </small>
                    </div>
                  ) : (
                    <p className="sentiment-empty">
                      The selected statistical model does not expose comparable technical-feature importance.
                    </p>
                  )}
                  <div className="factor-availability-list">
                    {Object.entries(comparison.factor_analysis)
                      .filter(
                        ([name]) =>
                          name !== "price_and_technical" &&
                          name !== "volume_liquidity",
                      )
                      .map(([name, factor]) => (
                        <span
                          className={
                            factor.available
                              ? "factor-available"
                              : "factor-unavailable"
                          }
                          key={name}
                        >
                          {name.replaceAll("_", " ")} ·{" "}
                          {factor.available
                            ? `${factor.used_by_selected_model.length} inputs selected`
                            : "no leakage-safe historical inputs"}
                        </span>
                      ))}
                    <span className="factor-available">
                      Price/technical · {comparison.factor_analysis.price_and_technical.used_by_selected_model.length} inputs used
                    </span>
                    <span
                      className={
                        comparison.factor_analysis.volume_liquidity.available
                          ? "factor-available"
                          : "factor-unavailable"
                      }
                    >
                      Volume/liquidity ·{" "}
                      {comparison.factor_analysis.volume_liquidity.used_by_selected_model.length} inputs used
                    </span>
                  </div>
                  <small className="sentiment-correlation-note">
                    {comparison.factor_analysis.price_and_technical.limitation}{" "}
                    {comparison.factor_analysis.volume_liquidity.limitation}{" "}
                    {comparison.factor_analysis.fundamentals.limitation}{" "}
                    {comparison.factor_analysis.news_sentiment.limitation}{" "}
                    {comparison.factor_analysis.corporate_events.limitation}
                  </small>
                </div>
              )}
              <Suspense fallback={<p className="activity-message">Loading interactive forecast chart…</p>}>
                <ForecastChart
                  bars={bars}
                  symbol={symbol}
                  future={comparison.forecast.future}
                  lastClose={comparison.last_close}
                />
              </Suspense>
            </div>
          )}

          {comparison ? (
            <div className="forecast-result">
              <div>
                <span className="result-label">
                  {comparison.horizon_label} ·{" "}
                  {comparison.selected_model.replaceAll("_", " ")}
                </span>
                <strong>
                  {marketPrice(comparison.forecast.predicted_close)}
                </strong>
                <small>
                  Target time: {formatTimestamp(comparison.forecast.timestamp)}
                </small>
              </div>
              <div>
                <span className="result-label">Last observed close</span>
                <strong>{marketPrice(comparison.last_close)}</strong>
                <small>
                  Observed: {formatTimestamp(comparison.last_bar_timestamp)}
                </small>
              </div>
              <div>
                <span className="result-label">Validation status</span>
                <strong className="generated-value">
                  {comparison.selected_model_beats_naive
                    ? "Beat naive baseline"
                    : "Did not beat naive baseline"}
                </strong>
                <small>
                  Generated: {formatTimestamp(comparison.generated_at)}
                </small>
              </div>
              <div>
                <span className="result-label">80% empirical validation range</span>
                {finalForecastPoint?.lower_80 !== null &&
                finalForecastPoint?.upper_80 !== null &&
                finalForecastPoint ? (
                  <>
                    <strong className="generated-value">
                      {marketPrice(finalForecastPoint.lower_80)}{" "}
                      –{" "}
                      {marketPrice(finalForecastPoint.upper_80)}
                    </strong>
                    <small>
                      Based on {finalForecastPoint.calibration_samples} walk-forward errors; not a guarantee.
                    </small>
                  </>
                ) : (
                  <>
                    <strong className="generated-value">Unavailable</strong>
                    <small>Insufficient validation residuals for this estimate.</small>
                  </>
                )}
              </div>
            </div>
          ) : forecast ? (
            <div className="forecast-result">
              <div>
                <span className="result-label">Predicted next close</span>
                <strong>
                  {marketPrice(forecast.predicted_next_close)}
                </strong>
                <small>Forecast from {forecast.model}</small>
              </div>
              <div>
                <span className="result-label">Reference close</span>
                <strong>{marketPrice(forecast.last_close)}</strong>
                <small>Last bar: {formatTimestamp(forecast.last_bar_timestamp)}</small>
              </div>
              <div>
                <span className="result-label">Generated</span>
                <strong className="generated-value">{formatTimestamp(forecast.generated_at)}</strong>
                <small>{forecast.interpretation}</small>
              </div>
            </div>
          ) : (
            <p className="placeholder-message">
              Run a model comparison for a horizon-based recommendation, or train the Random Forest before requesting its next-bar forecast.
            </p>
          )}

          {metrics && (
            <div className="evaluation-block">
              <div className="evaluation-heading">
                <h3>Chronological holdout evaluation</h3>
                <span className={metrics.beats_baseline ? "evaluation-badge better" : "evaluation-badge"}>
                  {metrics.beats_baseline ? "Beat naive baseline" : "Did not beat naive baseline"}
                </span>
              </div>
              <div className="evaluation-grid">
                <div><span>MAE</span><strong>{metrics.mae.toFixed(4)}</strong></div>
                <div><span>RMSE</span><strong>{metrics.rmse.toFixed(4)}</strong></div>
                <div><span>Naive MAE</span><strong>{metrics.baseline_mae.toFixed(4)}</strong></div>
                <div><span>Holdout rows</span><strong>{metrics.test_rows}</strong></div>
              </div>
              <small className="evaluation-note">
                These holdout metrics are from the separate Random Forest training endpoint, not the walk-forward model comparison above.
              </small>
            </div>
          )}
        </section>

        <Suspense fallback={<p className="activity-message">Loading news sentiment tools…</p>}>
          <NewsSentimentPanel
            symbol={symbol}
            data={sentiment}
            loading={sentimentLoading}
            error={sentimentError}
            days={sentimentDays}
            onAnalyze={(days) => void analyzeSentiment(days)}
          />
        </Suspense>

        <footer className="app-footer">
          <span>MarketScope AI · Educational research prototype</span>
          <span>Market data availability and delay depend on the provider.</span>
        </footer>
          </>
        )}
      </main>
    </div>
  );
}
