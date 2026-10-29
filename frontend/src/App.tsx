import { useEffect, useMemo, useState, type ReactNode } from "react";
import { isAxiosError } from "axios";
import {
  compareStock,
  getHistory,
  predictStock,
  trainModel,
  type ComparisonResult,
  type ForecastResponse,
  type HistoryResponse,
  type TrainingMetrics,
} from "./api";
import StockChart from "./components/StockChart";
import HelpPage from "./components/HelpPage";
import "./App.css";

const EMPTY_VALIDATION: NonNullable<
  ComparisonResult["models"][string]["validation_predictions"]
> = [];

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

function formatPrice(value: number, ticker: string) {
  const currency = /\.(NS|BO)$/i.test(ticker) ? "INR" : "USD";
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
  const [interval, setInterval] = useState("1d");
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [horizon, setHorizon] = useState(1);
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

    setActiveAction("comparison");
    setError("");
    setComparison(null);
    setForecast(null);
    setMetrics(null);
    try {
      const historyPeriod = interval === "1d" ? "1y" : "5d";
      const comparisonPeriod = interval === "1d" ? "5y" : "60d";
      const [marketHistory, result] = await Promise.all([
        getHistory(normalizedSymbol, interval, historyPeriod),
        compareStock(
          normalizedSymbol,
          interval,
          comparisonPeriod,
          requestedHorizon,
          requestedModel,
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

  const isBusy = activeAction !== null;
  const supportedHorizons = interval === "1d" ? [1, 5, 10, 20] : [1, 3, 6, 12];
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
            timestamp: comparison.forecast.timestamp,
            predicted_close: comparison.forecast.predicted_close,
            last_bar_timestamp: comparison.last_bar_timestamp,
            last_close: comparison.last_close,
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
              />
            </label>
            <label className="field">
              <span>Interval</span>
              <select
                disabled={isBusy}
                value={interval}
                onChange={(event) => {
                  setInterval(event.target.value);
                  setHorizon(1);
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
                ? formatPrice(latestBar.close, history?.symbol ?? symbol)
                : "—"}
            </strong>
            <small>{latestBar ? formatTimestamp(latestBar.timestamp) : "Load data to begin"}</small>
          </article>
          <article className="metric-card">
            <div className="metric-heading"><span>Change</span><span className="metric-icon">↕</span></div>
            <strong className={change ? (change.amount >= 0 ? "positive" : "negative") : ""}>
              {change
                ? `${change.amount >= 0 ? "+" : ""}${formatPrice(
                    change.amount,
                    history?.symbol ?? symbol,
                  )}`
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
                Compare real baseline, statistical, and machine-learning forecasts for this ticker and horizon.
              </p>
            </div>
            <div className="action-group">
              <label className="field">
                <span>Forecast horizon</span>
                <select
                  aria-label="Forecast horizon"
                  disabled={isBusy}
                  value={horizon}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    setHorizon(value);
                    if (comparison) void runComparison(modelChoice, value);
                  }}
                >
                  {supportedHorizons.map((value) => (
                    <option key={value} value={value}>
                      {value} {interval === "1d" ? "trading day(s)" : "bar(s)"}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Forecast model</span>
                <select
                  aria-label="Forecast model"
                  disabled={isBusy}
                  value={modelChoice}
                  onChange={(event) => {
                    const value = event.target.value;
                    setModelChoice(value);
                    if (comparison) void runComparison(value);
                  }}
                >
                  <option value="auto">Recommended automatically</option>
                  {Object.entries(comparison?.models ?? {}).map(([name, result]) => (
                    <option key={name} value={name} disabled={!result.eligible}>
                      {name.replaceAll("_", " ")}
                      {!result.eligible ? " (unavailable)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <Tooltip text="Evaluate eligible models on chronological folds and generate a forecast for the chosen horizon.">
                <button
                  className="button button-primary"
                  disabled={isBusy}
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
                {comparison.selection_objective}. Scores use {comparison.validation_folds} expanding chronological folds;
                {` `}selected-model MAE {selectedModelResult?.metrics?.mae.toFixed(4) ?? "unavailable"} vs naive MAE{" "}
                {comparison.models.naive?.metrics?.mae.toFixed(4) ?? "unavailable"}.
              </p>
              <div className="comparison-table-wrap">
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Family / status</th>
                      <th>Validation MAE</th>
                      <th>RMSE</th>
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
                          <td>{result.metrics?.mae.toFixed(4) ?? result.reason ?? "—"}</td>
                          <td>{result.metrics?.rmse.toFixed(4) ?? "—"}</td>
                          <td>{result.metrics ? `${result.metrics.smape.toFixed(2)}%` : "—"}</td>
                          <td>
                            {result.metrics
                              ? `${(result.metrics.directional_accuracy * 100).toFixed(0)}%`
                              : "—"}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              {selectedValidation.length > 0 && (
                <div className="validation-chart">
                  <h3>Out-of-sample actual vs forecast</h3>
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
                        aria-label="Held-out actual prices compared with selected model and naive forecasts"
                      >
                        <line x1="40" y1="28" x2="740" y2="28" className="plot-grid" />
                        <line x1="40" y1="106" x2="740" y2="106" className="plot-grid" />
                        <line x1="40" y1="184" x2="740" y2="184" className="plot-grid" />
                        <text x="4" y="31" className="plot-label">
                          {formatPrice(validationPlot.maximum, symbol)}
                        </text>
                        <text x="4" y="187" className="plot-label">
                          {formatPrice(validationPlot.minimum, symbol)}
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
                        <span>Actual {formatPrice(point.actual, symbol)}</span>
                        <span>Forecast {formatPrice(point.predicted, symbol)}</span>
                        <span>
                          Naive{" "}
                          {formatPrice(
                            validationBaseline[index]?.predicted ?? point.actual,
                            symbol,
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
            </div>
          )}

          {comparison ? (
            <div className="forecast-result">
              <div>
                <span className="result-label">
                  {comparison.horizon}-bar forecast ·{" "}
                  {comparison.selected_model.replaceAll("_", " ")}
                </span>
                <strong>
                  {formatPrice(comparison.forecast.predicted_close, symbol)}
                </strong>
                <small>
                  Target time: {formatTimestamp(comparison.forecast.timestamp)}
                </small>
              </div>
              <div>
                <span className="result-label">Last observed close</span>
                <strong>{formatPrice(comparison.last_close, symbol)}</strong>
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
            </div>
          ) : forecast ? (
            <div className="forecast-result">
              <div>
                <span className="result-label">Predicted next close</span>
                <strong>
                  {formatPrice(forecast.predicted_next_close, symbol)}
                </strong>
                <small>Forecast from {forecast.model}</small>
              </div>
              <div>
                <span className="result-label">Reference close</span>
                <strong>{formatPrice(forecast.last_close, symbol)}</strong>
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
