import type { ReactNode } from "react";

type Props = {
  onBack: () => void;
};

function Tooltip({ text, children }: { text: string; children: ReactNode }) {
  return (
    <span className="tooltip-wrap" data-tooltip={text} title={text}>
      {children}
    </span>
  );
}

export default function HelpPage({ onBack }: Props) {
  return (
    <div className="help-page">
      <section className="help-hero">
        <div>
          <p className="eyebrow">MARKETSCOPE AI · QUICK START</p>
          <h2>Understand the dashboard. Use forecasts responsibly.</h2>
          <p>
            This guide walks through the working market-data and forecasting
            features, explains the controls, and describes what the model
            comparison results mean.
          </p>
        </div>
        <Tooltip text="Return to the dashboard to load market data or run an analysis.">
          <button className="button button-primary" type="button" onClick={onBack}>
            Back to overview
          </button>
        </Tooltip>
      </section>

      <section className="help-section">
        <div className="help-section-heading">
          <span className="help-step">01</span>
          <div>
            <h2>Quick start</h2>
            <p>Load market history, then explore the chart or run a forecast.</p>
          </div>
        </div>
        <ol className="tutorial-steps">
          <li>
            <strong>Choose a ticker.</strong>
            <span>
              Enter a provider-supported symbol. Examples include <code>AAPL</code>,
              {" "}<code>MSFT</code>, or Indian listings such as <code>RELIANCE.NS</code>.
              Availability depends on Yahoo Finance.
            </span>
          </li>
          <li>
            <strong>Choose an exchange calendar.</strong>
            <span>
              Auto detect uses the Indian ticker suffix or Yahoo Finance exchange
              metadata for international tickers. You may select an available
              exchange explicitly if detection is unavailable.
            </span>
          </li>
          <li>
            <strong>Select an interval and load history.</strong>
            <span>
              Daily loads approximately one year for the chart; 5-minute and
              15-minute intervals load approximately five days. Provider history
              and intraday retention can vary.
            </span>
          </li>
          <li>
            <strong>Inspect the market chart.</strong>
            <span>
              Use the chart type selector for candlesticks, OHLC bars, a close
              line, or an area chart. The volume pane uses provider-supplied volume.
              Drag to pan and use the chart controls to zoom.
            </span>
          </li>
          <li>
            <strong>Compare models and forecast.</strong>
            <span>
              Pick a 1-trading-day horizon, a 3-day, 7-day, 15-day, or 30-day
              calendar window, a 3- or 6-month window, or enter a custom count
              of 1–132 trading sessions.
              Select Daily data, leave the mode on automatic recommendation or
              choose an eligible model after a comparison, then use
              <em> Compare &amp; forecast</em>.
            </span>
          </li>
          <li>
            <strong>Review the evidence.</strong>
            <span>
              Read the leaderboard, held-out actual-versus-prediction plot, naive
              baseline comparison, and the forecast target timestamp. An unavailable
              or failed model is not silently treated as a successful forecast.
            </span>
          </li>
        </ol>
      </section>

      <section className="help-section">
        <div className="help-section-heading">
          <span className="help-step">02</span>
          <div>
            <h2>What the controls do</h2>
            <p>Each dashboard action works with the currently selected ticker and interval.</p>
          </div>
        </div>
        <div className="help-cards">
          <article className="help-card">
            <h3>Load data</h3>
            <p>Requests OHLCV history from the backend provider and updates the chart and summary cards.</p>
          </article>
          <article className="help-card">
            <h3>Forecast horizon</h3>
            <p>
            Forecast options include 1, 3, 7, 15, and 30 trading sessions,
            3 or 6 calendar months, and custom counts of 1–132 sessions.
            Dates come from the selected exchange calendar. Press Compare
            &amp; forecast after entering a custom number of sessions.
            </p>
          </article>
          <article className="help-card">
            <h3>News sentiment</h3>
            <p>
              Analyze recent provider news over a 30- or 90-day window. Finviz
              is tried first and Yahoo Finance is the fallback. Groq
              classifications require a backend-only GROQ_API_KEY. Sentiment
              and next-session return correlation are exploratory and do not
              establish causation.
            </p>
          </article>
          <article className="help-card">
            <h3>Forecast model</h3>
            <p>
              Automatically Recommend Best Model is the default. It selects by
              walk-forward MAE and requires a non-naive model to beat the naive
              baseline. Choosing an eligible model manually reruns the comparison
              and forecast for that model.
            </p>
          </article>
          <article className="help-card">
            <h3>Compare &amp; forecast</h3>
            <p>
              Evaluates naive baselines, Ridge, Random Forest, Extra Trees,
              HistGradientBoosting, XGBoost, ARIMA, SARIMA, Exponential Smoothing,
              Prophet, and PyTorch LSTM when dependencies and data permit.
            </p>
          </article>
          <article className="help-card">
            <h3>Train model</h3>
            <p>
              Trains the separate Random Forest model and reports a chronological
              holdout evaluation. This is independent of the comparison leaderboard.
            </p>
          </article>
          <article className="help-card">
            <h3>Predict next bar</h3>
            <p>
              Uses the saved Random Forest artifact for a next-bar estimate. Train
              first if the API reports that no saved model exists.
            </p>
          </article>
        </div>
      </section>

      <section className="help-section">
        <div className="help-section-heading">
          <span className="help-step">03</span>
          <div>
            <h2>Models and scores</h2>
            <p>The leaderboard reports measured validation results, not promises of future performance.</p>
          </div>
        </div>
        <div className="help-model-list">
          <p><strong>Naive:</strong> predicts the latest close; it is the key reference baseline.</p>
          <p><strong>Seasonal naive:</strong> uses a prior observation as a simple seasonal reference.</p>
          <p><strong>Ridge:</strong> linear regression with regularization over engineered price/volume features.</p>
          <p><strong>Random Forest and Extra Trees:</strong> tree ensembles trained on lagged/rolling features.</p>
          <p><strong>HistGradientBoosting:</strong> a boosted-tree regression model.</p>
          <p><strong>ARIMA:</strong> a statistical autoregressive integrated moving-average model.</p>
          <p><strong>SARIMA and Exponential Smoothing:</strong> classical seasonal/trend models evaluated against the same validation dates.</p>
          <p><strong>XGBoost:</strong> gradient-boosted trees over leakage-free lagged returns; unavailable if its package cannot load.</p>
          <p><strong>LSTM:</strong> PyTorch sequence model with training-only return scaling and early stopping; optional and unavailable when PyTorch is not installed.</p>
          <p><strong>Prophet:</strong> a trend/seasonality model when its dependency is available and data is eligible.</p>
        </div>
        <div className="help-cards metric-help-grid">
          <article className="help-card">
            <h3>MAE and RMSE</h3>
            <p>Average absolute error and square-root mean squared error, in the same units as the price. Lower is better.</p>
          </article>
          <article className="help-card">
            <h3>sMAPE and bias</h3>
            <p>sMAPE reports percentage-scaled error; bias indicates whether predictions tend to be above or below actual values.</p>
          </article>
          <article className="help-card">
            <h3>Directional accuracy</h3>
            <p>Share of evaluated moves with the correct direction. It does not replace price-error metrics or indicate profitability.</p>
          </article>
        </div>
      </section>

      <section className="help-section help-warning">
        <div className="help-section-heading">
          <span className="help-step">!</span>
          <div>
            <h2>Data and forecast limitations</h2>
            <p>Use this application for education and research, not as investment advice.</p>
          </div>
        </div>
        <ul>
          <li>Yahoo Finance data can be delayed, incomplete, or unavailable; this app does not guarantee a live feed.</li>
          <li>Validation scores describe historical folds and do not guarantee future accuracy or investment returns.</li>
          <li>Automatic selection uses validation MAE and requires advanced candidates to beat the naive baseline; otherwise it can recommend the naive model.</li>
          <li>Forecast dates use supported US, Indian, UK, Canadian, Australian, Hong Kong, Japanese, and German session calendars. Automatic exchange lookup depends on ticker suffixes or Yahoo Finance metadata; select an exchange explicitly if lookup fails.</li>
          <li>The 80% interval is an empirical walk-forward residual range. It is not a guaranteed coverage probability; it may be unavailable when too few validation folds can be formed.</li>
          <li>Forecast comparisons require daily bars. Intraday charts are supported, but multi-day intraday model comparisons are not.</li>
        </ul>
      </section>

      <footer className="help-footer">
        <span>Need to try a workflow? Return to the overview and start with a ticker.</span>
        <Tooltip text="Return to the dashboard.">
          <button className="button button-secondary" type="button" onClick={onBack}>
            Go to overview
          </button>
        </Tooltip>
      </footer>
    </div>
  );
}
