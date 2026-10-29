import type { SentimentResponse } from "../api";

type Props = {
  symbol: string;
  data: SentimentResponse | null;
  loading: boolean;
  error: string;
  days: number;
  onAnalyze: (days: number) => void;
};

function formatTime(value?: string | null) {
  if (!value) return "Unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      });
}

function displayFactor(value: number | string) {
  if (typeof value !== "number") return value;
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 3,
    notation: Math.abs(value) >= 1_000_000 ? "compact" : "standard",
  }).format(value);
}

function SentimentTimeline({ data }: { data: SentimentResponse }) {
  const rows = data.sentiment.timeline;
  if (!rows.length) {
    return <p className="sentiment-empty">No classified articles are available for the timeline.</p>;
  }
  const x = (index: number) =>
    rows.length < 2 ? 400 : 30 + (index * 740) / (rows.length - 1);
  const y = (score: number) => 20 + ((1 - score) / 2) * 145;
  const points = rows.map((row, index) => `${x(index)},${y(row.score)}`).join(" ");

  return (
    <div className="sentiment-chart-wrap">
      <svg
        className="sentiment-timeline-chart"
        viewBox="0 0 800 205"
        role="img"
        aria-label="Daily AI news sentiment score timeline from negative one to positive one"
      >
        {[-1, 0, 1].map((score) => (
          <g key={score}>
            <line
              x1="30"
              x2="770"
              y1={y(score)}
              y2={y(score)}
              className="sentiment-gridline"
            />
            <text x="5" y={y(score) + 4} className="sentiment-axis-label">
              {score > 0 ? `+${score}` : score}
            </text>
          </g>
        ))}
        <polyline points={points} className="sentiment-timeline-line" />
        {rows.map((row, index) => (
          <circle
            key={row.date}
            cx={x(index)}
            cy={y(row.score)}
            r="4"
            className="sentiment-timeline-dot"
          >
            <title>
              {row.date} · Score {row.score.toFixed(2)} · {row.article_count} article(s)
            </title>
          </circle>
        ))}
        {rows.length > 0 && (
          <>
            <text x="30" y="195" className="sentiment-axis-label">
              {rows[0].date}
            </text>
            <text x="770" y="195" textAnchor="end" className="sentiment-axis-label">
              {rows.at(-1)?.date}
            </text>
          </>
        )}
      </svg>
    </div>
  );
}

function CorrelationScatter({ data }: { data: SentimentResponse }) {
  const pairs = data.correlation.pairs;
  if (pairs.length < 2) {
    return <p className="sentiment-empty">Not enough timestamp-aligned observations to draw a scatter plot.</p>;
  }
  const returns = pairs.map((pair) => pair.forward_return);
  const minReturn = Math.min(...returns);
  const maxReturn = Math.max(...returns);
  const span = maxReturn - minReturn || Math.max(Math.abs(minReturn) * 0.1, 0.01);
  const x = (score: number) => 28 + ((score + 1) / 2) * 340;
  const y = (value: number) => 22 + ((maxReturn - value) / span) * 166;
  const meanX =
    pairs.reduce((sum, pair) => sum + pair.sentiment_score, 0) / pairs.length;
  const meanY = returns.reduce((sum, value) => sum + value, 0) / pairs.length;
  const varianceX = pairs.reduce(
    (sum, pair) => sum + (pair.sentiment_score - meanX) ** 2,
    0,
  );
  const covariance = pairs.reduce(
    (sum, pair) =>
      sum + (pair.sentiment_score - meanX) * (pair.forward_return - meanY),
    0,
  );
  const slope = varianceX > 1e-12 ? covariance / varianceX : 0;
  const regressionLine =
    data.correlation.coefficient === null
      ? ""
      : `${x(-1)},${y(meanY + slope * (-1 - meanX))} ${x(1)},${y(meanY + slope * (1 - meanX))}`;

  return (
    <div className="sentiment-chart-wrap">
      <svg
        className="correlation-scatter-chart"
        viewBox="0 0 390 225"
        role="img"
        aria-label="Scatter plot of news sentiment against post-publication next-session returns"
      >
        <line x1="28" x2="368" y1="190" y2="190" className="sentiment-gridline" />
        <line x1="28" x2="28" y1="22" y2="190" className="sentiment-gridline" />
        <text x="198" y="218" textAnchor="middle" className="sentiment-axis-label">
          Daily news sentiment score
        </text>
        <text
          x="10"
          y="105"
          textAnchor="middle"
          className="sentiment-axis-label"
          transform="rotate(-90 10 105)"
        >
          Forward return
        </text>
        {regressionLine && (
          <polyline points={regressionLine} className="correlation-regression-line" />
        )}
        {pairs.map((pair) => (
          <circle
            key={`${pair.date}:${pair.available_at}`}
            cx={x(pair.sentiment_score)}
            cy={y(pair.forward_return)}
            r="4"
            className="correlation-scatter-dot"
          >
            <title>
              {pair.date} · Sentiment {pair.sentiment_score.toFixed(2)} · Forward return{" "}
              {(pair.forward_return * 100).toFixed(2)}%
            </title>
          </circle>
        ))}
      </svg>
    </div>
  );
}

export default function NewsSentimentPanel({
  symbol,
  data,
  loading,
  error,
  days,
  onAnalyze,
}: Props) {
  const distribution = data?.sentiment.distribution;
  const totalArticles = distribution
    ? distribution.positive + distribution.negative + distribution.neutral
    : 0;
  const factors = data?.factors.fundamentals.values ?? {};
  const availableFactors = Object.entries(factors);

  return (
    <section className="panel sentiment-panel" id="news-sentiment">
      <div className="sentiment-heading">
        <div>
          <p className="eyebrow">GLOBAL NEWS &amp; FACTORS</p>
          <h2>{symbol} news sentiment</h2>
          <p className="muted">
            Finviz with Yahoo Finance fallback; headline classification requires a configured Groq key.
          </p>
          <label className="sentiment-window-field">
            <span>News window</span>
            <select
              aria-label="Sentiment date range"
              disabled={loading}
              value={days}
              onChange={(event) => onAnalyze(Number(event.target.value))}
            >
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
            </select>
          </label>
        </div>
        <button
          className="button button-primary"
          type="button"
          disabled={loading || !symbol.trim()}
          onClick={() => onAnalyze(days)}
        >
          {loading ? "Analyzing…" : data ? "Refresh analysis" : "Analyze news sentiment"}
        </button>
      </div>

      {loading && (
        <p className="sentiment-status" role="status">
          Retrieving recent news, market context, and timestamp-aligned prices…
        </p>
      )}
      {error && (
        <div className="sentiment-error" role="alert">
          <span>{error}</span>
          <button type="button" className="button button-secondary" onClick={() => onAnalyze(days)}>
            Retry
          </button>
        </div>
      )}
      {!data && !loading && !error && (
        <p className="sentiment-empty">
          Run an analysis to retrieve provider news, sentiment classifications, correlation, and available market factors.
        </p>
      )}

      {data && (
        <>
          <div className="sentiment-source-row">
            <span>
              News: <strong>{data.news.source ?? "Unavailable"}</strong>
              {data.news.fallback_used ? " · fallback attempted" : ""}
            </span>
            <span>Retrieved {formatTime(data.news.retrieved_at)}</span>
            <span>
              Sentiment: <strong>{data.sentiment.status}</strong>
              {data.sentiment.model ? ` · ${data.sentiment.model}` : ""}
            </span>
          </div>

          {data.sentiment.reason && (
            <p className="sentiment-notice" role="status">{data.sentiment.reason}</p>
          )}

          <div className="sentiment-summary-grid">
            <article className="sentiment-card">
              <span>Overall classified sentiment</span>
              <strong className={
                data.sentiment.overall_score === null
                  ? ""
                  : data.sentiment.overall_score > 0
                    ? "sentiment-positive"
                    : data.sentiment.overall_score < 0
                      ? "sentiment-negative"
                      : ""
              }>
                {data.sentiment.overall_score === null
                  ? "Unavailable"
                  : data.sentiment.overall_score.toFixed(2)}
              </strong>
              <small>Article-level scale: −1 to +1</small>
            </article>
            <article className="sentiment-card">
              <span>Pearson correlation</span>
              <strong>
                {data.correlation.coefficient === null
                  ? "Unavailable"
                  : data.correlation.coefficient.toFixed(3)}
              </strong>
              <small>
                n={data.correlation.sample_size}
                {data.correlation.p_value === null
                  ? " · p-value unavailable"
                  : ` · nominal p=${data.correlation.p_value.toFixed(3)}`}
              </small>
            </article>
            <article className="sentiment-card">
              <span>Retrieved articles</span>
              <strong>{data.news.items.length}</strong>
              <small>
                {totalArticles} classified · {data.exchange} local time
              </small>
            </article>
          </div>

          <div className="sentiment-charts-grid">
            <div className="sentiment-chart-card">
              <h3>Daily sentiment timeline</h3>
              <SentimentTimeline data={data} />
              <div className="sentiment-distribution" aria-label="Sentiment distribution">
                {(["positive", "neutral", "negative"] as const).map((label) => {
                  const count = distribution?.[label] ?? 0;
                  const width = totalArticles ? (count / totalArticles) * 100 : 0;
                  return (
                    <div className={`sentiment-distribution-item ${label}`} key={label}>
                      <span>{label} · {count}</span>
                      <div className="sentiment-distribution-track">
                        <i style={{ width: `${width}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="sentiment-chart-card">
              <h3>Sentiment vs forward return</h3>
              <CorrelationScatter data={data} />
              <p className="sentiment-correlation-note">
                {data.correlation.note}
              </p>
            </div>
          </div>

          <div className="sentiment-detail-grid">
            <div className="sentiment-detail-card">
              <h3>Main news themes</h3>
              {data.sentiment.themes.length ? (
                <ul className="sentiment-theme-list">
                  {data.sentiment.themes.map((theme) => (
                    <li key={theme.theme}>
                      <span>{theme.theme}</span><strong>{theme.count}</strong>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="sentiment-empty">Themes are unavailable without classified news.</p>
              )}
              {data.sentiment.significant_changes.length > 0 && (
                <div className="sentiment-changes">
                  <strong>Large daily sentiment shifts</strong>
                  {data.sentiment.significant_changes.map((change) => (
                    <p key={change.date}>
                      {change.date}: {change.from_score.toFixed(2)} → {change.to_score.toFixed(2)}
                    </p>
                  ))}
                </div>
              )}
            </div>

            <div className="sentiment-detail-card">
              <h3>Available fundamentals &amp; market context</h3>
              {availableFactors.length ? (
                <dl className="fundamental-list">
                  {availableFactors.map(([name, value]) => (
                    <div key={name}>
                      <dt>{name.replaceAll("_", " ")}</dt>
                      <dd>{displayFactor(value)}</dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p className="sentiment-empty">Yahoo Finance did not provide a current fundamental snapshot.</p>
              )}
              <div className="benchmark-list">
                {data.factors.market_indices.map((index) => (
                  <span key={index.symbol}>
                    {index.symbol}:{" "}
                    {index.change_20_sessions === null
                      ? "unavailable"
                      : `${(index.change_20_sessions * 100).toFixed(2)}% over ~20 sessions`}
                    {index.last_observed_at
                      ? ` · ${formatTime(index.last_observed_at)}`
                      : ""}
                  </span>
                ))}
              </div>
              <div className="factor-availability-list">
                <span className="factor-unavailable">
                  Sector performance · {data.factors.sector_performance.status}
                </span>
                <span className="factor-unavailable">
                  Macro data · {data.factors.macroeconomic.status}
                </span>
                <span className="factor-unavailable">
                  Corporate events · {data.factors.corporate_events.status}
                </span>
              </div>
              <p className="sentiment-correlation-note">
                {data.factors.forecast_usage}
              </p>
            </div>
          </div>

          <div className="sentiment-news-list">
            <h3>Recent news</h3>
            {data.news.items.length ? (
              data.news.items.map((item) => (
                <article className="sentiment-news-item" key={item.id}>
                  <div className="sentiment-news-meta">
                    <span>{item.source}</span>
                    <time dateTime={item.published_at}>{formatTime(item.published_at)}</time>
                    {item.sentiment && (
                      <span className={`sentiment-label ${item.sentiment.toLowerCase()}`}>
                        {item.sentiment} · {item.score?.toFixed(2)}
                      </span>
                    )}
                  </div>
                  <h4>
                    {item.url ? (
                      <a href={item.url} target="_blank" rel="noreferrer">
                        {item.title}
                      </a>
                    ) : item.title}
                  </h4>
                  {item.summary && <p>{item.summary}</p>}
                  {item.rationale && (
                    <small>{item.theme ? `${item.theme}: ` : ""}{item.rationale}</small>
                  )}
                </article>
              ))
            ) : (
              <p className="sentiment-empty">
                No timestamped articles were returned. Provider failure or symbol coverage may vary; retry later.
              </p>
            )}
          </div>

          <ul className="sentiment-limitations">
            {data.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
          </ul>
        </>
      )}
    </section>
  );
}
