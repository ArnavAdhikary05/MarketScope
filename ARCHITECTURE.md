# Architecture

MarketScope AI currently uses a small synchronous client/server architecture:

```text
React + TypeScript + Lightweight Charts
           |
           | HTTP / JSON (VITE_API_BASE_URL)
           v
FastAPI routers (backend/app/main.py)
           |
     +-----+-------------------+
     |                         |
Yahoo Finance provider    Forecast services
(OHLCV normalization)     (features, comparison)
     |                         |
     +---------- pandas / scikit-learn / statsmodels / Prophet / XGBoost
                   pandas-market-calendars / Finviz / Yahoo news / Groq API
```

The frontend owns ticker, interval, horizon, exchange, model-selection, loading, and error state. `api.ts` is the HTTP client. `StockChart` manages Lightweight Charts instances and displays provider bars, volume, and an optional selected-model forecast segment. `ForecastChart` displays the history, recursive forecast path, and available empirical residual ranges with SVG-based pan, zoom, and hover interaction.

The backend exposes health, history, legacy train/predict, comparison, and sentiment routes. Price provider integration is isolated in `providers/yahoo.py`; news retrieval, Groq classification, time-aligned correlation, and context snapshots are in `services/news_sentiment.py`. Feature calculation is in `services/features.py`; horizon-aware validation and model selection are in `services/model_comparison.py`. Sentiment responses are cached briefly in process and not persisted because news retention/provider restrictions may apply. Existing SQLAlchemy models are otherwise unchanged; routes do not persist analytics records.

Comparison and fitting execute synchronously inside a request. The API has no queue, cache, cancellation, or durable comparison/job store. Daily forecast dates are derived from NYSE, NASDAQ, NSE, or BSE session calendars. The Docker file and Compose stack run PostgreSQL and the backend; the frontend is a separate development process.

## Boundaries

- The provider returns adjusted OHLCV from yfinance, not licensed exchange-grade live data.
- The comparison includes only models that produce valid full paths for all common validation origins.
- Calendar-based multi-day forecasts require daily bars; optional LSTM forecasts require PyTorch.
- Empirical 80% residual ranges are withheld when fewer than four calibration errors are available and are not formal coverage guarantees.
- Model artifact loading in the legacy prediction route uses the local artifact directory.
- Background workers, persistence workflows, and production authentication are not implemented.
