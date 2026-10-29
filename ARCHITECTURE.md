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
     +---------- pandas / scikit-learn / statsmodels / Prophet
```

The frontend owns ticker, interval, horizon, model-selection, loading, and error state. `api.ts` is the HTTP client. `StockChart` manages Lightweight Charts instances and displays provider bars, volume, and an optional selected-model forecast segment.

The backend exposes health, history, legacy train/predict, and comparison routes. Provider integration is isolated in `providers/yahoo.py`. Feature calculation is in `services/features.py`; horizon-aware validation and model selection are in `services/model_comparison.py`. Existing SQLAlchemy models and database configuration are present, but routes do not persist records. `create_all` is currently used at startup instead of schema migrations.

Comparison and fitting execute synchronously inside a request. The API has no queue, cache, cancellation, or durable comparison/job store. The Docker file and Compose stack run PostgreSQL and the backend; the frontend is a separate development process.

## Boundaries

- The provider returns adjusted OHLCV from yfinance, not licensed exchange-grade live data.
- The comparison includes only models that produce outputs for the common validation origins.
- Model artifact loading in the legacy prediction route uses the local artifact directory.
- Forecast intervals, exchange calendars, background workers, persistence workflows, and production authentication are not implemented.
