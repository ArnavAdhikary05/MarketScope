# MarketScope AI

MarketScope AI is an educational market-research prototype for viewing provider-supplied OHLCV history and comparing forecasting models. The React dashboard displays interactive price/volume charts and exposes a chronological model comparison for supported tickers.

Market data availability, delay, history, and intraday coverage depend on Yahoo Finance. This is not a guaranteed real-time feed. Forecasts are experimental, are not investment advice, and do not imply profitable trading.

## Current functionality

- React, TypeScript, Vite, and TradingView Lightweight Charts frontend.
- FastAPI endpoints for health, historical prices, model training, next-bar prediction, and model comparison.
- Yahoo Finance history for provider-supported US and Indian symbols and supported intervals.
- Global stock-news analysis using Finviz where available and Yahoo Finance news as a fallback, with optional Groq headline classification, timestamp-aware aggregation, and Pearson correlation against post-publication returns.
- Current Yahoo Finance fundamentals and news sentiment are displayed as context only; supervised models use point-in-time price/volume factors and exchange benchmark history when available.
- OHLCV cleaning, rolling-return/moving-average/volatility/RSI/volume features.
- Multi-horizon daily forecasts for 1, 3, 7, 15, and 30 trading sessions and 3 or 6 calendar months, mapped to supported exchange calendars.
- Walk-forward comparison of naive baselines, Ridge, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, ARIMA, SARIMA, Exponential Smoothing, Prophet, and an optional LSTM.
- Automatic model recommendation is the default. It minimizes full-path MAE across earlier chronological validation folds, requires non-naive models to beat the naive baseline, and reports a later rolling-origin test period separately.
- Manual eligible-model selection, full-path validation and separate later test MAE/RMSE/sMAPE/bias/directional accuracy, an explicit naive-baseline comparison, forecast timestamps, and a daily recursive forecast path.
- Interactive candlestick, OHLC, line, and area price/volume charts plus a forecast chart with pan, zoom, hover values, and an empirical 80% walk-forward residual range when enough calibration errors exist.
- A Help & tutorial page with a step-by-step workflow, explanations of controls and metrics, model summaries, and provider/forecast limitations. Dashboard actions have hover and keyboard-focus tooltips.
- SQLite by default and PostgreSQL via `DATABASE_URL`; SQLAlchemy tables are created at startup.

Prophet, XGBoost, exchange calendars, and Finviz news integration are declared in `backend\requirements.txt`; unavailable optional models and provider data are identified rather than silently substituted. PyTorch is optional and is required to enable LSTM. Forecasts are daily-only. The horizon selector offers 1/3/7/15/30 trading sessions, 3/6 calendar months, and custom 1–132 trading sessions. Forecast timestamps use scheduled future exchange closes.

### Configure Groq sentiment (optional)

Copy `backend\.env.example` to `backend\.env` and set `GROQ_API_KEY` locally. Keep `.env` files private; the key is read only by the backend and is never returned by its APIs. Without a configured key, the app still shows provider news and available market factors but explicitly marks AI sentiment as unconfigured. For Docker Compose, export `GROQ_API_KEY` in the shell or set it in the root Compose `.env`; do not commit real credentials.

## Known limitations

This repository is still a prototype and does not implement the full original project vision:

- Walk-forward comparison uses at most eight expanding origins, not a final untouched test set or persisted experiment. Evaluation is exploratory and may be unstable.
- The displayed 80% interval is an empirical absolute-error range from same-horizon walk-forward residuals, not a formal coverage guarantee; it is unavailable when fewer than four calibration residuals exist for a forecast step.
- Multi-day model comparison supports daily observations only. Intraday price history remains available, but intraday multi-day forecasting is not supported.
- LSTM requires optional PyTorch installation and can be computationally expensive. A model that is unavailable or fails any fold is excluded from automatic selection and reports its status/reason.
- Market bars, model runs, comparisons, forecasts, or forecast outcomes are not persisted through the API. SQLAlchemy schemas exist, but there are no migrations or repository workflows.
- The legacy Random Forest training/prediction endpoints persist a local artifact; comparison uses an in-request model fit.
- Model comparisons run synchronously in the API process. There are no background jobs or cancellation.
- Technical-indicator selection, model-run history, watchlists, alerts, strategy backtesting APIs/UI, and application settings are not implemented.
- Docker Compose currently defines PostgreSQL and the backend only; run the frontend separately.

See [ARCHITECTURE.md](./ARCHITECTURE.md), [MODELING.md](./MODELING.md), [MODEL_SELECTION.md](./MODEL_SELECTION.md), [DATA_SOURCES.md](./DATA_SOURCES.md), [BACKTESTING.md](./BACKTESTING.md), and [TESTING.md](./TESTING.md) for implementation detail.

## Run locally on Windows PowerShell

### Backend

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location .\backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API listens at `http://127.0.0.1:8000`; OpenAPI documentation is at `http://127.0.0.1:8000/docs`. Without `DATABASE_URL`, the backend uses SQLite at `backend\stocks.db`.

### Frontend

In another PowerShell window:

```powershell
Set-Location .\frontend
npm install
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
npm run dev
```

The development server normally runs at `http://localhost:5173`. `VITE_API_BASE_URL` is optional and is only required when the API is not at the default URL.

## Docker

From the repository root:

```powershell
$env:POSTGRES_PASSWORD = "choose-a-local-password"
docker compose up --build
```

This starts PostgreSQL and the backend. Run the frontend separately using the preceding steps and configure its API URL to a backend address reachable by the browser. PostgreSQL is not published to a host port by the supplied Compose file.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Basic API health response |
| `GET` | `/api/stocks/{symbol}/history?interval=1d&period=1y` | Fetch cleaned provider history |
| `POST` | `/api/stocks/{symbol}/train?interval=1d` | Train and chronological-holdout evaluate the legacy Random Forest |
| `GET` | `/api/stocks/{symbol}/predict?interval=1d` | Predict the next bar using a trained artifact |
| `GET` | `/api/stocks/{symbol}/compare?interval=1d&period=10y&horizon=3d&model=auto&exchange=auto` | Compare models and forecast; `model` may be `auto` or an eligible model identifier |
| `GET` | `/api/stocks/{symbol}/sentiment?exchange=auto&days=90` | Retrieve 30/90-day timestamped news, optional Groq classifications, daily sentiment, forward-return correlation, current fundamentals, and benchmark snapshots |

Supported horizons: `3d`, `7d`, `15d`, `30d`, `3m`, and `6m`; numeric `horizon` values represent trading sessions from 1 to 132. Model comparisons include earlier full-path walk-forward validation and a later rolling-origin test period excluded from selection. Supported explicit exchanges are `NYSE`, `NASDAQ`, `NSE`, `BSE`, `LSE`, `TSX`, `TSXV`, `ASX`, `HKEX`, `JPX`, and `XETR`; `auto` uses ticker suffixes or Yahoo Finance exchange metadata. Comparison identifiers: `naive`, `seasonal_naive`, `ridge`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `xgboost`, `lstm`, `arima`, `sarima`, `exponential_smoothing`, and `prophet`; actual eligibility depends on data and installed dependencies.

## Validation commands

Frontend:

```powershell
Set-Location .\frontend
npm run build
npm run lint
```

Backend deterministic unit tests:

```powershell
Set-Location .\backend
python -m unittest discover -s tests -v
```

These tests use generated local fixture data and do not make market-provider requests. `backend\test_provider.py` is a separate manual live-provider smoke check.
