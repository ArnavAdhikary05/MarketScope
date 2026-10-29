# MarketScope AI

MarketScope AI is an educational market-research prototype for viewing provider-supplied OHLCV history and comparing forecasting models. The React dashboard displays interactive price/volume charts and exposes a chronological model comparison for supported tickers.

Market data availability, delay, history, and intraday coverage depend on Yahoo Finance. This is not a guaranteed real-time feed. Forecasts are experimental, are not investment advice, and do not imply profitable trading.

## Current functionality

- React, TypeScript, Vite, and TradingView Lightweight Charts frontend.
- FastAPI endpoints for health, historical prices, model training, next-bar prediction, and model comparison.
- Yahoo Finance history for provider-supported US and Indian symbols and supported intervals.
- OHLCV cleaning, rolling-return/moving-average/volatility/RSI/volume features.
- Model comparison for last-value and seasonal-naive baselines, Ridge, Random Forest, Extra Trees, HistGradientBoosting, ARIMA(1,1,1), and Prophet.
- Five expanding chronological validation origins with a horizon-aware label purge. Automatic selection minimizes validation MAE and requires a non-naive candidate to beat the naive MAE; otherwise naive is recommended.
- Manual model selection, per-model validation metrics, held-out actual/prediction values, and an endpoint forecast for the chosen horizon.
- Interactive candlestick, OHLC, line, and area charts with real volume; the selected model forecast is drawn as a distinct dotted segment.
- A Help & tutorial page with a step-by-step workflow, explanations of controls and metrics, model summaries, and provider/forecast limitations. Dashboard actions have hover and keyboard-focus tooltips.
- SQLite by default and PostgreSQL via `DATABASE_URL`; SQLAlchemy tables are created at startup.

Prophet is declared in `backend\requirements.txt`; the model is marked unavailable if the package cannot be imported. The UI offers daily horizons of 1, 5, 10, and 20 bars and intraday horizons of 1, 3, 6, and 12 bars. The backend accepts horizons from 1 through 20.

## Known limitations

This repository is still a prototype and does not implement the full original project vision:

- The current comparison uses a small fixed number of expanding validation origins, not a final untouched test set or persisted walk-forward experiment.
- Future daily timestamps use weekdays and do not apply exchange holiday calendars. Intraday future timestamps are arithmetic intervals and do not model exchange sessions.
- Prophet intervals are not surfaced, and no model currently supplies a calibrated uncertainty interval.
- XGBoost, LSTM/PyTorch, and additional model families are not implemented.
- Market bars, model runs, comparisons, forecasts, or forecast outcomes are not persisted through the API. SQLAlchemy schemas exist, but there are no migrations or repository workflows.
- The legacy Random Forest training/prediction endpoints persist a local artifact; comparison uses an in-request model fit.
- Model comparisons run synchronously in the API process. There are no background jobs or cancellation.
- Technical-indicator selection, actual-versus-predicted charts, model-run history, watchlists, alerts, strategy backtesting APIs/UI, and application settings are not implemented.
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
| `GET` | `/api/stocks/{symbol}/compare?interval=1d&period=5y&horizon=1&model=auto` | Compare models and forecast; `model` may be `auto` or an eligible model identifier |

Comparison identifiers: `naive`, `seasonal_naive`, `ridge`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `arima`, and `prophet`.

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
