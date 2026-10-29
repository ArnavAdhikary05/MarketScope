# Model comparison and selection

`GET /api/stocks/{symbol}/compare` fetches daily history, determines future session closes from the requested exchange calendar, and compares available model paths at up to eight expanding chronological origins. Horizons `3d`, `7d`, `15d`, and `30d` mean exact trading-session counts; `3m` and `6m` span calendar months and include scheduled exchange sessions through the end date. Numeric horizons from 1 to 132 also represent trading sessions.

Every candidate uses the same origins and target paths. A missing optional dependency, insufficient history, failed fit, or invalid forecast marks that candidate unavailable or failed and excludes it from ranking. No placeholder forecast is substituted.

## Automatic mode

Automatic mode is the default (`model=auto`). Selection minimizes full-path walk-forward validation MAE, with naive last-close forecasting as the baseline. A non-naive recommendation must strictly beat naive MAE; if none does, naive is recommended. RMSE, sMAPE, bias, directional accuracy, and baseline comparisons are also returned.

Selection uses up to eight expanding origins in the earlier 80% of history. A later rolling-origin test period is reported separately and is not used to select the model. At later test origins, already observed earlier test-period closes are included in training, as in repeated real-world forecasting. These metrics describe the available sample and do not establish stable future performance or a generally best model.

## Manual mode

The `model` parameter may specify any eligible model identifier. The chosen model is refit on all available history, then generates a recursive estimate for every future exchange session in the requested horizon. The response retains the automatic recommendation and separately identifies the manual selection.

## Uncertainty

The forecast includes empirical 80% absolute-error ranges at each future session when at least four same-horizon walk-forward residuals are available. These are small-sample residual ranges, not formal prediction intervals or guarantees of coverage. Missing ranges are explicitly returned as `null`.

## API example

```text
GET /api/stocks/AAPL/compare?interval=1d&period=10y&horizon=3d&model=auto&exchange=NYSE
```

`exchange` may be `auto`, `NYSE`, `NASDAQ`, `NSE`, `BSE`, `LSE`, `TSX`, `TSXV`, `ASX`, `HKEX`, `JPX`, or `XETR`. Supported model identifiers are `naive`, `seasonal_naive`, `ridge`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `xgboost`, `lstm`, `arima`, `sarima`, `exponential_smoothing`, and `prophet`.

The sentiment endpoint is `GET /api/stocks/{symbol}/sentiment?exchange=auto&days=90`; `days` may be 30 or 90. It returns provider news, optional Groq classifications, exchange-local daily sentiment, correlation pairs, current Yahoo Finance fundamentals, and relevant broad-index snapshots. Current fundamentals and market/news snapshots are context only, not features in historical validation, because the service does not have point-in-time history for them.
