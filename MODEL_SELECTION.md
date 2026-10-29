# Model comparison and selection

`GET /api/stocks/{symbol}/compare` retrieves historical bars, builds a horizon-specific target, and evaluates model candidates at five expanding chronological origins. The first origin is after the initial 60% of usable history; later origins span the remainder. For each origin, model training rows are purged so that every training target is observable by that origin.

All models are scored on the same origin/target pairs. A model that fails to fit, lacks sufficient training rows, or lacks its optional dependency is marked failed or unavailable and is excluded from ranking. No placeholder predictions are substituted.

## Automatic mode

The objective is validation MAE. The naive last-close forecast is always the reference. The system recommends the lowest-MAE candidate only when it is the naive model itself or has lower MAE than naive; if no candidate beats naive, the naive model is selected. RMSE, sMAPE, bias, directional accuracy, and baseline-relative metrics remain visible as context and do not override the stated objective.

This is a prototype comparison with five folds, not a final unbiased benchmark. Repeatedly querying symbols/horizons and observing results can still lead to selection bias. There is no final holdout set, persisted experiment, cache, or model versioning.

## Manual mode

The `model` query parameter can name any eligible model. The selected model is fit against all target labels available as of the most recent observation and returns a single direct forecast at the requested horizon. The forecast response and UI label manual versus automatic selection.

## API example

```text
GET /api/stocks/AAPL/compare?interval=1d&period=5y&horizon=5&model=auto
```

The result includes the selected and recommended identifiers, per-model metrics/status, held-out fold forecasts, last input timestamp, generated time, and the forecast target timestamp/value.
