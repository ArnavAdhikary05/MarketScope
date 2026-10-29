# Forecast models and features

## Features and target

Features use observations available at the forecast origin: one- and five-bar close returns, SMA(5), SMA(20), EMA(12), rolling 20-return volatility, RSI(14), and percentage volume change. Infinite and incomplete feature rows are removed. RSI is set to 100 for gain-only windows, 0 for loss-only windows, and 50 for flat windows.

For comparison horizon `h`, the target is `close[t+h]`. This is a direct horizon-specific target. It is not a recursive path and does not create intermediate OHLC bars.

## Implemented models

| Identifier | Family | Method |
| --- | --- | --- |
| `naive` | Baseline | Last observed close |
| `seasonal_naive` | Baseline | Close five daily bars or 78 intraday bars before origin |
| `ridge` | Machine learning | Training-fold-fitted standardization followed by Ridge regression on the shared engineered features |
| `random_forest` | Machine learning | Fixed, seeded scikit-learn Random Forest |
| `extra_trees` | Machine learning | Fixed, seeded Extra Trees regressor |
| `hist_gradient_boosting` | Machine learning | Fixed, seeded histogram gradient boosting regressor |
| `arima` | Statistical | Statsmodels ARIMA(1,1,1), refit at each validation origin |
| `prophet` | Statistical | Prophet using `ds`/`y`, explicit future timestamps, and seasonality enabled only with minimum coverage |

Prophet is optional at import time so deployments without the package can still serve the API; comparison marks it unavailable. It is listed as a backend requirement.

## Evaluation and limitations

The model comparison uses the same chronological forecast origins and target timestamps for all successful candidates. Training target rows are purged so their target timestamp does not extend beyond the validation origin. All feature transformations are backward-looking; the tree and linear regressors are refit from the training fold only.

Metrics are MAE, RMSE, sMAPE, forecast bias, and directional accuracy. The latter is based on the change from the origin close to the horizon target. The legacy Random Forest training endpoint remains a separate one-step chronological holdout workflow.

No interval estimates are returned. The models do not implement LSTM, XGBoost, ensembles, log-return targets, or a separately reserved final test period. Small validation fold counts are exploratory and not evidence of stable future performance.
