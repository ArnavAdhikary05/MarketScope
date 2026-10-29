# Forecast models and evaluation

## Horizons and data

Forecasting currently accepts daily bars only. The horizon selector offers 1, 3, 7, 15, or 30 trading sessions, 3 or 6 calendar months, and a custom count of 1–132 trading sessions. Named day horizons are exact counts of scheduled sessions from the selected exchange calendar; month horizons end on the corresponding calendar date and include only scheduled session closes.

The exchange may be selected as NYSE, NASDAQ, NSE, BSE, LSE, TSX, TSXV, ASX, HKEX, JPX, or XETR. Automatic resolution uses Indian ticker suffixes where available and otherwise provider exchange metadata. If automatic resolution is unavailable, choose an exchange explicitly. Historical observations are cleaned and sorted; missing market sessions are not imputed.

## Forecast methods

Every model generates a recursive daily price path from the latest observation. The naive baselines repeat the last close or a five-session seasonal value. Statistical methods operate on log prices: ARIMA(1,1,1), SARIMA(1,1,1)(1,0,1,5), and additive Exponential Smoothing. Prophet fits log prices against dates. Supervised learners predict one-step log returns from trailing return, mean, volatility, and price-deviation features and recursively feed estimates back into their own history.

Machine-learning forecast models additionally use lagged OHLCV-derived factors: opening gap, intraday range, 5/20-session volatility and moving-average deviations, EMA deviation, RSI, MACD histogram, Bollinger position, ATR, 20-session daily VWAP proxy, volume change/z-score, OBV change, support/resistance distances, turnover deviation, and 10/20-session momentum. Each training target uses features no later than its preceding observation. A training-fold-only selector drops constant inputs and retains one target-correlated representative of highly redundant features; those same columns are used for that fold's recursive forecast. During the future recursive path, the last observed technical/volume factor vector is held fixed because future OHLCV is not forecast. Statistical models use close history only. Tree feature importance is model-specific split importance; Ridge reports absolute standardized coefficient magnitudes, and neither is a causal explanation.

Current fundamentals and current news sentiment are surfaced separately and excluded from model selection because historical point-in-time financial releases and archived news are unavailable. When Yahoo provides the appropriate exchange benchmark history, its returns and rolling volatility are aligned backward to each stock-bar timestamp and can be selected as lagged factors for supervised tree/Ridge models. The latest known benchmark features are held fixed in recursive future paths; future market returns are never assumed. Sector history, FX, rates, macro indicators, and point-in-time event data remain unavailable as forecast inputs.

| Identifier | Family | Method |
| --- | --- | --- |
| `naive` | Baseline | Repeat the last observed close |
| `seasonal_naive` | Baseline | Repeat the value from five sessions earlier |
| `ridge` | Machine learning | Standardized Ridge regression on trailing log-return features |
| `random_forest` | Machine learning | Seeded Random Forest regressor |
| `extra_trees` | Machine learning | Seeded Extra Trees regressor |
| `hist_gradient_boosting` | Machine learning | Seeded histogram gradient boosting regressor |
| `xgboost` | Machine learning | XGBoost regressor |
| `lstm` | Deep learning | Small PyTorch LSTM on standardized return sequences; optional dependency |
| `arima` | Statistical | Statsmodels ARIMA(1,1,1) on log prices |
| `sarima` | Statistical | Statsmodels SARIMAX with a five-session seasonal component |
| `exponential_smoothing` | Statistical | Additive trend and (when enough data exist) five-session seasonality |
| `prophet` | Statistical | Prophet on log prices and explicit future session dates |

Prophet and XGBoost are declared backend dependencies. PyTorch is optional. When an optional dependency is missing, that model is marked unavailable with an explanatory reason rather than replaced by a fabricated result.

## Walk-forward validation and recommendation

Up to eight expanding chronological forecast origins in the earlier 80% of history are used. Each origin predicts the same number of exchange sessions as the requested final horizon; ranking uses the full forecast paths, not just their endpoints. Future targets are not present in the model's training history at an origin. Feature construction and redundancy filtering use only training-fold information. A candidate is eligible only if it yields a finite positive full path at every validation fold.

Reported validation and later rolling-origin test metrics include MAE, RMSE, sMAPE, bias, and directional accuracy over all forecast steps. The latest 20% of observations is excluded from selection; as in live rolling evaluation, earlier test-period closes become available to later test origins. Automatic mode is the default. It selects the lowest full-path validation MAE only if a non-naive model strictly beats the naive baseline; otherwise the naive baseline is recommended. Manual mode permits any validation-eligible model. Test results are reported separately and do not influence selection.

## Prediction ranges and limitations

Per-step ranges are empirical 80th-percentile absolute-error widths around the forecast path, calculated from that selected model's same-horizon walk-forward residuals. A range is withheld if fewer than four residual observations are available at that step. This is a small-sample empirical estimate, not a calibrated statistical coverage guarantee and not a promise that future values will fall within the range. Prophet's native intervals are not used.

There is no persisted experiment archive, model versioning, or live performance monitoring. Long horizons leave relatively few historical origins and should be treated as especially uncertain. All outputs are estimates for educational research, not investment advice.
