# Backtesting status

`backend/app/services/backtesting.py` contains two calculation helpers:

- `evaluate_forecasts` computes MAE, RMSE, and a directional-accuracy estimate for aligned series.
- `simulate_strategy` demonstrates a long/cash momentum signal based on the prior bar's return, charges a configurable fee in basis points on turnover, and returns total return, maximum drawdown, equity values, and strategy returns.

These helpers are not exposed through FastAPI or the frontend. The strategy is illustrative: it does not model spreads, slippage, borrow costs, position sizing, exchange calendars, or buy-and-hold comparison. Its result is not proof of profitability. Forecasting metrics and strategy returns should not be conflated.

The model comparison endpoint evaluates forecasts, not a trading strategy. It does not expose a final untouched test set or persist any backtest run.
