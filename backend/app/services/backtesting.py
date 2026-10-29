
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecasts(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    actual = actual.astype(float)
    predicted = predicted.astype(float)

    valid = actual.notna() & predicted.notna()
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No valid forecast/actual pairs")

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    actual_direction = np.sign(actual.diff())
    predicted_direction = np.sign(predicted.diff())
    direction_valid = actual_direction.notna()

    directional_accuracy = (
        actual_direction[direction_valid].to_numpy()
        == predicted_direction[direction_valid].to_numpy()
    ).mean() if direction_valid.any() else float("nan")

    return {
        "observations": int(len(actual)),
        "mae": float(mae),
        "rmse": float(rmse),
        "directional_accuracy": float(directional_accuracy),
    }


def simulate_strategy(
    close: pd.Series,
    fee_bps: float = 5.0,
    threshold: float = 0.0,
) -> dict:
    """
    Illustrative long/cash strategy.

    Position at t is determined by the return observed
    through t, and earns the next bar's return.
    This avoids using the next return to choose the position.
    """
    close = close.astype(float).dropna()

    if len(close) < 3 or (close <= 0).any():
        raise ValueError("Need at least 3 positive closing prices")

    returns = close.pct_change().fillna(0)

    # Simple momentum signal based only on past return.
    signal = np.sign(returns).where(
        returns.abs() > threshold, 0
    )
    position = signal.shift(1).fillna(0)

    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * (fee_bps / 10000)

    strategy_returns = position * returns - costs
    equity = (1 + strategy_returns).cumprod()

    drawdown = equity / equity.cummax() - 1
    peak_to_trough = float(drawdown.min())

    return {
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": peak_to_trough,
        "equity_curve": equity.tolist(),
        "strategy_returns": strategy_returns.tolist(),
        "bars": int(len(close)),
        "fee_bps": fee_bps,
    }
