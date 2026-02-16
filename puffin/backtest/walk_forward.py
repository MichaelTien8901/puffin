"""Walk-forward analysis for strategy validation."""

import pandas as pd

from puffin.backtest.engine import Backtester, BacktestResult
from puffin.strategies.base import Strategy


def walk_forward(
    strategy: Strategy,
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    n_splits: int = 5,
    initial_capital: float = 100_000.0,
    **backtester_kwargs,
) -> list[dict]:
    """Run walk-forward analysis with rolling train/test splits.

    Args:
        strategy: Strategy to test.
        data: OHLCV DataFrame.
        train_ratio: Fraction of each window used for training.
        n_splits: Number of rolling windows.
        initial_capital: Starting capital per window.

    Returns:
        List of dicts with 'train_metrics' and 'test_metrics' per split.
    """
    n = len(data)
    window_size = n // n_splits
    results = []

    for i in range(n_splits):
        start = i * window_size
        end = min(start + window_size, n)
        if end - start < 20:
            continue

        split_point = start + int((end - start) * train_ratio)
        train_data = data.iloc[start:split_point]
        test_data = data.iloc[split_point:end]

        if len(train_data) < 10 or len(test_data) < 5:
            continue

        bt = Backtester(initial_capital=initial_capital, **backtester_kwargs)

        train_result = bt.run(strategy, train_data)
        test_result = bt.run(strategy, test_data)

        results.append({
            "split": i + 1,
            "train_start": train_data.index[0],
            "train_end": train_data.index[-1],
            "test_start": test_data.index[0],
            "test_end": test_data.index[-1],
            "train_metrics": train_result.metrics(),
            "test_metrics": test_result.metrics(),
        })

    return results
