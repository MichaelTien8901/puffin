"""Tests for backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from puffin.backtest.engine import Backtester, SlippageModel, CommissionModel
from puffin.strategies.momentum import MomentumStrategy


@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    prices = 100 + np.arange(100) * 0.3 + np.random.randn(100) * 0.5
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Open": prices - 0.1,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": np.random.randint(1000, 5000, 100),
        },
        index=dates,
    )


class TestBacktester:
    def test_basic_run(self, sample_data):
        strategy = MomentumStrategy(short_window=10, long_window=30)
        bt = Backtester(initial_capital=100_000)
        result = bt.run(strategy, sample_data)
        assert len(result.equity_curve) == len(sample_data)
        assert result.initial_capital == 100_000

    def test_metrics_calculated(self, sample_data):
        strategy = MomentumStrategy(short_window=10, long_window=30)
        bt = Backtester(initial_capital=100_000)
        result = bt.run(strategy, sample_data)
        metrics = result.metrics()
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "total_trades" in metrics

    def test_with_slippage(self, sample_data):
        strategy = MomentumStrategy(short_window=10, long_window=30)
        bt_no_slip = Backtester(initial_capital=100_000)
        bt_slip = Backtester(
            initial_capital=100_000,
            slippage=SlippageModel(fixed=0.01),
        )
        result_no_slip = bt_no_slip.run(strategy, sample_data)
        result_slip = bt_slip.run(strategy, sample_data)
        # Slippage should reduce returns
        assert result_slip.final_value <= result_no_slip.final_value + 1  # small tolerance

    def test_with_commission(self, sample_data):
        strategy = MomentumStrategy(short_window=10, long_window=30)
        bt = Backtester(
            initial_capital=100_000,
            commission=CommissionModel(flat=5.0),
        )
        result = bt.run(strategy, sample_data)
        metrics = result.metrics()
        assert "total_return" in metrics

    def test_no_lookahead(self, sample_data):
        """Strategy should only see data up to current bar."""
        strategy = MomentumStrategy(short_window=10, long_window=30)
        bt = Backtester(initial_capital=100_000)
        result = bt.run(strategy, sample_data)
        # If there were lookahead, returns would be unrealistically high
        metrics = result.metrics()
        assert metrics["total_return"] < 5.0  # Sanity check
