"""
Tests for portfolio rebalancing.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from puffin.portfolio.rebalance import (
    RebalanceEngine,
    Trade,
    CostModel,
    rebalance_schedule,
    backtest_rebalancing
)


@pytest.fixture
def cost_model():
    """Create a cost model for testing."""
    return CostModel(
        commission_pct=0.001,
        commission_fixed=1.0,
        slippage_pct=0.0005,
        min_commission=1.0
    )


@pytest.fixture
def sample_returns():
    """Create sample returns for backtesting."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')

    returns = pd.DataFrame(
        np.random.randn(252, 4) * 0.01,
        index=dates,
        columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    )

    return returns


def test_cost_model_init():
    """Test CostModel initialization."""
    model = CostModel()
    assert model.commission_pct == 0.001
    assert model.commission_fixed == 0.0
    assert model.slippage_pct == 0.0005
    assert model.min_commission == 1.0


def test_cost_model_calculate_cost(cost_model):
    """Test cost calculation."""
    trade_value = 10000.0
    cost = cost_model.calculate_cost(trade_value)

    # Cost should include commission and slippage
    expected_cost = (0.001 * 10000 + 1.0) + (0.0005 * 10000)
    assert np.isclose(cost, expected_cost)


def test_cost_model_min_commission(cost_model):
    """Test minimum commission."""
    # Small trade
    trade_value = 100.0
    cost = cost_model.calculate_cost(trade_value)

    # Should apply minimum commission
    assert cost >= cost_model.min_commission


def test_trade_creation():
    """Test Trade creation."""
    trade = Trade(
        symbol='AAPL',
        quantity=100.0,
        price=150.0,
        value=15000.0
    )

    assert trade.symbol == 'AAPL'
    assert trade.quantity == 100.0
    assert trade.price == 150.0
    assert trade.value == 15000.0
    assert trade.timestamp is not None


def test_rebalance_engine_init():
    """Test RebalanceEngine initialization."""
    engine = RebalanceEngine()
    assert engine.cost_model is not None


def test_compute_trades_basic():
    """Test basic trade computation."""
    engine = RebalanceEngine()

    current_weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
    target_weights = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'AMZN': 0.25}
    portfolio_value = 100000.0
    prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 300.0, 'AMZN': 3200.0}

    trades = engine.compute_trades(
        current_weights,
        target_weights,
        portfolio_value,
        prices
    )

    # Should generate trades
    assert len(trades) > 0

    # Total trade value should approximately maintain portfolio value
    total_buy = sum(t.value for t in trades if t.value > 0)
    total_sell = sum(abs(t.value) for t in trades if t.value < 0)
    assert np.isclose(total_buy, total_sell, rtol=0.01)


def test_compute_trades_no_change():
    """Test trade computation when no rebalancing needed."""
    engine = RebalanceEngine()

    weights = {'AAPL': 0.5, 'GOOGL': 0.5}
    portfolio_value = 100000.0
    prices = {'AAPL': 150.0, 'GOOGL': 2800.0}

    trades = engine.compute_trades(
        weights,
        weights,  # Same current and target
        portfolio_value,
        prices
    )

    # Should not generate trades
    assert len(trades) == 0


def test_compute_trades_min_trade_value():
    """Test minimum trade value filtering."""
    engine = RebalanceEngine()

    current_weights = {'AAPL': 0.5001, 'GOOGL': 0.4999}
    target_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
    portfolio_value = 100000.0
    prices = {'AAPL': 150.0, 'GOOGL': 2800.0}

    # Without min trade value
    trades1 = engine.compute_trades(
        current_weights,
        target_weights,
        portfolio_value,
        prices,
        min_trade_value=0.0
    )

    # With min trade value
    trades2 = engine.compute_trades(
        current_weights,
        target_weights,
        portfolio_value,
        prices,
        min_trade_value=1000.0
    )

    # Should filter out small trades
    assert len(trades2) <= len(trades1)


def test_apply_transaction_costs():
    """Test transaction cost application."""
    engine = RebalanceEngine()

    trades = [
        Trade('AAPL', 100, 150.0, 15000.0),
        Trade('GOOGL', -10, 2800.0, -28000.0)
    ]

    result = engine.apply_transaction_costs(trades)

    assert 'trades' in result
    assert 'total_cost' in result
    assert 'cost_by_symbol' in result

    # Total cost should be positive
    assert result['total_cost'] > 0

    # Should have costs for each symbol
    assert 'AAPL' in result['cost_by_symbol']
    assert 'GOOGL' in result['cost_by_symbol']


def test_optimize_with_costs():
    """Test cost-aware optimization."""
    engine = RebalanceEngine()

    current_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
    target_weights = {'AAPL': 0.49, 'GOOGL': 0.51}  # Small change
    portfolio_value = 100000.0
    prices = {'AAPL': 150.0, 'GOOGL': 2800.0}

    result = engine.optimize_with_costs(
        current_weights,
        target_weights,
        portfolio_value,
        prices,
        cost_threshold=0.001
    )

    assert 'should_rebalance' in result
    assert 'trades' in result
    assert 'total_cost' in result
    assert 'expected_benefit' in result

    # For small changes, might not rebalance
    assert isinstance(result['should_rebalance'], bool)


def test_rebalance_schedule_monthly():
    """Test monthly rebalancing schedule."""
    schedule = rebalance_schedule(strategy='monthly')

    weights = {'AAPL': 0.5, 'GOOGL': 0.5}

    # First call should rebalance
    date1 = datetime(2020, 1, 15)
    assert schedule(date1, weights, weights) == True

    # Same month should not rebalance
    date2 = datetime(2020, 1, 20)
    assert schedule(date2, weights, weights) == False

    # Next month should rebalance
    date3 = datetime(2020, 2, 1)
    assert schedule(date3, weights, weights) == True


def test_rebalance_schedule_quarterly():
    """Test quarterly rebalancing schedule."""
    schedule = rebalance_schedule(strategy='quarterly')

    weights = {'AAPL': 0.5, 'GOOGL': 0.5}

    # First call should rebalance
    date1 = datetime(2020, 1, 15)
    assert schedule(date1, weights, weights) == True

    # Same quarter should not rebalance
    date2 = datetime(2020, 2, 15)
    assert schedule(date2, weights, weights) == False

    # Next quarter should rebalance
    date3 = datetime(2020, 4, 1)
    assert schedule(date3, weights, weights) == True


def test_rebalance_schedule_annual():
    """Test annual rebalancing schedule."""
    schedule = rebalance_schedule(strategy='annual')

    weights = {'AAPL': 0.5, 'GOOGL': 0.5}

    # First call should rebalance
    date1 = datetime(2020, 1, 15)
    assert schedule(date1, weights, weights) == True

    # Same year should not rebalance
    date2 = datetime(2020, 6, 15)
    assert schedule(date2, weights, weights) == False

    # Next year should rebalance
    date3 = datetime(2021, 1, 1)
    assert schedule(date3, weights, weights) == True


def test_rebalance_schedule_threshold():
    """Test threshold-based rebalancing."""
    schedule = rebalance_schedule(strategy='threshold', threshold=0.05)

    current_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
    target_weights = {'AAPL': 0.5, 'GOOGL': 0.5}

    # Small deviation - should not rebalance
    date1 = datetime(2020, 1, 15)
    current_weights_small_dev = {'AAPL': 0.52, 'GOOGL': 0.48}
    assert schedule(date1, current_weights_small_dev, target_weights) == False

    # Large deviation - should rebalance
    current_weights_large_dev = {'AAPL': 0.6, 'GOOGL': 0.4}
    assert schedule(date1, current_weights_large_dev, target_weights) == True


def test_backtest_rebalancing(sample_returns):
    """Test rebalancing backtest."""
    target_weights = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'AMZN': 0.25}
    rebalance_fn = rebalance_schedule(strategy='monthly')

    result = backtest_rebalancing(
        sample_returns,
        target_weights,
        rebalance_fn,
        initial_value=100000.0
    )

    # Should return DataFrame
    assert isinstance(result, pd.DataFrame)

    # Should have required columns
    assert 'portfolio_value' in result.columns
    assert 'rebalanced' in result.columns
    assert 'transaction_costs' in result.columns

    # Portfolio value should be positive
    assert (result['portfolio_value'] > 0).all()

    # Transaction costs should be non-negative and monotonic
    assert (result['transaction_costs'] >= 0).all()
    assert result['transaction_costs'].is_monotonic_increasing


def test_backtest_no_rebalancing(sample_returns):
    """Test backtest with no rebalancing."""
    target_weights = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'AMZN': 0.25}

    # Schedule that never rebalances
    def never_rebalance(date, current, target):
        return False

    result = backtest_rebalancing(
        sample_returns,
        target_weights,
        never_rebalance,
        initial_value=100000.0
    )

    # Should not have any rebalancing
    assert result['rebalanced'].sum() == 0

    # Transaction costs should be zero (except potential initial)
    assert result['transaction_costs'].iloc[-1] <= 0.01


def test_trade_dataclass_timestamp():
    """Test Trade timestamp handling."""
    # Without timestamp
    trade1 = Trade('AAPL', 100, 150.0, 15000.0)
    assert trade1.timestamp is not None

    # With timestamp
    custom_time = datetime(2020, 1, 15, 10, 30)
    trade2 = Trade('GOOGL', 50, 2800.0, 140000.0, timestamp=custom_time)
    assert trade2.timestamp == custom_time


def test_rebalance_engine_with_custom_cost_model():
    """Test RebalanceEngine with custom cost model."""
    custom_model = CostModel(
        commission_pct=0.002,
        commission_fixed=5.0,
        slippage_pct=0.001,
        min_commission=5.0
    )

    engine = RebalanceEngine(cost_model=custom_model)

    assert engine.cost_model.commission_pct == 0.002
    assert engine.cost_model.min_commission == 5.0
