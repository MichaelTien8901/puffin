"""Tests for P&L tracking."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from puffin.monitor.pnl import PnLTracker, Position


class TestPosition:
    """Tests for Position dataclass."""

    def test_create_position(self):
        """Test creating position."""
        pos = Position(
            ticker='AAPL',
            quantity=100,
            avg_price=150.0,
            current_price=155.0,
            strategy='momentum'
        )

        assert pos.ticker == 'AAPL'
        assert pos.quantity == 100
        assert pos.avg_price == 150.0

    def test_market_value(self):
        """Test market value calculation."""
        pos = Position(
            ticker='AAPL',
            quantity=100,
            avg_price=150.0,
            current_price=155.0
        )

        assert pos.market_value == 15500.0

    def test_cost_basis(self):
        """Test cost basis calculation."""
        pos = Position(
            ticker='AAPL',
            quantity=100,
            avg_price=150.0,
            current_price=155.0
        )

        assert pos.cost_basis == 15000.0

    def test_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        pos = Position(
            ticker='AAPL',
            quantity=100,
            avg_price=150.0,
            current_price=155.0
        )

        assert pos.unrealized_pnl == 500.0


class TestPnLTracker:
    """Tests for P&L tracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = PnLTracker(initial_cash=100000.0)

        assert tracker.cash == 100000.0
        assert tracker.initial_cash == 100000.0
        assert tracker.realized_pnl == 0.0
        assert len(tracker.positions) == 0

    def test_record_buy_trade(self):
        """Test recording buy trade."""
        tracker = PnLTracker(initial_cash=100000.0)

        tracker.record_trade(
            ticker='AAPL',
            quantity=100,
            price=150.0,
            side='buy',
            commission=1.0
        )

        assert tracker.cash == 100000.0 - 15000.0 - 1.0
        assert 'AAPL' in tracker.positions
        assert tracker.positions['AAPL'].quantity == 100
        assert tracker.positions['AAPL'].avg_price == 150.0

    def test_record_sell_trade(self):
        """Test recording sell trade."""
        tracker = PnLTracker(initial_cash=100000.0)

        # Buy first
        tracker.record_trade(
            ticker='AAPL',
            quantity=100,
            price=150.0,
            side='buy',
            commission=1.0
        )

        # Then sell
        tracker.record_trade(
            ticker='AAPL',
            quantity=50,
            price=155.0,
            side='sell',
            commission=1.0
        )

        # Realized P&L = (155 - 150) * 50 - 1 = 249
        assert tracker.realized_pnl == 249.0
        assert tracker.positions['AAPL'].quantity == 50

    def test_sell_entire_position(self):
        """Test selling entire position."""
        tracker = PnLTracker(initial_cash=100000.0)

        tracker.record_trade(
            ticker='AAPL',
            quantity=100,
            price=150.0,
            side='buy',
            commission=1.0
        )

        tracker.record_trade(
            ticker='AAPL',
            quantity=100,
            price=155.0,
            side='sell',
            commission=1.0
        )

        # Position should be closed
        assert 'AAPL' not in tracker.positions
        assert tracker.realized_pnl == 499.0  # (155-150)*100 - 1 (sell commission only)

    def test_sell_without_position(self):
        """Test selling without position raises error."""
        tracker = PnLTracker(initial_cash=100000.0)

        with pytest.raises(ValueError):
            tracker.record_trade(
                ticker='AAPL',
                quantity=100,
                price=150.0,
                side='sell',
                commission=1.0
            )

    def test_sell_more_than_position(self):
        """Test selling more than position raises error."""
        tracker = PnLTracker(initial_cash=100000.0)

        tracker.record_trade(
            ticker='AAPL',
            quantity=50,
            price=150.0,
            side='buy',
            commission=1.0
        )

        with pytest.raises(ValueError):
            tracker.record_trade(
                ticker='AAPL',
                quantity=100,
                price=155.0,
                side='sell',
                commission=1.0
            )

    def test_equity_calculation(self):
        """Test equity calculation."""
        tracker = PnLTracker(initial_cash=100000.0)

        tracker.record_trade(
            ticker='AAPL',
            quantity=100,
            price=150.0,
            side='buy',
            commission=1.0
        )

        # Update position price
        tracker.positions['AAPL'].current_price = 155.0

        equity = tracker.equity()
        # Equity = cash + position value
        # Cash = 100000 - 15001 = 84999
        # Position = 100 * 155 = 15500
        # Total = 100499
        assert equity == 100499.0

    def test_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        tracker = PnLTracker(initial_cash=100000.0)

        tracker.record_trade(
            ticker='AAPL',
            quantity=100,
            price=150.0,
            side='buy',
            commission=1.0
        )

        tracker.positions['AAPL'].current_price = 155.0

        unrealized = tracker.unrealized_pnl()
        assert unrealized == 500.0

    def test_total_pnl(self):
        """Test total P&L calculation."""
        tracker = PnLTracker(initial_cash=100000.0)

        # Buy 200 shares
        tracker.record_trade(
            ticker='AAPL',
            quantity=200,
            price=150.0,
            side='buy',
            commission=1.0
        )

        # Sell 100 shares at profit
        tracker.record_trade(
            ticker='AAPL',
            quantity=100,
            price=155.0,
            side='sell',
            commission=1.0
        )

        # Update remaining position price
        tracker.positions['AAPL'].current_price = 160.0

        # Realized = (155-150)*100 - 1 = 499
        # Unrealized = (160-150)*100 = 1000
        # Total = 1499
        total = tracker.total_pnl()
        assert total == 1499.0

    def test_update_snapshot(self):
        """Test update creates snapshot."""
        tracker = PnLTracker(initial_cash=100000.0)

        pos = Position(
            ticker='AAPL',
            quantity=100,
            avg_price=150.0,
            current_price=155.0
        )

        tracker.update(
            positions={'AAPL': pos},
            prices={'AAPL': 155.0}
        )

        assert len(tracker.history) == 1
        assert 'equity' in tracker.history[0]
        assert 'total_pnl' in tracker.history[0]

    def test_attribution_by_strategy(self):
        """Test P&L attribution by strategy."""
        tracker = PnLTracker(initial_cash=100000.0)

        pos1 = Position(
            ticker='AAPL',
            quantity=100,
            avg_price=150.0,
            current_price=155.0,
            strategy='momentum'
        )

        pos2 = Position(
            ticker='GOOGL',
            quantity=50,
            avg_price=200.0,
            current_price=195.0,
            strategy='mean_reversion'
        )

        tracker.positions = {'AAPL': pos1, 'GOOGL': pos2}

        attribution = tracker.attribution_by_strategy()

        assert len(attribution) == 2
        assert 'momentum' in attribution['strategy'].values
        assert 'mean_reversion' in attribution['strategy'].values

    def test_attribution_by_asset(self):
        """Test P&L attribution by asset."""
        tracker = PnLTracker(initial_cash=100000.0)

        pos1 = Position(
            ticker='AAPL',
            quantity=100,
            avg_price=150.0,
            current_price=155.0
        )

        pos2 = Position(
            ticker='GOOGL',
            quantity=50,
            avg_price=200.0,
            current_price=195.0
        )

        tracker.positions = {'AAPL': pos1, 'GOOGL': pos2}

        attribution = tracker.attribution_by_asset()

        assert len(attribution) == 2
        assert 'AAPL' in attribution['ticker'].values
        assert 'GOOGL' in attribution['ticker'].values
        assert 'unrealized_pnl' in attribution.columns
        assert 'return_pct' in attribution.columns

    def test_performance_summary(self):
        """Test performance summary."""
        tracker = PnLTracker(initial_cash=100000.0)

        tracker.record_trade(
            ticker='AAPL',
            quantity=100,
            price=150.0,
            side='buy',
            commission=1.0
        )

        tracker.positions['AAPL'].current_price = 155.0

        summary = tracker.performance_summary()

        assert 'initial_cash' in summary
        assert 'current_equity' in summary
        assert 'total_pnl' in summary
        assert 'total_return' in summary
        assert summary['num_positions'] == 1
