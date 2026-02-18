"""Tests for stop loss implementations."""

import pytest
from datetime import datetime, timedelta
from puffin.risk.stop_loss import (
    Position,
    FixedStop,
    TrailingStop,
    ATRStop,
    TimeStop,
    StopLossManager,
)


class TestFixedStop:
    """Tests for fixed stop loss."""

    def test_long_position_triggered(self):
        """Test fixed stop triggered for long position."""
        stop = FixedStop(stop_distance=5.0, price_based=True)
        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long'
        )

        # Stop at 95, current price 94 -> triggered
        assert stop.check(94.0, 100.0, position) == True

        # Stop at 95, current price 96 -> not triggered
        assert stop.check(96.0, 100.0, position) == False

    def test_short_position_triggered(self):
        """Test fixed stop triggered for short position."""
        stop = FixedStop(stop_distance=5.0, price_based=True)
        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='short'
        )

        # Stop at 105, current price 106 -> triggered
        assert stop.check(106.0, 100.0, position) == True

        # Stop at 105, current price 104 -> not triggered
        assert stop.check(104.0, 100.0, position) == False

    def test_percentage_stop(self):
        """Test percentage-based stop."""
        stop = FixedStop(stop_distance=0.05, price_based=False)  # 5% stop
        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long'
        )

        # 6% loss -> triggered
        assert stop.check(94.0, 100.0, position) == True

        # 4% loss -> not triggered
        assert stop.check(96.0, 100.0, position) == False


class TestTrailingStop:
    """Tests for trailing stop loss."""

    def test_long_position_trailing(self):
        """Test trailing stop for long position."""
        stop = TrailingStop(trail_distance=5.0, price_based=True)
        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long'
        )

        # Price goes up to 110
        assert stop.check(110.0, 100.0, position) == False
        assert position.highest_price == 110.0

        # Price falls to 106 (within 5 of high 110) -> not triggered
        assert stop.check(106.0, 100.0, position) == False

        # Price falls to 105.5 (within 5 of 110) -> not triggered
        assert stop.check(105.5, 100.0, position) == False

        # Price falls to 104 (more than 5 from 110) -> triggered
        assert stop.check(104.0, 100.0, position) == True

    def test_short_position_trailing(self):
        """Test trailing stop for short position."""
        stop = TrailingStop(trail_distance=5.0, price_based=True)
        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='short'
        )

        # Price goes down to 90
        assert stop.check(90.0, 100.0, position) == False
        assert position.lowest_price == 90.0

        # Price rises to 94 (within 5 of low) -> not triggered
        assert stop.check(94.0, 100.0, position) == False

        # Price rises to 96 (more than 5 from 90) -> triggered
        assert stop.check(96.0, 100.0, position) == True


class TestATRStop:
    """Tests for ATR-based stop loss."""

    def test_fixed_atr_stop(self):
        """Test fixed ATR stop."""
        stop = ATRStop(atr_multiplier=2.0, trailing=False)
        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long',
            metadata={'atr': 3.0}
        )

        # Stop at 100 - 2*3 = 94
        assert stop.check(93.0, 100.0, position) == True
        assert stop.check(95.0, 100.0, position) == False

    def test_trailing_atr_stop(self):
        """Test trailing ATR stop."""
        stop = ATRStop(atr_multiplier=2.0, trailing=True)
        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long',
            metadata={'atr': 3.0}
        )

        # Price goes up to 110
        assert stop.check(110.0, 100.0, position) == False
        assert position.highest_price == 110.0

        # Stop now at 110 - 2*3 = 104
        assert stop.check(103.0, 100.0, position) == True
        assert stop.check(105.0, 100.0, position) == False

    def test_missing_atr(self):
        """Test error when ATR not provided."""
        stop = ATRStop(atr_multiplier=2.0)
        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long'
        )

        with pytest.raises(ValueError):
            stop.check(95.0, 100.0, position)


class TestTimeStop:
    """Tests for time-based stop loss."""

    def test_bar_based_stop(self):
        """Test bar-based time stop."""
        stop = TimeStop(max_bars=5)
        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long'
        )

        # First 4 bars -> not triggered
        for _ in range(4):
            assert stop.check(100.0, 100.0, position) == False

        # 5th bar -> triggered
        assert stop.check(100.0, 100.0, position) == True

    def test_time_based_stop(self):
        """Test time-based stop."""
        stop = TimeStop(max_seconds=3600)  # 1 hour
        entry_time = datetime.now()
        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=entry_time,
            quantity=100,
            side='long',
            metadata={'current_time': entry_time + timedelta(minutes=30)}
        )

        # 30 minutes -> not triggered
        assert stop.check(100.0, 100.0, position) == False

        # 70 minutes -> triggered
        position.metadata['current_time'] = entry_time + timedelta(minutes=70)
        assert stop.check(100.0, 100.0, position) == True


class TestStopLossManager:
    """Tests for stop loss manager."""

    def test_add_and_check_stop(self):
        """Test adding and checking stops."""
        manager = StopLossManager()

        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long'
        )

        manager.add_position(position)
        manager.add_stop('AAPL', FixedStop(stop_distance=5.0))

        # Not triggered
        assert manager.check_stops('AAPL', 96.0) == False

        # Triggered
        assert manager.check_stops('AAPL', 94.0) == True

    def test_multiple_stops(self):
        """Test multiple stops on same position."""
        manager = StopLossManager()

        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long'
        )

        manager.add_position(position)
        manager.add_stop('AAPL', FixedStop(stop_distance=5.0))
        manager.add_stop('AAPL', TimeStop(max_bars=10))

        # Neither triggered
        assert manager.check_stops('AAPL', 96.0) == False

        # Price stop triggered
        assert manager.check_stops('AAPL', 94.0) == True

    def test_remove_position(self):
        """Test removing position."""
        manager = StopLossManager()

        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long'
        )

        manager.add_position(position)
        manager.add_stop('AAPL', FixedStop(stop_distance=5.0))

        # Remove position
        manager.remove_position('AAPL')

        # Check should return False (no position)
        assert manager.check_stops('AAPL', 94.0) == False

    def test_get_stop_prices(self):
        """Test getting stop prices."""
        manager = StopLossManager()

        position = Position(
            ticker='AAPL',
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=100,
            side='long'
        )

        manager.add_position(position)
        manager.add_stop('AAPL', FixedStop(stop_distance=5.0))

        stop_prices = manager.get_stop_prices('AAPL', 100.0)

        assert 'FixedStop_0' in stop_prices
        assert stop_prices['FixedStop_0'] == 95.0
