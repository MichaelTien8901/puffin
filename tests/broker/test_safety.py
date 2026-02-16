"""Tests for SafetyController."""

from datetime import datetime, date
from unittest.mock import Mock, patch

import pytest

from puffin.broker import (
    SafetyController,
    Order,
    Position,
    AccountInfo,
    OrderSide,
    OrderType,
    BrokerError,
    PositionSizingValidator,
    TradingHoursValidator,
    SymbolWhitelistValidator,
)


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = Mock()
    broker.get_position = Mock(return_value=None)
    broker.get_positions = Mock(return_value={})
    broker.get_account = Mock(return_value=AccountInfo(
        equity=100000.0,
        cash=50000.0,
        buying_power=100000.0,
        portfolio_value=50000.0,
    ))
    return broker


@pytest.fixture
def safety_controller(mock_broker):
    """Create a SafetyController."""
    controller = SafetyController(
        broker=mock_broker,
        max_order_size=1000,
        max_position_size=5000,
        max_daily_loss=5000.0,
        require_confirmation=False,  # Disable for testing
    )
    controller._confirmation_given = True  # Skip confirmation for tests
    return controller


@pytest.fixture
def sample_order():
    """Create a sample order."""
    return Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        type=OrderType.MARKET,
    )


def test_validate_order_size_pass(safety_controller, sample_order):
    """Test order size validation passes."""
    is_valid, reason = safety_controller.validate_order(sample_order)
    assert is_valid is True
    assert reason == ""


def test_validate_order_size_fail(safety_controller):
    """Test order size validation fails."""
    large_order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=2000,  # Exceeds max_order_size=1000
        type=OrderType.MARKET,
    )

    is_valid, reason = safety_controller.validate_order(large_order)
    assert is_valid is False
    assert "max_order_size" in reason


def test_validate_position_size_pass(safety_controller, mock_broker, sample_order):
    """Test position size validation passes."""
    # Setup existing position
    mock_broker.get_position.return_value = Position(
        symbol="AAPL",
        qty=500,
        avg_price=150.0,
        market_value=75000.0,
        unrealized_pnl=0.0,
    )

    # Order would result in 600 shares (under max_position_size=5000)
    is_valid, reason = safety_controller.validate_order(sample_order)
    assert is_valid is True


def test_validate_position_size_fail(safety_controller, mock_broker):
    """Test position size validation fails."""
    # Setup existing large position
    mock_broker.get_position.return_value = Position(
        symbol="AAPL",
        qty=4900,
        avg_price=150.0,
        market_value=735000.0,
        unrealized_pnl=0.0,
    )

    # Order would exceed max_position_size=5000
    large_order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=200,
        type=OrderType.MARKET,
    )

    is_valid, reason = safety_controller.validate_position_size_fail(large_order)
    assert is_valid is False
    assert "max_position_size" in reason


def test_validate_confirmation_required(mock_broker):
    """Test that confirmation is required."""
    controller = SafetyController(
        broker=mock_broker,
        require_confirmation=True,
    )

    order = Order(symbol="AAPL", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)

    is_valid, reason = controller.validate_order(order)
    assert is_valid is False
    assert "confirmation" in reason.lower()


def test_confirm_live_trading(mock_broker):
    """Test live trading confirmation."""
    controller = SafetyController(broker=mock_broker, require_confirmation=True)

    # Confirm with code
    success = controller.confirm_live_trading("CONFIRM")
    assert success is True
    assert controller._confirmation_given is True


def test_daily_loss_limit(safety_controller, mock_broker):
    """Test daily loss limit circuit breaker."""
    # Setup positions with large loss
    mock_broker.get_positions.return_value = {
        "AAPL": Position(
            symbol="AAPL",
            qty=100,
            avg_price=150.0,
            market_value=14000.0,
            unrealized_pnl=-6000.0,  # Exceeds max_daily_loss=5000
        )
    }

    # Check daily P&L (should trigger circuit breaker)
    pnl = safety_controller.check_daily_pnl()
    assert pnl == -6000.0
    assert safety_controller.is_circuit_breaker_active()

    # Try to place order (should fail)
    order = Order(symbol="TSLA", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)
    is_valid, reason = safety_controller.validate_order(order)
    assert is_valid is False
    assert "circuit breaker" in reason.lower()


def test_reset_circuit_breaker(safety_controller):
    """Test resetting circuit breaker."""
    safety_controller._circuit_breaker_active = True
    assert safety_controller.is_circuit_breaker_active()

    safety_controller.reset_circuit_breaker()
    assert not safety_controller.is_circuit_breaker_active()


def test_custom_validator(safety_controller):
    """Test adding custom validator."""
    # Define custom validator that rejects even quantities
    def reject_even_qty(order: Order) -> tuple[bool, str]:
        if order.qty % 2 == 0:
            return False, "Even quantities not allowed"
        return True, ""

    safety_controller.add_validator(reject_even_qty)

    # Even quantity (should fail)
    even_order = Order(symbol="AAPL", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)
    is_valid, reason = safety_controller.validate_order(even_order)
    assert is_valid is False
    assert "even" in reason.lower()

    # Odd quantity (should pass)
    odd_order = Order(symbol="AAPL", side=OrderSide.BUY, qty=101, type=OrderType.MARKET)
    is_valid, reason = safety_controller.validate_order(odd_order)
    assert is_valid is True


def test_position_sizing_validator(mock_broker):
    """Test PositionSizingValidator."""
    validator = PositionSizingValidator(max_position_pct=0.25)

    # Setup account
    mock_broker.get_account.return_value = AccountInfo(
        equity=100000.0,
        cash=50000.0,
        buying_power=100000.0,
        portfolio_value=100000.0,
    )

    # Setup existing position
    mock_broker.get_position.return_value = Position(
        symbol="AAPL",
        qty=100,
        avg_price=150.0,
        market_value=15000.0,
        unrealized_pnl=0.0,
    )

    # Small order (should pass - would be ~20% of portfolio)
    order = Order(symbol="AAPL", side=OrderSide.BUY, qty=50, type=OrderType.LIMIT, limit_price=150.0)
    is_valid, reason = validator(mock_broker, order)
    assert is_valid is True

    # Large order (should fail - would be >25% of portfolio)
    large_order = Order(symbol="AAPL", side=OrderSide.BUY, qty=200, type=OrderType.LIMIT, limit_price=150.0)
    is_valid, reason = validator(mock_broker, large_order)
    assert is_valid is False
    assert "%" in reason


def test_trading_hours_validator_market_open(mock_broker):
    """Test TradingHoursValidator when market is open."""
    validator = TradingHoursValidator(allow_extended_hours=False)

    order = Order(symbol="AAPL", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)

    # Mock market as open
    with patch("puffin.broker.safety.TradingSession") as mock_session:
        mock_session.return_value.is_market_open.return_value = True

        is_valid, reason = validator(mock_broker, order)
        assert is_valid is True


def test_trading_hours_validator_market_closed(mock_broker):
    """Test TradingHoursValidator when market is closed."""
    validator = TradingHoursValidator(allow_extended_hours=False)

    order = Order(symbol="AAPL", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)

    # Mock market as closed
    with patch("puffin.broker.safety.TradingSession") as mock_session:
        mock_session.return_value.is_market_open.return_value = False
        mock_session.return_value.next_open.return_value = datetime(2024, 1, 2, 9, 30)

        is_valid, reason = validator(mock_broker, order)
        assert is_valid is False
        assert "closed" in reason.lower()


def test_symbol_whitelist_validator(mock_broker):
    """Test SymbolWhitelistValidator."""
    validator = SymbolWhitelistValidator(allowed_symbols=["AAPL", "TSLA", "NVDA"])

    # Allowed symbol
    order = Order(symbol="AAPL", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)
    is_valid, reason = validator(mock_broker, order)
    assert is_valid is True

    # Not allowed symbol
    bad_order = Order(symbol="GME", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)
    is_valid, reason = validator(mock_broker, bad_order)
    assert is_valid is False
    assert "whitelist" in reason.lower()


def test_get_stats(safety_controller):
    """Test getting safety controller stats."""
    stats = safety_controller.get_stats()

    assert "confirmation_given" in stats
    assert "circuit_breaker_active" in stats
    assert "max_order_size" in stats
    assert stats["max_order_size"] == 1000


def test_max_total_position_value(safety_controller, mock_broker):
    """Test max total position value check."""
    safety_controller.max_total_position_value = 40000.0

    # Account portfolio value is 50000 (exceeds limit)
    order = Order(symbol="AAPL", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)

    is_valid, reason = safety_controller.validate_order(order)
    assert is_valid is False
    assert "position value" in reason.lower()
