"""Tests for OrderManager."""

from datetime import datetime
from unittest.mock import Mock, MagicMock

import pytest

from puffin.broker import (
    OrderManager,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    OrderStatusInfo,
    Position,
    BrokerError,
)


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = Mock()
    broker.submit_order = Mock(return_value="order_123")
    broker.cancel_order = Mock(return_value=True)
    broker.get_order_status = Mock()
    broker.get_positions = Mock(return_value={})
    broker.modify_order = Mock(return_value="order_124")
    return broker


@pytest.fixture
def order_manager(mock_broker):
    """Create an OrderManager with mock broker."""
    return OrderManager(mock_broker)


@pytest.fixture
def sample_order():
    """Create a sample order."""
    return Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        type=OrderType.MARKET,
    )


def test_submit_order(order_manager, mock_broker, sample_order):
    """Test submitting an order."""
    # Setup mock
    mock_broker.get_order_status.return_value = OrderStatusInfo(
        order_id="order_123",
        client_order_id=None,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        filled_qty=0,
        type=OrderType.MARKET,
        status=OrderStatus.PENDING,
    )

    # Submit order
    order_id = order_manager.submit(sample_order)

    # Verify
    assert order_id == "order_123"
    mock_broker.submit_order.assert_called_once_with(sample_order)
    assert order_id in order_manager._pending_orders


def test_submit_order_rejection(order_manager, mock_broker, sample_order):
    """Test order rejection."""
    # Setup mock to raise error
    mock_broker.submit_order.side_effect = BrokerError("Rejected")

    # Submit order (should raise)
    with pytest.raises(BrokerError):
        order_manager.submit(sample_order)


def test_cancel_order(order_manager, mock_broker):
    """Test canceling an order."""
    order_id = "order_123"

    # Setup mock
    mock_broker.get_order_status.return_value = OrderStatusInfo(
        order_id=order_id,
        client_order_id=None,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        filled_qty=0,
        type=OrderType.MARKET,
        status=OrderStatus.CANCELED,
    )

    # Add order to pending
    order_manager._pending_orders.add(order_id)

    # Cancel
    success = order_manager.cancel(order_id)

    # Verify
    assert success is True
    mock_broker.cancel_order.assert_called_once_with(order_id)
    assert order_id not in order_manager._pending_orders


def test_track_order_fill(order_manager, mock_broker):
    """Test tracking an order that gets filled."""
    order_id = "order_123"

    # Setup mock - order is filled
    mock_broker.get_order_status.return_value = OrderStatusInfo(
        order_id=order_id,
        client_order_id=None,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        filled_qty=100,
        type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        avg_fill_price=150.0,
    )

    # Add to pending
    order_manager._pending_orders.add(order_id)

    # Track order
    status = order_manager.track_order(order_id)

    # Verify
    assert status.status == OrderStatus.FILLED
    assert order_id not in order_manager._pending_orders
    assert order_id in order_manager._filled_orders


def test_on_fill_callback(order_manager, mock_broker):
    """Test on_fill callback is triggered."""
    order_id = "order_123"
    callback = Mock()

    # Register callback
    order_manager.on_fill(callback)

    # Setup mock - order is filled
    mock_broker.get_order_status.return_value = OrderStatusInfo(
        order_id=order_id,
        client_order_id=None,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        filled_qty=100,
        type=OrderType.MARKET,
        status=OrderStatus.FILLED,
    )

    # Add to pending
    order_manager._pending_orders.add(order_id)

    # Track order
    order_manager.track_order(order_id)

    # Verify callback was called
    callback.assert_called_once()


def test_pending_orders(order_manager, mock_broker):
    """Test getting pending orders."""
    # Create multiple orders
    order_ids = ["order_1", "order_2", "order_3"]

    for order_id in order_ids:
        order_manager._pending_orders.add(order_id)
        order_manager._orders[order_id] = OrderStatusInfo(
            order_id=order_id,
            client_order_id=None,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=100,
            filled_qty=0,
            type=OrderType.MARKET,
            status=OrderStatus.PENDING,
        )

    # Get pending orders
    pending = order_manager.pending_orders()

    # Verify
    assert len(pending) == 3
    assert all(o.status == OrderStatus.PENDING for o in pending)


def test_filled_orders(order_manager, mock_broker):
    """Test getting filled orders."""
    # Create filled orders
    order_ids = ["order_1", "order_2"]

    for order_id in order_ids:
        order_manager._filled_orders.add(order_id)
        order_manager._orders[order_id] = OrderStatusInfo(
            order_id=order_id,
            client_order_id=None,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=100,
            filled_qty=100,
            type=OrderType.MARKET,
            status=OrderStatus.FILLED,
        )

    # Get filled orders
    filled = order_manager.filled_orders()

    # Verify
    assert len(filled) == 2
    assert all(o.status == OrderStatus.FILLED for o in filled)


def test_reconcile_positions_match(order_manager, mock_broker):
    """Test position reconciliation when positions match."""
    # Setup filled order
    order_manager._filled_orders.add("order_1")
    order_manager._orders["order_1"] = OrderStatusInfo(
        order_id="order_1",
        client_order_id=None,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        filled_qty=100,
        type=OrderType.MARKET,
        status=OrderStatus.FILLED,
    )

    # Setup broker position
    mock_broker.get_positions.return_value = {
        "AAPL": Position(
            symbol="AAPL",
            qty=100,
            avg_price=150.0,
            market_value=15000.0,
            unrealized_pnl=0.0,
        )
    }

    # Reconcile
    discrepancies = order_manager.reconcile_positions()

    # Verify no discrepancies
    assert len(discrepancies) == 0


def test_reconcile_positions_mismatch(order_manager, mock_broker):
    """Test position reconciliation when positions don't match."""
    # Setup filled order - expect 100 shares
    order_manager._filled_orders.add("order_1")
    order_manager._orders["order_1"] = OrderStatusInfo(
        order_id="order_1",
        client_order_id=None,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        filled_qty=100,
        type=OrderType.MARKET,
        status=OrderStatus.FILLED,
    )

    # Setup broker position - actually have 150 shares
    mock_broker.get_positions.return_value = {
        "AAPL": Position(
            symbol="AAPL",
            qty=150,
            avg_price=150.0,
            market_value=22500.0,
            unrealized_pnl=0.0,
        )
    }

    # Reconcile
    discrepancies = order_manager.reconcile_positions()

    # Verify discrepancy found
    assert len(discrepancies) == 1
    assert discrepancies[0]["symbol"] == "AAPL"
    assert discrepancies[0]["expected_qty"] == 100
    assert discrepancies[0]["actual_qty"] == 150
    assert discrepancies[0]["diff"] == 50


def test_cancel_all_pending(order_manager, mock_broker):
    """Test canceling all pending orders."""
    # Setup pending orders
    order_ids = ["order_1", "order_2", "order_3"]
    for order_id in order_ids:
        order_manager._pending_orders.add(order_id)

    # Setup mock
    mock_broker.get_order_status.return_value = OrderStatusInfo(
        order_id="order_1",
        client_order_id=None,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        filled_qty=0,
        type=OrderType.MARKET,
        status=OrderStatus.CANCELED,
    )

    # Cancel all
    results = order_manager.cancel_all_pending()

    # Verify
    assert len(results) == 3
    assert all(results.values())
    assert len(order_manager._pending_orders) == 0


def test_get_stats(order_manager):
    """Test getting order statistics."""
    # Setup various orders
    order_manager._orders = {
        "order_1": OrderStatusInfo(
            order_id="order_1", client_order_id=None, symbol="AAPL",
            side=OrderSide.BUY, qty=100, filled_qty=100,
            type=OrderType.MARKET, status=OrderStatus.FILLED,
        ),
        "order_2": OrderStatusInfo(
            order_id="order_2", client_order_id=None, symbol="TSLA",
            side=OrderSide.BUY, qty=50, filled_qty=0,
            type=OrderType.MARKET, status=OrderStatus.PENDING,
        ),
        "order_3": OrderStatusInfo(
            order_id="order_3", client_order_id=None, symbol="NVDA",
            side=OrderSide.BUY, qty=75, filled_qty=0,
            type=OrderType.MARKET, status=OrderStatus.CANCELED,
        ),
    }
    order_manager._pending_orders = {"order_2"}
    order_manager._filled_orders = {"order_1"}

    # Get stats
    stats = order_manager.get_stats()

    # Verify
    assert stats["total_orders"] == 3
    assert stats["pending"] == 1
    assert stats["filled"] == 1
    assert stats["canceled"] == 1
    assert stats["fill_rate"] == pytest.approx(1/3)


def test_modify_order(order_manager, mock_broker):
    """Test modifying an order."""
    order_id = "order_123"
    new_order_id = "order_124"

    # Setup mock
    mock_broker.get_order_status.return_value = OrderStatusInfo(
        order_id=new_order_id,
        client_order_id=None,
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=150,
        filled_qty=0,
        type=OrderType.LIMIT,
        status=OrderStatus.PENDING,
        limit_price=155.0,
    )

    # Add original order to pending
    order_manager._pending_orders.add(order_id)

    # Modify
    result_id = order_manager.modify(order_id, qty=150, limit_price=155.0)

    # Verify
    assert result_id == new_order_id
    assert order_id not in order_manager._pending_orders
    assert new_order_id in order_manager._pending_orders
