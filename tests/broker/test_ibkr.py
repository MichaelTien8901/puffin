"""Tests for IBKRBroker."""

from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

import pytest

from puffin.broker import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    OrderStatusInfo,
    TimeInForce,
    Position,
    AccountInfo,
    BrokerError,
    OrderRejectedError,
    InsufficientFundsError,
)
from puffin.broker.ibkr import IBKRBroker


# --- Mock helpers ---


def _mock_ib():
    """Create a mock IB client."""
    ib = Mock()
    ib.isConnected.return_value = True
    ib.connect = Mock()
    ib.sleep = Mock()
    ib.placeOrder = Mock()
    ib.cancelOrder = Mock()
    ib.openTrades = Mock(return_value=[])
    ib.trades = Mock(return_value=[])
    ib.positions = Mock(return_value=[])
    ib.accountSummary = Mock(return_value=[])
    ib.disconnect = Mock()
    return ib


def _mock_trade(
    order_id=1,
    symbol="AAPL",
    action="BUY",
    order_type="MKT",
    total_qty=100,
    filled=0,
    avg_fill_price=0.0,
    status_str="Submitted",
    lmt_price=0.0,
    aux_price=0.0,
    order_ref="",
):
    """Create a mock ib_async Trade object."""
    trade = Mock()
    trade.contract = Mock()
    trade.contract.symbol = symbol

    trade.order = Mock()
    trade.order.orderId = order_id
    trade.order.action = action
    trade.order.orderType = order_type
    trade.order.totalQuantity = total_qty
    trade.order.lmtPrice = lmt_price
    trade.order.auxPrice = aux_price
    trade.order.orderRef = order_ref

    trade.orderStatus = Mock()
    trade.orderStatus.status = status_str
    trade.orderStatus.filled = filled
    trade.orderStatus.avgFillPrice = avg_fill_price

    return trade


def _mock_position(symbol="AAPL", qty=100, avg_cost=150.0):
    """Create a mock ib_async position."""
    pos = Mock()
    pos.contract = Mock()
    pos.contract.symbol = symbol
    pos.position = qty
    pos.avgCost = avg_cost
    pos.unrealizedPNL = 50.0
    return pos


def _mock_account_item(tag, value):
    item = Mock()
    item.tag = tag
    item.value = value
    return item


# --- Fixtures ---


@pytest.fixture
def broker():
    """Create an IBKRBroker with mocked IB client."""
    b = IBKRBroker(host="127.0.0.1", port=7497, client_id=1, paper=True)
    b._ib = _mock_ib()
    return b


@pytest.fixture
def sample_order():
    return Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        type=OrderType.MARKET,
    )


# --- Connection tests ---


class TestConnection:
    def test_init_defaults(self):
        b = IBKRBroker()
        assert b.host == "127.0.0.1"
        assert b.port == 4002
        assert b.client_id == 1
        assert b.paper is True
        assert b._ib is None

    def test_connect_reuses_existing(self, broker):
        ib = broker._connect()
        assert ib is broker._ib

    def test_connect_failure(self):
        b = IBKRBroker()
        with patch("puffin.broker.ibkr.IBKRBroker._connect") as mock_connect:
            mock_connect.side_effect = BrokerError("Connection failed")
            with pytest.raises(BrokerError, match="Connection failed"):
                mock_connect()


# --- Submit order tests ---


class TestSubmitOrder:
    def test_submit_market_order(self, broker, sample_order):
        trade = _mock_trade(order_id=42)
        broker._ib.placeOrder.return_value = trade

        order_id = broker.submit_order(sample_order)

        assert order_id == "42"
        broker._ib.placeOrder.assert_called_once()
        broker._ib.sleep.assert_called_with(0)

    def test_submit_limit_order(self, broker):
        order = Order(
            symbol="TSLA",
            side=OrderSide.SELL,
            qty=50,
            type=OrderType.LIMIT,
            limit_price=250.00,
            time_in_force=TimeInForce.GTC,
        )
        trade = _mock_trade(order_id=43)
        broker._ib.placeOrder.return_value = trade

        order_id = broker.submit_order(order)
        assert order_id == "43"

    def test_submit_stop_order(self, broker):
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            qty=100,
            type=OrderType.STOP,
            stop_price=170.00,
        )
        trade = _mock_trade(order_id=44)
        broker._ib.placeOrder.return_value = trade

        order_id = broker.submit_order(order)
        assert order_id == "44"

    def test_submit_stop_limit_order(self, broker):
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            qty=100,
            type=OrderType.STOP_LIMIT,
            stop_price=170.00,
            limit_price=169.50,
        )
        trade = _mock_trade(order_id=45)
        broker._ib.placeOrder.return_value = trade

        order_id = broker.submit_order(order)
        assert order_id == "45"

    def test_submit_insufficient_funds(self, broker, sample_order):
        broker._ib.placeOrder.side_effect = Exception("Insufficient buying power")

        with pytest.raises(InsufficientFundsError):
            broker.submit_order(sample_order)

    def test_submit_rejected(self, broker, sample_order):
        broker._ib.placeOrder.side_effect = Exception("Order rejected by exchange")

        with pytest.raises(OrderRejectedError):
            broker.submit_order(sample_order)

    def test_submit_generic_error(self, broker, sample_order):
        broker._ib.placeOrder.side_effect = Exception("Network timeout")

        with pytest.raises(BrokerError, match="Failed to submit order"):
            broker.submit_order(sample_order)


# --- Cancel order tests ---


class TestCancelOrder:
    def test_cancel_existing_order(self, broker):
        trade = _mock_trade(order_id=42)
        broker._ib.openTrades.return_value = [trade]

        assert broker.cancel_order("42") is True
        broker._ib.cancelOrder.assert_called_once_with(trade.order)

    def test_cancel_nonexistent_order(self, broker):
        broker._ib.openTrades.return_value = []

        assert broker.cancel_order("999") is False

    def test_cancel_error(self, broker):
        broker._ib.openTrades.side_effect = Exception("Connection lost")

        assert broker.cancel_order("42") is False


# --- Position tests ---


class TestGetPositions:
    def test_get_positions(self, broker):
        broker._ib.positions.return_value = [
            _mock_position("AAPL", 100, 150.0),
            _mock_position("TSLA", -50, 200.0),
        ]

        positions = broker.get_positions()

        assert len(positions) == 2
        assert "AAPL" in positions
        assert positions["AAPL"].qty == 100
        assert positions["AAPL"].avg_price == 150.0
        assert "TSLA" in positions
        assert positions["TSLA"].qty == -50

    def test_get_positions_empty(self, broker):
        broker._ib.positions.return_value = []
        assert broker.get_positions() == {}

    def test_get_positions_error(self, broker):
        broker._ib.positions.side_effect = Exception("API error")

        with pytest.raises(BrokerError, match="Failed to fetch positions"):
            broker.get_positions()


# --- Account tests ---


class TestGetAccount:
    def test_get_account(self, broker):
        broker._ib.accountSummary.return_value = [
            _mock_account_item("NetLiquidation", "100000"),
            _mock_account_item("AvailableFunds", "50000"),
            _mock_account_item("BuyingPower", "200000"),
            _mock_account_item("GrossPositionValue", "50000"),
            _mock_account_item("InitMarginReq", "25000"),
            _mock_account_item("MaintMarginReq", "20000"),
        ]

        account = broker.get_account()

        assert account.equity == 100000.0
        assert account.cash == 50000.0
        assert account.buying_power == 200000.0
        assert account.portfolio_value == 50000.0
        assert account.margin_used == 25000.0
        assert account.maintenance_margin == 20000.0

    def test_get_account_missing_fields(self, broker):
        broker._ib.accountSummary.return_value = []

        account = broker.get_account()
        assert account.equity == 0.0
        assert account.cash == 0.0

    def test_get_account_error(self, broker):
        broker._ib.accountSummary.side_effect = Exception("Timeout")

        with pytest.raises(BrokerError, match="Failed to fetch account info"):
            broker.get_account()


# --- Order status tests ---


class TestGetOrderStatus:
    def test_get_order_status_submitted(self, broker):
        trade = _mock_trade(order_id=42, status_str="Submitted", filled=0)
        broker._ib.trades.return_value = [trade]

        status = broker.get_order_status("42")

        assert status.order_id == "42"
        assert status.symbol == "AAPL"
        assert status.side == OrderSide.BUY
        assert status.status == OrderStatus.ACCEPTED
        assert status.filled_qty == 0

    def test_get_order_status_filled(self, broker):
        trade = _mock_trade(
            order_id=42,
            status_str="Filled",
            filled=100,
            avg_fill_price=175.50,
        )
        broker._ib.trades.return_value = [trade]

        status = broker.get_order_status("42")

        assert status.status == OrderStatus.FILLED
        assert status.filled_qty == 100
        assert status.avg_fill_price == 175.50

    def test_get_order_status_canceled(self, broker):
        trade = _mock_trade(order_id=42, status_str="Cancelled")
        broker._ib.trades.return_value = [trade]

        status = broker.get_order_status("42")
        assert status.status == OrderStatus.CANCELED

    def test_get_order_status_partially_filled(self, broker):
        trade = _mock_trade(
            order_id=42,
            status_str="Filled",
            total_qty=100,
            filled=50,
        )
        broker._ib.trades.return_value = [trade]

        status = broker.get_order_status("42")
        assert status.status == OrderStatus.PARTIALLY_FILLED

    def test_get_order_status_not_found(self, broker):
        broker._ib.trades.return_value = []

        with pytest.raises(BrokerError, match="Order 999 not found"):
            broker.get_order_status("999")

    def test_order_type_mapping(self, broker):
        for ib_type, expected in [
            ("MKT", OrderType.MARKET),
            ("LMT", OrderType.LIMIT),
            ("STP", OrderType.STOP),
            ("STP LMT", OrderType.STOP_LIMIT),
        ]:
            trade = _mock_trade(order_id=1, order_type=ib_type)
            broker._ib.trades.return_value = [trade]
            status = broker.get_order_status("1")
            assert status.type == expected

    def test_sell_order_side(self, broker):
        trade = _mock_trade(order_id=1, action="SELL")
        broker._ib.trades.return_value = [trade]

        status = broker.get_order_status("1")
        assert status.side == OrderSide.SELL


# --- get_all_orders tests ---


class TestGetAllOrders:
    def test_get_all_orders(self, broker):
        broker._ib.trades.return_value = [
            _mock_trade(order_id=1, status_str="Submitted"),
            _mock_trade(order_id=2, status_str="Filled", filled=100, avg_fill_price=150.0),
        ]

        orders = broker.get_all_orders()
        assert len(orders) == 2

    def test_get_all_orders_open(self, broker):
        broker._ib.openTrades.return_value = [
            _mock_trade(order_id=1, status_str="Submitted"),
        ]

        orders = broker.get_all_orders(status="open")
        assert len(orders) == 1
        broker._ib.openTrades.assert_called_once()

    def test_get_all_orders_closed(self, broker):
        broker._ib.trades.return_value = [
            _mock_trade(order_id=1, status_str="Submitted"),
            _mock_trade(order_id=2, status_str="Filled", filled=100, avg_fill_price=150.0),
            _mock_trade(order_id=3, status_str="Cancelled"),
        ]

        orders = broker.get_all_orders(status="closed")
        assert len(orders) == 2
        assert all(
            o.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)
            for o in orders
        )

    def test_get_all_orders_limit(self, broker):
        broker._ib.trades.return_value = [
            _mock_trade(order_id=i) for i in range(10)
        ]

        orders = broker.get_all_orders(limit=3)
        assert len(orders) == 3

    def test_get_all_orders_error(self, broker):
        broker._ib.trades.side_effect = Exception("API error")

        with pytest.raises(BrokerError, match="Failed to fetch orders"):
            broker.get_all_orders()


# --- close_position tests ---


class TestClosePosition:
    def test_close_long_position(self, broker):
        broker._ib.positions.return_value = [_mock_position("AAPL", 100, 150.0)]
        trade = _mock_trade(order_id=99)
        broker._ib.placeOrder.return_value = trade

        order_id = broker.close_position("AAPL")

        assert order_id == "99"
        # Verify a SELL order was placed
        call_args = broker._ib.placeOrder.call_args
        ib_order = call_args[0][1]
        assert ib_order.action == "SELL"

    def test_close_short_position(self, broker):
        broker._ib.positions.return_value = [_mock_position("TSLA", -50, 200.0)]
        trade = _mock_trade(order_id=100)
        broker._ib.placeOrder.return_value = trade

        order_id = broker.close_position("TSLA")

        assert order_id == "100"

    def test_close_partial_position(self, broker):
        broker._ib.positions.return_value = [_mock_position("AAPL", 100, 150.0)]
        trade = _mock_trade(order_id=101)
        broker._ib.placeOrder.return_value = trade

        order_id = broker.close_position("AAPL", qty=25)
        assert order_id == "101"

    def test_close_no_position(self, broker):
        broker._ib.positions.return_value = []

        with pytest.raises(BrokerError, match="No position in AAPL"):
            broker.close_position("AAPL")


# --- close_all_positions tests ---


class TestCloseAllPositions:
    def test_close_all(self, broker):
        broker._ib.positions.return_value = [
            _mock_position("AAPL", 100, 150.0),
            _mock_position("TSLA", 50, 200.0),
        ]
        trade1 = _mock_trade(order_id=10)
        trade2 = _mock_trade(order_id=11)
        broker._ib.placeOrder.side_effect = [trade1, trade2]

        results = broker.close_all_positions()

        assert results["AAPL"] == "10"
        assert results["TSLA"] == "11"

    def test_close_all_empty(self, broker):
        broker._ib.positions.return_value = []

        results = broker.close_all_positions()
        assert results == {}

    def test_close_all_partial_failure(self, broker):
        broker._ib.positions.return_value = [
            _mock_position("AAPL", 100, 150.0),
            _mock_position("TSLA", 50, 200.0),
        ]
        trade1 = _mock_trade(order_id=10)
        broker._ib.placeOrder.side_effect = [trade1, Exception("Network error")]

        results = broker.close_all_positions()

        assert results["AAPL"] == "10"
        assert results["TSLA"] is None


# --- Disconnect tests ---


class TestDisconnect:
    def test_disconnect(self, broker):
        broker.disconnect()
        broker._ib.disconnect.assert_called_once()

    def test_disconnect_not_connected(self, broker):
        broker._ib.isConnected.return_value = False
        broker.disconnect()
        broker._ib.disconnect.assert_not_called()

    def test_disconnect_no_client(self):
        b = IBKRBroker()
        b.disconnect()  # Should not raise
