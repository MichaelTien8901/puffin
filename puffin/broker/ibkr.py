"""Interactive Brokers broker implementation using ib_async."""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from puffin.broker.base import (
    Broker,
    Order,
    Position,
    AccountInfo,
    OrderStatusInfo,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    BrokerError,
    OrderRejectedError,
    InsufficientFundsError,
)

logger = logging.getLogger(__name__)


@dataclass
class ContractSpec:
    """Specification for an IBKR contract beyond simple US stocks.

    Attributes:
        asset_type: One of "STK", "OPT", "FUT", "CASH".
        exchange: Exchange (default "SMART" for stocks).
        currency: Currency (default "USD").
        expiry: Expiry date string "YYYYMMDD" for options/futures.
        strike: Strike price for options.
        right: "C" or "P" for options.
        multiplier: Contract multiplier (e.g., "100" for options).
        pair_currency: Second currency for forex pairs (e.g., "JPY" for USD/JPY).
    """

    asset_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    expiry: str | None = None
    strike: float | None = None
    right: str | None = None
    multiplier: str | None = None
    pair_currency: str | None = None


class IBKRBroker(Broker):
    """Interactive Brokers implementation using ib_async.

    Requires IB Gateway or TWS running locally.
    Default ports: 4002 (Gateway paper), 4001 (Gateway live),
                   7497 (TWS paper), 7496 (TWS live).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        paper: bool = True,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.paper = paper
        self._ib = None

        logger.info(f"IBKR broker initialized (host={host}, port={port}, paper={paper})")

    def _connect(self):
        """Lazy-connect to IB Gateway/TWS."""
        if self._ib is not None and self._ib.isConnected():
            return self._ib

        from ib_async import IB
        self._ib = IB()
        try:
            self._ib.connect(self.host, self.port, clientId=self.client_id)
        except Exception as e:
            self._ib = None
            raise BrokerError(
                f"Failed to connect to IB Gateway at {self.host}:{self.port}. "
                f"Ensure IB Gateway or TWS is running. Error: {e}"
            )
        return self._ib

    def _make_contract(self, symbol: str, spec: ContractSpec | None = None):
        """Create an IB contract from symbol and optional spec.

        When spec is None, defaults to a US stock on SMART/USD (backward compatible).
        """
        if spec is None:
            from ib_async import Stock
            return Stock(symbol, "SMART", "USD")

        asset_type = spec.asset_type.upper()

        if asset_type == "STK":
            from ib_async import Stock
            return Stock(symbol, spec.exchange, spec.currency)

        if asset_type == "OPT":
            from ib_async import Option
            return Option(
                symbol,
                spec.expiry,
                spec.strike,
                spec.right,
                spec.exchange,
                spec.currency,
            )

        if asset_type == "FUT":
            from ib_async import Future
            return Future(
                symbol,
                spec.expiry,
                spec.exchange,
                currency=spec.currency,
                multiplier=spec.multiplier,
            )

        if asset_type == "CASH":
            from ib_async import Forex
            return Forex(symbol + spec.pair_currency)

        raise BrokerError(f"Unsupported asset type: {asset_type}")

    def _make_order(self, order: Order):
        """Convert Order to an ib_async order object."""
        from ib_async import MarketOrder, LimitOrder, StopOrder, StopLimitOrder

        action = "BUY" if order.side == OrderSide.BUY else "SELL"

        tif_map = {
            TimeInForce.DAY: "DAY",
            TimeInForce.GTC: "GTC",
            TimeInForce.IOC: "IOC",
            TimeInForce.FOK: "FOK",
        }
        tif = tif_map[order.time_in_force]

        if order.type == OrderType.MARKET:
            ib_order = MarketOrder(action, order.qty)
        elif order.type == OrderType.LIMIT:
            ib_order = LimitOrder(action, order.qty, order.limit_price)
        elif order.type == OrderType.STOP:
            ib_order = StopOrder(action, order.qty, order.stop_price)
        elif order.type == OrderType.STOP_LIMIT:
            ib_order = StopLimitOrder(action, order.qty, order.stop_price, order.limit_price)
        else:
            raise BrokerError(f"Unsupported order type: {order.type}")

        ib_order.tif = tif
        if order.client_order_id:
            ib_order.orderRef = order.client_order_id

        return ib_order

    def submit_order(self, order: Order) -> str:
        try:
            ib = self._connect()
            contract = self._make_contract(order.symbol)
            ib_order = self._make_order(order)

            trade = ib.placeOrder(contract, ib_order)
            ib.sleep(0)  # Allow event loop to process

            order_id = str(trade.order.orderId)
            logger.info(
                f"Submitted {order.side.value} order for {order.qty} {order.symbol} "
                f"(order_id={order_id})"
            )
            return order_id

        except BrokerError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "insufficient" in error_msg or "buying power" in error_msg:
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            elif "rejected" in error_msg:
                raise OrderRejectedError(f"Order rejected: {e}")
            else:
                raise BrokerError(f"Failed to submit order: {e}")

    def submit_order_with_spec(self, order: Order, spec: ContractSpec) -> str:
        """Submit an order with a specific contract specification.

        Use this for options, futures, forex, or non-US stocks.
        """
        try:
            ib = self._connect()
            contract = self._make_contract(order.symbol, spec)
            ib_order = self._make_order(order)

            trade = ib.placeOrder(contract, ib_order)
            ib.sleep(0)

            order_id = str(trade.order.orderId)
            logger.info(
                f"Submitted {spec.asset_type} {order.side.value} order for "
                f"{order.qty} {order.symbol} (order_id={order_id})"
            )
            return order_id

        except BrokerError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "insufficient" in error_msg or "buying power" in error_msg:
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            elif "rejected" in error_msg:
                raise OrderRejectedError(f"Order rejected: {e}")
            else:
                raise BrokerError(f"Failed to submit order: {e}")

    def cancel_order(self, order_id: str) -> bool:
        try:
            ib = self._connect()
            # Find the trade by order ID
            for trade in ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    ib.cancelOrder(trade.order)
                    ib.sleep(0)
                    logger.info(f"Canceled order {order_id}")
                    return True
            logger.warning(f"Order {order_id} not found in open trades")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_positions(self) -> dict[str, Position]:
        try:
            ib = self._connect()
            # Use portfolio() instead of positions() — it includes marketValue and unrealizedPNL
            portfolio_items = ib.portfolio()

            positions = {}
            for item in portfolio_items:
                symbol = item.contract.symbol
                qty = int(item.position)
                avg_price = float(item.averageCost)
                positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    avg_price=avg_price,
                    market_value=float(item.marketValue),
                    unrealized_pnl=float(item.unrealizedPNL),
                    cost_basis=abs(qty) * avg_price,
                    last_update=datetime.now(),
                )

            return positions

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to fetch positions: {e}")

    def get_account(self) -> AccountInfo:
        try:
            ib = self._connect()
            summary = ib.accountSummary()

            values = {}
            for item in summary:
                values[item.tag] = item.value

            return AccountInfo(
                equity=float(values.get("NetLiquidation", 0)),
                cash=float(values.get("AvailableFunds", 0)),
                buying_power=float(values.get("BuyingPower", 0)),
                portfolio_value=float(values.get("GrossPositionValue", 0)),
                margin_used=float(values.get("InitMarginReq", 0)),
                maintenance_margin=float(values.get("MaintMarginReq", 0)),
                last_update=datetime.now(),
            )

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to fetch account info: {e}")

    def get_order_status(self, order_id: str) -> OrderStatusInfo:
        try:
            ib = self._connect()

            # Search open and completed trades
            for trade in ib.trades():
                if str(trade.order.orderId) == order_id:
                    return self._trade_to_status(trade)

            raise BrokerError(f"Order {order_id} not found")

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to fetch order status: {e}")

    def _trade_to_status(self, trade) -> OrderStatusInfo:
        """Convert an ib_async Trade to OrderStatusInfo."""
        order = trade.order
        status = trade.orderStatus

        side = OrderSide.BUY if order.action == "BUY" else OrderSide.SELL

        type_map = {
            "MKT": OrderType.MARKET,
            "LMT": OrderType.LIMIT,
            "STP": OrderType.STOP,
            "STP LMT": OrderType.STOP_LIMIT,
        }
        order_type = type_map.get(order.orderType, OrderType.MARKET)

        status_map = {
            "PendingSubmit": OrderStatus.PENDING,
            "PendingCancel": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.ACCEPTED,
            "Submitted": OrderStatus.ACCEPTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELED,
            "Inactive": OrderStatus.REJECTED,
        }
        order_status = status_map.get(status.status, OrderStatus.PENDING)

        if status.status == "Filled" and status.filled < order.totalQuantity:
            order_status = OrderStatus.PARTIALLY_FILLED

        return OrderStatusInfo(
            order_id=str(order.orderId),
            client_order_id=order.orderRef if order.orderRef else None,
            symbol=trade.contract.symbol,
            side=side,
            qty=int(order.totalQuantity),
            filled_qty=int(status.filled),
            type=order_type,
            status=order_status,
            limit_price=float(order.lmtPrice) if order.lmtPrice else None,
            stop_price=float(order.auxPrice) if order.auxPrice else None,
            avg_fill_price=float(status.avgFillPrice) if status.avgFillPrice else None,
            submitted_at=self._find_log_time(trade, ("PendingSubmit", "PreSubmitted", "Submitted")),
            filled_at=self._find_log_time(trade, ("Filled",)),
            canceled_at=self._find_log_time(trade, ("Cancelled",)),
        )

    @staticmethod
    def _find_log_time(trade, statuses: tuple[str, ...]) -> datetime | None:
        """Find the earliest timestamp in trade.log matching any of the given statuses."""
        for entry in trade.log:
            if entry.status in statuses:
                return entry.time
        return None

    def get_all_orders(
        self, status: str | None = None, limit: int = 100
    ) -> list[OrderStatusInfo]:
        """Get all orders, optionally filtered by status.

        Args:
            status: Filter by status ("open", "closed", "all"). Default None (all).
            limit: Maximum number of orders to return.

        Returns:
            List of OrderStatusInfo.
        """
        try:
            ib = self._connect()
            if status == "open":
                trades = ib.openTrades()
            else:
                trades = ib.trades()

            results = [self._trade_to_status(t) for t in trades[:limit]]

            if status == "closed":
                results = [
                    r
                    for r in results
                    if r.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)
                ]

            return results

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to fetch orders: {e}")

    def close_position(self, symbol: str, qty: int | None = None) -> str:
        """Close a position (or part of it).

        Args:
            symbol: Symbol to close.
            qty: Quantity to close (None = close all).

        Returns:
            Order ID.
        """
        try:
            position = self.get_position(symbol)
            if position is None or position.qty == 0:
                raise BrokerError(f"No position in {symbol}")

            side = OrderSide.SELL if position.qty > 0 else OrderSide.BUY
            close_qty = qty if qty is not None else abs(position.qty)

            order = Order(
                symbol=symbol,
                side=side,
                qty=close_qty,
                type=OrderType.MARKET,
            )

            return self.submit_order(order)

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(f"Failed to close position: {e}")

    def close_all_positions(self) -> dict[str, str]:
        """Close all open positions.

        Returns:
            Dict mapping symbol to order ID.
        """
        positions = self.get_positions()
        results = {}

        for symbol in positions:
            try:
                order_id = self.close_position(symbol)
                results[symbol] = order_id
            except Exception as e:
                logger.error(f"Failed to close position in {symbol}: {e}")
                results[symbol] = None

        return results

    def disconnect(self):
        """Disconnect from IB Gateway/TWS."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IB Gateway")

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass
