"""Alpaca broker implementation."""

import logging
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import OrderType as AlpacaType
from alpaca.trading.enums import TimeInForce as AlpacaTIF
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest
from alpaca.trading.stream import TradingStream

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


class AlpacaBroker(Broker):
    """Alpaca broker implementation using alpaca-py SDK."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        stream_updates: bool = False,
    ):
        """Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key.
            secret_key: Alpaca secret key.
            paper: Use paper trading (default True).
            stream_updates: Enable WebSocket order status tracking (default False).
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        # Initialize trading client
        self.client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

        # WebSocket stream for order updates
        self.stream = None
        self._stream_active = False
        if stream_updates:
            self._init_stream()

        logger.info(f"Alpaca broker initialized (paper={paper})")

    def _init_stream(self):
        """Initialize WebSocket stream for order updates."""
        self.stream = TradingStream(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )

        @self.stream.on_trade_update
        async def on_trade_update(update):
            """Handle trade updates from WebSocket."""
            logger.info(f"Trade update: {update.event} for order {update.order.id}")
            # Store in local cache or trigger callbacks
            # This can be extended for real-time order tracking

        self._stream_active = True
        logger.info("WebSocket order stream initialized")

    def submit_order(self, order: Order) -> str:
        """Submit an order to Alpaca.

        Args:
            order: Order specification.

        Returns:
            Order ID from Alpaca.

        Raises:
            BrokerError: If order submission fails.
        """
        try:
            # Convert to Alpaca request
            request = self._build_order_request(order)

            # Submit order
            alpaca_order = self.client.submit_order(request)

            logger.info(
                f"Submitted {order.side.value} order for {order.qty} {order.symbol} "
                f"(order_id={alpaca_order.id})"
            )

            return alpaca_order.id

        except Exception as e:
            error_msg = str(e).lower()
            if "insufficient" in error_msg or "buying power" in error_msg:
                raise InsufficientFundsError(f"Insufficient funds: {e}")
            elif "rejected" in error_msg:
                raise OrderRejectedError(f"Order rejected: {e}")
            else:
                raise BrokerError(f"Failed to submit order: {e}")

    def _build_order_request(self, order: Order):
        """Build Alpaca order request from Order object."""
        # Convert enums
        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
        tif = self._convert_tif(order.time_in_force)

        # Build request based on order type
        if order.type == OrderType.MARKET:
            return MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                client_order_id=order.client_order_id,
            )
        elif order.type == OrderType.LIMIT:
            return LimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                limit_price=order.limit_price,
                client_order_id=order.client_order_id,
            )
        elif order.type == OrderType.STOP:
            return StopOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                stop_price=order.stop_price,
                client_order_id=order.client_order_id,
            )
        elif order.type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                client_order_id=order.client_order_id,
            )

    def _convert_tif(self, tif: TimeInForce):
        """Convert TimeInForce to Alpaca enum."""
        mapping = {
            TimeInForce.DAY: AlpacaTIF.DAY,
            TimeInForce.GTC: AlpacaTIF.GTC,
            TimeInForce.IOC: AlpacaTIF.IOC,
            TimeInForce.FOK: AlpacaTIF.FOK,
        }
        return mapping[tif]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Alpaca order ID.

        Returns:
            True if cancellation successful, False otherwise.
        """
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"Canceled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_positions(self) -> dict[str, Position]:
        """Get all current positions.

        Returns:
            Dict mapping symbol to Position.
        """
        try:
            alpaca_positions = self.client.get_all_positions()

            positions = {}
            for pos in alpaca_positions:
                positions[pos.symbol] = Position(
                    symbol=pos.symbol,
                    qty=int(pos.qty),
                    avg_price=float(pos.avg_entry_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    cost_basis=float(pos.cost_basis),
                    last_update=datetime.now(),
                )

            return positions

        except Exception as e:
            raise BrokerError(f"Failed to fetch positions: {e}")

    def get_account(self) -> AccountInfo:
        """Get account information.

        Returns:
            AccountInfo with current account status.
        """
        try:
            account = self.client.get_account()

            return AccountInfo(
                equity=float(account.equity),
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                portfolio_value=float(account.portfolio_value),
                margin_used=float(getattr(account, "initial_margin", 0)),
                maintenance_margin=float(getattr(account, "maintenance_margin", 0)),
                last_update=datetime.now(),
            )

        except Exception as e:
            raise BrokerError(f"Failed to fetch account info: {e}")

    def get_order_status(self, order_id: str) -> OrderStatusInfo:
        """Get status of a specific order.

        Args:
            order_id: Alpaca order ID.

        Returns:
            OrderStatusInfo with current order status.
        """
        try:
            order = self.client.get_order_by_id(order_id)

            return OrderStatusInfo(
                order_id=order.id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=OrderSide.BUY if order.side == AlpacaSide.BUY else OrderSide.SELL,
                qty=int(order.qty),
                filled_qty=int(order.filled_qty or 0),
                type=self._convert_order_type(order.type),
                status=self._convert_order_status(order.status),
                limit_price=float(order.limit_price) if order.limit_price else None,
                stop_price=float(order.stop_price) if order.stop_price else None,
                avg_fill_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
                canceled_at=order.canceled_at,
            )

        except Exception as e:
            raise BrokerError(f"Failed to fetch order status: {e}")

    def _convert_order_type(self, alpaca_type) -> OrderType:
        """Convert Alpaca order type to our OrderType."""
        mapping = {
            AlpacaType.MARKET: OrderType.MARKET,
            AlpacaType.LIMIT: OrderType.LIMIT,
            AlpacaType.STOP: OrderType.STOP,
            AlpacaType.STOP_LIMIT: OrderType.STOP_LIMIT,
        }
        return mapping.get(alpaca_type, OrderType.MARKET)

    def _convert_order_status(self, alpaca_status) -> OrderStatus:
        """Convert Alpaca order status to our OrderStatus."""
        status_str = str(alpaca_status).lower()

        if "pending" in status_str or "new" in status_str:
            return OrderStatus.PENDING
        elif "accepted" in status_str:
            return OrderStatus.ACCEPTED
        elif "filled" in status_str:
            if "partially" in status_str:
                return OrderStatus.PARTIALLY_FILLED
            return OrderStatus.FILLED
        elif "canceled" in status_str or "cancelled" in status_str:
            return OrderStatus.CANCELED
        elif "rejected" in status_str:
            return OrderStatus.REJECTED
        elif "expired" in status_str:
            return OrderStatus.EXPIRED
        else:
            return OrderStatus.PENDING

    def get_all_orders(self, status: str | None = None, limit: int = 100) -> list[OrderStatusInfo]:
        """Get all orders, optionally filtered by status.

        Args:
            status: Filter by status ("open", "closed", "all"). Default None (all).
            limit: Maximum number of orders to return.

        Returns:
            List of OrderStatusInfo.
        """
        try:
            orders = self.client.get_orders(status=status, limit=limit)
            return [self.get_order_status(order.id) for order in orders]
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
            # Get current position
            position = self.get_position(symbol)
            if position is None or position.qty == 0:
                raise BrokerError(f"No position in {symbol}")

            # Determine side and qty
            side = OrderSide.SELL if position.qty > 0 else OrderSide.BUY
            close_qty = qty if qty is not None else abs(position.qty)

            # Submit order
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

    def __del__(self):
        """Cleanup on destruction."""
        if getattr(self, 'stream', None) and getattr(self, '_stream_active', False):
            try:
                self.stream.stop()
            except Exception:
                pass
