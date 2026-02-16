"""Order management and tracking."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Callable

from puffin.broker.base import Broker, Order, OrderStatus, OrderStatusInfo, BrokerError

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order lifecycle with tracking and callbacks."""

    def __init__(self, broker: Broker):
        """Initialize order manager.

        Args:
            broker: Broker instance to execute orders.
        """
        self.broker = broker
        self._orders: dict[str, OrderStatusInfo] = {}
        self._pending_orders: set[str] = set()
        self._filled_orders: set[str] = set()

        # Event callbacks
        self._on_fill_callbacks: list[Callable] = []
        self._on_cancel_callbacks: list[Callable] = []
        self._on_reject_callbacks: list[Callable] = []

    def submit(self, order: Order) -> str:
        """Submit an order.

        Args:
            order: Order to submit.

        Returns:
            Order ID.

        Raises:
            BrokerError: If submission fails.
        """
        try:
            order_id = self.broker.submit_order(order)
            self._pending_orders.add(order_id)

            # Fetch initial status
            status = self.broker.get_order_status(order_id)
            self._orders[order_id] = status

            logger.info(
                f"Submitted order {order_id}: {order.side.value} {order.qty} {order.symbol}"
            )

            return order_id

        except BrokerError as e:
            logger.error(f"Order submission failed: {e}")
            # Trigger reject callback
            for callback in self._on_reject_callbacks:
                try:
                    callback(order, str(e))
                except Exception as cb_error:
                    logger.error(f"Reject callback error: {cb_error}")
            raise

    def cancel(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if canceled, False otherwise.
        """
        success = self.broker.cancel_order(order_id)

        if success:
            logger.info(f"Canceled order {order_id}")
            self._pending_orders.discard(order_id)

            # Update status
            status = self.broker.get_order_status(order_id)
            self._orders[order_id] = status

            # Trigger cancel callback
            for callback in self._on_cancel_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Cancel callback error: {e}")

        return success

    def modify(self, order_id: str, qty: int | None = None, limit_price: float | None = None) -> str:
        """Modify an existing order.

        Args:
            order_id: Order ID to modify.
            qty: New quantity (optional).
            limit_price: New limit price (optional).

        Returns:
            New order ID (may be same as original).

        Raises:
            BrokerError: If modification fails.
        """
        try:
            new_order_id = self.broker.modify_order(order_id, qty=qty, limit_price=limit_price)

            # Remove old order from pending
            self._pending_orders.discard(order_id)

            # Add new order to pending
            self._pending_orders.add(new_order_id)

            # Update status
            status = self.broker.get_order_status(new_order_id)
            self._orders[new_order_id] = status

            logger.info(f"Modified order {order_id} → {new_order_id}")

            return new_order_id

        except BrokerError:
            raise

    def track_order(self, order_id: str) -> OrderStatusInfo:
        """Track an order and update its status.

        Args:
            order_id: Order ID to track.

        Returns:
            Current OrderStatusInfo.

        Raises:
            BrokerError: If unable to fetch status.
        """
        status = self.broker.get_order_status(order_id)
        self._orders[order_id] = status

        # Update tracking sets
        if status.status == OrderStatus.FILLED:
            self._pending_orders.discard(order_id)
            self._filled_orders.add(order_id)

            # Trigger fill callback
            for callback in self._on_fill_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Fill callback error: {e}")

        elif status.status in {OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}:
            self._pending_orders.discard(order_id)

        return status

    def track_all_pending(self) -> dict[str, OrderStatusInfo]:
        """Track all pending orders and update their statuses.

        Returns:
            Dict mapping order_id to OrderStatusInfo.
        """
        results = {}
        for order_id in list(self._pending_orders):
            try:
                results[order_id] = self.track_order(order_id)
            except BrokerError as e:
                logger.error(f"Failed to track order {order_id}: {e}")
                results[order_id] = self._orders.get(order_id)

        return results

    def pending_orders(self) -> list[OrderStatusInfo]:
        """Get all pending orders.

        Returns:
            List of pending OrderStatusInfo.
        """
        return [self._orders[oid] for oid in self._pending_orders if oid in self._orders]

    def filled_orders(self) -> list[OrderStatusInfo]:
        """Get all filled orders.

        Returns:
            List of filled OrderStatusInfo.
        """
        return [self._orders[oid] for oid in self._filled_orders if oid in self._orders]

    def get_order(self, order_id: str) -> OrderStatusInfo | None:
        """Get cached order status.

        Args:
            order_id: Order ID.

        Returns:
            OrderStatusInfo if cached, None otherwise.
        """
        return self._orders.get(order_id)

    def reconcile_positions(self) -> list[dict]:
        """Reconcile tracked orders with actual positions.

        Returns:
            List of discrepancies {symbol, expected_qty, actual_qty, diff}.
        """
        # Calculate expected positions from filled orders
        expected_positions = defaultdict(int)

        for order_id in self._filled_orders:
            status = self._orders.get(order_id)
            if status and status.status == OrderStatus.FILLED:
                qty = status.filled_qty
                if status.side.value == "sell":
                    qty = -qty
                expected_positions[status.symbol] += qty

        # Get actual positions
        actual_positions = self.broker.get_positions()

        # Find discrepancies
        discrepancies = []
        all_symbols = set(expected_positions.keys()) | set(actual_positions.keys())

        for symbol in all_symbols:
            expected_qty = expected_positions.get(symbol, 0)
            actual_qty = actual_positions.get(symbol, Position(symbol, 0, 0, 0, 0)).qty
            diff = actual_qty - expected_qty

            if diff != 0:
                discrepancies.append({
                    "symbol": symbol,
                    "expected_qty": expected_qty,
                    "actual_qty": actual_qty,
                    "diff": diff,
                })

        if discrepancies:
            logger.warning(f"Found {len(discrepancies)} position discrepancies")

        return discrepancies

    def on_fill(self, callback: Callable[[OrderStatusInfo], None]):
        """Register callback for order fills.

        Args:
            callback: Function to call when order is filled.
                      Signature: callback(order_status: OrderStatusInfo)
        """
        self._on_fill_callbacks.append(callback)

    def on_cancel(self, callback: Callable[[OrderStatusInfo], None]):
        """Register callback for order cancellations.

        Args:
            callback: Function to call when order is canceled.
                      Signature: callback(order_status: OrderStatusInfo)
        """
        self._on_cancel_callbacks.append(callback)

    def on_reject(self, callback: Callable[[Order, str], None]):
        """Register callback for order rejections.

        Args:
            callback: Function to call when order is rejected.
                      Signature: callback(order: Order, reason: str)
        """
        self._on_reject_callbacks.append(callback)

    def cancel_all_pending(self) -> dict[str, bool]:
        """Cancel all pending orders.

        Returns:
            Dict mapping order_id to success status.
        """
        results = {}
        for order_id in list(self._pending_orders):
            results[order_id] = self.cancel(order_id)

        return results

    def get_order_history(self, symbol: str | None = None) -> list[OrderStatusInfo]:
        """Get order history, optionally filtered by symbol.

        Args:
            symbol: Filter by symbol (optional).

        Returns:
            List of OrderStatusInfo ordered by submission time.
        """
        orders = list(self._orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        # Sort by submission time
        orders.sort(key=lambda o: o.submitted_at or datetime.min, reverse=True)

        return orders

    def get_stats(self) -> dict:
        """Get order statistics.

        Returns:
            Dict with order statistics.
        """
        filled = len(self._filled_orders)
        pending = len(self._pending_orders)
        total = len(self._orders)

        canceled = sum(
            1 for o in self._orders.values()
            if o.status == OrderStatus.CANCELED
        )
        rejected = sum(
            1 for o in self._orders.values()
            if o.status == OrderStatus.REJECTED
        )

        # Calculate fill rate
        fill_rate = filled / total if total > 0 else 0.0

        return {
            "total_orders": total,
            "pending": pending,
            "filled": filled,
            "canceled": canceled,
            "rejected": rejected,
            "fill_rate": fill_rate,
        }

    def clear_history(self, keep_pending: bool = True):
        """Clear order history.

        Args:
            keep_pending: Keep pending orders in history (default True).
        """
        if keep_pending:
            # Keep only pending orders
            self._orders = {
                oid: status for oid, status in self._orders.items()
                if oid in self._pending_orders
            }
            self._filled_orders.clear()
        else:
            # Clear everything
            self._orders.clear()
            self._pending_orders.clear()
            self._filled_orders.clear()

        logger.info("Cleared order history")
