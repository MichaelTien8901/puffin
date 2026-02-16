"""Abstract broker interface and data models for live trading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    """Time in force."""
    DAY = "day"  # Good for day
    GTC = "gtc"  # Good til canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order specification."""
    symbol: str
    side: OrderSide
    qty: int
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: str | None = None

    def __post_init__(self):
        """Validate order parameters."""
        if self.qty <= 0:
            raise ValueError("Quantity must be positive")

        if self.type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require limit_price")

        if self.type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require stop_price")

        if self.type == OrderType.STOP_LIMIT:
            if self.limit_price is None or self.stop_price is None:
                raise ValueError("Stop limit orders require both limit_price and stop_price")


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    qty: int  # Positive for long, negative for short
    avg_price: float
    market_value: float
    unrealized_pnl: float
    cost_basis: float | None = None
    last_update: datetime | None = None

    @property
    def side(self) -> str:
        """Position side (long/short)."""
        return "long" if self.qty > 0 else "short" if self.qty < 0 else "flat"


@dataclass
class AccountInfo:
    """Account information."""
    equity: float  # Total account value
    cash: float  # Available cash
    buying_power: float  # Available buying power (may include margin)
    portfolio_value: float | None = None  # Market value of positions
    margin_used: float = 0.0
    maintenance_margin: float = 0.0
    last_update: datetime | None = None

    def __post_init__(self):
        if self.portfolio_value is None:
            self.portfolio_value = self.equity - self.cash


@dataclass
class OrderStatusInfo:
    """Detailed order status."""
    order_id: str
    client_order_id: str | None
    symbol: str
    side: OrderSide
    qty: int
    filled_qty: int
    type: OrderType
    status: OrderStatus
    limit_price: float | None = None
    stop_price: float | None = None
    avg_fill_price: float | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    canceled_at: datetime | None = None
    reject_reason: str | None = None


class Broker(ABC):
    """Abstract broker interface for live trading."""

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order to the broker.

        Args:
            order: Order specification.

        Returns:
            Order ID assigned by the broker.

        Raises:
            BrokerError: If order submission fails.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancellation successful, False otherwise.

        Raises:
            BrokerError: If cancellation fails.
        """

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Get all current positions.

        Returns:
            Dict mapping symbol to Position.

        Raises:
            BrokerError: If unable to fetch positions.
        """

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Get account information.

        Returns:
            AccountInfo with current account status.

        Raises:
            BrokerError: If unable to fetch account info.
        """

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatusInfo:
        """Get status of a specific order.

        Args:
            order_id: Order ID to check.

        Returns:
            OrderStatusInfo with current order status.

        Raises:
            BrokerError: If unable to fetch order status.
        """

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol.

        Args:
            symbol: Symbol to get position for.

        Returns:
            Position if exists, None otherwise.
        """
        positions = self.get_positions()
        return positions.get(symbol)

    def modify_order(self, order_id: str, qty: int | None = None,
                    limit_price: float | None = None) -> str:
        """Modify an existing order.

        Default implementation cancels and replaces. Override for native modify support.

        Args:
            order_id: Order ID to modify.
            qty: New quantity (optional).
            limit_price: New limit price (optional).

        Returns:
            New order ID.

        Raises:
            BrokerError: If modification fails.
        """
        # Get original order
        original = self.get_order_status(order_id)

        # Cancel it
        if not self.cancel_order(order_id):
            raise BrokerError(f"Failed to cancel order {order_id}")

        # Submit new order
        new_order = Order(
            symbol=original.symbol,
            side=original.side,
            qty=qty if qty is not None else original.qty,
            type=original.type,
            limit_price=limit_price if limit_price is not None else original.limit_price,
            stop_price=original.stop_price,
        )

        return self.submit_order(new_order)


class BrokerError(Exception):
    """Base exception for broker errors."""
    pass


class OrderRejectedError(BrokerError):
    """Order was rejected by the broker."""
    pass


class InsufficientFundsError(BrokerError):
    """Insufficient funds to place order."""
    pass


class InvalidOrderError(BrokerError):
    """Order parameters are invalid."""
    pass
