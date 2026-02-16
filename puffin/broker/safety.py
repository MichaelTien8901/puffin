"""Safety controls and risk checks for live trading."""

import logging
from datetime import datetime, date
from typing import Callable

from puffin.broker.base import Broker, Order, BrokerError

logger = logging.getLogger(__name__)


class SafetyController:
    """Safety controls and validation for live trading."""

    def __init__(
        self,
        broker: Broker,
        max_order_size: int | None = None,
        max_position_size: int | None = None,
        max_daily_loss: float | None = None,
        max_total_position_value: float | None = None,
        require_confirmation: bool = True,
    ):
        """Initialize safety controller.

        Args:
            broker: Broker instance to check positions/account.
            max_order_size: Maximum shares per order (default None = unlimited).
            max_position_size: Maximum shares per position (default None = unlimited).
            max_daily_loss: Maximum loss allowed per day (default None = unlimited).
            max_total_position_value: Maximum total position value (default None = unlimited).
            require_confirmation: Require explicit confirmation for live trading (default True).
        """
        self.broker = broker
        self.max_order_size = max_order_size
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_total_position_value = max_total_position_value
        self.require_confirmation = require_confirmation

        # Daily loss tracking
        self._daily_pnl: dict[date, float] = {}
        self._circuit_breaker_active = False
        self._confirmation_given = False

        # Custom validators
        self._custom_validators: list[Callable[[Order], tuple[bool, str]]] = []

    def validate_order(self, order: Order) -> tuple[bool, str]:
        """Validate an order against safety rules.

        Args:
            order: Order to validate.

        Returns:
            Tuple of (is_valid, reason). reason is empty string if valid.
        """
        # Check confirmation requirement
        if self.require_confirmation and not self._confirmation_given:
            return False, "Live trading confirmation not given. Call confirm_live_trading() first."

        # Check circuit breaker
        if self._circuit_breaker_active:
            return False, "Circuit breaker active due to daily loss limit."

        # Check order size
        if self.max_order_size and order.qty > self.max_order_size:
            return False, f"Order size {order.qty} exceeds max_order_size {self.max_order_size}"

        # Check position size
        if self.max_position_size:
            try:
                position = self.broker.get_position(order.symbol)
                current_qty = position.qty if position else 0

                # Calculate new position size
                if order.side.value == "buy":
                    new_qty = current_qty + order.qty
                else:
                    new_qty = current_qty - order.qty

                if abs(new_qty) > self.max_position_size:
                    return False, (
                        f"Order would result in position size {new_qty}, "
                        f"exceeding max_position_size {self.max_position_size}"
                    )
            except BrokerError as e:
                logger.error(f"Failed to check position size: {e}")
                return False, f"Unable to validate position size: {e}"

        # Check total position value
        if self.max_total_position_value:
            try:
                account = self.broker.get_account()
                if account.portfolio_value > self.max_total_position_value:
                    return False, (
                        f"Total position value ${account.portfolio_value:,.2f} "
                        f"exceeds limit ${self.max_total_position_value:,.2f}"
                    )
            except BrokerError as e:
                logger.error(f"Failed to check total position value: {e}")
                return False, f"Unable to validate total position value: {e}"

        # Check daily loss limit
        if self.max_daily_loss:
            today = datetime.now().date()
            today_pnl = self._daily_pnl.get(today, 0.0)

            # Update today's P&L from broker
            try:
                account = self.broker.get_account()
                # Note: This assumes broker provides today's P&L
                # You may need to track this differently
                # For now, we'll use a simple approximation
                positions = self.broker.get_positions()
                total_unrealized_pnl = sum(p.unrealized_pnl for p in positions.values())
                self._daily_pnl[today] = total_unrealized_pnl

                if total_unrealized_pnl < -abs(self.max_daily_loss):
                    self._circuit_breaker_active = True
                    return False, (
                        f"Daily loss ${total_unrealized_pnl:,.2f} exceeds limit "
                        f"${self.max_daily_loss:,.2f}. Circuit breaker activated."
                    )
            except BrokerError as e:
                logger.error(f"Failed to check daily loss: {e}")

        # Run custom validators
        for validator in self._custom_validators:
            try:
                is_valid, reason = validator(order)
                if not is_valid:
                    return False, f"Custom validator failed: {reason}"
            except Exception as e:
                logger.error(f"Custom validator error: {e}")
                return False, f"Custom validator error: {e}"

        return True, ""

    def confirm_live_trading(self, confirmation_code: str | None = None) -> bool:
        """Explicitly confirm live trading.

        Args:
            confirmation_code: Optional confirmation code for extra safety.
                              If None, will prompt for "CONFIRM".

        Returns:
            True if confirmation given.
        """
        if confirmation_code is None:
            # Prompt for confirmation
            print("\n" + "=" * 60)
            print("  LIVE TRADING CONFIRMATION REQUIRED")
            print("=" * 60)
            print("\nYou are about to enable live trading with real money.")
            print("Please review your safety settings:")
            print(f"  - Max order size: {self.max_order_size or 'UNLIMITED'}")
            print(f"  - Max position size: {self.max_position_size or 'UNLIMITED'}")
            print(f"  - Max daily loss: ${self.max_daily_loss or 'UNLIMITED'}")
            print(f"  - Max total position value: ${self.max_total_position_value or 'UNLIMITED'}")
            print("\nType 'CONFIRM' to proceed: ", end="")

            response = input().strip()
            confirmed = response == "CONFIRM"
        else:
            confirmed = confirmation_code == "CONFIRM"

        if confirmed:
            self._confirmation_given = True
            logger.warning("Live trading confirmation given")
            print("\nLive trading enabled. Trade safely!")
        else:
            logger.info("Live trading confirmation denied")
            print("\nLive trading NOT enabled.")

        return confirmed

    def daily_loss_limit(self, max_loss: float):
        """Set or update daily loss limit (circuit breaker).

        Args:
            max_loss: Maximum daily loss before circuit breaker activates.
        """
        self.max_daily_loss = abs(max_loss)
        logger.info(f"Daily loss limit set to ${self.max_daily_loss:,.2f}")

    def reset_circuit_breaker(self):
        """Reset circuit breaker (admin override).

        Use with caution.
        """
        logger.warning("Circuit breaker manually reset")
        self._circuit_breaker_active = False

    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is active.

        Returns:
            True if circuit breaker is active.
        """
        return self._circuit_breaker_active

    def add_validator(self, validator: Callable[[Order], tuple[bool, str]]):
        """Add a custom order validator.

        Args:
            validator: Function that takes Order and returns (is_valid, reason).
        """
        self._custom_validators.append(validator)
        logger.info(f"Added custom validator: {validator.__name__}")

    def check_daily_pnl(self) -> float:
        """Get current daily P&L.

        Returns:
            Today's P&L.
        """
        today = datetime.now().date()

        try:
            positions = self.broker.get_positions()
            total_unrealized_pnl = sum(p.unrealized_pnl for p in positions.values())
            self._daily_pnl[today] = total_unrealized_pnl

            # Check against limit
            if self.max_daily_loss and total_unrealized_pnl < -abs(self.max_daily_loss):
                if not self._circuit_breaker_active:
                    logger.critical(
                        f"Daily loss limit breached: ${total_unrealized_pnl:,.2f}. "
                        "Activating circuit breaker."
                    )
                    self._circuit_breaker_active = True

            return total_unrealized_pnl

        except BrokerError as e:
            logger.error(f"Failed to check daily P&L: {e}")
            return 0.0

    def get_stats(self) -> dict:
        """Get safety controller statistics.

        Returns:
            Dict with safety stats.
        """
        return {
            "confirmation_given": self._confirmation_given,
            "circuit_breaker_active": self._circuit_breaker_active,
            "max_order_size": self.max_order_size,
            "max_position_size": self.max_position_size,
            "max_daily_loss": self.max_daily_loss,
            "max_total_position_value": self.max_total_position_value,
            "custom_validators": len(self._custom_validators),
        }

    def reset_confirmation(self):
        """Reset live trading confirmation (for testing).

        Use with extreme caution in production.
        """
        logger.warning("Live trading confirmation reset")
        self._confirmation_given = False


class PositionSizingValidator:
    """Pre-built validator for position sizing rules."""

    def __init__(self, max_position_pct: float = 0.25):
        """Initialize position sizing validator.

        Args:
            max_position_pct: Maximum position size as fraction of portfolio (default 0.25 = 25%).
        """
        self.max_position_pct = max_position_pct

    def __call__(self, broker: Broker, order: Order) -> tuple[bool, str]:
        """Validate order against position sizing rules.

        Args:
            broker: Broker instance.
            order: Order to validate.

        Returns:
            Tuple of (is_valid, reason).
        """
        try:
            account = broker.get_account()
            position = broker.get_position(order.symbol)

            # Estimate order value (use current price approximation)
            # In production, you'd fetch real-time quote
            if position and position.market_value:
                current_price = abs(position.market_value / position.qty)
            else:
                # Fallback to limit price if available
                current_price = order.limit_price if order.limit_price else 100.0

            order_value = order.qty * current_price

            # Calculate new position value
            current_position_value = abs(position.market_value) if position else 0.0

            if order.side.value == "buy":
                new_position_value = current_position_value + order_value
            else:
                new_position_value = max(0, current_position_value - order_value)

            # Check against portfolio
            max_allowed = account.portfolio_value * self.max_position_pct

            if new_position_value > max_allowed:
                return False, (
                    f"Position would be ${new_position_value:,.2f} "
                    f"({new_position_value/account.portfolio_value*100:.1f}% of portfolio), "
                    f"exceeding {self.max_position_pct*100:.1f}% limit"
                )

            return True, ""

        except Exception as e:
            return False, f"Position sizing validation error: {e}"


class TradingHoursValidator:
    """Pre-built validator for trading hours restrictions."""

    def __init__(self, allow_extended_hours: bool = False):
        """Initialize trading hours validator.

        Args:
            allow_extended_hours: Allow trading outside regular hours (default False).
        """
        self.allow_extended_hours = allow_extended_hours

    def __call__(self, broker: Broker, order: Order) -> tuple[bool, str]:
        """Validate order against trading hours.

        Args:
            broker: Broker instance (unused but required for signature).
            order: Order to validate.

        Returns:
            Tuple of (is_valid, reason).
        """
        from puffin.broker.session import TradingSession

        session = TradingSession(extended_hours=self.allow_extended_hours)

        if not session.is_market_open():
            next_open = session.next_open()
            return False, f"Market is closed. Next open: {next_open}"

        return True, ""


class SymbolWhitelistValidator:
    """Pre-built validator for symbol whitelist."""

    def __init__(self, allowed_symbols: list[str]):
        """Initialize symbol whitelist validator.

        Args:
            allowed_symbols: List of allowed symbols.
        """
        self.allowed_symbols = set(s.upper() for s in allowed_symbols)

    def __call__(self, broker: Broker, order: Order) -> tuple[bool, str]:
        """Validate order against symbol whitelist.

        Args:
            broker: Broker instance (unused but required for signature).
            order: Order to validate.

        Returns:
            Tuple of (is_valid, reason).
        """
        if order.symbol.upper() not in self.allowed_symbols:
            return False, f"Symbol {order.symbol} not in whitelist"

        return True, ""
