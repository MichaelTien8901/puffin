"""Stop loss implementations for trade exit management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class Position:
    """Position information for stop loss calculations."""
    ticker: str
    entry_price: float
    entry_time: datetime
    quantity: float
    side: str  # 'long' or 'short'
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.highest_price is None:
            self.highest_price = self.entry_price
        if self.lowest_price is None:
            self.lowest_price = self.entry_price


class StopLoss(ABC):
    """Abstract base class for stop loss strategies."""

    @abstractmethod
    def check(
        self,
        current_price: float,
        entry_price: float,
        position: Position
    ) -> bool:
        """
        Check if stop loss is triggered.

        Parameters
        ----------
        current_price : float
            Current market price
        entry_price : float
            Entry price of the position
        position : Position
            Position object with additional information

        Returns
        -------
        bool
            True if stop loss is triggered, False otherwise
        """
        pass


class FixedStop(StopLoss):
    """Fixed stop loss at a specified distance from entry."""

    def __init__(self, stop_distance: float, price_based: bool = True):
        """
        Initialize fixed stop loss.

        Parameters
        ----------
        stop_distance : float
            Stop distance in price units (if price_based=True) or percentage (if False)
        price_based : bool, default=True
            If True, stop_distance is in price units; if False, it's a percentage
        """
        if stop_distance <= 0:
            raise ValueError("Stop distance must be positive")
        self.stop_distance = stop_distance
        self.price_based = price_based

    def check(
        self,
        current_price: float,
        entry_price: float,
        position: Position
    ) -> bool:
        """Check if fixed stop loss is triggered."""
        if self.price_based:
            # Price-based stop
            if position.side == 'long':
                stop_price = entry_price - self.stop_distance
                return current_price <= stop_price
            else:  # short
                stop_price = entry_price + self.stop_distance
                return current_price >= stop_price
        else:
            # Percentage-based stop
            if position.side == 'long':
                pct_change = (current_price - entry_price) / entry_price
                return pct_change <= -self.stop_distance
            else:  # short
                pct_change = (entry_price - current_price) / entry_price
                return pct_change <= -self.stop_distance


class TrailingStop(StopLoss):
    """Trailing stop loss that follows favorable price movement."""

    def __init__(self, trail_distance: float, price_based: bool = True):
        """
        Initialize trailing stop loss.

        Parameters
        ----------
        trail_distance : float
            Trailing distance in price units (if price_based=True) or percentage (if False)
        price_based : bool, default=True
            If True, trail_distance is in price units; if False, it's a percentage
        """
        if trail_distance <= 0:
            raise ValueError("Trail distance must be positive")
        self.trail_distance = trail_distance
        self.price_based = price_based

    def check(
        self,
        current_price: float,
        entry_price: float,
        position: Position
    ) -> bool:
        """Check if trailing stop loss is triggered."""
        # Update highest/lowest price
        if position.side == 'long':
            position.highest_price = max(position.highest_price, current_price)

            if self.price_based:
                stop_price = position.highest_price - self.trail_distance
            else:
                stop_price = position.highest_price * (1 - self.trail_distance)

            return current_price <= stop_price
        else:  # short
            position.lowest_price = min(position.lowest_price, current_price)

            if self.price_based:
                stop_price = position.lowest_price + self.trail_distance
            else:
                stop_price = position.lowest_price * (1 + self.trail_distance)

            return current_price >= stop_price


class ATRStop(StopLoss):
    """ATR-based stop loss that adapts to volatility."""

    def __init__(self, atr_multiplier: float = 2.0, trailing: bool = False):
        """
        Initialize ATR-based stop loss.

        Parameters
        ----------
        atr_multiplier : float, default=2.0
            Multiplier for ATR to set stop distance
        trailing : bool, default=False
            If True, acts as a trailing stop using ATR distance
        """
        if atr_multiplier <= 0:
            raise ValueError("ATR multiplier must be positive")
        self.atr_multiplier = atr_multiplier
        self.trailing = trailing

    def check(
        self,
        current_price: float,
        entry_price: float,
        position: Position
    ) -> bool:
        """Check if ATR stop loss is triggered."""
        # ATR should be stored in position metadata
        atr = position.metadata.get('atr')
        if atr is None:
            raise ValueError("ATR value must be provided in position.metadata['atr']")

        stop_distance = atr * self.atr_multiplier

        if self.trailing:
            # Trailing ATR stop
            if position.side == 'long':
                position.highest_price = max(position.highest_price, current_price)
                stop_price = position.highest_price - stop_distance
                return current_price <= stop_price
            else:  # short
                position.lowest_price = min(position.lowest_price, current_price)
                stop_price = position.lowest_price + stop_distance
                return current_price >= stop_price
        else:
            # Fixed ATR stop
            if position.side == 'long':
                stop_price = entry_price - stop_distance
                return current_price <= stop_price
            else:  # short
                stop_price = entry_price + stop_distance
                return current_price >= stop_price


class TimeStop(StopLoss):
    """Time-based stop loss that exits after a specified duration."""

    def __init__(self, max_bars: Optional[int] = None, max_seconds: Optional[float] = None):
        """
        Initialize time-based stop loss.

        Parameters
        ----------
        max_bars : int, optional
            Maximum number of bars to hold position
        max_seconds : float, optional
            Maximum number of seconds to hold position
        """
        if max_bars is None and max_seconds is None:
            raise ValueError("Must specify either max_bars or max_seconds")
        self.max_bars = max_bars
        self.max_seconds = max_seconds
        self.bar_count = 0

    def check(
        self,
        current_price: float,
        entry_price: float,
        position: Position
    ) -> bool:
        """Check if time stop is triggered."""
        # Bar-based stop
        if self.max_bars is not None:
            self.bar_count += 1
            if self.bar_count >= self.max_bars:
                return True

        # Time-based stop
        if self.max_seconds is not None:
            current_time = position.metadata.get('current_time')
            if current_time is None:
                raise ValueError("current_time must be provided in position.metadata['current_time']")

            elapsed = (current_time - position.entry_time).total_seconds()
            if elapsed >= self.max_seconds:
                return True

        return False

    def reset(self):
        """Reset bar count for new position."""
        self.bar_count = 0


class StopLossManager:
    """Manages multiple stop loss strategies per position."""

    def __init__(self):
        """Initialize stop loss manager."""
        self.stops: Dict[str, List[StopLoss]] = {}  # ticker -> list of stops
        self.positions: Dict[str, Position] = {}  # ticker -> position

    def add_stop(self, ticker: str, stop: StopLoss) -> None:
        """
        Add a stop loss strategy for a position.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        stop : StopLoss
            Stop loss strategy to add
        """
        if ticker not in self.stops:
            self.stops[ticker] = []
        self.stops[ticker].append(stop)

    def add_position(self, position: Position) -> None:
        """
        Add a position to track.

        Parameters
        ----------
        position : Position
            Position to track
        """
        self.positions[position.ticker] = position

    def check_stops(self, ticker: str, current_price: float) -> bool:
        """
        Check all stop losses for a position.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        current_price : float
            Current market price

        Returns
        -------
        bool
            True if any stop loss is triggered, False otherwise
        """
        if ticker not in self.stops or ticker not in self.positions:
            return False

        position = self.positions[ticker]

        # Check all stops; if any trigger, return True
        for stop in self.stops[ticker]:
            if stop.check(current_price, position.entry_price, position):
                return True

        return False

    def remove_position(self, ticker: str) -> None:
        """
        Remove a position and its stops.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        """
        if ticker in self.stops:
            # Reset time stops before removal
            for stop in self.stops[ticker]:
                if isinstance(stop, TimeStop):
                    stop.reset()
            del self.stops[ticker]

        if ticker in self.positions:
            del self.positions[ticker]

    def get_stop_prices(self, ticker: str, current_price: float) -> Dict[str, float]:
        """
        Get current stop prices for all stops on a position.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        current_price : float
            Current market price

        Returns
        -------
        dict
            Dictionary of stop type -> stop price
        """
        if ticker not in self.stops or ticker not in self.positions:
            return {}

        position = self.positions[ticker]
        stop_prices = {}

        for i, stop in enumerate(self.stops[ticker]):
            stop_name = f"{stop.__class__.__name__}_{i}"

            if isinstance(stop, FixedStop):
                if stop.price_based:
                    if position.side == 'long':
                        stop_prices[stop_name] = position.entry_price - stop.stop_distance
                    else:
                        stop_prices[stop_name] = position.entry_price + stop.stop_distance
                else:
                    if position.side == 'long':
                        stop_prices[stop_name] = position.entry_price * (1 - stop.stop_distance)
                    else:
                        stop_prices[stop_name] = position.entry_price * (1 + stop.stop_distance)

            elif isinstance(stop, TrailingStop):
                if stop.price_based:
                    if position.side == 'long':
                        stop_prices[stop_name] = position.highest_price - stop.trail_distance
                    else:
                        stop_prices[stop_name] = position.lowest_price + stop.trail_distance
                else:
                    if position.side == 'long':
                        stop_prices[stop_name] = position.highest_price * (1 - stop.trail_distance)
                    else:
                        stop_prices[stop_name] = position.lowest_price * (1 + stop.trail_distance)

            elif isinstance(stop, ATRStop):
                atr = position.metadata.get('atr', 0)
                stop_distance = atr * stop.atr_multiplier
                if stop.trailing:
                    if position.side == 'long':
                        stop_prices[stop_name] = position.highest_price - stop_distance
                    else:
                        stop_prices[stop_name] = position.lowest_price + stop_distance
                else:
                    if position.side == 'long':
                        stop_prices[stop_name] = position.entry_price - stop_distance
                    else:
                        stop_prices[stop_name] = position.entry_price + stop_distance

        return stop_prices
