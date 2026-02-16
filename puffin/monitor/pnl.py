"""Profit and Loss (P&L) tracking and attribution."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """Position information for P&L tracking."""
    ticker: str
    quantity: float
    avg_price: float
    current_price: float
    strategy: str = 'default'
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Calculate cost basis."""
        return self.quantity * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        return self.market_value - self.cost_basis


class PnLTracker:
    """Track profit and loss over time with attribution."""

    def __init__(self, initial_cash: float = 0.0):
        """
        Initialize P&L tracker.

        Parameters
        ----------
        initial_cash : float, default=0.0
            Initial cash balance
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.history: List[Dict[str, Any]] = []
        self.realized_pnl = 0.0

    def update(self, positions: Dict[str, Position], prices: Dict[str, float]) -> None:
        """
        Update positions and prices, recording P&L snapshot.

        Parameters
        ----------
        positions : dict
            Dictionary of ticker -> Position
        prices : dict
            Dictionary of ticker -> current price
        """
        # Update positions
        self.positions = positions.copy()

        # Update prices
        for ticker, position in self.positions.items():
            if ticker in prices:
                position.current_price = prices[ticker]

        # Record snapshot
        snapshot = {
            'timestamp': datetime.now(),
            'equity': self.equity(),
            'cash': self.cash,
            'positions_value': self.positions_value(),
            'unrealized_pnl': self.unrealized_pnl(),
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl(),
        }
        self.history.append(snapshot)

    def equity(self) -> float:
        """Calculate total equity (cash + positions value)."""
        return self.cash + self.positions_value()

    def positions_value(self) -> float:
        """Calculate total value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())

    def unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl()

    def record_trade(
        self,
        ticker: str,
        quantity: float,
        price: float,
        side: str,
        commission: float = 0.0
    ) -> None:
        """
        Record a trade and update P&L.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        quantity : float
            Quantity traded (positive)
        price : float
            Execution price
        side : str
            'buy' or 'sell'
        commission : float, default=0.0
            Commission paid
        """
        trade_value = quantity * price + commission

        if side == 'buy':
            # Update cash
            self.cash -= trade_value

            # Update or create position
            if ticker in self.positions:
                pos = self.positions[ticker]
                # Update average price
                total_qty = pos.quantity + quantity
                total_cost = pos.cost_basis + (quantity * price)
                pos.avg_price = total_cost / total_qty if total_qty > 0 else price
                pos.quantity = total_qty
            else:
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price
                )

        elif side == 'sell':
            if ticker not in self.positions:
                raise ValueError(f"Cannot sell {ticker}: no position exists")

            pos = self.positions[ticker]
            if quantity > pos.quantity:
                raise ValueError(f"Cannot sell {quantity} of {ticker}: only {pos.quantity} available")

            # Calculate realized P&L
            cost_basis = pos.avg_price * quantity
            proceeds = price * quantity
            realized = proceeds - cost_basis - commission
            self.realized_pnl += realized

            # Update cash
            self.cash += (proceeds - commission)

            # Update position
            pos.quantity -= quantity
            if pos.quantity == 0:
                del self.positions[ticker]

    def daily_pnl(self) -> pd.Series:
        """
        Calculate daily P&L from history.

        Returns
        -------
        pd.Series
            Daily P&L time series
        """
        if len(self.history) == 0:
            return pd.Series(dtype=float)

        df = pd.DataFrame(self.history)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily = df.groupby('date')['total_pnl'].last()

        # Calculate daily change
        daily_change = daily.diff()
        daily_change.iloc[0] = daily.iloc[0]  # First day is absolute P&L

        return daily_change

    def cumulative_pnl(self) -> pd.Series:
        """
        Calculate cumulative P&L from history.

        Returns
        -------
        pd.Series
            Cumulative P&L time series
        """
        if len(self.history) == 0:
            return pd.Series(dtype=float)

        df = pd.DataFrame(self.history)
        df.set_index('timestamp', inplace=True)
        return df['total_pnl']

    def attribution_by_strategy(self) -> pd.DataFrame:
        """
        Calculate P&L attribution by strategy.

        Returns
        -------
        pd.DataFrame
            P&L attributed to each strategy
        """
        if len(self.positions) == 0:
            return pd.DataFrame(columns=['strategy', 'unrealized_pnl', 'market_value'])

        attribution = []
        for ticker, pos in self.positions.items():
            attribution.append({
                'ticker': ticker,
                'strategy': pos.strategy,
                'unrealized_pnl': pos.unrealized_pnl,
                'market_value': pos.market_value
            })

        df = pd.DataFrame(attribution)
        strategy_attribution = df.groupby('strategy').agg({
            'unrealized_pnl': 'sum',
            'market_value': 'sum'
        }).reset_index()

        return strategy_attribution

    def attribution_by_asset(self) -> pd.DataFrame:
        """
        Calculate P&L attribution by asset.

        Returns
        -------
        pd.DataFrame
            P&L attributed to each asset
        """
        if len(self.positions) == 0:
            return pd.DataFrame(columns=['ticker', 'unrealized_pnl', 'market_value', 'return_pct'])

        attribution = []
        for ticker, pos in self.positions.items():
            return_pct = (pos.unrealized_pnl / pos.cost_basis * 100) if pos.cost_basis != 0 else 0.0
            attribution.append({
                'ticker': ticker,
                'strategy': pos.strategy,
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
                'cost_basis': pos.cost_basis,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'return_pct': return_pct
            })

        df = pd.DataFrame(attribution)
        df = df.sort_values('unrealized_pnl', ascending=False)

        return df

    def performance_summary(self) -> Dict[str, float]:
        """
        Generate performance summary statistics.

        Returns
        -------
        dict
            Summary statistics including:
            - total_return: Total return percentage
            - total_pnl: Total P&L
            - realized_pnl: Realized P&L
            - unrealized_pnl: Unrealized P&L
            - current_equity: Current equity
            - num_positions: Number of open positions
        """
        current_equity = self.equity()
        total_return = ((current_equity - self.initial_cash) / self.initial_cash * 100) if self.initial_cash > 0 else 0.0

        return {
            'initial_cash': self.initial_cash,
            'current_equity': current_equity,
            'cash': self.cash,
            'positions_value': self.positions_value(),
            'total_pnl': self.total_pnl(),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl(),
            'total_return': total_return,
            'num_positions': len(self.positions)
        }
