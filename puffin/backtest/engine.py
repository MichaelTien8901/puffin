"""Event-driven backtesting engine."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from puffin.strategies.base import Strategy


@dataclass
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    order_type: str = "market"  # "market", "limit", "stop"
    limit_price: float | None = None
    stop_price: float | None = None
    timestamp: datetime | None = None


@dataclass
class Fill:
    symbol: str
    side: str
    qty: int
    price: float
    commission: float
    slippage: float
    timestamp: datetime


@dataclass
class Position:
    symbol: str
    qty: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class SlippageModel:
    """Slippage configuration."""
    fixed: float = 0.0      # Fixed dollar amount per share
    pct: float = 0.0        # Percentage of price

    def calculate(self, price: float) -> float:
        return self.fixed + price * self.pct


@dataclass
class CommissionModel:
    """Commission configuration."""
    flat: float = 0.0       # Flat fee per order
    per_share: float = 0.0  # Per-share fee
    pct: float = 0.0        # Percentage of trade value

    def calculate(self, price: float, qty: int) -> float:
        return self.flat + self.per_share * qty + price * qty * self.pct


class Backtester:
    """Event-driven backtesting engine.

    Processes bars sequentially with no lookahead bias.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
    ):
        self.initial_capital = initial_capital
        self.slippage = slippage or SlippageModel()
        self.commission = commission or CommissionModel()

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        symbols: list[str] | None = None,
    ) -> "BacktestResult":
        """Run a backtest.

        Args:
            strategy: Strategy instance.
            data: OHLCV DataFrame. For multi-asset, expects MultiIndex (Date, Symbol).
            symbols: List of symbols (auto-detected from data if MultiIndex).

        Returns:
            BacktestResult with performance metrics.
        """
        is_multi = isinstance(data.index, pd.MultiIndex)

        if is_multi and symbols is None:
            symbols = data.index.get_level_values("Symbol").unique().tolist()
        elif symbols is None:
            symbols = ["default"]

        cash = self.initial_capital
        positions: dict[str, Position] = {}
        equity_history = []
        fills: list[Fill] = []
        pending_orders: list[Order] = []

        if is_multi:
            dates = data.index.get_level_values("Date" if "Date" in data.index.names else 0).unique().sort_values()
        else:
            dates = data.index.sort_values()

        for i, date in enumerate(dates):
            # Get data up to current bar (no lookahead)
            if is_multi:
                current_bar = data.loc[date] if date in data.index.get_level_values(0) else None
                history = data.loc[:date]
            else:
                current_bar = data.loc[date]
                history = data.loc[:date]

            if current_bar is None:
                continue

            # Execute pending orders against current bar
            new_pending = []
            for order in pending_orders:
                fill = self._try_fill(order, current_bar, is_multi, date)
                if fill:
                    cash, positions = self._apply_fill(fill, cash, positions)
                    fills.append(fill)
                else:
                    new_pending.append(order)
            pending_orders = new_pending

            # Generate signals from strategy
            signals = strategy.generate_signals(history)

            # Create orders from latest signals
            if is_multi:
                for symbol in symbols:
                    if symbol in signals.index.get_level_values(-1):
                        sig_val = float(signals.loc[(date, symbol), "signal"]) if (date, symbol) in signals.index else 0
                    else:
                        sig_val = 0
                    order = self._signal_to_order(symbol, sig_val, positions, current_bar, is_multi, date)
                    if order:
                        pending_orders.append(order)
            else:
                if date in signals.index:
                    sig_val = float(signals.loc[date, "signal"])
                    symbol = symbols[0]
                    order = self._signal_to_order(symbol, sig_val, positions, current_bar, is_multi, date)
                    if order:
                        pending_orders.append(order)

            # Calculate portfolio value
            portfolio_value = cash
            for sym, pos in positions.items():
                if pos.qty != 0:
                    if is_multi:
                        try:
                            price = float(current_bar.loc[sym, "Close"]) if sym in current_bar.index else pos.avg_price
                        except (KeyError, TypeError):
                            price = pos.avg_price
                    else:
                        price = float(current_bar["Close"])
                    portfolio_value += pos.qty * price

            equity_history.append({"date": date, "equity": portfolio_value, "cash": cash})

        equity_df = pd.DataFrame(equity_history).set_index("date")
        return BacktestResult(
            equity_curve=equity_df["equity"],
            fills=fills,
            initial_capital=self.initial_capital,
            final_value=equity_df["equity"].iloc[-1] if len(equity_df) > 0 else self.initial_capital,
        )

    def _signal_to_order(self, symbol, signal, positions, current_bar, is_multi, date):
        if signal == 0:
            return None

        pos = positions.get(symbol, Position(symbol=symbol))

        if signal > 0 and pos.qty <= 0:
            # Determine position size (simple: fixed fraction of portfolio)
            if is_multi:
                try:
                    price = float(current_bar.loc[symbol, "Close"])
                except (KeyError, TypeError):
                    return None
            else:
                price = float(current_bar["Close"])
            qty = max(1, int(self.initial_capital * 0.1 / price))
            return Order(symbol=symbol, side="buy", qty=qty, timestamp=date)
        elif signal < 0 and pos.qty > 0:
            return Order(symbol=symbol, side="sell", qty=pos.qty, timestamp=date)

        return None

    def _try_fill(self, order, bar, is_multi, date):
        if is_multi:
            try:
                row = bar.loc[order.symbol]
            except KeyError:
                return None
        else:
            row = bar

        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])

        if order.order_type == "market":
            fill_price = open_price
        elif order.order_type == "limit":
            if order.side == "buy" and low <= order.limit_price:
                fill_price = min(open_price, order.limit_price)
            elif order.side == "sell" and high >= order.limit_price:
                fill_price = max(open_price, order.limit_price)
            else:
                return None
        elif order.order_type == "stop":
            if order.side == "sell" and low <= order.stop_price:
                fill_price = order.stop_price
            elif order.side == "buy" and high >= order.stop_price:
                fill_price = order.stop_price
            else:
                return None
        else:
            return None

        # Apply slippage
        slip = self.slippage.calculate(fill_price)
        if order.side == "buy":
            fill_price += slip
        else:
            fill_price -= slip

        comm = self.commission.calculate(fill_price, order.qty)

        return Fill(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            commission=comm,
            slippage=slip * order.qty,
            timestamp=date,
        )

    def _apply_fill(self, fill, cash, positions):
        pos = positions.get(fill.symbol, Position(symbol=fill.symbol))

        if fill.side == "buy":
            total_cost = fill.price * fill.qty + fill.commission
            new_qty = pos.qty + fill.qty
            if new_qty > 0:
                pos.avg_price = (pos.avg_price * pos.qty + fill.price * fill.qty) / new_qty
            pos.qty = new_qty
            cash -= total_cost
        else:
            proceeds = fill.price * fill.qty - fill.commission
            pos.realized_pnl += (fill.price - pos.avg_price) * fill.qty
            pos.qty -= fill.qty
            if pos.qty == 0:
                pos.avg_price = 0.0
            cash += proceeds

        positions[fill.symbol] = pos
        return cash, positions


class BacktestResult:
    """Results from a backtest run."""

    def __init__(
        self,
        equity_curve: pd.Series,
        fills: list[Fill],
        initial_capital: float,
        final_value: float,
    ):
        self.equity_curve = equity_curve
        self.fills = fills
        self.initial_capital = initial_capital
        self.final_value = final_value

    def metrics(self, risk_free_rate: float = 0.05) -> dict:
        """Calculate performance metrics."""
        returns = self.equity_curve.pct_change().dropna()

        if len(returns) == 0:
            return {"total_return": 0, "trades": 0}

        total_return = (self.final_value / self.initial_capital) - 1
        ann_return = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * np.sqrt(252)

        excess = returns - risk_free_rate / 252
        sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0

        peak = self.equity_curve.cummax()
        dd = (self.equity_curve - peak) / peak
        max_dd = dd.min()

        # Trade metrics
        trade_pnls = []
        for fill in self.fills:
            if fill.side == "sell":
                trade_pnls.append(fill.price * fill.qty - fill.commission)

        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.fills),
        }

    def plot(self):
        """Plot equity curve and drawdown."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        self.equity_curve.plot(ax=ax1, title="Equity Curve")
        ax1.set_ylabel("Portfolio Value ($)")

        peak = self.equity_curve.cummax()
        dd = (self.equity_curve - peak) / peak
        dd.plot(ax=ax2, title="Drawdown", color="red")
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, color="red")
        ax2.set_ylabel("Drawdown (%)")

        plt.tight_layout()
        plt.show()
