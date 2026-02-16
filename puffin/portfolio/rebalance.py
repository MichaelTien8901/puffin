"""
Portfolio Rebalancing

Implements portfolio rebalancing strategies and transaction cost modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Trade:
    """
    Represents a single trade.

    Attributes
    ----------
    symbol : str
        Asset symbol
    quantity : float
        Number of shares/units to trade (positive for buy, negative for sell)
    price : float
        Execution price
    value : float
        Trade value (quantity * price)
    timestamp : datetime
        Trade timestamp
    """
    symbol: str
    quantity: float
    price: float
    value: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CostModel:
    """
    Transaction cost model.

    Attributes
    ----------
    commission_pct : float
        Commission as a percentage of trade value
    commission_fixed : float
        Fixed commission per trade
    slippage_pct : float
        Slippage as a percentage of trade value
    min_commission : float
        Minimum commission per trade
    """
    commission_pct: float = 0.001  # 0.1%
    commission_fixed: float = 0.0
    slippage_pct: float = 0.0005  # 0.05%
    min_commission: float = 1.0

    def calculate_cost(self, trade_value: float) -> float:
        """
        Calculate total transaction cost for a trade.

        Parameters
        ----------
        trade_value : float
            Absolute value of the trade

        Returns
        -------
        float
            Total transaction cost
        """
        # Commission
        commission = max(
            self.commission_pct * trade_value + self.commission_fixed,
            self.min_commission
        )

        # Slippage
        slippage = self.slippage_pct * trade_value

        return commission + slippage


class RebalanceEngine:
    """
    Portfolio rebalancing engine.

    Handles computation of trades needed to rebalance a portfolio from current
    weights to target weights, with transaction cost considerations.
    """

    def __init__(self, cost_model: Optional[CostModel] = None):
        """
        Initialize rebalance engine.

        Parameters
        ----------
        cost_model : CostModel, optional
            Transaction cost model. If None, uses default CostModel.
        """
        self.cost_model = cost_model if cost_model is not None else CostModel()

    def compute_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        min_trade_value: float = 0.0
    ) -> List[Trade]:
        """
        Compute trades needed to rebalance portfolio.

        Parameters
        ----------
        current_weights : dict
            Current portfolio weights {symbol: weight}
        target_weights : dict
            Target portfolio weights {symbol: weight}
        portfolio_value : float
            Total portfolio value
        prices : dict
            Current prices for each symbol {symbol: price}
        min_trade_value : float, optional
            Minimum trade value to execute (filters out tiny trades), by default 0.0

        Returns
        -------
        list of Trade
            List of trades to execute
        """
        # Get all symbols
        all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))

        trades = []

        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)

            # Weight difference
            weight_diff = target_weight - current_weight

            if abs(weight_diff) < 1e-6:
                continue

            # Value to trade
            trade_value = weight_diff * portfolio_value

            if abs(trade_value) < min_trade_value:
                continue

            # Get price
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue

            # Compute quantity
            quantity = trade_value / price

            # Create trade
            trade = Trade(
                symbol=symbol,
                quantity=quantity,
                price=price,
                value=trade_value
            )
            trades.append(trade)

        return trades

    def apply_transaction_costs(
        self,
        trades: List[Trade]
    ) -> Dict:
        """
        Calculate transaction costs for a list of trades.

        Parameters
        ----------
        trades : list of Trade
            List of trades

        Returns
        -------
        dict
            Dictionary containing:
            - trades: list of trades with costs
            - total_cost: total transaction cost
            - cost_by_symbol: dict of costs per symbol
        """
        total_cost = 0.0
        cost_by_symbol = {}

        for trade in trades:
            trade_cost = self.cost_model.calculate_cost(abs(trade.value))
            total_cost += trade_cost
            cost_by_symbol[trade.symbol] = trade_cost

        return {
            'trades': trades,
            'total_cost': total_cost,
            'cost_by_symbol': cost_by_symbol
        }

    def optimize_with_costs(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        cost_threshold: float = 0.001
    ) -> Dict:
        """
        Optimize rebalancing considering transaction costs.

        Only rebalance if the benefit exceeds the cost by a threshold.

        Parameters
        ----------
        current_weights : dict
            Current portfolio weights
        target_weights : dict
            Target portfolio weights
        portfolio_value : float
            Total portfolio value
        prices : dict
            Current prices
        cost_threshold : float, optional
            Minimum benefit-to-cost ratio to execute rebalance, by default 0.001

        Returns
        -------
        dict
            Dictionary containing:
            - should_rebalance: bool
            - trades: list of trades (if should_rebalance)
            - total_cost: total transaction cost
            - expected_benefit: estimated benefit from rebalancing
        """
        # Compute trades
        trades = self.compute_trades(
            current_weights,
            target_weights,
            portfolio_value,
            prices
        )

        # Calculate costs
        cost_info = self.apply_transaction_costs(trades)
        total_cost = cost_info['total_cost']

        # Estimate benefit (simplified: tracking error reduction)
        tracking_error = sum(
            abs(target_weights.get(s, 0) - current_weights.get(s, 0))
            for s in set(list(current_weights.keys()) + list(target_weights.keys()))
        )
        expected_benefit = tracking_error * portfolio_value

        # Decision
        should_rebalance = (expected_benefit - total_cost) / portfolio_value > cost_threshold

        return {
            'should_rebalance': should_rebalance,
            'trades': trades if should_rebalance else [],
            'total_cost': total_cost,
            'expected_benefit': expected_benefit,
            'benefit_cost_ratio': expected_benefit / total_cost if total_cost > 0 else np.inf
        }


def rebalance_schedule(
    strategy: str = 'monthly',
    threshold: float = 0.05,
    **kwargs
) -> Callable[[datetime, Dict, Dict], bool]:
    """
    Create a rebalancing schedule function.

    Parameters
    ----------
    strategy : str, optional
        Rebalancing strategy: 'monthly', 'quarterly', 'annual', or 'threshold'
        Default is 'monthly'
    threshold : float, optional
        Weight deviation threshold for threshold-based rebalancing, by default 0.05
    **kwargs
        Additional strategy-specific parameters

    Returns
    -------
    Callable
        Function that takes (date, current_weights, target_weights) and returns
        bool indicating whether to rebalance
    """
    if strategy == 'monthly':
        last_rebalance = {'date': None}

        def monthly_rebalance(date: datetime, current_weights: Dict, target_weights: Dict) -> bool:
            if last_rebalance['date'] is None:
                last_rebalance['date'] = date
                return True

            # Check if month changed
            if date.month != last_rebalance['date'].month or date.year != last_rebalance['date'].year:
                last_rebalance['date'] = date
                return True

            return False

        return monthly_rebalance

    elif strategy == 'quarterly':
        last_rebalance = {'date': None}

        def quarterly_rebalance(date: datetime, current_weights: Dict, target_weights: Dict) -> bool:
            if last_rebalance['date'] is None:
                last_rebalance['date'] = date
                return True

            # Check if quarter changed
            current_quarter = (date.month - 1) // 3
            last_quarter = (last_rebalance['date'].month - 1) // 3

            if current_quarter != last_quarter or date.year != last_rebalance['date'].year:
                last_rebalance['date'] = date
                return True

            return False

        return quarterly_rebalance

    elif strategy == 'annual':
        last_rebalance = {'date': None}

        def annual_rebalance(date: datetime, current_weights: Dict, target_weights: Dict) -> bool:
            if last_rebalance['date'] is None:
                last_rebalance['date'] = date
                return True

            # Check if year changed
            if date.year != last_rebalance['date'].year:
                last_rebalance['date'] = date
                return True

            return False

        return annual_rebalance

    elif strategy == 'threshold':
        def threshold_rebalance(date: datetime, current_weights: Dict, target_weights: Dict) -> bool:
            # Calculate maximum weight deviation
            all_symbols = set(list(current_weights.keys()) + list(target_weights.keys()))

            max_deviation = max(
                abs(target_weights.get(s, 0) - current_weights.get(s, 0))
                for s in all_symbols
            )

            return max_deviation > threshold

        return threshold_rebalance

    else:
        raise ValueError(f"Unknown rebalancing strategy: {strategy}")


def backtest_rebalancing(
    returns: pd.DataFrame,
    target_weights: Dict[str, float],
    rebalance_fn: Callable,
    initial_value: float = 100000.0,
    cost_model: Optional[CostModel] = None
) -> pd.DataFrame:
    """
    Backtest a rebalancing strategy.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset (rows: dates, columns: symbols)
    target_weights : dict
        Target portfolio weights {symbol: weight}
    rebalance_fn : Callable
        Rebalancing schedule function
    initial_value : float, optional
        Initial portfolio value, by default 100000.0
    cost_model : CostModel, optional
        Transaction cost model

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - portfolio_value: portfolio value over time
        - rebalanced: bool indicating rebalance dates
        - transaction_costs: cumulative transaction costs
    """
    engine = RebalanceEngine(cost_model)

    # Initialize
    dates = returns.index
    portfolio_value = initial_value
    current_weights = target_weights.copy()
    cumulative_costs = 0.0

    results = []

    for i, date in enumerate(dates):
        # Get today's returns
        today_returns = returns.iloc[i]

        # Update portfolio value based on returns
        for symbol in current_weights:
            if symbol in today_returns.index:
                portfolio_value *= (1 + today_returns[symbol] * current_weights[symbol])

        # Update current weights (drift due to returns)
        for symbol in current_weights:
            if symbol in today_returns.index:
                current_weights[symbol] *= (1 + today_returns[symbol])

        # Renormalize weights
        total_weight = sum(current_weights.values())
        if total_weight > 0:
            current_weights = {k: v / total_weight for k, v in current_weights.items()}

        # Check if should rebalance
        should_rebalance = rebalance_fn(date, current_weights, target_weights)

        transaction_cost = 0.0
        if should_rebalance:
            # Compute trades (use closing prices from previous day as approximation)
            prices = {symbol: 1.0 for symbol in returns.columns}  # Simplified
            trades = engine.compute_trades(
                current_weights,
                target_weights,
                portfolio_value,
                prices
            )

            # Apply costs
            cost_info = engine.apply_transaction_costs(trades)
            transaction_cost = cost_info['total_cost']
            cumulative_costs += transaction_cost
            portfolio_value -= transaction_cost

            # Reset to target weights
            current_weights = target_weights.copy()

        # Record results
        results.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'rebalanced': should_rebalance,
            'transaction_costs': cumulative_costs
        })

    return pd.DataFrame(results).set_index('date')


def compare_rebalancing_strategies(
    returns: pd.DataFrame,
    target_weights: Dict[str, float],
    strategies: List[str] = ['monthly', 'quarterly', 'threshold'],
    initial_value: float = 100000.0,
    cost_model: Optional[CostModel] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compare multiple rebalancing strategies.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns
    target_weights : dict
        Target portfolio weights
    strategies : list, optional
        List of strategy names to compare
    initial_value : float, optional
        Initial portfolio value
    cost_model : CostModel, optional
        Transaction cost model

    Returns
    -------
    dict
        Dictionary mapping strategy name to backtest results DataFrame
    """
    results = {}

    for strategy in strategies:
        rebalance_fn = rebalance_schedule(strategy=strategy)
        backtest_df = backtest_rebalancing(
            returns,
            target_weights,
            rebalance_fn,
            initial_value,
            cost_model
        )
        results[strategy] = backtest_df

    return results
