"""
Pairs trading strategy implementation.

This module provides a complete pairs trading strategy based on cointegration
and mean reversion, including pair selection, spread calculation, signal
generation, and backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from .cointegration import (
    find_cointegrated_pairs,
    engle_granger_test,
    calculate_spread,
    half_life,
    adf_test_spread
)


class PairsTradingStrategy:
    """
    Pairs trading strategy based on cointegration and mean reversion.

    The strategy identifies cointegrated pairs, monitors their spread,
    and generates trading signals when the spread deviates from its mean.

    Parameters
    ----------
    entry_z : float, default 2.0
        Z-score threshold for entering positions
    exit_z : float, default 0.5
        Z-score threshold for exiting positions
    lookback : int, default 20
        Lookback period for calculating spread statistics
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lookback: int = 20
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
        self.pairs_ = None
        self.spreads_ = {}
        self.hedge_ratios_ = {}

    def find_pairs(
        self,
        prices: pd.DataFrame,
        significance: float = 0.05,
        min_half_life: float = 1,
        max_half_life: float = 252
    ) -> List[Tuple[str, str, float, float]]:
        """
        Identify cointegrated pairs from a universe of assets.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (each column is an asset)
        significance : float, default 0.05
            Significance level for cointegration test
        min_half_life : float, default 1
            Minimum acceptable half-life (in periods)
        max_half_life : float, default 252
            Maximum acceptable half-life (in periods)

        Returns
        -------
        list
            List of tuples (ticker1, ticker2, p_value, hedge_ratio)

        Examples
        --------
        >>> strategy = PairsTradingStrategy()
        >>> pairs = strategy.find_pairs(prices, significance=0.05)
        """
        # Find cointegrated pairs
        pairs = find_cointegrated_pairs(prices, significance=significance)

        # Filter by half-life
        filtered_pairs = []
        for ticker1, ticker2, p_value, hedge_ratio in pairs:
            # Calculate spread
            spread = calculate_spread(
                prices[ticker1],
                prices[ticker2],
                hedge_ratio=hedge_ratio
            )

            # Calculate half-life
            try:
                hl = half_life(spread)

                # Filter by half-life range
                if min_half_life <= hl <= max_half_life:
                    filtered_pairs.append((ticker1, ticker2, p_value, hedge_ratio))

            except Exception:
                # Skip pairs with problematic spreads
                continue

        self.pairs_ = filtered_pairs
        return filtered_pairs

    def compute_spread(
        self,
        pair: Tuple[str, str],
        prices: pd.DataFrame,
        hedge_ratio: Optional[float] = None
    ) -> pd.Series:
        """
        Compute the spread for a given pair.

        Parameters
        ----------
        pair : tuple
            Tuple of (ticker1, ticker2)
        prices : pd.DataFrame
            Price data
        hedge_ratio : float, optional
            Hedge ratio. If None, estimated from data.

        Returns
        -------
        pd.Series
            Spread time series

        Examples
        --------
        >>> strategy = PairsTradingStrategy()
        >>> spread = strategy.compute_spread(('AAPL', 'MSFT'), prices)
        """
        ticker1, ticker2 = pair

        if ticker1 not in prices.columns or ticker2 not in prices.columns:
            raise ValueError(f"Tickers not found in prices DataFrame")

        spread = calculate_spread(
            prices[ticker1],
            prices[ticker2],
            hedge_ratio=hedge_ratio
        )

        # Store spread and hedge ratio
        self.spreads_[pair] = spread
        if hedge_ratio is not None:
            self.hedge_ratios_[pair] = hedge_ratio

        return spread

    def generate_signals(
        self,
        spread: pd.Series,
        entry_z: Optional[float] = None,
        exit_z: Optional[float] = None,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        Generate trading signals based on spread z-scores.

        Signals:
        - 1: Long the spread (buy ticker1, sell ticker2)
        - -1: Short the spread (sell ticker1, buy ticker2)
        - 0: No position

        Parameters
        ----------
        spread : pd.Series
            Spread time series
        entry_z : float, optional
            Entry z-score threshold
        exit_z : float, optional
            Exit z-score threshold
        lookback : int, optional
            Lookback period for statistics

        Returns
        -------
        pd.Series
            Trading signals

        Examples
        --------
        >>> strategy = PairsTradingStrategy(entry_z=2.0, exit_z=0.5)
        >>> signals = strategy.generate_signals(spread)
        """
        entry_z = entry_z if entry_z is not None else self.entry_z
        exit_z = exit_z if exit_z is not None else self.exit_z
        lookback = lookback if lookback is not None else self.lookback

        # Calculate rolling mean and std
        spread_mean = spread.rolling(window=lookback).mean()
        spread_std = spread.rolling(window=lookback).std()

        # Calculate z-scores
        z_scores = (spread - spread_mean) / spread_std

        # Generate signals
        signals = pd.Series(0, index=spread.index)
        position = 0  # Current position

        for i in range(len(z_scores)):
            z = z_scores.iloc[i]

            if pd.isna(z):
                signals.iloc[i] = position
                continue

            # Entry logic
            if position == 0:
                if z > entry_z:
                    position = -1  # Short the spread (spread too high)
                elif z < -entry_z:
                    position = 1  # Long the spread (spread too low)

            # Exit logic
            elif position == 1:
                if z > -exit_z:
                    position = 0  # Exit long position

            elif position == -1:
                if z < exit_z:
                    position = 0  # Exit short position

            signals.iloc[i] = position

        return signals

    def backtest_pair(
        self,
        pair: Tuple[str, str],
        prices: pd.DataFrame,
        hedge_ratio: Optional[float] = None,
        transaction_cost: float = 0.001
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Backtest the pairs trading strategy for a single pair.

        Parameters
        ----------
        pair : tuple
            Tuple of (ticker1, ticker2)
        prices : pd.DataFrame
            Price data
        hedge_ratio : float, optional
            Hedge ratio. If None, estimated from in-sample data.
        transaction_cost : float, default 0.001
            Transaction cost as fraction of trade value

        Returns
        -------
        dict
            Dictionary with performance statistics:
            - 'returns': Daily returns series
            - 'cumulative_returns': Cumulative returns series
            - 'sharpe_ratio': Sharpe ratio
            - 'total_return': Total return
            - 'max_drawdown': Maximum drawdown
            - 'num_trades': Number of trades
            - 'win_rate': Percentage of profitable trades

        Examples
        --------
        >>> strategy = PairsTradingStrategy()
        >>> results = strategy.backtest_pair(('AAPL', 'MSFT'), prices)
        >>> print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        """
        ticker1, ticker2 = pair

        # Get price series
        price1 = prices[ticker1]
        price2 = prices[ticker2]

        # Align prices
        pair_prices = pd.DataFrame({
            'price1': price1,
            'price2': price2
        }).dropna()

        if len(pair_prices) < 100:
            raise ValueError("Need at least 100 observations for backtesting")

        # Split into in-sample (first 60%) and out-of-sample (last 40%)
        split_idx = int(len(pair_prices) * 0.6)
        in_sample = pair_prices.iloc[:split_idx]
        out_of_sample = pair_prices.iloc[split_idx:]

        # Estimate hedge ratio from in-sample data if not provided
        if hedge_ratio is None:
            test_result = engle_granger_test(
                in_sample['price1'],
                in_sample['price2']
            )
            hedge_ratio = test_result['hedge_ratio']

        # Calculate spread on out-of-sample data
        spread = calculate_spread(
            out_of_sample['price1'],
            out_of_sample['price2'],
            hedge_ratio=hedge_ratio
        )

        # Generate signals
        signals = self.generate_signals(spread)

        # Calculate returns
        # Return for spread = (price1[t+1] - price1[t]) - hedge_ratio * (price2[t+1] - price2[t])
        price1_returns = out_of_sample['price1'].pct_change()
        price2_returns = out_of_sample['price2'].pct_change()

        # Portfolio returns (assuming unit investment)
        portfolio_returns = signals.shift(1) * (
            price1_returns - hedge_ratio * price2_returns
        )

        # Apply transaction costs
        position_changes = signals.diff().abs()
        costs = position_changes * transaction_cost
        portfolio_returns = portfolio_returns - costs

        # Calculate performance metrics
        portfolio_returns = portfolio_returns.dropna()

        if len(portfolio_returns) == 0:
            return {
                'returns': pd.Series(),
                'cumulative_returns': pd.Series(),
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
                'win_rate': 0.0
            }

        cumulative_returns = (1 + portfolio_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        # Sharpe ratio (annualized, assuming 252 trading days)
        sharpe_ratio = (
            portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            if portfolio_returns.std() > 0 else 0.0
        )

        # Maximum drawdown
        cum_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - cum_max) / cum_max
        max_drawdown = drawdown.min()

        # Trade statistics
        num_trades = (signals.diff() != 0).sum()
        winning_trades = (portfolio_returns > 0).sum()
        win_rate = winning_trades / len(portfolio_returns) if len(portfolio_returns) > 0 else 0.0

        return {
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate
        }

    def backtest_portfolio(
        self,
        pairs: List[Tuple[str, str]],
        prices: pd.DataFrame,
        transaction_cost: float = 0.001
    ) -> Dict[str, Union[float, pd.Series, pd.DataFrame]]:
        """
        Backtest a portfolio of pairs.

        Parameters
        ----------
        pairs : list
            List of pairs to trade
        prices : pd.DataFrame
            Price data
        transaction_cost : float, default 0.001
            Transaction cost

        Returns
        -------
        dict
            Portfolio performance statistics

        Examples
        --------
        >>> strategy = PairsTradingStrategy()
        >>> pairs = [('AAPL', 'MSFT'), ('GOOGL', 'META')]
        >>> results = strategy.backtest_portfolio(pairs, prices)
        """
        all_returns = []
        pair_results = {}

        for pair in pairs:
            try:
                result = self.backtest_pair(
                    pair,
                    prices,
                    transaction_cost=transaction_cost
                )
                all_returns.append(result['returns'])
                pair_results[f"{pair[0]}-{pair[1]}"] = result

            except Exception as e:
                print(f"Failed to backtest pair {pair}: {str(e)}")
                continue

        if not all_returns:
            return {
                'returns': pd.Series(),
                'cumulative_returns': pd.Series(),
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'pair_results': {}
            }

        # Combine returns (equal weight)
        combined_returns = pd.concat(all_returns, axis=1).mean(axis=1)
        cumulative_returns = (1 + combined_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        # Sharpe ratio
        sharpe_ratio = (
            combined_returns.mean() / combined_returns.std() * np.sqrt(252)
            if combined_returns.std() > 0 else 0.0
        )

        # Maximum drawdown
        cum_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - cum_max) / cum_max
        max_drawdown = drawdown.min()

        return {
            'returns': combined_returns,
            'cumulative_returns': cumulative_returns,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'pair_results': pair_results
        }


def rank_pairs_by_performance(
    pairs: List[Tuple[str, str]],
    prices: pd.DataFrame,
    metric: str = 'sharpe_ratio'
) -> pd.DataFrame:
    """
    Rank pairs by backtest performance.

    Parameters
    ----------
    pairs : list
        List of pairs to evaluate
    prices : pd.DataFrame
        Price data
    metric : str, default 'sharpe_ratio'
        Metric to rank by: 'sharpe_ratio', 'total_return', 'win_rate'

    Returns
    -------
    pd.DataFrame
        DataFrame with pair performance metrics, sorted by specified metric

    Examples
    --------
    >>> pairs = [('AAPL', 'MSFT'), ('GOOGL', 'META')]
    >>> rankings = rank_pairs_by_performance(pairs, prices, metric='sharpe_ratio')
    """
    strategy = PairsTradingStrategy()
    results = []

    for pair in pairs:
        try:
            result = strategy.backtest_pair(pair, prices)
            results.append({
                'pair': f"{pair[0]}-{pair[1]}",
                'ticker1': pair[0],
                'ticker2': pair[1],
                'sharpe_ratio': result['sharpe_ratio'],
                'total_return': result['total_return'],
                'max_drawdown': result['max_drawdown'],
                'num_trades': result['num_trades'],
                'win_rate': result['win_rate']
            })

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=metric, ascending=False)

    return results_df
