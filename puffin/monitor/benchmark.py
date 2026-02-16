"""Benchmark comparison and performance attribution."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


class BenchmarkComparison:
    """Compare strategy performance against benchmarks."""

    def __init__(self):
        """Initialize benchmark comparison."""
        pass

    def compare(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Compare strategy returns against benchmark.

        Parameters
        ----------
        strategy_returns : pd.Series
            Strategy returns time series
        benchmark_returns : pd.Series
            Benchmark returns time series

        Returns
        -------
        dict
            Performance metrics:
            - alpha: Jensen's alpha (excess return)
            - beta: Market beta (systematic risk)
            - ir: Information ratio (risk-adjusted outperformance)
            - tracking_error: Standard deviation of excess returns

        Examples
        --------
        >>> strategy = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> benchmark = pd.Series([0.005, 0.015, -0.005, 0.02])
        >>> bc = BenchmarkComparison()
        >>> metrics = bc.compare(strategy, benchmark)
        >>> 'alpha' in metrics
        True
        """
        # Align time series
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned) == 0:
            return {
                'alpha': 0.0,
                'beta': 0.0,
                'ir': 0.0,
                'tracking_error': 0.0,
                'correlation': 0.0,
                'outperformance': 0.0
            }

        strategy_ret = aligned['strategy']
        benchmark_ret = aligned['benchmark']

        # Calculate beta using covariance
        covariance = np.cov(strategy_ret, benchmark_ret)[0, 1]
        benchmark_variance = np.var(benchmark_ret)

        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
        else:
            beta = 0.0

        # Calculate alpha (Jensen's alpha)
        mean_strategy = strategy_ret.mean()
        mean_benchmark = benchmark_ret.mean()
        alpha = mean_strategy - beta * mean_benchmark

        # Calculate tracking error (volatility of excess returns)
        excess_returns = strategy_ret - benchmark_ret
        tracking_error = excess_returns.std()

        # Calculate information ratio
        if tracking_error > 0:
            ir = excess_returns.mean() / tracking_error
        else:
            ir = 0.0

        # Calculate correlation
        correlation = strategy_ret.corr(benchmark_ret)

        # Calculate total outperformance
        outperformance = (strategy_ret.sum() - benchmark_ret.sum()) * 100

        return {
            'alpha': alpha,
            'beta': beta,
            'ir': ir,
            'tracking_error': tracking_error,
            'correlation': correlation,
            'outperformance': outperformance
        }

    def plot_comparison(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        figsize: tuple = (12, 8)
    ) -> plt.Figure:
        """
        Plot strategy vs benchmark comparison.

        Parameters
        ----------
        strategy_returns : pd.Series
            Strategy returns time series
        benchmark_returns : pd.Series
            Benchmark returns time series
        figsize : tuple, default=(12, 8)
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
            Comparison plot figure
        """
        # Align time series
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
            return fig

        # Calculate cumulative returns
        cum_strategy = (1 + aligned['strategy']).cumprod() - 1
        cum_benchmark = (1 + aligned['benchmark']).cumprod() - 1

        # Calculate metrics
        metrics = self.compare(aligned['strategy'], aligned['benchmark'])

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Cumulative returns
        ax1 = axes[0, 0]
        ax1.plot(cum_strategy.index, cum_strategy * 100, label='Strategy', linewidth=2)
        ax1.plot(cum_benchmark.index, cum_benchmark * 100, label='Benchmark', linewidth=2)
        ax1.set_title('Cumulative Returns')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Excess returns
        ax2 = axes[0, 1]
        excess_returns = aligned['strategy'] - aligned['benchmark']
        ax2.bar(excess_returns.index, excess_returns * 100, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_title('Excess Returns (Strategy - Benchmark)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Excess Return (%)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Rolling correlation
        ax3 = axes[1, 0]
        rolling_corr = aligned['strategy'].rolling(window=20).corr(aligned['benchmark'])
        ax3.plot(rolling_corr.index, rolling_corr, linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_title('Rolling 20-Period Correlation')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Correlation')
        ax3.set_ylim(-1, 1)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Performance metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrics_text = f"""
Performance Metrics

Alpha:              {metrics['alpha']:.4f}
Beta:               {metrics['beta']:.4f}
Information Ratio:  {metrics['ir']:.4f}
Tracking Error:     {metrics['tracking_error']:.4f}
Correlation:        {metrics['correlation']:.4f}
Outperformance:     {metrics['outperformance']:.2f}%

Strategy Total:     {cum_strategy.iloc[-1] * 100:.2f}%
Benchmark Total:    {cum_benchmark.iloc[-1] * 100:.2f}%
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace')

        plt.tight_layout()
        return fig

    def rolling_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Parameters
        ----------
        strategy_returns : pd.Series
            Strategy returns time series
        benchmark_returns : pd.Series
            Benchmark returns time series
        window : int, default=60
            Rolling window size

        Returns
        -------
        pd.DataFrame
            Rolling metrics over time
        """
        # Align time series
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned) < window:
            return pd.DataFrame()

        # Calculate rolling metrics
        results = []

        for i in range(window, len(aligned) + 1):
            window_data = aligned.iloc[i - window:i]
            metrics = self.compare(window_data['strategy'], window_data['benchmark'])
            metrics['date'] = aligned.index[i - 1]
            results.append(metrics)

        return pd.DataFrame(results).set_index('date')
