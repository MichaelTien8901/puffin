"""
Portfolio Tearsheet and Performance Analysis

Provides portfolio performance analysis and visualization, with pyfolio-reloaded
integration and graceful fallback to manual computation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def compute_stats(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict:
    """
    Compute comprehensive portfolio statistics.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns (as a time series)
    risk_free_rate : float, optional
        Annualized risk-free rate, by default 0.0
    periods_per_year : int, optional
        Number of periods per year (252 for daily, 12 for monthly), by default 252

    Returns
    -------
    dict
        Dictionary containing:
        - annual_return: annualized return
        - annual_vol: annualized volatility
        - sharpe: Sharpe ratio
        - sortino: Sortino ratio
        - max_dd: maximum drawdown
        - calmar: Calmar ratio
        - skew: return skewness
        - kurtosis: return kurtosis
        - var_95: Value at Risk (95%)
        - cvar_95: Conditional Value at Risk (95%)
    """
    if len(returns) == 0:
        return {}

    # Basic statistics
    mean_return = returns.mean()
    std_return = returns.std()

    # Annualized metrics
    annual_return = (1 + mean_return) ** periods_per_year - 1
    annual_vol = std_return * np.sqrt(periods_per_year)

    # Sharpe ratio
    excess_return = mean_return - (risk_free_rate / periods_per_year)
    sharpe = (excess_return / std_return * np.sqrt(periods_per_year)) if std_return > 0 else 0.0

    # Sortino ratio (using downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    sortino = (excess_return / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar ratio (return / max drawdown)
    calmar = abs(annual_return / max_dd) if max_dd != 0 else 0.0

    # Higher moments
    skew = returns.skew()
    kurtosis = returns.kurtosis()

    # Value at Risk (VaR) and Conditional VaR (CVaR)
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    # Average win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    return {
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'skew': skew,
        'kurtosis': kurtosis,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }


def generate_tearsheet(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict:
    """
    Generate a comprehensive tearsheet with portfolio performance statistics.

    Attempts to use pyfolio-reloaded if available, otherwise falls back to
    manual computation.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    benchmark : pd.Series, optional
        Benchmark returns for comparison
    risk_free_rate : float, optional
        Annualized risk-free rate, by default 0.0
    periods_per_year : int, optional
        Number of periods per year, by default 252

    Returns
    -------
    dict
        Dictionary containing portfolio statistics and benchmark comparison
    """
    stats = compute_stats(returns, risk_free_rate, periods_per_year)

    result = {
        'portfolio': stats
    }

    # Compute benchmark statistics if provided
    if benchmark is not None:
        # Align returns and benchmark
        aligned_returns, aligned_benchmark = returns.align(benchmark, join='inner')

        if len(aligned_returns) > 0:
            benchmark_stats = compute_stats(aligned_benchmark, risk_free_rate, periods_per_year)
            result['benchmark'] = benchmark_stats

            # Compute relative statistics
            excess_returns = aligned_returns - aligned_benchmark
            result['excess'] = compute_stats(excess_returns, 0.0, periods_per_year)

            # Information ratio
            tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
            information_ratio = (excess_returns.mean() * periods_per_year / tracking_error) if tracking_error > 0 else 0.0
            result['information_ratio'] = information_ratio

            # Beta and alpha
            cov = aligned_returns.cov(aligned_benchmark)
            benchmark_var = aligned_benchmark.var()
            beta = cov / benchmark_var if benchmark_var > 0 else 0.0

            alpha = (aligned_returns.mean() - risk_free_rate / periods_per_year -
                    beta * (aligned_benchmark.mean() - risk_free_rate / periods_per_year)) * periods_per_year

            result['beta'] = beta
            result['alpha'] = alpha

    # Try to use pyfolio if available
    try:
        import pyfolio as pf

        # Create pyfolio tearsheet data
        pyfolio_stats = pf.timeseries.perf_stats(returns)
        result['pyfolio_stats'] = pyfolio_stats.to_dict()

    except ImportError:
        result['pyfolio_available'] = False

    return result


def plot_returns(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    figsize: tuple = (12, 6)
) -> Figure:
    """
    Plot cumulative returns with optional benchmark comparison.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    benchmark : pd.Series, optional
        Benchmark returns
    figsize : tuple, optional
        Figure size, by default (12, 6)

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Cumulative returns
    cumulative = (1 + returns).cumprod()
    cumulative.plot(ax=ax, label='Portfolio', linewidth=2)

    if benchmark is not None:
        aligned_returns, aligned_benchmark = returns.align(benchmark, join='inner')
        if len(aligned_benchmark) > 0:
            benchmark_cumulative = (1 + aligned_benchmark).cumprod()
            benchmark_cumulative.plot(ax=ax, label='Benchmark', linewidth=2, alpha=0.7)

    ax.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_drawdown(
    returns: pd.Series,
    figsize: tuple = (12, 6)
) -> Figure:
    """
    Plot drawdown over time.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    figsize : tuple, optional
        Figure size, by default (12, 6)

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    # Plot
    drawdown.plot(ax=ax, color='red', linewidth=2)
    ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')

    ax.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.grid(True, alpha=0.3)

    # Add max drawdown annotation
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax.annotate(f'Max DD: {max_dd_val:.2%}',
                xy=(max_dd_idx, max_dd_val),
                xytext=(10, -30),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    return fig


def plot_monthly_returns(
    returns: pd.Series,
    figsize: tuple = (12, 8)
) -> Figure:
    """
    Plot monthly returns heatmap.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns (daily)
    figsize : tuple, optional
        Figure size, by default (12, 8)

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Resample to monthly
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

    # Create pivot table
    monthly_returns_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })

    pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticklabels(pivot.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.1%}',
                             ha='center', va='center', color='black', fontsize=8)

    ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    figsize: tuple = (12, 10)
) -> Figure:
    """
    Plot rolling performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    window : int, optional
        Rolling window size, by default 252 (1 year for daily data)
    figsize : tuple, optional
        Figure size, by default (12, 10)

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Rolling return
    rolling_return = returns.rolling(window).mean() * 252
    rolling_return.plot(ax=axes[0], linewidth=2)
    axes[0].set_title('Rolling Annual Return', fontweight='bold')
    axes[0].set_ylabel('Return')
    axes[0].grid(True, alpha=0.3)

    # Rolling volatility
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_vol.plot(ax=axes[1], linewidth=2, color='orange')
    axes[1].set_title('Rolling Annual Volatility', fontweight='bold')
    axes[1].set_ylabel('Volatility')
    axes[1].grid(True, alpha=0.3)

    # Rolling Sharpe ratio
    rolling_sharpe = rolling_return / rolling_vol
    rolling_sharpe.plot(ax=axes[2], linewidth=2, color='green')
    axes[2].set_title('Rolling Sharpe Ratio', fontweight='bold')
    axes[2].set_ylabel('Sharpe Ratio')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_tearsheet_summary(tearsheet: Dict) -> None:
    """
    Print a formatted summary of tearsheet statistics.

    Parameters
    ----------
    tearsheet : dict
        Tearsheet dictionary from generate_tearsheet
    """
    portfolio_stats = tearsheet.get('portfolio', {})

    print("=" * 60)
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 60)
    print()

    print("Returns & Risk:")
    print(f"  Annual Return:       {portfolio_stats.get('annual_return', 0):.2%}")
    print(f"  Annual Volatility:   {portfolio_stats.get('annual_vol', 0):.2%}")
    print(f"  Sharpe Ratio:        {portfolio_stats.get('sharpe', 0):.3f}")
    print(f"  Sortino Ratio:       {portfolio_stats.get('sortino', 0):.3f}")
    print()

    print("Drawdown:")
    print(f"  Maximum Drawdown:    {portfolio_stats.get('max_dd', 0):.2%}")
    print(f"  Calmar Ratio:        {portfolio_stats.get('calmar', 0):.3f}")
    print()

    print("Distribution:")
    print(f"  Skewness:            {portfolio_stats.get('skew', 0):.3f}")
    print(f"  Kurtosis:            {portfolio_stats.get('kurtosis', 0):.3f}")
    print(f"  VaR (95%):           {portfolio_stats.get('var_95', 0):.2%}")
    print(f"  CVaR (95%):          {portfolio_stats.get('cvar_95', 0):.2%}")
    print()

    print("Win/Loss:")
    print(f"  Win Rate:            {portfolio_stats.get('win_rate', 0):.2%}")
    print(f"  Average Win:         {portfolio_stats.get('avg_win', 0):.2%}")
    print(f"  Average Loss:        {portfolio_stats.get('avg_loss', 0):.2%}")

    # Benchmark comparison
    if 'benchmark' in tearsheet:
        print()
        print("=" * 60)
        print("BENCHMARK COMPARISON")
        print("=" * 60)
        print()
        print(f"  Information Ratio:   {tearsheet.get('information_ratio', 0):.3f}")
        print(f"  Beta:                {tearsheet.get('beta', 0):.3f}")
        print(f"  Alpha:               {tearsheet.get('alpha', 0):.2%}")

    print()
    print("=" * 60)
