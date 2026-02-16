#!/usr/bin/env python
"""
Example usage of Bayesian ML models for trading.

This script demonstrates:
1. Bayesian Sharpe ratio estimation
2. Strategy comparison
3. Dynamic hedge ratios for pairs trading
4. Stochastic volatility modeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check if PyMC is available
try:
    import pymc as pm
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("PyMC not installed. Install with: pip install pymc arviz")
    print("This example requires PyMC to run.")
    exit(1)

from puffin.models.bayesian import (
    BayesianLinearRegression,
    bayesian_sharpe,
    compare_strategies_bayesian,
    BayesianPairsTrading
)
from puffin.models.stochastic_vol import StochasticVolatilityModel, estimate_volatility_regime


def example_1_bayesian_sharpe():
    """Example 1: Bayesian Sharpe ratio estimation."""
    print("\n" + "="*60)
    print("Example 1: Bayesian Sharpe Ratio Estimation")
    print("="*60)

    # Generate synthetic returns
    np.random.seed(42)
    n = 252  # One year of daily returns

    # Strategy with positive expected return
    returns = np.random.randn(n) * 0.015 + 0.001

    # Estimate Sharpe ratio
    sharpe_stats = bayesian_sharpe(returns, samples=2000)

    print(f"\nPosterior Mean Sharpe Ratio: {sharpe_stats['mean']:.2f}")
    print(f"Posterior Std: {sharpe_stats['std']:.2f}")
    print(f"94% Credible Interval: [{sharpe_stats['hdi_low']:.2f}, {sharpe_stats['hdi_high']:.2f}]")
    print(f"P(Sharpe > 0): {sharpe_stats['prob_positive']:.1%}")

    # Traditional estimate for comparison
    traditional_sharpe = returns.mean() / returns.std() * np.sqrt(252)
    print(f"\nTraditional Sharpe (point estimate): {traditional_sharpe:.2f}")
    print("\nNote: Bayesian approach provides uncertainty quantification!")


def example_2_strategy_comparison():
    """Example 2: Compare multiple strategies."""
    print("\n" + "="*60)
    print("Example 2: Strategy Comparison")
    print("="*60)

    np.random.seed(42)
    n = 252

    # Create three strategies with different characteristics
    strategies = {
        'Momentum': np.random.randn(n) * 0.012 + 0.001,
        'Mean Reversion': np.random.randn(n) * 0.010 + 0.0008,
        'ML Strategy': np.random.randn(n) * 0.015 + 0.0012,
    }

    # Compare strategies
    print("\nComparing strategies (this may take a minute)...")
    results = compare_strategies_bayesian(strategies, samples=2000)

    print("\nStrategy Rankings:")
    print(results.to_string(index=False))


def example_3_pairs_trading():
    """Example 3: Bayesian pairs trading."""
    print("\n" + "="*60)
    print("Example 3: Bayesian Pairs Trading")
    print("="*60)

    np.random.seed(42)
    n = 252

    # Create synthetic cointegrated pair
    x = np.cumsum(np.random.randn(n) * 0.5) + 100
    y = 1.5 * x + np.cumsum(np.random.randn(n) * 0.3) + 50

    # Add dates for better visualization
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    y_series = pd.Series(y, index=dates)
    x_series = pd.Series(x, index=dates)

    print("\nFitting dynamic hedge ratios...")
    pairs = BayesianPairsTrading()
    hedge_df = pairs.fit_dynamic_hedge(y_series, x_series, window=60)

    print(f"\nMean hedge ratio: {hedge_df['hedge_ratio_mean'].mean():.3f}")
    print(f"Hedge ratio std: {hedge_df['hedge_ratio_std'].mean():.3f}")

    # Generate signals
    signals = pairs.generate_signals(entry_threshold=2.0, exit_threshold=0.5)
    print(f"\nSignal distribution:")
    print(signals.value_counts())

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot prices
    axes[0].plot(dates, y, label='Stock Y', alpha=0.7)
    axes[0].plot(dates, x, label='Stock X', alpha=0.7)
    axes[0].set_title('Synthetic Pair Prices')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot hedge ratio
    axes[1].plot(hedge_df.index, hedge_df['hedge_ratio_mean'], label='Hedge Ratio')
    axes[1].fill_between(
        hedge_df.index,
        hedge_df['hedge_ratio_mean'] - 2 * hedge_df['hedge_ratio_std'],
        hedge_df['hedge_ratio_mean'] + 2 * hedge_df['hedge_ratio_std'],
        alpha=0.3,
        label='95% Credible Interval'
    )
    axes[1].set_title('Dynamic Hedge Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot spread
    axes[2].plot(hedge_df.index, hedge_df['spread'], label='Spread')
    axes[2].axhline(y=hedge_df['spread'].mean(), color='r', linestyle='--', alpha=0.5)
    axes[2].set_title('Spread (Y - β*X)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/pairs_trading_example.png', dpi=100, bbox_inches='tight')
    print("\nPlot saved to /tmp/pairs_trading_example.png")


def example_4_stochastic_volatility():
    """Example 4: Stochastic volatility estimation."""
    print("\n" + "="*60)
    print("Example 4: Stochastic Volatility Model")
    print("="*60)

    np.random.seed(42)
    n = 200

    # Generate returns with time-varying volatility (GARCH-like)
    vol = np.ones(n) * 0.01
    returns = np.zeros(n)

    for t in range(1, n):
        vol[t] = 0.005 + 0.1 * returns[t-1]**2 + 0.85 * vol[t-1]
        returns[t] = vol[t] * np.random.randn()

    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    returns_series = pd.Series(returns, index=dates)

    print("\nFitting stochastic volatility model (this may take 1-2 minutes)...")
    sv_model = StochasticVolatilityModel()
    sv_model.fit(returns_series, samples=1000, tune=500)

    if sv_model.volatility_path is not None:
        print(f"\nCurrent volatility: {sv_model.volatility_path[-1]:.4f}")
        print(f"Volatility forecast: {sv_model.volatility_forecast:.4f}")

        # Visualize
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot returns
        axes[0].plot(dates, returns, label='Returns', alpha=0.7)
        axes[0].set_title('Returns')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot volatility
        axes[1].plot(dates, sv_model.volatility_path, label='Estimated Volatility', linewidth=2)
        axes[1].plot(dates, vol, label='True Volatility', linestyle='--', alpha=0.7)
        axes[1].fill_between(dates, 0, sv_model.volatility_path, alpha=0.3)
        axes[1].set_title('Stochastic Volatility')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/stochastic_vol_example.png', dpi=100, bbox_inches='tight')
        print("Plot saved to /tmp/stochastic_vol_example.png")


def example_5_linear_regression():
    """Example 5: Bayesian linear regression."""
    print("\n" + "="*60)
    print("Example 5: Bayesian Linear Regression")
    print("="*60)

    np.random.seed(42)
    n = 150

    # Generate data with known relationship
    X = np.random.randn(n, 2)
    true_coef = np.array([3.0, -1.5])
    y = 2.0 + X @ true_coef + np.random.randn(n) * 0.5

    # Split into train/test
    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("\nFitting Bayesian linear regression...")
    model = BayesianLinearRegression()
    model.fit(X_train, y_train, samples=1000, tune=500)

    # Predictions
    mean_pred, (lower, upper) = model.predict(X_test)

    print(f"\nTrue coefficients: {true_coef}")
    print("\nPosterior statistics:")
    summary = model.summary()
    if isinstance(summary, dict):
        for param in ['alpha', 'beta']:
            if param in summary:
                print(f"{param}: {summary[param]}")

    # Compare predictions
    print(f"\nTest set predictions (first 5):")
    print(f"True values: {y_test[:5]}")
    print(f"Predicted:   {mean_pred[:5]}")
    print(f"95% CI lower: {lower[:5]}")
    print(f"95% CI upper: {upper[:5]}")


def main():
    """Run all examples."""
    print("="*60)
    print("Bayesian ML Examples for Trading")
    print("="*60)

    # Run all examples
    example_1_bayesian_sharpe()
    example_2_strategy_comparison()
    example_3_pairs_trading()
    example_4_stochastic_volatility()
    example_5_linear_regression()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
