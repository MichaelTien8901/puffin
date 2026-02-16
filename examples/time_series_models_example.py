"""
Example demonstrating time series models in Puffin.

This script shows how to use:
- Stationarity testing
- ARIMA models
- GARCH models for volatility
- Cointegration analysis
- Pairs trading strategy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)


def example_stationarity_testing():
    """Example: Testing for stationarity."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Stationarity Testing")
    print("="*70)

    from puffin.models import test_stationarity, check_stationarity

    # Generate stationary series (white noise)
    stationary = pd.Series(np.random.randn(252), name='Returns')

    # Generate non-stationary series (random walk)
    non_stationary = pd.Series(np.random.randn(252).cumsum(), name='Prices')

    print("\nTesting white noise (should be stationary):")
    result = test_stationarity(stationary)
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Is stationary: {result['is_stationary']}")

    print("\nTesting random walk (should be non-stationary):")
    result = test_stationarity(non_stationary)
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Is stationary: {result['is_stationary']}")

    # Comprehensive check
    print("\nComprehensive stationarity check:")
    check_stationarity(non_stationary, verbose=True)


def example_arima_forecasting():
    """Example: ARIMA forecasting."""
    print("\n" + "="*70)
    print("EXAMPLE 2: ARIMA Forecasting")
    print("="*70)

    from puffin.models import ARIMAModel, auto_arima

    # Generate AR(1) process
    n = 300
    phi = 0.7
    ar1 = np.zeros(n)
    for i in range(1, n):
        ar1[i] = phi * ar1[i-1] + np.random.randn()

    series = pd.Series(ar1)

    # Auto-select ARIMA order
    print("\nAuto-selecting ARIMA order...")
    model = auto_arima(series, max_p=3, max_d=1, max_q=3)
    print(f"Selected order: {model.order_}")
    print(f"AIC: {model.aic_:.2f}")

    # Forecast
    forecast = model.predict(steps=20)
    print(f"\n20-step ahead forecast:")
    print(forecast.head(10))

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values, label='Historical', alpha=0.7)
    forecast_index = range(len(series), len(series) + len(forecast))
    plt.plot(forecast_index, forecast.values, 'r--', label='Forecast')
    plt.title('ARIMA Forecast')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('arima_forecast.png')
    print("\nPlot saved to 'arima_forecast.png'")
    plt.close()


def example_garch_volatility():
    """Example: GARCH volatility modeling."""
    print("\n" + "="*70)
    print("EXAMPLE 3: GARCH Volatility Modeling")
    print("="*70)

    from puffin.models import GARCHModel

    # Generate returns with volatility clustering
    n = 500
    returns = []
    sigma = 1.0

    for i in range(n):
        # GARCH(1,1) process
        epsilon = np.random.randn()
        sigma = np.sqrt(0.1 + 0.1 * (returns[-1]**2 if i > 0 else 0) + 0.8 * sigma**2)
        returns.append(sigma * epsilon)

    returns_series = pd.Series(returns)

    # Fit GARCH model
    print("\nFitting GARCH(1,1) model...")
    garch = GARCHModel(p=1, q=1)
    garch.fit(returns_series)

    print(f"Model AIC: {garch.results_.aic:.2f}")

    # Get conditional volatility
    conditional_vol = garch.conditional_volatility

    # Forecast volatility
    vol_forecast = garch.forecast_volatility(horizon=10)
    print(f"\n10-period volatility forecast:")
    print(vol_forecast)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(returns_series, alpha=0.6, label='Returns')
    ax1.set_title('Returns')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(conditional_vol, label='Conditional Volatility', color='red')
    ax2.set_title('GARCH Conditional Volatility')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('garch_volatility.png')
    print("\nPlot saved to 'garch_volatility.png'")
    plt.close()


def example_cointegration():
    """Example: Cointegration analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Cointegration Analysis")
    print("="*70)

    from puffin.models import engle_granger_test, find_cointegrated_pairs, calculate_spread, half_life

    # Generate cointegrated pair
    n = 252
    common_factor = np.random.randn(n).cumsum()

    prices = pd.DataFrame({
        'Stock_A': common_factor + np.random.randn(n) * 0.5,
        'Stock_B': 1.5 * common_factor + np.random.randn(n) * 0.5,
        'Stock_C': np.random.randn(n).cumsum(),  # Independent
    })

    # Test for cointegration
    print("\nTesting Stock_A and Stock_B for cointegration:")
    result = engle_granger_test(prices['Stock_A'], prices['Stock_B'])
    print(f"  Cointegrated: {result['is_cointegrated']}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Hedge ratio: {result['hedge_ratio']:.4f}")

    # Find all cointegrated pairs
    print("\nSearching for all cointegrated pairs...")
    pairs = find_cointegrated_pairs(prices, significance=0.05)

    for ticker1, ticker2, p_value, hedge_ratio in pairs:
        print(f"  {ticker1}-{ticker2}: p={p_value:.4f}, hedge={hedge_ratio:.4f}")

    # Analyze spread
    if pairs:
        ticker1, ticker2, p_value, hedge_ratio = pairs[0]
        spread = calculate_spread(prices[ticker1], prices[ticker2], hedge_ratio)

        hl = half_life(spread)
        print(f"\nSpread half-life: {hl:.2f} periods")

        # Plot spread
        plt.figure(figsize=(12, 5))
        spread.plot(title=f'Spread: {ticker1} - {hedge_ratio:.2f} * {ticker2}')
        plt.axhline(y=spread.mean(), color='r', linestyle='--', label='Mean')
        plt.axhline(y=spread.mean() + 2*spread.std(), color='g', linestyle='--', label='+2 Std')
        plt.axhline(y=spread.mean() - 2*spread.std(), color='g', linestyle='--', label='-2 Std')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cointegration_spread.png')
        print("\nPlot saved to 'cointegration_spread.png'")
        plt.close()


def example_pairs_trading():
    """Example: Pairs trading strategy."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Pairs Trading Strategy")
    print("="*70)

    from puffin.models import PairsTradingStrategy

    # Generate price data
    n = 500
    factor1 = np.random.randn(n).cumsum()
    factor2 = np.random.randn(n).cumsum()

    prices = pd.DataFrame({
        'Stock_A': factor1 + np.random.randn(n) * 0.3,
        'Stock_B': 1.5 * factor1 + np.random.randn(n) * 0.3,
        'Stock_C': factor2 + np.random.randn(n) * 0.3,
        'Stock_D': 0.8 * factor2 + np.random.randn(n) * 0.3,
    })
    prices.index = pd.date_range('2020-01-01', periods=n, freq='D')

    # Create strategy
    strategy = PairsTradingStrategy(entry_z=2.0, exit_z=0.5, lookback=20)

    # Find tradable pairs
    print("\nFinding tradable pairs...")
    pairs = strategy.find_pairs(prices, significance=0.05, min_half_life=5, max_half_life=100)

    print(f"Found {len(pairs)} tradable pairs:")
    for ticker1, ticker2, p_value, hedge_ratio in pairs:
        print(f"  {ticker1}-{ticker2}: p={p_value:.4f}")

    # Backtest
    if pairs:
        print("\nBacktesting pairs...")
        results = strategy.backtest_portfolio(pairs, prices, transaction_cost=0.001)

        print("\nPortfolio Performance:")
        print(f"  Total Return: {results['total_return']:.2%}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2%}")

        # Plot performance
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        results['cumulative_returns'].plot(ax=ax1, title='Pairs Trading: Cumulative Returns')
        ax1.set_ylabel('Cumulative Returns')
        ax1.grid(True)

        results['returns'].hist(bins=50, ax=ax2)
        ax2.set_title('Daily Returns Distribution')
        ax2.set_xlabel('Returns')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('pairs_trading_performance.png')
        print("\nPlot saved to 'pairs_trading_performance.png'")
        plt.close()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TIME SERIES MODELS EXAMPLES")
    print("="*70)

    try:
        example_stationarity_testing()
    except Exception as e:
        print(f"\nError in stationarity testing example: {e}")

    try:
        example_arima_forecasting()
    except Exception as e:
        print(f"\nError in ARIMA example: {e}")

    try:
        example_garch_volatility()
    except Exception as e:
        print(f"\nError in GARCH example: {e}")

    try:
        example_cointegration()
    except Exception as e:
        print(f"\nError in cointegration example: {e}")

    try:
        example_pairs_trading()
    except Exception as e:
        print(f"\nError in pairs trading example: {e}")

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
