"""
Cointegration tests and analysis for pairs trading.

This module provides tests for cointegration relationships between time series,
which is fundamental for statistical arbitrage and pairs trading strategies.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Union, Optional
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations


def engle_granger_test(
    y1: pd.Series,
    y2: pd.Series,
    trend: str = 'c',
    autolag: Optional[str] = 'AIC'
) -> Dict[str, Union[float, bool]]:
    """
    Perform Engle-Granger two-step cointegration test.

    Tests whether two time series are cointegrated by testing if their
    linear combination is stationary.

    Parameters
    ----------
    y1 : pd.Series
        First time series
    y2 : pd.Series
        Second time series
    trend : str, default 'c'
        Trend parameter: 'c' (constant), 'ct' (constant + trend), 'nc' (none)
    autolag : str, optional
        Automatic lag selection method

    Returns
    -------
    dict
        Dictionary containing:
        - 'test_statistic': Test statistic
        - 'p_value': p-value
        - 'is_cointegrated': True if p_value < 0.05
        - 'hedge_ratio': Optimal hedge ratio (beta from regression)
        - 'critical_values': Critical values at 1%, 5%, 10%

    Examples
    --------
    >>> y1 = pd.Series(np.random.randn(252).cumsum())
    >>> y2 = pd.Series(np.random.randn(252).cumsum())
    >>> result = engle_granger_test(y1, y2)
    >>> print(f"Cointegrated: {result['is_cointegrated']}")
    >>> print(f"Hedge ratio: {result['hedge_ratio']:.4f}")
    """
    # Align series
    data = pd.DataFrame({'y1': y1, 'y2': y2}).dropna()

    if len(data) < 20:
        raise ValueError("Need at least 20 observations for cointegration test")

    y1_clean = data['y1'].values
    y2_clean = data['y2'].values

    # Calculate hedge ratio using OLS regression
    # y1 = alpha + beta * y2 + epsilon
    X = np.column_stack([np.ones(len(y2_clean)), y2_clean])
    beta_hat = np.linalg.lstsq(X, y1_clean, rcond=None)[0]
    hedge_ratio = beta_hat[1]

    # Calculate residuals
    residuals = y1_clean - beta_hat[0] - beta_hat[1] * y2_clean

    # Test residuals for stationarity (ADF test)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        adf_result = adfuller(residuals, regression=trend, autolag=autolag)

    # Use statsmodels coint for proper critical values
    coint_result = coint(y1_clean, y2_clean, trend=trend, autolag=autolag)

    return {
        'test_statistic': coint_result[0],
        'p_value': coint_result[1],
        'is_cointegrated': coint_result[1] < 0.05,
        'hedge_ratio': hedge_ratio,
        'critical_values': {
            '1%': coint_result[2][0],
            '5%': coint_result[2][1],
            '10%': coint_result[2][2]
        }
    }


def johansen_test(
    data: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1
) -> Dict[str, Union[np.ndarray, int]]:
    """
    Perform Johansen cointegration test for multiple time series.

    The Johansen test can detect multiple cointegration relationships
    among a set of variables.

    Parameters
    ----------
    data : pd.DataFrame
        Multivariate time series (each column is a variable)
    det_order : int, default 0
        Deterministic trend order:
        -1: no deterministic terms
        0: constant term
        1: linear trend
    k_ar_diff : int, default 1
        Number of lagged differences in the model

    Returns
    -------
    dict
        Dictionary containing:
        - 'trace_statistic': Trace test statistics
        - 'trace_critical_values': Critical values for trace test
        - 'max_eigen_statistic': Maximum eigenvalue test statistics
        - 'max_eigen_critical_values': Critical values for max eigenvalue test
        - 'n_cointegrated': Number of cointegration relationships at 5% level
        - 'eigenvalues': Eigenvalues

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'price1': np.random.randn(252).cumsum(),
    ...     'price2': np.random.randn(252).cumsum(),
    ...     'price3': np.random.randn(252).cumsum()
    ... })
    >>> result = johansen_test(data)
    >>> print(f"Number of cointegration relationships: {result['n_cointegrated']}")
    """
    # Remove NaN values
    data_clean = data.dropna()

    if len(data_clean) < 20:
        raise ValueError("Need at least 20 observations for Johansen test")

    if data_clean.shape[1] < 2:
        raise ValueError("Need at least 2 variables for Johansen test")

    # Perform Johansen test
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        result = coint_johansen(data_clean, det_order=det_order, k_ar_diff=k_ar_diff)

    # Count number of cointegration relationships at 5% level
    # Compare trace statistic to 5% critical value (index 1)
    trace_stats = result.trace_stat
    trace_crit = result.trace_stat_crit_vals[:, 1]  # 5% critical values
    n_cointegrated = np.sum(trace_stats > trace_crit)

    return {
        'trace_statistic': result.trace_stat,
        'trace_critical_values': result.trace_stat_crit_vals,
        'max_eigen_statistic': result.max_eig_stat,
        'max_eigen_critical_values': result.max_eig_stat_crit_vals,
        'n_cointegrated': n_cointegrated,
        'eigenvalues': result.eig
    }


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05,
    min_observations: int = 50
) -> List[Tuple[str, str, float, float]]:
    """
    Find all cointegrated pairs in a set of price series.

    Tests all possible pairs of price series for cointegration using
    the Engle-Granger test.

    Parameters
    ----------
    prices : pd.DataFrame
        Price series (each column is a different asset)
    significance : float, default 0.05
        Significance level for cointegration test
    min_observations : int, default 50
        Minimum number of overlapping observations required

    Returns
    -------
    list
        List of tuples (ticker1, ticker2, p_value, hedge_ratio) for
        cointegrated pairs, sorted by p-value

    Examples
    --------
    >>> prices = pd.DataFrame({
    ...     'AAPL': np.random.randn(252).cumsum() + 100,
    ...     'MSFT': np.random.randn(252).cumsum() + 100,
    ...     'GOOGL': np.random.randn(252).cumsum() + 100
    ... })
    >>> pairs = find_cointegrated_pairs(prices, significance=0.05)
    >>> for ticker1, ticker2, p_value, hedge_ratio in pairs:
    ...     print(f"{ticker1}-{ticker2}: p={p_value:.4f}, hedge={hedge_ratio:.4f}")
    """
    tickers = prices.columns
    n_tickers = len(tickers)
    cointegrated_pairs = []

    # Test all pairs
    for i in range(n_tickers):
        for j in range(i + 1, n_tickers):
            ticker1 = tickers[i]
            ticker2 = tickers[j]

            # Get overlapping data
            pair_data = prices[[ticker1, ticker2]].dropna()

            if len(pair_data) < min_observations:
                continue

            try:
                # Test for cointegration
                result = engle_granger_test(
                    pair_data[ticker1],
                    pair_data[ticker2]
                )

                # Store if cointegrated
                if result['p_value'] < significance:
                    cointegrated_pairs.append((
                        ticker1,
                        ticker2,
                        result['p_value'],
                        result['hedge_ratio']
                    ))

            except Exception:
                # Skip pairs that cause errors
                continue

    # Sort by p-value (most significant first)
    cointegrated_pairs.sort(key=lambda x: x[2])

    return cointegrated_pairs


def calculate_spread(
    y1: pd.Series,
    y2: pd.Series,
    hedge_ratio: Optional[float] = None
) -> pd.Series:
    """
    Calculate the spread between two cointegrated series.

    Parameters
    ----------
    y1 : pd.Series
        First time series
    y2 : pd.Series
        Second time series
    hedge_ratio : float, optional
        Hedge ratio. If None, estimated from data.

    Returns
    -------
    pd.Series
        Spread series (y1 - hedge_ratio * y2)

    Examples
    --------
    >>> y1 = pd.Series(np.random.randn(252).cumsum())
    >>> y2 = pd.Series(np.random.randn(252).cumsum())
    >>> spread = calculate_spread(y1, y2)
    """
    # Align series
    data = pd.DataFrame({'y1': y1, 'y2': y2}).dropna()

    if hedge_ratio is None:
        # Estimate hedge ratio
        y1_clean = data['y1'].values
        y2_clean = data['y2'].values

        X = np.column_stack([np.ones(len(y2_clean)), y2_clean])
        beta_hat = np.linalg.lstsq(X, y1_clean, rcond=None)[0]
        hedge_ratio = beta_hat[1]

    # Calculate spread
    spread = data['y1'] - hedge_ratio * data['y2']

    return spread


def half_life(spread: pd.Series) -> float:
    """
    Calculate the half-life of mean reversion for a spread.

    The half-life indicates how long it takes for the spread to revert
    halfway back to its mean.

    Parameters
    ----------
    spread : pd.Series
        Spread time series

    Returns
    -------
    float
        Half-life in number of periods

    Examples
    --------
    >>> spread = pd.Series(np.random.randn(252))
    >>> hl = half_life(spread)
    >>> print(f"Half-life: {hl:.2f} periods")
    """
    spread_clean = spread.dropna()

    if len(spread_clean) < 2:
        raise ValueError("Spread must have at least 2 observations")

    # Fit AR(1) model: spread_t = mu + phi * spread_{t-1} + epsilon_t
    spread_lag = spread_clean.shift(1).dropna()
    spread_diff = spread_clean.diff().dropna()

    # Align
    data = pd.DataFrame({
        'spread_lag': spread_lag,
        'spread_diff': spread_diff
    }).dropna()

    # OLS regression
    X = data['spread_lag'].values
    y = data['spread_diff'].values

    # Add constant
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Get phi coefficient
    phi = beta[1] + 1  # Because we used differenced data

    # Calculate half-life
    if phi >= 1 or phi <= 0:
        return np.inf

    half_life_val = -np.log(2) / np.log(phi)

    return half_life_val


def test_cointegration_all_pairs(
    prices: pd.DataFrame,
    method: str = 'engle-granger',
    significance: float = 0.05
) -> pd.DataFrame:
    """
    Test cointegration for all pairs and return a matrix of p-values.

    Parameters
    ----------
    prices : pd.DataFrame
        Price series (each column is a different asset)
    method : str, default 'engle-granger'
        Test method: 'engle-granger' or 'johansen'
    significance : float, default 0.05
        Significance level

    Returns
    -------
    pd.DataFrame
        Matrix of p-values for all pairs

    Examples
    --------
    >>> prices = pd.DataFrame({
    ...     'AAPL': np.random.randn(252).cumsum() + 100,
    ...     'MSFT': np.random.randn(252).cumsum() + 100,
    ...     'GOOGL': np.random.randn(252).cumsum() + 100
    ... })
    >>> p_value_matrix = test_cointegration_all_pairs(prices)
    """
    tickers = prices.columns
    n_tickers = len(tickers)

    # Initialize matrix
    p_values = np.full((n_tickers, n_tickers), np.nan)

    for i in range(n_tickers):
        for j in range(i + 1, n_tickers):
            ticker1 = tickers[i]
            ticker2 = tickers[j]

            # Get overlapping data
            pair_data = prices[[ticker1, ticker2]].dropna()

            if len(pair_data) < 20:
                continue

            try:
                if method == 'engle-granger':
                    result = engle_granger_test(
                        pair_data[ticker1],
                        pair_data[ticker2]
                    )
                    p_value = result['p_value']
                else:
                    raise ValueError(f"Unknown method: {method}")

                p_values[i, j] = p_value
                p_values[j, i] = p_value

            except Exception:
                continue

    # Create DataFrame
    p_value_df = pd.DataFrame(
        p_values,
        index=tickers,
        columns=tickers
    )

    return p_value_df


def adf_test_spread(
    spread: pd.Series,
    regression: str = 'c',
    autolag: Optional[str] = 'AIC'
) -> Dict[str, Union[float, bool]]:
    """
    Perform ADF test on a spread to check for mean reversion.

    Parameters
    ----------
    spread : pd.Series
        Spread time series
    regression : str, default 'c'
        Regression type
    autolag : str, optional
        Automatic lag selection

    Returns
    -------
    dict
        Dictionary with test results

    Examples
    --------
    >>> spread = pd.Series(np.random.randn(252))
    >>> result = adf_test_spread(spread)
    >>> print(f"Is mean reverting: {result['is_stationary']}")
    """
    spread_clean = spread.dropna()

    if len(spread_clean) < 10:
        raise ValueError("Spread must have at least 10 observations")

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        result = adfuller(spread_clean, regression=regression, autolag=autolag)

    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }
