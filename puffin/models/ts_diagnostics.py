"""
Time series diagnostics and analysis tools.

This module provides functions for analyzing time series properties including
decomposition, stationarity testing, and autocorrelation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def decompose_series(
    series: pd.Series,
    period: int = 252,
    model: str = 'additive',
    extrapolate_trend: Optional[str] = 'freq'
) -> Dict[str, pd.Series]:
    """
    Decompose a time series into trend, seasonal, and residual components.

    Parameters
    ----------
    series : pd.Series
        Time series to decompose
    period : int, default 252
        Period of the seasonal component (252 for daily trading data)
    model : str, default 'additive'
        Type of decomposition: 'additive' or 'multiplicative'
    extrapolate_trend : str or int, optional
        How to extrapolate trend for missing values

    Returns
    -------
    dict
        Dictionary with keys 'trend', 'seasonal', 'residual', and 'observed'

    Examples
    --------
    >>> prices = pd.Series(np.random.randn(500).cumsum())
    >>> components = decompose_series(prices, period=20)
    >>> trend = components['trend']
    """
    if len(series) < 2 * period:
        raise ValueError(f"Series length ({len(series)}) must be at least 2*period ({2*period})")

    # Ensure series has datetime index for decomposition
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.date_range(start='2020-01-01', periods=len(series), freq='D')

    result = seasonal_decompose(
        series,
        model=model,
        period=period,
        extrapolate_trend=extrapolate_trend
    )

    return {
        'observed': result.observed,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid
    }


def test_stationarity(
    series: pd.Series,
    regression: str = 'c',
    autolag: Optional[str] = 'AIC'
) -> Dict[str, Union[float, bool, dict]]:
    """
    Test for stationarity using the Augmented Dickey-Fuller test.

    Parameters
    ----------
    series : pd.Series
        Time series to test
    regression : str, default 'c'
        Constant and trend order to include in regression:
        'c' : constant only (default)
        'ct' : constant and trend
        'ctt' : constant, and linear and quadratic trend
        'n' : no constant, no trend
    autolag : str, optional
        Method to use for automatic lag selection

    Returns
    -------
    dict
        Dictionary containing:
        - 'test_statistic': ADF test statistic
        - 'p_value': p-value
        - 'n_lags': number of lags used
        - 'n_obs': number of observations
        - 'critical_values': dict of critical values at 1%, 5%, 10%
        - 'is_stationary': True if p_value < 0.05

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(252))
    >>> result = test_stationarity(returns)
    >>> print(f"Is stationary: {result['is_stationary']}")
    """
    # Remove NaN values
    series_clean = series.dropna()

    if len(series_clean) < 10:
        raise ValueError("Series must have at least 10 non-NaN observations")

    result = adfuller(series_clean, regression=regression, autolag=autolag)

    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }


def test_kpss(
    series: pd.Series,
    regression: str = 'c',
    nlags: Optional[str] = 'auto'
) -> Dict[str, Union[float, bool, dict]]:
    """
    Test for stationarity using the KPSS test.

    Note: KPSS has null hypothesis that the series is stationary (opposite of ADF).

    Parameters
    ----------
    series : pd.Series
        Time series to test
    regression : str, default 'c'
        Trend component to include in regression:
        'c' : constant only (default)
        'ct' : constant and trend
    nlags : str or int, optional
        Number of lags to use

    Returns
    -------
    dict
        Dictionary containing:
        - 'test_statistic': KPSS test statistic
        - 'p_value': p-value
        - 'n_lags': number of lags used
        - 'critical_values': dict of critical values at 10%, 5%, 2.5%, 1%
        - 'is_stationary': True if p_value > 0.05 (opposite of ADF)

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(252))
    >>> result = test_kpss(returns)
    >>> print(f"Is stationary: {result['is_stationary']}")
    """
    # Remove NaN values
    series_clean = series.dropna()

    if len(series_clean) < 10:
        raise ValueError("Series must have at least 10 non-NaN observations")

    result = kpss(series_clean, regression=regression, nlags=nlags)

    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'critical_values': result[3],
        'is_stationary': result[1] > 0.05  # Note: opposite interpretation from ADF
    }


def plot_acf_pacf(
    series: pd.Series,
    nlags: int = 40,
    alpha: float = 0.05,
    figsize: tuple = (15, 5)
) -> plt.Figure:
    """
    Plot autocorrelation function (ACF) and partial autocorrelation function (PACF).

    Parameters
    ----------
    series : pd.Series
        Time series to analyze
    nlags : int, default 40
        Number of lags to include
    alpha : float, default 0.05
        Significance level for confidence intervals
    figsize : tuple, default (15, 5)
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        Figure with ACF and PACF plots

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(252))
    >>> fig = plot_acf_pacf(returns, nlags=20)
    >>> plt.show()
    """
    # Remove NaN values
    series_clean = series.dropna()

    if len(series_clean) < nlags + 1:
        nlags = len(series_clean) - 1
        if nlags < 1:
            raise ValueError("Series too short for autocorrelation analysis")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot ACF
    plot_acf(series_clean, lags=nlags, alpha=alpha, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Correlation')

    # Plot PACF
    plot_pacf(series_clean, lags=nlags, alpha=alpha, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Correlation')

    plt.tight_layout()
    return fig


def autocorrelation(
    series: pd.Series,
    nlags: int = 40,
    adjusted: bool = False
) -> np.ndarray:
    """
    Calculate autocorrelation coefficients for a time series.

    Parameters
    ----------
    series : pd.Series
        Time series to analyze
    nlags : int, default 40
        Number of lags to calculate
    adjusted : bool, default False
        If True, use adjusted autocorrelation (divide by N-k instead of N)

    Returns
    -------
    np.ndarray
        Array of autocorrelation coefficients (length nlags + 1)

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(252))
    >>> acf_values = autocorrelation(returns, nlags=20)
    >>> print(f"First-order autocorrelation: {acf_values[1]:.3f}")
    """
    # Remove NaN values
    series_clean = series.dropna()

    if len(series_clean) < nlags + 1:
        raise ValueError(f"Series length ({len(series_clean)}) must be > nlags ({nlags})")

    # Calculate autocorrelation using pandas
    acf_values = np.array([series_clean.autocorr(lag=i) for i in range(nlags + 1)])

    if adjusted:
        # Adjust for degrees of freedom
        n = len(series_clean)
        adjustment = np.array([n / (n - i) if i < n else np.nan for i in range(nlags + 1)])
        acf_values = acf_values * adjustment

    return acf_values


def check_stationarity(
    series: pd.Series,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Comprehensive stationarity check using both ADF and KPSS tests.

    Parameters
    ----------
    series : pd.Series
        Time series to test
    verbose : bool, default True
        If True, print test results

    Returns
    -------
    dict
        Dictionary with 'adf' and 'kpss' test results

    Examples
    --------
    >>> prices = pd.Series(np.random.randn(252).cumsum())
    >>> results = check_stationarity(prices)
    """
    adf_result = test_stationarity(series)
    kpss_result = test_kpss(series)

    if verbose:
        print("=" * 60)
        print("STATIONARITY TESTS")
        print("=" * 60)
        print("\nAugmented Dickey-Fuller Test:")
        print(f"  Test Statistic: {adf_result['test_statistic']:.4f}")
        print(f"  p-value: {adf_result['p_value']:.4f}")
        print(f"  Lags used: {adf_result['n_lags']}")
        print(f"  Critical Values:")
        for key, value in adf_result['critical_values'].items():
            print(f"    {key}: {value:.4f}")
        print(f"  Is Stationary (ADF): {adf_result['is_stationary']}")

        print("\nKPSS Test:")
        print(f"  Test Statistic: {kpss_result['test_statistic']:.4f}")
        print(f"  p-value: {kpss_result['p_value']:.4f}")
        print(f"  Lags used: {kpss_result['n_lags']}")
        print(f"  Critical Values:")
        for key, value in kpss_result['critical_values'].items():
            print(f"    {key}: {value:.4f}")
        print(f"  Is Stationary (KPSS): {kpss_result['is_stationary']}")

        print("\n" + "=" * 60)
        if adf_result['is_stationary'] and kpss_result['is_stationary']:
            print("CONCLUSION: Series is STATIONARY (both tests agree)")
        elif not adf_result['is_stationary'] and not kpss_result['is_stationary']:
            print("CONCLUSION: Series is NON-STATIONARY (both tests agree)")
        else:
            print("CONCLUSION: Tests DISAGREE - further investigation needed")
        print("=" * 60)

    return {
        'adf': adf_result,
        'kpss': kpss_result
    }
