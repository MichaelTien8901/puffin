"""
Alpha factor computation module.

This module provides functions to compute various types of alpha factors:
- Momentum factors (trend-following signals)
- Value factors (fundamental valuation metrics)
- Volatility factors (realized volatility estimators)
- Quality factors (profitability and efficiency metrics)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


def compute_momentum_factors(
    prices: pd.DataFrame,
    windows: List[int] = [5, 10, 21, 63, 252]
) -> pd.DataFrame:
    """
    Compute momentum factors over various lookback windows.

    Momentum factors capture the tendency of assets to continue moving
    in the same direction. This function computes simple returns over
    multiple time horizons.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with datetime index and symbols as columns.
    windows : List[int], optional
        List of lookback windows in periods (default: [5,10,21,63,252]).
        Common windows: 5 (week), 10 (2 weeks), 21 (month), 63 (quarter), 252 (year)

    Returns
    -------
    pd.DataFrame
        Multi-level DataFrame with (date, symbol) index and momentum factors as columns.
        Columns: mom_5, mom_10, mom_21, mom_63, mom_252, mom_ratio, mom_diff

    Examples
    --------
    >>> prices = pd.DataFrame({'AAPL': [100, 101, 102], 'MSFT': [50, 51, 52]})
    >>> factors = compute_momentum_factors(prices, windows=[5, 10])
    """
    factors_list = []

    for symbol in prices.columns:
        symbol_prices = prices[symbol].dropna()
        factor_data = pd.DataFrame(index=symbol_prices.index)
        factor_data['symbol'] = symbol

        # Basic momentum: returns over various horizons
        for window in windows:
            if len(symbol_prices) > window:
                factor_data[f'mom_{window}'] = (
                    symbol_prices / symbol_prices.shift(window) - 1
                )

        # Cross-sectional momentum features
        if 21 in windows and 252 in windows:
            # Momentum ratio: short-term vs long-term
            factor_data['mom_ratio'] = (
                factor_data['mom_21'] / (factor_data['mom_252'].abs() + 1e-8)
            )
            # Momentum differential
            factor_data['mom_diff'] = factor_data['mom_21'] - factor_data['mom_252']

        # Acceleration: second derivative of momentum
        if 21 in windows:
            factor_data['mom_accel'] = factor_data['mom_21'].diff()

        # Recent trend strength (sum of positive days / total days)
        if len(symbol_prices) > 21:
            returns = symbol_prices.pct_change()
            factor_data['trend_strength'] = (
                returns.rolling(21).apply(lambda x: (x > 0).sum() / len(x))
            )

        factors_list.append(factor_data)

    # Combine all symbols
    all_factors = pd.concat(factors_list, ignore_index=False)
    all_factors = all_factors.reset_index().set_index(['index', 'symbol'])
    all_factors.index.names = ['date', 'symbol']

    return all_factors


def compute_value_factors(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Compute value factors from fundamental data.

    Value factors identify undervalued or overvalued securities based on
    fundamental metrics relative to market price.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Fundamental data with columns: price, earnings, book_value,
        enterprise_value, ebitda, revenue, market_cap.
        Index should be (date, symbol) MultiIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with value factors as columns:
        - pe_ratio: Price to Earnings
        - pb_ratio: Price to Book
        - ev_ebitda: Enterprise Value to EBITDA
        - ps_ratio: Price to Sales
        - earnings_yield: E/P ratio (inverse of P/E)
        - book_yield: B/P ratio (inverse of P/B)

    Examples
    --------
    >>> fundamentals = pd.DataFrame({
    ...     'price': [100, 110],
    ...     'earnings': [10, 12],
    ...     'book_value': [50, 55]
    ... })
    >>> factors = compute_value_factors(fundamentals)
    """
    factors = pd.DataFrame(index=fundamentals.index)

    # Price to Earnings (P/E)
    if 'price' in fundamentals.columns and 'earnings' in fundamentals.columns:
        factors['pe_ratio'] = fundamentals['price'] / (fundamentals['earnings'] + 1e-8)
        factors['earnings_yield'] = fundamentals['earnings'] / (fundamentals['price'] + 1e-8)

    # Price to Book (P/B)
    if 'price' in fundamentals.columns and 'book_value' in fundamentals.columns:
        factors['pb_ratio'] = fundamentals['price'] / (fundamentals['book_value'] + 1e-8)
        factors['book_yield'] = fundamentals['book_value'] / (fundamentals['price'] + 1e-8)

    # Enterprise Value to EBITDA
    if 'enterprise_value' in fundamentals.columns and 'ebitda' in fundamentals.columns:
        factors['ev_ebitda'] = (
            fundamentals['enterprise_value'] / (fundamentals['ebitda'] + 1e-8)
        )

    # Price to Sales (P/S)
    if 'price' in fundamentals.columns and 'revenue' in fundamentals.columns:
        factors['ps_ratio'] = fundamentals['price'] / (fundamentals['revenue'] + 1e-8)

    # Market Cap to Revenue
    if 'market_cap' in fundamentals.columns and 'revenue' in fundamentals.columns:
        factors['mcap_revenue'] = (
            fundamentals['market_cap'] / (fundamentals['revenue'] + 1e-8)
        )

    # Replace inf with NaN
    factors = factors.replace([np.inf, -np.inf], np.nan)

    return factors


def compute_volatility_factors(
    prices: pd.DataFrame,
    windows: List[int] = [21, 63]
) -> pd.DataFrame:
    """
    Compute volatility factors using various estimators.

    This function computes multiple volatility estimators:
    - Realized volatility (close-to-close)
    - Parkinson volatility (high-low range)
    - Garman-Klass volatility (OHLC)

    Parameters
    ----------
    prices : pd.DataFrame or dict
        If DataFrame: close prices with datetime index and symbols as columns.
        If dict: Must contain 'close', 'high', 'low', 'open' DataFrames.
    windows : List[int], optional
        List of rolling windows for volatility computation (default: [21, 63])

    Returns
    -------
    pd.DataFrame
        Multi-level DataFrame with volatility factors as columns:
        - realized_vol_{window}: Close-to-close realized volatility
        - parkinson_vol_{window}: Parkinson high-low estimator
        - gk_vol_{window}: Garman-Klass OHLC estimator
        - vol_ratio: Short-term vs long-term volatility ratio

    Examples
    --------
    >>> prices = pd.DataFrame({'AAPL': [100, 102, 101, 103, 105]})
    >>> factors = compute_volatility_factors(prices, windows=[3])
    """
    factors_list = []

    # Determine if we have OHLC data
    has_ohlc = isinstance(prices, dict) and all(
        k in prices for k in ['close', 'high', 'low', 'open']
    )

    if has_ohlc:
        close_prices = prices['close']
        high_prices = prices['high']
        low_prices = prices['low']
        open_prices = prices['open']
        symbols = close_prices.columns
    else:
        close_prices = prices
        symbols = prices.columns

    for symbol in symbols:
        symbol_close = close_prices[symbol].dropna()
        factor_data = pd.DataFrame(index=symbol_close.index)
        factor_data['symbol'] = symbol

        # Log returns for realized volatility
        log_returns = np.log(symbol_close / symbol_close.shift(1))

        # Realized volatility (annualized)
        for window in windows:
            if len(symbol_close) > window:
                factor_data[f'realized_vol_{window}'] = (
                    log_returns.rolling(window).std() * np.sqrt(252)
                )

        # Parkinson volatility (high-low estimator)
        if has_ohlc:
            symbol_high = high_prices[symbol].dropna()
            symbol_low = low_prices[symbol].dropna()

            for window in windows:
                if len(symbol_high) > window:
                    hl_ratio = np.log(symbol_high / symbol_low) ** 2
                    factor_data[f'parkinson_vol_{window}'] = (
                        np.sqrt(hl_ratio.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)
                    )

        # Garman-Klass volatility (OHLC estimator)
        if has_ohlc:
            symbol_open = open_prices[symbol].dropna()

            for window in windows:
                if len(symbol_open) > window:
                    # Garman-Klass formula
                    hl_term = 0.5 * (np.log(symbol_high / symbol_low) ** 2)
                    co_term = -(2 * np.log(2) - 1) * (np.log(symbol_close / symbol_open) ** 2)
                    gk_vol_sq = hl_term + co_term

                    factor_data[f'gk_vol_{window}'] = (
                        np.sqrt(gk_vol_sq.rolling(window).mean()) * np.sqrt(252)
                    )

        # Volatility ratio (regime detection)
        if len(windows) >= 2:
            short_window = min(windows)
            long_window = max(windows)
            factor_data['vol_ratio'] = (
                factor_data[f'realized_vol_{short_window}'] /
                (factor_data[f'realized_vol_{long_window}'] + 1e-8)
            )

        # Volatility trend (change in volatility)
        if len(windows) >= 1:
            window = windows[0]
            vol_col = f'realized_vol_{window}'
            if vol_col in factor_data.columns:
                factor_data['vol_trend'] = factor_data[vol_col].pct_change()

        factors_list.append(factor_data)

    # Combine all symbols
    all_factors = pd.concat(factors_list, ignore_index=False)
    all_factors = all_factors.reset_index().set_index(['index', 'symbol'])
    all_factors.index.names = ['date', 'symbol']

    return all_factors


def compute_quality_factors(financials: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quality factors from financial statement data.

    Quality factors measure the financial health and profitability
    of a company.

    Parameters
    ----------
    financials : pd.DataFrame
        Financial statement data with columns: net_income, revenue, assets,
        equity, operating_cash_flow, total_accruals, liabilities.
        Index should be (date, symbol) MultiIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with quality factors as columns:
        - roe: Return on Equity (net income / equity)
        - roa: Return on Assets (net income / assets)
        - profit_margin: Net income / revenue
        - asset_turnover: Revenue / assets
        - accruals_ratio: Total accruals / assets
        - cash_flow_to_income: Operating cash flow / net income

    Examples
    --------
    >>> financials = pd.DataFrame({
    ...     'net_income': [1000, 1100],
    ...     'equity': [5000, 5500],
    ...     'assets': [10000, 11000]
    ... })
    >>> factors = compute_quality_factors(financials)
    """
    factors = pd.DataFrame(index=financials.index)

    # Return on Equity (ROE)
    if 'net_income' in financials.columns and 'equity' in financials.columns:
        factors['roe'] = financials['net_income'] / (financials['equity'] + 1e-8)

    # Return on Assets (ROA)
    if 'net_income' in financials.columns and 'assets' in financials.columns:
        factors['roa'] = financials['net_income'] / (financials['assets'] + 1e-8)

    # Profit Margin
    if 'net_income' in financials.columns and 'revenue' in financials.columns:
        factors['profit_margin'] = (
            financials['net_income'] / (financials['revenue'] + 1e-8)
        )

    # Asset Turnover
    if 'revenue' in financials.columns and 'assets' in financials.columns:
        factors['asset_turnover'] = (
            financials['revenue'] / (financials['assets'] + 1e-8)
        )

    # Accruals Ratio (lower is better - indicates quality of earnings)
    if 'total_accruals' in financials.columns and 'assets' in financials.columns:
        factors['accruals_ratio'] = (
            financials['total_accruals'] / (financials['assets'] + 1e-8)
        )

    # Cash Flow to Income (higher is better)
    if 'operating_cash_flow' in financials.columns and 'net_income' in financials.columns:
        factors['cash_flow_to_income'] = (
            financials['operating_cash_flow'] / (financials['net_income'] + 1e-8)
        )

    # Debt to Equity
    if 'liabilities' in financials.columns and 'equity' in financials.columns:
        factors['debt_to_equity'] = (
            financials['liabilities'] / (financials['equity'] + 1e-8)
        )

    # Financial Leverage
    if 'assets' in financials.columns and 'equity' in financials.columns:
        factors['leverage'] = financials['assets'] / (financials['equity'] + 1e-8)

    # Replace inf with NaN
    factors = factors.replace([np.inf, -np.inf], np.nan)

    return factors


def compute_all_factors(
    prices: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    fundamentals: Optional[pd.DataFrame] = None,
    financials: Optional[pd.DataFrame] = None,
    momentum_windows: List[int] = [5, 10, 21, 63, 252],
    volatility_windows: List[int] = [21, 63]
) -> pd.DataFrame:
    """
    Compute all alpha factors in one call.

    Parameters
    ----------
    prices : pd.DataFrame or dict
        Price data. If dict, should contain 'close', 'high', 'low', 'open'.
    fundamentals : pd.DataFrame, optional
        Fundamental data for value factors.
    financials : pd.DataFrame, optional
        Financial statement data for quality factors.
    momentum_windows : List[int], optional
        Windows for momentum computation.
    volatility_windows : List[int], optional
        Windows for volatility computation.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all computed factors.
    """
    all_factors = []

    # Momentum factors
    close_prices = prices['close'] if isinstance(prices, dict) else prices
    momentum = compute_momentum_factors(close_prices, momentum_windows)
    all_factors.append(momentum)

    # Volatility factors
    volatility = compute_volatility_factors(prices, volatility_windows)
    all_factors.append(volatility)

    # Value factors
    if fundamentals is not None:
        value = compute_value_factors(fundamentals)
        all_factors.append(value)

    # Quality factors
    if financials is not None:
        quality = compute_quality_factors(financials)
        all_factors.append(quality)

    # Merge all factors
    result = all_factors[0]
    for factor_df in all_factors[1:]:
        result = result.join(factor_df, how='outer', rsuffix='_dup')
        # Remove duplicate symbol column if it exists
        dup_cols = [col for col in result.columns if col.endswith('_dup')]
        result = result.drop(columns=dup_cols)

    return result
