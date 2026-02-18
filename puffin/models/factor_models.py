"""
Factor models for asset pricing and risk analysis.

This module provides implementations of popular factor models including
CAPM, Fama-French 3-factor and 5-factor models, and Fama-MacBeth regression.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union
import statsmodels.api as sm
from datetime import datetime
import warnings


class FamaFrenchModel:
    """
    Fama-French factor models for asset pricing.

    Supports CAPM, 3-factor, and 5-factor models using data from
    Kenneth French's data library.

    The models explain asset returns using common risk factors:
    - Mkt-RF: Market excess return
    - SMB: Small Minus Big (size factor)
    - HML: High Minus Low (value factor)
    - RMW: Robust Minus Weak (profitability factor)
    - CMA: Conservative Minus Aggressive (investment factor)
    - RF: Risk-free rate
    """

    def __init__(self):
        """Initialize Fama-French model."""
        self._factors_cache = None
        self._cache_start = None
        self._cache_end = None

    def fetch_factors(self, start: Union[str, datetime], end: Union[str, datetime]) -> pd.DataFrame:
        """
        Fetch Fama-French factors from Kenneth French data library.

        Args:
            start: Start date (YYYY-MM-DD or datetime)
            end: End date (YYYY-MM-DD or datetime)

        Returns:
            DataFrame: Factor returns with columns [Mkt-RF, SMB, HML, RMW, CMA, RF]
        """
        # Convert to datetime
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        # Check cache
        if (self._factors_cache is not None and
            self._cache_start is not None and
            self._cache_end is not None and
            start >= self._cache_start and
            end <= self._cache_end):
            return self._factors_cache.loc[start:end].copy()

        try:
            # Try to use pandas_datareader
            import pandas_datareader as pdr

            # Fetch 5-factor data (daily)
            ff5 = pdr.DataReader('F-F_Research_Data_5_Factors_2x3_daily',
                                'famafrench',
                                start=start,
                                end=end)[0]

            # Convert from percentage to decimal
            ff5 = ff5 / 100.0

            # Ensure column names are correct
            ff5.columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

            # Cache the data
            self._factors_cache = ff5
            self._cache_start = ff5.index.min()
            self._cache_end = ff5.index.max()

            return ff5

        except ImportError:
            warnings.warn(
                "pandas_datareader not installed. Generating synthetic factor data. "
                "Install with: pip install pandas-datareader"
            )
            return self._generate_synthetic_factors(start, end)

        except Exception as e:
            warnings.warn(
                f"Failed to fetch Fama-French data: {e}. "
                "Generating synthetic factor data."
            )
            return self._generate_synthetic_factors(start, end)

    def _generate_synthetic_factors(self, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Generate synthetic factor data for testing/demonstration.

        This is used when real data cannot be fetched.

        Args:
            start: Start date
            end: End date

        Returns:
            DataFrame: Synthetic factor returns
        """
        # Create date range
        dates = pd.date_range(start=start, end=end, freq='D')

        # Generate realistic factor returns (based on historical statistics)
        np.random.seed(42)
        n = len(dates)

        data = {
            'Mkt-RF': np.random.normal(0.0003, 0.01, n),  # ~8% annual, 16% vol
            'SMB': np.random.normal(0.0001, 0.005, n),    # ~2.5% annual, 8% vol
            'HML': np.random.normal(0.0001, 0.005, n),    # ~2.5% annual, 8% vol
            'RMW': np.random.normal(0.0001, 0.004, n),    # ~2.5% annual, 6% vol
            'CMA': np.random.normal(0.0001, 0.004, n),    # ~2.5% annual, 6% vol
            'RF': np.full(n, 0.00008),                     # ~2% annual risk-free rate
        }

        return pd.DataFrame(data, index=dates)

    def fit_capm(self, returns: Union[pd.Series, np.ndarray],
                 market_returns: Optional[Union[pd.Series, np.ndarray]] = None,
                 start: Optional[Union[str, datetime]] = None,
                 end: Optional[Union[str, datetime]] = None) -> Dict[str, Any]:
        """
        Fit Capital Asset Pricing Model (CAPM).

        CAPM: R_i - R_f = alpha + beta * (R_m - R_f) + epsilon

        Args:
            returns: Asset returns (should be excess returns if market_returns not provided)
            market_returns: Market returns (optional, will fetch if not provided)
            start: Start date for factor data (required if market_returns not provided)
            end: End date for factor data (required if market_returns not provided)

        Returns:
            dict: Results containing alpha, beta, r2, and statistics
        """
        # Convert to Series if array
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        # Get market factor if not provided
        if market_returns is None:
            if start is None or end is None:
                raise ValueError("start and end dates required when market_returns not provided")

            factors = self.fetch_factors(start, end)
            # Align dates
            common_idx = returns.index.intersection(factors.index)
            returns = returns.loc[common_idx]
            mkt_rf = factors.loc[common_idx, 'Mkt-RF']
            rf = factors.loc[common_idx, 'RF']
            # Calculate excess returns
            excess_returns = returns - rf
        else:
            if isinstance(market_returns, np.ndarray):
                market_returns = pd.Series(market_returns, index=returns.index)
            excess_returns = returns
            mkt_rf = market_returns

        # Fit regression
        X = sm.add_constant(mkt_rf)
        model = sm.OLS(excess_returns, X).fit()

        # Extract results
        alpha = model.params[0]
        beta = model.params[1]

        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'alpha_pvalue': model.pvalues[0],
            'beta_pvalue': model.pvalues[1],
            'alpha_tstat': model.tvalues[0],
            'beta_tstat': model.tvalues[1],
            'residuals': model.resid,
            'fitted_values': model.fittedvalues,
        }

    def fit_three_factor(self, returns: Union[pd.Series, np.ndarray],
                        start: Optional[Union[str, datetime]] = None,
                        end: Optional[Union[str, datetime]] = None,
                        factors: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Fit Fama-French 3-factor model.

        Model: R_i - R_f = alpha + beta_mkt*(R_m - R_f) + beta_smb*SMB + beta_hml*HML + epsilon

        Args:
            returns: Asset returns
            start: Start date for factor data (required if factors not provided)
            end: End date for factor data (required if factors not provided)
            factors: Pre-loaded factor data (optional)

        Returns:
            dict: Results containing alpha, betas, r2, and statistics
        """
        # Convert to Series if array
        if isinstance(returns, np.ndarray):
            if start is None or end is None:
                raise ValueError("start and end required to create date index")
            date_range = pd.date_range(start=start, end=end, periods=len(returns))
            returns = pd.Series(returns, index=date_range)

        # Get factors if not provided
        if factors is None:
            if start is None or end is None:
                start = returns.index.min()
                end = returns.index.max()
            factors = self.fetch_factors(start, end)

        # Align dates
        common_idx = returns.index.intersection(factors.index)
        returns = returns.loc[common_idx]
        factors = factors.loc[common_idx]

        # Calculate excess returns
        excess_returns = returns - factors['RF']

        # Prepare X matrix
        X = factors[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)

        # Fit regression
        model = sm.OLS(excess_returns, X).fit()

        return {
            'alpha': model.params[0],
            'beta_mkt': model.params[1],
            'beta_smb': model.params[2],
            'beta_hml': model.params[3],
            'betas': {
                'Mkt-RF': model.params[1],
                'SMB': model.params[2],
                'HML': model.params[3],
            },
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'pvalues': {
                'alpha': model.pvalues[0],
                'Mkt-RF': model.pvalues[1],
                'SMB': model.pvalues[2],
                'HML': model.pvalues[3],
            },
            'tstats': {
                'alpha': model.tvalues[0],
                'Mkt-RF': model.tvalues[1],
                'SMB': model.tvalues[2],
                'HML': model.tvalues[3],
            },
            'residuals': model.resid,
            'fitted_values': model.fittedvalues,
        }

    def fit_five_factor(self, returns: Union[pd.Series, np.ndarray],
                       start: Optional[Union[str, datetime]] = None,
                       end: Optional[Union[str, datetime]] = None,
                       factors: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Fit Fama-French 5-factor model.

        Model: R_i - R_f = alpha + beta_mkt*(R_m-R_f) + beta_smb*SMB +
                          beta_hml*HML + beta_rmw*RMW + beta_cma*CMA + epsilon

        Args:
            returns: Asset returns
            start: Start date for factor data (required if factors not provided)
            end: End date for factor data (required if factors not provided)
            factors: Pre-loaded factor data (optional)

        Returns:
            dict: Results containing alpha, betas, r2, and statistics
        """
        # Convert to Series if array
        if isinstance(returns, np.ndarray):
            if start is None or end is None:
                raise ValueError("start and end required to create date index")
            date_range = pd.date_range(start=start, end=end, periods=len(returns))
            returns = pd.Series(returns, index=date_range)

        # Get factors if not provided
        if factors is None:
            if start is None or end is None:
                start = returns.index.min()
                end = returns.index.max()
            factors = self.fetch_factors(start, end)

        # Align dates
        common_idx = returns.index.intersection(factors.index)
        returns = returns.loc[common_idx]
        factors = factors.loc[common_idx]

        # Calculate excess returns
        excess_returns = returns - factors['RF']

        # Prepare X matrix
        X = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        X = sm.add_constant(X)

        # Fit regression
        model = sm.OLS(excess_returns, X).fit()

        return {
            'alpha': model.params[0],
            'beta_mkt': model.params[1],
            'beta_smb': model.params[2],
            'beta_hml': model.params[3],
            'beta_rmw': model.params[4],
            'beta_cma': model.params[5],
            'betas': {
                'Mkt-RF': model.params[1],
                'SMB': model.params[2],
                'HML': model.params[3],
                'RMW': model.params[4],
                'CMA': model.params[5],
            },
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'pvalues': {
                'alpha': model.pvalues[0],
                'Mkt-RF': model.pvalues[1],
                'SMB': model.pvalues[2],
                'HML': model.pvalues[3],
                'RMW': model.pvalues[4],
                'CMA': model.pvalues[5],
            },
            'tstats': {
                'alpha': model.tvalues[0],
                'Mkt-RF': model.tvalues[1],
                'SMB': model.tvalues[2],
                'HML': model.tvalues[3],
                'RMW': model.tvalues[4],
                'CMA': model.tvalues[5],
            },
            'residuals': model.resid,
            'fitted_values': model.fittedvalues,
        }


def fama_macbeth(panel_data: pd.DataFrame,
                 factors: list,
                 return_col: str = 'returns',
                 time_col: str = 'date',
                 asset_col: str = 'asset') -> Dict[str, Any]:
    """
    Fama-MacBeth two-step regression procedure.

    Step 1: Time-series regression to estimate betas for each asset
    Step 2: Cross-sectional regression at each time to estimate risk premiums
    Final: Average the risk premiums across time

    This method accounts for cross-sectional correlation and provides
    robust estimates of factor risk premiums.

    Args:
        panel_data: Panel data with columns [time_col, asset_col, return_col, factor_cols]
        factors: List of factor column names
        return_col: Name of returns column
        time_col: Name of time/date column
        asset_col: Name of asset identifier column

    Returns:
        dict: Risk premiums, t-statistics, R-squared, and detailed results
    """
    # Validate inputs
    required_cols = [time_col, asset_col, return_col] + factors
    missing_cols = set(required_cols) - set(panel_data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Step 1: Time-series regression for each asset to get betas
    assets = panel_data[asset_col].unique()
    betas = {}

    for asset in assets:
        asset_data = panel_data[panel_data[asset_col] == asset].copy()
        asset_data = asset_data.sort_values(time_col)

        y = asset_data[return_col].values
        X = asset_data[factors].values
        X = sm.add_constant(X)

        # Run time-series regression
        model = sm.OLS(y, X).fit()
        betas[asset] = model.params[1:]  # Exclude constant

    # Create beta DataFrame with prefixed column names to avoid collision during merge
    beta_cols = [f'_beta_{f}' for f in factors]
    beta_df = pd.DataFrame(betas, index=factors).T
    beta_df.columns = beta_cols
    beta_df[asset_col] = beta_df.index
    beta_df = beta_df.reset_index(drop=True)

    # Also store a version with original column names for the return value
    beta_df_return = beta_df.copy()
    beta_df_return.columns = [c.replace('_beta_', '') if c.startswith('_beta_') else c
                              for c in beta_df_return.columns]

    # Step 2: Cross-sectional regression at each time period
    times = sorted(panel_data[time_col].unique())
    cross_sectional_results = []

    for t in times:
        # Get returns and betas for this time period
        t_data = panel_data[panel_data[time_col] == t].copy()
        t_data = t_data.merge(beta_df, on=asset_col, how='inner')

        if len(t_data) < len(factors) + 1:
            continue  # Skip if not enough observations

        # Run cross-sectional regression: returns ~ betas
        y = t_data[return_col].values
        X = t_data[beta_cols].values
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
            result = {
                'time': t,
                'lambda_0': model.params[0],  # Constant (zero-beta rate)
            }
            for i, factor in enumerate(factors):
                result[f'lambda_{factor}'] = model.params[i + 1]
            result['r_squared'] = model.rsquared
            cross_sectional_results.append(result)
        except:
            continue

    # Convert to DataFrame
    cs_results = pd.DataFrame(cross_sectional_results)

    if len(cs_results) == 0:
        raise ValueError("No successful cross-sectional regressions")

    # Step 3: Average risk premiums across time
    risk_premiums = {}
    t_stats = {}

    # Zero-beta rate
    lambda_0 = cs_results['lambda_0'].mean()
    lambda_0_std = cs_results['lambda_0'].std()
    lambda_0_se = lambda_0_std / np.sqrt(len(cs_results))
    t_stats['lambda_0'] = lambda_0 / lambda_0_se if lambda_0_se > 0 else np.nan
    risk_premiums['lambda_0'] = lambda_0

    # Factor risk premiums
    for factor in factors:
        col = f'lambda_{factor}'
        premium = cs_results[col].mean()
        premium_std = cs_results[col].std()
        premium_se = premium_std / np.sqrt(len(cs_results))
        t_stats[factor] = premium / premium_se if premium_se > 0 else np.nan
        risk_premiums[factor] = premium

    # Average R-squared
    avg_r2 = cs_results['r_squared'].mean()

    return {
        'risk_premiums': risk_premiums,
        't_stats': t_stats,
        'r_squared': avg_r2,
        'n_periods': len(cs_results),
        'cross_sectional_results': cs_results,
        'betas': beta_df_return,
    }
