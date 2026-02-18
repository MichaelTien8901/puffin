"""
Factor evaluation module with Alphalens integration.

This module provides tools for evaluating alpha factors:
- Information Coefficient (IC) analysis
- Factor returns and decay
- Turnover analysis
- Comprehensive factor tearsheets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import warnings


# Try to import alphalens
try:
    import alphalens
    ALPHALENS_AVAILABLE = True
except ImportError:
    try:
        # Try alphalens-reloaded
        import alphalens_reloaded as alphalens
        ALPHALENS_AVAILABLE = True
    except ImportError:
        ALPHALENS_AVAILABLE = False
        warnings.warn(
            "Alphalens not available. Install alphalens-reloaded for full functionality: "
            "pip install alphalens-reloaded",
            ImportWarning
        )


class FactorEvaluator:
    """
    Factor evaluator with comprehensive performance metrics.

    This class provides methods to evaluate the quality and predictive
    power of alpha factors using various statistical metrics and
    visualization tools.

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        Factor values with (date, asset) MultiIndex
    prices : pd.DataFrame
        Asset prices with date index and assets as columns
    quantiles : int or sequence, optional
        Number of quantiles for factor analysis (default: 5)
    periods : list of int, optional
        Forward return periods to analyze (default: [1, 5, 21])

    Examples
    --------
    >>> factor = pd.Series([...], index=pd.MultiIndex.from_tuples([...]))
    >>> prices = pd.DataFrame([...])
    >>> evaluator = FactorEvaluator(factor, prices)
    >>> ic = evaluator.compute_ic()
    >>> tearsheet = evaluator.full_tearsheet()
    """

    def __init__(
        self,
        quantiles: int = 5,
        periods: List[int] = [1, 5, 21]
    ):
        self.quantiles = quantiles
        self.periods = periods

    def compute_ic(
        self,
        factor: pd.Series,
        returns: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.Series:
        """
        Compute Information Coefficient (IC) between factor and forward returns.

        IC measures the correlation between factor values and subsequent returns.
        Higher IC (in absolute value) indicates stronger predictive power.

        Parameters
        ----------
        factor : pd.Series
            Factor values with (date, asset) MultiIndex
        returns : pd.DataFrame
            Forward returns with date index and assets as columns
        method : str, optional
            Correlation method: 'pearson' or 'spearman' (default: 'pearson')

        Returns
        -------
        pd.Series
            Information Coefficient time series

        Examples
        --------
        >>> factor = pd.Series([0.5, -0.2, 0.8], index=multi_index)
        >>> returns = pd.DataFrame({'AAPL': [0.01, -0.02, 0.03]})
        >>> ic = evaluator.compute_ic(factor, returns)
        """
        if not isinstance(factor.index, pd.MultiIndex):
            raise ValueError("Factor must have MultiIndex (date, asset)")

        # Align factor with returns
        ic_series = []

        for date in factor.index.get_level_values(0).unique():
            factor_date = factor.xs(date, level=0)

            if date in returns.index:
                returns_date = returns.loc[date]

                # Align assets
                common_assets = factor_date.index.intersection(returns_date.index)
                if len(common_assets) > 2:
                    factor_values = factor_date.loc[common_assets]
                    return_values = returns_date.loc[common_assets]

                    if method == 'spearman':
                        ic = factor_values.corr(return_values, method='spearman')
                    else:
                        ic = factor_values.corr(return_values, method='pearson')

                    ic_series.append({'date': date, 'ic': ic})

        if not ic_series:
            return pd.Series(dtype=float)
        result = pd.DataFrame(ic_series).set_index('date')['ic']
        return result

    def compute_factor_returns(
        self,
        factor: pd.Series,
        prices: pd.DataFrame,
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute factor returns for multiple holding periods.

        Factor returns measure the returns of portfolios constructed by
        going long the top quantile and short the bottom quantile based
        on factor values.

        Parameters
        ----------
        factor : pd.Series
            Factor values with (date, asset) MultiIndex
        prices : pd.DataFrame
            Asset prices with date index and assets as columns
        periods : list of int, optional
            Forward return periods (default: use self.periods)

        Returns
        -------
        pd.DataFrame
            Factor returns for each period, with columns like 'return_1d', 'return_5d'

        Examples
        --------
        >>> factor_returns = evaluator.compute_factor_returns(factor, prices)
        """
        if periods is None:
            periods = self.periods

        if not isinstance(factor.index, pd.MultiIndex):
            raise ValueError("Factor must have MultiIndex (date, asset)")

        # Compute forward returns
        forward_returns = {}
        for period in periods:
            forward_returns[f'return_{period}d'] = prices.pct_change(period).shift(-period)

        # Compute factor returns
        results = []

        for date in factor.index.get_level_values(0).unique():
            factor_date = factor.xs(date, level=0)

            # Rank and create quantiles
            factor_quantiles = pd.qcut(
                factor_date,
                q=self.quantiles,
                labels=False,
                duplicates='drop'
            )

            row_result = {'date': date}

            for period_name, returns_df in forward_returns.items():
                if date in returns_df.index:
                    returns_date = returns_df.loc[date]

                    # Align assets
                    common_assets = factor_quantiles.index.intersection(returns_date.index)

                    if len(common_assets) > 0:
                        # Long-short return (top quantile - bottom quantile)
                        top_quantile = factor_quantiles.loc[common_assets] == (self.quantiles - 1)
                        bottom_quantile = factor_quantiles.loc[common_assets] == 0

                        if top_quantile.sum() > 0 and bottom_quantile.sum() > 0:
                            top_return = returns_date.loc[common_assets][top_quantile].mean()
                            bottom_return = returns_date.loc[common_assets][bottom_quantile].mean()
                            factor_return = top_return - bottom_return
                            row_result[period_name] = factor_return

            if len(row_result) > 1:  # Has at least date and one return
                results.append(row_result)

        if results:
            return pd.DataFrame(results).set_index('date')
        else:
            return pd.DataFrame()

    def compute_turnover(self, factor: pd.Series) -> pd.Series:
        """
        Compute factor turnover over time.

        Turnover measures how much the factor's quantile assignments change
        from one period to the next. High turnover may lead to higher
        transaction costs.

        Parameters
        ----------
        factor : pd.Series
            Factor values with (date, asset) MultiIndex

        Returns
        -------
        pd.Series
            Turnover percentage at each date

        Examples
        --------
        >>> turnover = evaluator.compute_turnover(factor)
        """
        if not isinstance(factor.index, pd.MultiIndex):
            raise ValueError("Factor must have MultiIndex (date, asset)")

        dates = sorted(factor.index.get_level_values(0).unique())
        turnover_series = []

        for i in range(1, len(dates)):
            prev_date = dates[i - 1]
            curr_date = dates[i]

            prev_factor = factor.xs(prev_date, level=0)
            curr_factor = factor.xs(curr_date, level=0)

            # Find common assets
            common_assets = prev_factor.index.intersection(curr_factor.index)

            if len(common_assets) > 2:
                # Compute quantiles
                prev_quantiles = pd.qcut(
                    prev_factor.loc[common_assets],
                    q=self.quantiles,
                    labels=False,
                    duplicates='drop'
                )
                curr_quantiles = pd.qcut(
                    curr_factor.loc[common_assets],
                    q=self.quantiles,
                    labels=False,
                    duplicates='drop'
                )

                # Compute turnover (percentage of assets that changed quantile)
                changed = (prev_quantiles != curr_quantiles).sum()
                turnover = (changed / len(common_assets)) * 100

                turnover_series.append({'date': curr_date, 'turnover': turnover})

        if turnover_series:
            return pd.DataFrame(turnover_series).set_index('date')['turnover']
        else:
            return pd.Series(dtype=float)

    def full_tearsheet(
        self,
        factor: pd.Series,
        prices: pd.DataFrame
    ) -> Dict[str, Union[pd.Series, pd.DataFrame, float]]:
        """
        Generate comprehensive factor evaluation tearsheet.

        This method computes all available metrics and organizes them
        into a dictionary for easy analysis and visualization.

        Parameters
        ----------
        factor : pd.Series
            Factor values with (date, asset) MultiIndex
        prices : pd.DataFrame
            Asset prices with date index and assets as columns

        Returns
        -------
        dict
            Dictionary containing:
            - 'ic': Information coefficient time series
            - 'ic_mean': Mean IC
            - 'ic_std': IC standard deviation
            - 'ic_ir': Information ratio (IC mean / IC std)
            - 'factor_returns': Factor returns DataFrame
            - 'mean_returns': Mean factor returns by period
            - 'turnover': Turnover time series
            - 'mean_turnover': Average turnover

        Examples
        --------
        >>> tearsheet = evaluator.full_tearsheet(factor, prices)
        >>> print(f"IC: {tearsheet['ic_mean']:.3f}")
        >>> print(f"IR: {tearsheet['ic_ir']:.3f}")
        """
        results = {}

        # Compute forward returns for IC
        forward_returns_1d = prices.pct_change(1).shift(-1)

        # Information Coefficient
        ic_pearson = self.compute_ic(factor, forward_returns_1d, method='pearson')
        ic_spearman = self.compute_ic(factor, forward_returns_1d, method='spearman')

        results['ic_pearson'] = ic_pearson
        results['ic_spearman'] = ic_spearman
        results['ic_mean'] = ic_pearson.mean()
        results['ic_std'] = ic_pearson.std()
        results['ic_ir'] = ic_pearson.mean() / (ic_pearson.std() + 1e-8)
        results['ic_spearman_mean'] = ic_spearman.mean()

        # Factor Returns
        factor_returns = self.compute_factor_returns(factor, prices)
        results['factor_returns'] = factor_returns

        if not factor_returns.empty:
            results['mean_returns'] = factor_returns.mean()
            results['std_returns'] = factor_returns.std()
            results['sharpe_ratios'] = (
                factor_returns.mean() / (factor_returns.std() + 1e-8)
            )

        # Turnover
        turnover = self.compute_turnover(factor)
        results['turnover'] = turnover
        results['mean_turnover'] = turnover.mean()

        # Summary statistics
        results['summary'] = {
            'mean_ic': results['ic_mean'],
            'ic_std': results['ic_std'],
            'information_ratio': results['ic_ir'],
            'mean_turnover': results['mean_turnover']
        }

        if not factor_returns.empty:
            results['summary']['mean_1d_return'] = factor_returns.iloc[:, 0].mean()
            results['summary']['sharpe_1d'] = results['sharpe_ratios'].iloc[0]

        return results

    def alphalens_tearsheet(
        self,
        factor: pd.Series,
        prices: pd.DataFrame,
        periods: Optional[List[int]] = None
    ):
        """
        Generate Alphalens tearsheet (requires alphalens-reloaded).

        This method uses the Alphalens library to create comprehensive
        factor analysis visualizations.

        Parameters
        ----------
        factor : pd.Series
            Factor values with (date, asset) MultiIndex
        prices : pd.DataFrame
            Asset prices with date index and assets as columns
        periods : list of int, optional
            Forward return periods (default: use self.periods)

        Returns
        -------
        None
            Displays tearsheet plots

        Raises
        ------
        ImportError
            If alphalens is not available
        """
        if not ALPHALENS_AVAILABLE:
            raise ImportError(
                "Alphalens not available. Install with: pip install alphalens-reloaded"
            )

        if periods is None:
            periods = self.periods

        # Create factor data for Alphalens
        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
            factor=factor,
            prices=prices,
            quantiles=self.quantiles,
            periods=periods
        )

        # Generate tearsheet
        alphalens.tears.create_full_tear_sheet(factor_data)


def factor_autocorrelation(factor: pd.Series, lags: int = 5) -> pd.Series:
    """
    Compute autocorrelation of factor values.

    High autocorrelation indicates that factor values persist over time,
    which may lead to lower turnover but also slower adaptation to
    changing market conditions.

    Parameters
    ----------
    factor : pd.Series
        Factor values with (date, asset) MultiIndex
    lags : int, optional
        Number of lags to compute (default: 5)

    Returns
    -------
    pd.Series
        Autocorrelation at each lag

    Examples
    --------
    >>> autocorr = factor_autocorrelation(factor, lags=10)
    """
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("Factor must have MultiIndex (date, asset)")

    # Pivot to wide format
    factor_wide = factor.unstack()

    # Compute autocorrelation for each asset and average
    autocorr_list = []

    for lag in range(1, lags + 1):
        lag_corr = []
        for asset in factor_wide.columns:
            series = factor_wide[asset].dropna()
            if len(series) > lag:
                corr = series.autocorr(lag=lag)
                if not np.isnan(corr):
                    lag_corr.append(corr)

        if lag_corr:
            autocorr_list.append(np.mean(lag_corr))
        else:
            autocorr_list.append(np.nan)

    return pd.Series(autocorr_list, index=range(1, lags + 1))


def factor_rank_autocorrelation(factor: pd.Series, lag: int = 1) -> float:
    """
    Compute rank autocorrelation of factor.

    This measures how stable the factor's ranking of assets is over time.

    Parameters
    ----------
    factor : pd.Series
        Factor values with (date, asset) MultiIndex
    lag : int, optional
        Lag period (default: 1)

    Returns
    -------
    float
        Average rank autocorrelation

    Examples
    --------
    >>> rank_autocorr = factor_rank_autocorrelation(factor, lag=1)
    """
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("Factor must have MultiIndex (date, asset)")

    dates = sorted(factor.index.get_level_values(0).unique())
    rank_corr_list = []

    for i in range(lag, len(dates)):
        prev_date = dates[i - lag]
        curr_date = dates[i]

        prev_factor = factor.xs(prev_date, level=0)
        curr_factor = factor.xs(curr_date, level=0)

        # Find common assets
        common_assets = prev_factor.index.intersection(curr_factor.index)

        if len(common_assets) > 2:
            # Rank correlation
            prev_ranks = prev_factor.loc[common_assets].rank()
            curr_ranks = curr_factor.loc[common_assets].rank()
            corr = prev_ranks.corr(curr_ranks, method='spearman')

            if not np.isnan(corr):
                rank_corr_list.append(corr)

    if rank_corr_list:
        return np.mean(rank_corr_list)
    else:
        return np.nan


def quantile_returns_analysis(
    factor: pd.Series,
    prices: pd.DataFrame,
    quantiles: int = 5,
    period: int = 1
) -> pd.DataFrame:
    """
    Analyze returns by factor quantile.

    This shows the average returns for each factor quantile, which helps
    verify that the factor has monotonic predictive power.

    Parameters
    ----------
    factor : pd.Series
        Factor values with (date, asset) MultiIndex
    prices : pd.DataFrame
        Asset prices
    quantiles : int, optional
        Number of quantiles (default: 5)
    period : int, optional
        Forward return period (default: 1)

    Returns
    -------
    pd.DataFrame
        Mean returns, std, and count for each quantile

    Examples
    --------
    >>> quantile_returns = quantile_returns_analysis(factor, prices, quantiles=5)
    """
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("Factor must have MultiIndex (date, asset)")

    # Compute forward returns
    forward_returns = prices.pct_change(period).shift(-period)

    quantile_data = []

    for date in factor.index.get_level_values(0).unique():
        factor_date = factor.xs(date, level=0)

        if date in forward_returns.index:
            returns_date = forward_returns.loc[date]

            # Align assets
            common_assets = factor_date.index.intersection(returns_date.index)

            if len(common_assets) > quantiles:
                factor_quantiles = pd.qcut(
                    factor_date.loc[common_assets],
                    q=quantiles,
                    labels=False,
                    duplicates='drop'
                )

                for q in range(quantiles):
                    mask = factor_quantiles == q
                    if mask.sum() > 0:
                        q_returns = returns_date.loc[common_assets][mask]
                        quantile_data.append({
                            'date': date,
                            'quantile': q,
                            'mean_return': q_returns.mean(),
                            'count': len(q_returns)
                        })

    if quantile_data:
        df = pd.DataFrame(quantile_data)
        summary = df.groupby('quantile').agg({
            'mean_return': ['mean', 'std'],
            'count': 'sum'
        })
        return summary
    else:
        return pd.DataFrame()
