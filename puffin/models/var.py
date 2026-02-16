"""
Vector Autoregression (VAR) models for multivariate time series.

This module provides VAR modeling for analyzing relationships between multiple
time series, including impulse response analysis and Granger causality testing.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Union
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests


class VARModel:
    """
    Vector Autoregression (VAR) model for multivariate time series.

    VAR models capture linear interdependencies among multiple time series.
    Each variable is modeled as a linear function of past values of itself
    and past values of all other variables.

    Parameters
    ----------
    maxlags : int, optional
        Maximum number of lags to consider for model fitting

    Attributes
    ----------
    model_ : statsmodels VAR model
        Fitted VAR model
    results_ : statsmodels VARResults
        Results from model fitting
    lags_ : int
        Number of lags used in fitted model
    """

    def __init__(self, maxlags: Optional[int] = None):
        self.maxlags = maxlags
        self.model_ = None
        self.results_ = None
        self.lags_ = None
        self.data_ = None

    def fit(
        self,
        data: pd.DataFrame,
        max_lags: Optional[int] = None,
        ic: str = 'aic',
        trend: str = 'c'
    ) -> 'VARModel':
        """
        Fit VAR model to multivariate time series data.

        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series data (each column is a variable)
        max_lags : int, optional
            Maximum number of lags to use. If None, automatically selected.
        ic : str, default 'aic'
            Information criterion for lag selection: 'aic', 'bic', 'hqic', 'fpe'
        trend : str, default 'c'
            Trend parameter: 'c' (constant), 'ct' (constant + trend), 'n' (none)

        Returns
        -------
        self
            Fitted model instance

        Examples
        --------
        >>> data = pd.DataFrame({
        ...     'returns1': np.random.randn(252),
        ...     'returns2': np.random.randn(252)
        ... })
        >>> model = VARModel()
        >>> model.fit(data, max_lags=5)
        """
        # Store data
        self.data_ = data.copy()

        # Remove any rows with NaN
        data_clean = data.dropna()

        if len(data_clean) < 20:
            raise ValueError("Need at least 20 observations to fit VAR model")

        if data_clean.shape[1] < 2:
            raise ValueError("VAR requires at least 2 variables")

        # Create VAR model
        self.model_ = VAR(data_clean)

        # Use provided max_lags or stored maxlags
        effective_max_lags = max_lags if max_lags is not None else self.maxlags

        # Fit model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            if effective_max_lags is None:
                # Auto-select lags
                self.results_ = self.model_.fit(ic=ic, trend=trend)
            else:
                self.results_ = self.model_.fit(maxlags=effective_max_lags, ic=ic, trend=trend)

        self.lags_ = self.results_.k_ar

        return self

    def predict(self, steps: int = 1) -> pd.DataFrame:
        """
        Generate forecasts for future time steps.

        Parameters
        ----------
        steps : int, default 1
            Number of steps ahead to forecast

        Returns
        -------
        pd.DataFrame
            Forecasted values for each variable

        Examples
        --------
        >>> model = VARModel()
        >>> model.fit(data)
        >>> forecast = model.predict(steps=10)
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        if steps < 1:
            raise ValueError("steps must be >= 1")

        # Get forecast
        forecast = self.results_.forecast(self.results_.endog[-self.lags_:], steps=steps)

        # Convert to DataFrame with proper column names
        forecast_df = pd.DataFrame(
            forecast,
            columns=self.data_.columns
        )

        return forecast_df

    def impulse_response(
        self,
        periods: int = 10,
        impulse: Optional[Union[str, int]] = None,
        response: Optional[Union[str, int]] = None,
        orthogonalized: bool = True
    ) -> np.ndarray:
        """
        Calculate impulse response functions.

        Shows how shocks to one variable affect other variables over time.

        Parameters
        ----------
        periods : int, default 10
            Number of periods for impulse response
        impulse : str or int, optional
            Variable to shock (column name or index). If None, all variables.
        response : str or int, optional
            Variable to measure response (column name or index). If None, all variables.
        orthogonalized : bool, default True
            Whether to use orthogonalized impulses (Cholesky decomposition)

        Returns
        -------
        np.ndarray
            Impulse response function array
            Shape: (periods+1, n_vars, n_vars) or subset based on impulse/response

        Examples
        --------
        >>> model = VARModel()
        >>> model.fit(data)
        >>> irf = model.impulse_response(periods=20)
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted first")

        # Get IRF
        if orthogonalized:
            irf = self.results_.irf(periods)
            irf_array = irf.orth_irfs
        else:
            irf = self.results_.irf(periods)
            irf_array = irf.irfs

        # Filter by impulse and response if specified
        if impulse is not None or response is not None:
            # Convert column names to indices if needed
            if isinstance(impulse, str):
                impulse_idx = list(self.data_.columns).index(impulse)
            else:
                impulse_idx = impulse

            if isinstance(response, str):
                response_idx = list(self.data_.columns).index(response)
            else:
                response_idx = response

            if impulse_idx is not None and response_idx is not None:
                irf_array = irf_array[:, response_idx, impulse_idx]
            elif impulse_idx is not None:
                irf_array = irf_array[:, :, impulse_idx]
            elif response_idx is not None:
                irf_array = irf_array[:, response_idx, :]

        return irf_array

    def forecast_error_variance_decomposition(
        self,
        periods: int = 10
    ) -> np.ndarray:
        """
        Calculate forecast error variance decomposition (FEVD).

        Shows the proportion of forecast error variance attributable to shocks
        in each variable.

        Parameters
        ----------
        periods : int, default 10
            Number of periods for FEVD

        Returns
        -------
        np.ndarray
            FEVD array (periods, n_vars, n_vars)

        Examples
        --------
        >>> model = VARModel()
        >>> model.fit(data)
        >>> fevd = model.forecast_error_variance_decomposition(periods=20)
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted first")

        fevd = self.results_.fevd(periods)
        return fevd.decomp

    def granger_causality(
        self,
        caused: Union[str, int],
        causing: Union[str, int, list],
        max_lag: int = 5
    ) -> Dict[int, Dict[str, float]]:
        """
        Test for Granger causality.

        Tests whether one variable helps predict another variable.

        Parameters
        ----------
        caused : str or int
            Variable being caused (dependent variable)
        causing : str, int, or list
            Variable(s) doing the causing (independent variable(s))
        max_lag : int, default 5
            Maximum lag to test

        Returns
        -------
        dict
            Dictionary with lag as key and test results (p-values) as values

        Examples
        --------
        >>> model = VARModel()
        >>> model.fit(data)
        >>> results = model.granger_causality('returns1', 'returns2', max_lag=5)
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted first")

        # Convert column names to indices if needed
        if isinstance(caused, str):
            caused_name = caused
            caused_idx = list(self.data_.columns).index(caused)
        else:
            caused_idx = caused
            caused_name = self.data_.columns[caused]

        if isinstance(causing, (list, tuple)):
            causing_names = [c if isinstance(c, str) else self.data_.columns[c] for c in causing]
        else:
            if isinstance(causing, str):
                causing_names = [causing]
            else:
                causing_names = [self.data_.columns[causing]]

        # Prepare data for Granger causality test
        test_data = self.data_[[caused_name] + causing_names].dropna()

        # Run test
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            results = grangercausalitytests(
                test_data,
                maxlag=max_lag,
                verbose=False
            )

        # Extract p-values
        causality_results = {}
        for lag in range(1, max_lag + 1):
            lag_results = results[lag][0]
            causality_results[lag] = {
                'ssr_ftest_pvalue': lag_results['ssr_ftest'][1],
                'ssr_chi2test_pvalue': lag_results['ssr_chi2test'][1],
                'lrtest_pvalue': lag_results['lrtest'][1],
                'params_ftest_pvalue': lag_results['params_ftest'][1]
            }

        return causality_results

    def residuals(self) -> pd.DataFrame:
        """
        Get model residuals.

        Returns
        -------
        pd.DataFrame
            Residuals for each variable

        Examples
        --------
        >>> model = VARModel()
        >>> model.fit(data)
        >>> resid = model.residuals()
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted first")

        return pd.DataFrame(
            self.results_.resid,
            columns=self.data_.columns,
            index=self.data_.index[self.lags_:]
        )

    def summary(self) -> str:
        """
        Get model summary statistics.

        Returns
        -------
        str
            Summary statistics

        Examples
        --------
        >>> model = VARModel()
        >>> model.fit(data)
        >>> print(model.summary())
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted first")

        return str(self.results_.summary())


def test_granger_causality_matrix(
    data: pd.DataFrame,
    max_lag: int = 5,
    significance: float = 0.05
) -> pd.DataFrame:
    """
    Test Granger causality between all pairs of variables.

    Parameters
    ----------
    data : pd.DataFrame
        Multivariate time series data
    max_lag : int, default 5
        Maximum lag to test
    significance : float, default 0.05
        Significance level for causality

    Returns
    -------
    pd.DataFrame
        Matrix showing p-values for Granger causality tests
        Row i, column j: does variable j Granger-cause variable i?

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'returns1': np.random.randn(252),
    ...     'returns2': np.random.randn(252),
    ...     'returns3': np.random.randn(252)
    ... })
    >>> causality_matrix = test_granger_causality_matrix(data, max_lag=3)
    """
    variables = data.columns
    n_vars = len(variables)

    # Initialize results matrix
    results = np.zeros((n_vars, n_vars))

    # Test each pair
    for i, caused in enumerate(variables):
        for j, causing in enumerate(variables):
            if i == j:
                results[i, j] = np.nan  # Variable doesn't cause itself
                continue

            # Prepare data
            test_data = data[[caused, causing]].dropna()

            if len(test_data) < 2 * max_lag + 1:
                results[i, j] = np.nan
                continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    gc_results = grangercausalitytests(
                        test_data,
                        maxlag=max_lag,
                        verbose=False
                    )

                # Get minimum p-value across all lags
                min_pvalue = min(
                    gc_results[lag][0]['ssr_ftest'][1]
                    for lag in range(1, max_lag + 1)
                )
                results[i, j] = min_pvalue

            except Exception:
                results[i, j] = np.nan

    # Create DataFrame
    results_df = pd.DataFrame(
        results,
        index=variables,
        columns=variables
    )

    return results_df
