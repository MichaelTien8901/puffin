"""
ARIMA (AutoRegressive Integrated Moving Average) models for time series forecasting.

This module provides ARIMA modeling capabilities with automatic order selection
and forecasting with confidence intervals.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple, Dict, Union
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from itertools import product


class ARIMAModel:
    """
    ARIMA model for time series forecasting.

    Parameters
    ----------
    order : tuple, optional
        The (p,d,q) order of the ARIMA model
    seasonal_order : tuple, optional
        The (P,D,Q,s) seasonal order of the ARIMA model
    trend : str, optional
        Trend parameter (default 'n' for no trend)

    Attributes
    ----------
    model_ : statsmodels ARIMA model
        Fitted ARIMA model
    order_ : tuple
        Order (p,d,q) used in the fitted model
    aic_ : float
        Akaike Information Criterion of fitted model
    bic_ : float
        Bayesian Information Criterion of fitted model
    """

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: str = 'n'
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model_ = None
        self.order_ = None
        self.aic_ = None
        self.bic_ = None
        self.series_ = None

    def fit(
        self,
        series: pd.Series,
        order: Optional[Tuple[int, int, int]] = None
    ) -> 'ARIMAModel':
        """
        Fit ARIMA model to time series data.

        Parameters
        ----------
        series : pd.Series
            Time series to fit
        order : tuple, optional
            ARIMA order (p,d,q). If None, uses auto-selection via AIC

        Returns
        -------
        self
            Fitted model instance

        Examples
        --------
        >>> returns = pd.Series(np.random.randn(252))
        >>> model = ARIMAModel()
        >>> model.fit(returns, order=(1, 0, 1))
        >>> predictions = model.predict(steps=10)
        """
        # Store the series
        self.series_ = series.copy()

        # Remove NaN values
        series_clean = series.dropna()

        if len(series_clean) < 10:
            raise ValueError("Need at least 10 observations to fit ARIMA model")

        # Auto-select order if not provided
        if order is None and self.order is None:
            print("Auto-selecting ARIMA order...")
            order = self.select_order(series_clean)
            print(f"Selected order: {order}")

        # Use provided order or stored order
        self.order_ = order if order is not None else self.order

        # Fit ARIMA model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            try:
                kwargs = dict(order=self.order_, trend=self.trend)
                if self.seasonal_order is not None:
                    kwargs['seasonal_order'] = self.seasonal_order
                model = ARIMA(series_clean, **kwargs)
                self.model_ = model.fit()
                self.aic_ = self.model_.aic
                self.bic_ = self.model_.bic

            except Exception as e:
                raise RuntimeError(f"Failed to fit ARIMA model: {str(e)}")

        return self

    def predict(self, steps: int = 1) -> pd.Series:
        """
        Generate point forecasts for future time steps.

        Parameters
        ----------
        steps : int, default 1
            Number of steps ahead to forecast

        Returns
        -------
        pd.Series
            Forecasted values

        Examples
        --------
        >>> model = ARIMAModel(order=(1, 0, 1))
        >>> model.fit(series)
        >>> forecast = model.predict(steps=5)
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        if steps < 1:
            raise ValueError("steps must be >= 1")

        # Get forecast
        forecast = self.model_.forecast(steps=steps)

        return forecast

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """
        Generate forecasts with confidence intervals.

        Parameters
        ----------
        series : pd.Series
            Time series to forecast
        horizon : int
            Number of steps to forecast
        confidence : float, default 0.95
            Confidence level for prediction intervals

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'forecast', 'lower', 'upper'

        Examples
        --------
        >>> model = ARIMAModel()
        >>> forecast_df = model.forecast(series, horizon=10, confidence=0.95)
        """
        # Fit model if not already fitted or if new series
        if self.model_ is None or not series.equals(self.series_):
            self.fit(series)

        # Get forecast with prediction intervals
        forecast_result = self.model_.get_forecast(steps=horizon)
        forecast_df = forecast_result.summary_frame(alpha=1 - confidence)

        # Rename columns for clarity
        result = pd.DataFrame({
            'forecast': forecast_df['mean'],
            'lower': forecast_df['mean_ci_lower'],
            'upper': forecast_df['mean_ci_upper']
        })

        return result

    def select_order(
        self,
        series: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal: bool = False,
        information_criterion: str = 'aic'
    ) -> Tuple[int, int, int]:
        """
        Select optimal ARIMA order using grid search over AIC/BIC.

        Parameters
        ----------
        series : pd.Series
            Time series for order selection
        max_p : int, default 5
            Maximum AR order to consider
        max_d : int, default 2
            Maximum integration order to consider
        max_q : int, default 5
            Maximum MA order to consider
        seasonal : bool, default False
            Whether to include seasonal component
        information_criterion : str, default 'aic'
            Criterion to minimize: 'aic' or 'bic'

        Returns
        -------
        tuple
            Optimal (p, d, q) order

        Examples
        --------
        >>> returns = pd.Series(np.random.randn(252))
        >>> model = ARIMAModel()
        >>> best_order = model.select_order(returns, max_p=3, max_d=1, max_q=3)
        """
        series_clean = series.dropna()

        if len(series_clean) < 20:
            raise ValueError("Need at least 20 observations for order selection")

        # Generate all combinations of p, d, q
        p_range = range(0, max_p + 1)
        d_range = range(0, max_d + 1)
        q_range = range(0, max_q + 1)

        best_score = np.inf
        best_order = (0, 0, 0)

        # Grid search
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            for p, d, q in product(p_range, d_range, q_range):
                # Skip if no parameters
                if p == 0 and q == 0:
                    continue

                try:
                    model = ARIMA(
                        series_clean,
                        order=(p, d, q),
                        trend=self.trend
                    )
                    fitted_model = model.fit()

                    # Get information criterion
                    if information_criterion.lower() == 'aic':
                        score = fitted_model.aic
                    elif information_criterion.lower() == 'bic':
                        score = fitted_model.bic
                    else:
                        raise ValueError(f"Unknown criterion: {information_criterion}")

                    # Update best order if better
                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)

                except Exception:
                    # Skip orders that fail to fit
                    continue

        if best_order == (0, 0, 0):
            # Fallback to simple order if nothing worked
            best_order = (1, 0, 0)

        return best_order

    def residuals(self) -> pd.Series:
        """
        Get model residuals.

        Returns
        -------
        pd.Series
            Residuals from fitted model

        Examples
        --------
        >>> model = ARIMAModel(order=(1, 0, 1))
        >>> model.fit(series)
        >>> resid = model.residuals()
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted first")

        return self.model_.resid

    def summary(self) -> str:
        """
        Get model summary statistics.

        Returns
        -------
        str
            Summary statistics

        Examples
        --------
        >>> model = ARIMAModel(order=(1, 0, 1))
        >>> model.fit(series)
        >>> print(model.summary())
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted first")

        return str(self.model_.summary())

    def get_params(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Get fitted model parameters.

        Returns
        -------
        dict
            Dictionary of model parameters

        Examples
        --------
        >>> model = ARIMAModel(order=(1, 0, 1))
        >>> model.fit(series)
        >>> params = model.get_params()
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted first")

        return {
            'params': self.model_.params,
            'order': self.order_,
            'aic': self.aic_,
            'bic': self.bic_,
            'sigma2': getattr(self.model_, 'sigma2', self.model_.mse)
        }


def auto_arima(
    series: pd.Series,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    seasonal: bool = False,
    information_criterion: str = 'aic',
    suppress_warnings: bool = True
) -> ARIMAModel:
    """
    Automatically select and fit the best ARIMA model.

    Parameters
    ----------
    series : pd.Series
        Time series to model
    max_p : int, default 5
        Maximum AR order
    max_d : int, default 2
        Maximum integration order
    max_q : int, default 5
        Maximum MA order
    seasonal : bool, default False
        Whether to use seasonal ARIMA
    information_criterion : str, default 'aic'
        Criterion for model selection
    suppress_warnings : bool, default True
        Whether to suppress convergence warnings

    Returns
    -------
    ARIMAModel
        Fitted ARIMA model with optimal order

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(252))
    >>> model = auto_arima(returns, max_p=3, max_d=1, max_q=3)
    >>> forecast = model.predict(steps=10)
    """
    model = ARIMAModel()

    # Select best order
    best_order = model.select_order(
        series,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        seasonal=seasonal,
        information_criterion=information_criterion
    )

    # Fit with best order
    model.order = best_order
    model.fit(series, order=best_order)

    return model
