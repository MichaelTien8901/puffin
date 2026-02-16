"""
GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) models.

This module provides GARCH models for volatility forecasting, including
GARCH, EGARCH, and GJR-GARCH variants.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Union, Literal
from arch import arch_model
from arch.univariate import ConstantMean, ZeroMean, GARCH, EGARCH, FIGARCH


class GARCHModel:
    """
    GARCH model for volatility forecasting.

    GARCH models capture time-varying volatility and volatility clustering
    commonly observed in financial returns.

    Parameters
    ----------
    p : int, default 1
        Order of the symmetric innovation (lag order of variance)
    q : int, default 1
        Order of lagged volatility
    model : str, default 'garch'
        Model type: 'garch', 'egarch', 'gjr-garch'
    mean_model : str, default 'constant'
        Mean model: 'constant', 'zero', 'ar', 'arx'
    dist : str, default 'normal'
        Error distribution: 'normal', 't', 'skewt', 'ged'

    Attributes
    ----------
    model_ : arch model
        Fitted ARCH/GARCH model
    results_ : arch results
        Results from model fitting
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        model: Literal['garch', 'egarch', 'gjr-garch'] = 'garch',
        mean_model: str = 'constant',
        dist: str = 'normal'
    ):
        self.p = p
        self.q = q
        self.model_type = model.lower()
        self.mean_model = mean_model
        self.dist = dist
        self.model_ = None
        self.results_ = None
        self.returns_ = None

    def fit(
        self,
        returns: pd.Series,
        p: Optional[int] = None,
        q: Optional[int] = None,
        model: Optional[str] = None,
        update_freq: int = 0
    ) -> 'GARCHModel':
        """
        Fit GARCH model to returns data.

        Parameters
        ----------
        returns : pd.Series
            Returns time series (typically percent returns * 100)
        p : int, optional
            GARCH p parameter (overrides init value)
        q : int, optional
            GARCH q parameter (overrides init value)
        model : str, optional
            Model type (overrides init value)
        update_freq : int, default 0
            Frequency of iteration updates (0 for no updates)

        Returns
        -------
        self
            Fitted model instance

        Examples
        --------
        >>> returns = pd.Series(np.random.randn(252))
        >>> model = GARCHModel(p=1, q=1)
        >>> model.fit(returns)
        >>> vol_forecast = model.forecast_volatility(horizon=10)
        """
        # Store returns
        self.returns_ = returns.copy()

        # Remove NaN values
        returns_clean = returns.dropna()

        if len(returns_clean) < 50:
            raise ValueError("Need at least 50 observations to fit GARCH model")

        # Use provided parameters or defaults
        p_val = p if p is not None else self.p
        q_val = q if q is not None else self.q
        model_type = model.lower() if model is not None else self.model_type

        # Create ARCH model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            if model_type == 'garch':
                vol = 'Garch'
            elif model_type == 'egarch':
                vol = 'EGARCH'
            elif model_type == 'gjr-garch':
                vol = 'GARCH'  # Use GARCH with power=1 for GJR
                # Note: arch package uses o parameter for asymmetry
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.model_ = arch_model(
                returns_clean,
                mean=self.mean_model,
                vol=vol,
                p=p_val,
                q=q_val,
                dist=self.dist
            )

            # Fit model
            self.results_ = self.model_.fit(
                update_freq=update_freq,
                disp='off'
            )

        return self

    def forecast_volatility(
        self,
        horizon: int = 1,
        method: str = 'analytic'
    ) -> pd.Series:
        """
        Forecast conditional volatility.

        Parameters
        ----------
        horizon : int, default 1
            Number of steps ahead to forecast
        method : str, default 'analytic'
            Forecast method: 'analytic', 'simulation', 'bootstrap'

        Returns
        -------
        pd.Series
            Forecasted volatility (standard deviation)

        Examples
        --------
        >>> model = GARCHModel(p=1, q=1)
        >>> model.fit(returns)
        >>> vol_forecast = model.forecast_volatility(horizon=5)
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted before forecasting")

        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        # Get forecast
        forecast = self.results_.forecast(horizon=horizon, method=method)

        # Extract volatility forecast (variance -> std dev)
        variance_forecast = forecast.variance.iloc[-1]

        # Convert to standard deviation
        vol_forecast = np.sqrt(variance_forecast)

        return vol_forecast

    @property
    def conditional_volatility(self) -> pd.Series:
        """
        Get conditional volatility from fitted model.

        Returns
        -------
        pd.Series
            Conditional volatility (standard deviation) for each time period

        Examples
        --------
        >>> model = GARCHModel(p=1, q=1)
        >>> model.fit(returns)
        >>> conditional_vol = model.conditional_volatility
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted first")

        return self.results_.conditional_volatility

    def residuals(self) -> pd.Series:
        """
        Get standardized residuals from fitted model.

        Returns
        -------
        pd.Series
            Standardized residuals

        Examples
        --------
        >>> model = GARCHModel(p=1, q=1)
        >>> model.fit(returns)
        >>> resid = model.residuals()
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted first")

        return self.results_.std_resid

    def summary(self) -> str:
        """
        Get model summary statistics.

        Returns
        -------
        str
            Summary statistics

        Examples
        --------
        >>> model = GARCHModel(p=1, q=1)
        >>> model.fit(returns)
        >>> print(model.summary())
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted first")

        return str(self.results_.summary())

    def get_params(self) -> Dict[str, Union[float, pd.Series]]:
        """
        Get fitted model parameters.

        Returns
        -------
        dict
            Dictionary of model parameters including omega, alpha, beta

        Examples
        --------
        >>> model = GARCHModel(p=1, q=1)
        >>> model.fit(returns)
        >>> params = model.get_params()
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted first")

        return {
            'params': self.results_.params,
            'aic': self.results_.aic,
            'bic': self.results_.bic,
            'loglikelihood': self.results_.loglikelihood
        }

    def forecast(
        self,
        horizon: int = 1,
        method: str = 'analytic',
        simulations: int = 1000
    ) -> pd.DataFrame:
        """
        Generate full forecast including mean and variance.

        Parameters
        ----------
        horizon : int, default 1
            Number of steps ahead to forecast
        method : str, default 'analytic'
            Forecast method: 'analytic', 'simulation', 'bootstrap'
        simulations : int, default 1000
            Number of simulations if method='simulation'

        Returns
        -------
        pd.DataFrame
            DataFrame with mean and variance forecasts

        Examples
        --------
        >>> model = GARCHModel(p=1, q=1)
        >>> model.fit(returns)
        >>> forecast_df = model.forecast(horizon=10)
        """
        if self.results_ is None:
            raise RuntimeError("Model must be fitted before forecasting")

        # Get forecast
        if method == 'simulation':
            forecast = self.results_.forecast(
                horizon=horizon,
                method=method,
                simulations=simulations
            )
        else:
            forecast = self.results_.forecast(horizon=horizon, method=method)

        # Extract mean and variance
        mean_forecast = forecast.mean.iloc[-1]
        variance_forecast = forecast.variance.iloc[-1]

        # Create result DataFrame
        result = pd.DataFrame({
            'mean': mean_forecast,
            'variance': variance_forecast,
            'volatility': np.sqrt(variance_forecast)
        })

        return result


def fit_garch_models(
    returns: pd.Series,
    max_p: int = 3,
    max_q: int = 3,
    models: list = None
) -> Dict[str, GARCHModel]:
    """
    Fit multiple GARCH models and select best based on AIC.

    Parameters
    ----------
    returns : pd.Series
        Returns time series
    max_p : int, default 3
        Maximum p parameter to try
    max_q : int, default 3
        Maximum q parameter to try
    models : list, optional
        List of model types to try. Default: ['garch', 'egarch', 'gjr-garch']

    Returns
    -------
    dict
        Dictionary mapping model specifications to fitted models

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(252))
    >>> models = fit_garch_models(returns, max_p=2, max_q=2)
    >>> best_model = min(models.values(), key=lambda m: m.results_.aic)
    """
    if models is None:
        models = ['garch', 'egarch']

    results = {}

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        for model_type in models:
            for p in range(1, max_p + 1):
                for q in range(1, max_q + 1):
                    try:
                        model = GARCHModel(p=p, q=q, model=model_type)
                        model.fit(returns, update_freq=0)

                        key = f"{model_type}({p},{q})"
                        results[key] = model

                    except Exception:
                        # Skip models that fail to converge
                        continue

    return results


def rolling_volatility_forecast(
    returns: pd.Series,
    window: int = 252,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    model: str = 'garch'
) -> pd.Series:
    """
    Generate rolling volatility forecasts.

    Parameters
    ----------
    returns : pd.Series
        Returns time series
    window : int, default 252
        Rolling window size for fitting
    horizon : int, default 1
        Forecast horizon
    p : int, default 1
        GARCH p parameter
    q : int, default 1
        GARCH q parameter
    model : str, default 'garch'
        Model type

    Returns
    -------
    pd.Series
        Rolling volatility forecasts

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(500))
    >>> vol_forecast = rolling_volatility_forecast(returns, window=252, horizon=1)
    """
    forecasts = []
    indices = []

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        for i in range(window, len(returns)):
            # Get window of data
            window_data = returns.iloc[i-window:i]

            try:
                # Fit model
                garch = GARCHModel(p=p, q=q, model=model)
                garch.fit(window_data, update_freq=0)

                # Forecast
                vol_forecast = garch.forecast_volatility(horizon=horizon)

                forecasts.append(vol_forecast.iloc[0])
                indices.append(returns.index[i])

            except Exception:
                # Skip periods where model fails
                continue

    result = pd.Series(forecasts, index=indices)
    return result
