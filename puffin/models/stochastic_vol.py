"""
Stochastic volatility models for trading using PyMC.

This module implements stochastic volatility (SV) models that allow
volatility to vary over time according to its own stochastic process.

The standard SV model:
    r_t = sigma_t * epsilon_t
    log(sigma_t^2) = mu + phi * (log(sigma_{t-1}^2) - mu) + eta_t

where:
    - r_t: returns at time t
    - sigma_t: volatility at time t (latent variable)
    - mu: long-run mean of log-volatility
    - phi: persistence parameter (0 < phi < 1)
    - epsilon_t, eta_t: standard normal innovations
"""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn(
        "PyMC and/or ArviZ not installed. Install with: pip install pymc arviz",
        ImportWarning
    )


def _check_pymc():
    """Check if PyMC is available and raise helpful error if not."""
    if not PYMC_AVAILABLE:
        raise ImportError(
            "PyMC is required for stochastic volatility models. "
            "Install with: pip install pymc arviz"
        )


class StochasticVolatilityModel:
    """
    Stochastic volatility model using Bayesian inference.

    Models time-varying volatility as a latent AR(1) process in log-space.
    More flexible than GARCH for capturing volatility dynamics.

    Attributes:
        trace: ArviZ InferenceData with posterior samples
        returns: Original return series used for fitting
        volatility_path: Posterior mean volatility over time
        volatility_forecast: One-step-ahead volatility forecast

    Example:
        >>> sv_model = StochasticVolatilityModel()
        >>> sv_model.fit(returns, samples=2000)
        >>> vol_path = sv_model.volatility_path
        >>> forecast = sv_model.volatility_forecast
    """

    def __init__(self):
        """Initialize stochastic volatility model."""
        _check_pymc()
        self.trace = None
        self.returns = None
        self.model = None
        self._volatility_path = None
        self._volatility_forecast = None

    def fit(
        self,
        returns: Union[np.ndarray, pd.Series],
        samples: int = 2000,
        tune: int = 1000,
        **kwargs
    ) -> 'StochasticVolatilityModel':
        """
        Fit stochastic volatility model to returns.

        Args:
            returns: Return series (not percentages)
            samples: Number of posterior samples
            tune: Number of tuning samples
            **kwargs: Additional arguments for pm.sample()

        Returns:
            self: Fitted model instance
        """
        # Convert to numpy and store
        if isinstance(returns, pd.Series):
            self.returns = returns
            returns_array = returns.values
        else:
            self.returns = pd.Series(returns)
            returns_array = returns

        # Center returns for numerical stability
        returns_centered = returns_array - returns_array.mean()
        n = len(returns_centered)

        try:
            with pm.Model() as model:
                # Prior on long-run mean of log-volatility
                # Initialized near observed variance
                log_var_init = np.log(np.var(returns_centered) + 1e-8)
                mu = pm.Normal('mu', mu=log_var_init, sigma=1)

                # Prior on persistence (should be < 1 for stationarity)
                # Use beta distribution transformed to (0, 1)
                phi = pm.Beta('phi', alpha=20, beta=1.5)  # Concentrates near 1

                # Prior on innovation std deviation
                sigma_eta = pm.HalfNormal('sigma_eta', sigma=0.5)

                # Initial log-volatility
                log_vol_init = pm.Normal('log_vol_init', mu=mu, sigma=sigma_eta / np.sqrt(1 - phi**2))

                # Define log-volatility as random walk
                log_vol_innovations = pm.Normal('log_vol_innovations', mu=0, sigma=1, shape=n-1)

                # Build log-volatility process using scan
                def transition(log_vol_prev, innovation):
                    return mu + phi * (log_vol_prev - mu) + sigma_eta * innovation

                # Compute log-volatility path
                log_vol_rest, _ = pm.scan(
                    fn=transition,
                    sequences=[log_vol_innovations],
                    outputs_info=[log_vol_init]
                )

                # Concatenate initial value with rest
                log_vol = pm.Deterministic('log_vol', pm.math.concatenate([[log_vol_init], log_vol_rest]))

                # Volatility (exp of log-vol)
                vol = pm.Deterministic('vol', pm.math.exp(log_vol / 2))

                # Likelihood: returns are normally distributed with time-varying volatility
                returns_likelihood = pm.Normal(
                    'returns',
                    mu=0,
                    sigma=vol,
                    observed=returns_centered
                )

                # Sample from posterior
                self.trace = pm.sample(
                    samples,
                    tune=tune,
                    return_inferencedata=True,
                    progressbar=False,
                    **kwargs
                )

            self.model = model

        except Exception as e:
            warnings.warn(
                f"MCMC sampling failed: {e}. Falling back to simpler estimation.",
                RuntimeWarning
            )
            # Fallback to EWMA-based volatility estimation
            self._fallback_estimation(returns_centered)
            return self

        # Extract volatility path (posterior mean)
        self._extract_volatility()

        return self

    def _fallback_estimation(self, returns: np.ndarray):
        """
        Fallback to exponentially weighted moving average if MCMC fails.

        Args:
            returns: Centered return series
        """
        # Use EWMA with reasonable halflife
        returns_series = pd.Series(returns)
        ewm_vol = returns_series.ewm(halflife=20).std()

        self._volatility_path = ewm_vol.values
        self._volatility_forecast = ewm_vol.iloc[-1]

        # Create a mock trace for consistency
        self.trace = None

    def _extract_volatility(self):
        """Extract volatility path from posterior samples."""
        if self.trace is None:
            return

        # Get posterior mean of volatility
        vol_samples = self.trace.posterior['vol'].values
        # Shape: (chains, draws, time)
        self._volatility_path = vol_samples.mean(axis=(0, 1))

        # Forecast: use posterior distribution of next-period volatility
        # Based on AR(1) process
        mu_post = self.trace.posterior['mu'].values.flatten()
        phi_post = self.trace.posterior['phi'].values.flatten()
        log_vol_last = self.trace.posterior['log_vol'].values[:, :, -1].flatten()

        # Next period log-vol = mu + phi * (current - mu)
        log_vol_forecast = mu_post + phi_post * (log_vol_last - mu_post)
        vol_forecast = np.exp(log_vol_forecast / 2)

        self._volatility_forecast = vol_forecast.mean()

    @property
    def volatility_path(self) -> Optional[np.ndarray]:
        """
        Posterior mean volatility over time.

        Returns:
            Array of volatility estimates for each time point
        """
        return self._volatility_path

    @property
    def volatility_forecast(self) -> Optional[float]:
        """
        One-step-ahead volatility forecast.

        Returns:
            Forecasted volatility for next period
        """
        return self._volatility_forecast

    def summary(self) -> pd.DataFrame:
        """
        Get summary statistics of model parameters.

        Returns:
            DataFrame with posterior statistics
        """
        if self.trace is None:
            raise ValueError("Model must be fitted with successful MCMC sampling")

        return az.summary(self.trace, var_names=['mu', 'phi', 'sigma_eta'])

    def plot_volatility(self):
        """
        Plot estimated volatility path over time.

        Returns:
            matplotlib figure
        """
        if self.volatility_path is None:
            raise ValueError("Model must be fitted first")

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))

            if isinstance(self.returns, pd.Series):
                index = self.returns.index
            else:
                index = range(len(self.volatility_path))

            ax.plot(index, self.volatility_path, label='Estimated Volatility', linewidth=2)
            ax.fill_between(
                index,
                0,
                self.volatility_path,
                alpha=0.3
            )

            ax.set_xlabel('Time')
            ax.set_ylabel('Volatility')
            ax.set_title('Stochastic Volatility Path')
            ax.legend()
            ax.grid(True, alpha=0.3)

            return fig

        except ImportError:
            warnings.warn("matplotlib not available for plotting")
            return None

    def plot_posterior(self):
        """
        Plot posterior distributions of model parameters.

        Returns:
            matplotlib figure
        """
        if self.trace is None:
            raise ValueError("Model must be fitted with successful MCMC sampling")

        return az.plot_posterior(self.trace, var_names=['mu', 'phi', 'sigma_eta'])


def estimate_volatility_regime(
    returns: Union[np.ndarray, pd.Series],
    samples: int = 1000
) -> pd.DataFrame:
    """
    Quick volatility regime estimation using stochastic volatility.

    Simplified interface for getting volatility estimates without
    storing the full model.

    Args:
        returns: Return series
        samples: Number of posterior samples (fewer for speed)

    Returns:
        DataFrame with columns:
            - volatility: Posterior mean volatility
            - vol_lower: Lower 94% HDI
            - vol_upper: Upper 94% HDI
    """
    _check_pymc()

    model = StochasticVolatilityModel()
    model.fit(returns, samples=samples, tune=500)

    if model.trace is None:
        # Fallback was used
        if isinstance(returns, pd.Series):
            index = returns.index
        else:
            index = pd.RangeIndex(len(returns))

        return pd.DataFrame({
            'volatility': model.volatility_path,
            'vol_lower': model.volatility_path * 0.8,
            'vol_upper': model.volatility_path * 1.2
        }, index=index)

    # Extract HDI from posterior
    vol_samples = model.trace.posterior['vol'].values
    vol_mean = vol_samples.mean(axis=(0, 1))
    vol_lower = np.percentile(vol_samples, 3, axis=(0, 1))
    vol_upper = np.percentile(vol_samples, 97, axis=(0, 1))

    if isinstance(returns, pd.Series):
        index = returns.index
    else:
        index = pd.RangeIndex(len(returns))

    return pd.DataFrame({
        'volatility': vol_mean,
        'vol_lower': vol_lower,
        'vol_upper': vol_upper
    }, index=index)
