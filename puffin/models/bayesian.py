"""
Bayesian models for algorithmic trading using PyMC.

This module provides Bayesian inference tools for trading applications:
- BayesianLinearRegression: MCMC-based linear regression with uncertainty quantification
- bayesian_sharpe: Bayesian Sharpe ratio estimation with credible intervals
- compare_strategies_bayesian: Compare multiple strategies using Bayesian methods
- BayesianPairsTrading: Dynamic hedge ratio estimation for pairs trading
"""

import warnings
from typing import Dict, Optional, Tuple, Union

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
            "PyMC is required for Bayesian models. "
            "Install with: pip install pymc arviz"
        )


class BayesianLinearRegression:
    """
    Bayesian linear regression using MCMC sampling.

    Provides uncertainty quantification for predictions through
    posterior distributions of parameters.

    Attributes:
        trace: ArviZ InferenceData object with posterior samples
        model: PyMC model object

    Example:
        >>> model = BayesianLinearRegression()
        >>> model.fit(X_train, y_train, samples=2000, tune=1000)
        >>> mean, hdi = model.predict(X_test)
        >>> stats = model.summary()
    """

    def __init__(self):
        """Initialize Bayesian linear regression model."""
        _check_pymc()
        self.trace = None
        self.model = None
        self._X_mean = None
        self._X_std = None
        self._y_mean = None
        self._y_std = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        samples: int = 2000,
        tune: int = 1000,
        **kwargs
    ) -> 'BayesianLinearRegression':
        """
        Fit Bayesian linear regression model using MCMC.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            samples: Number of posterior samples to draw
            tune: Number of tuning samples
            **kwargs: Additional arguments passed to pm.sample()

        Returns:
            self: Fitted model instance
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Standardize features for better sampling
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0) + 1e-8  # Avoid division by zero
        X_scaled = (X - self._X_mean) / self._X_std

        self._y_mean = y.mean()
        self._y_std = y.std() + 1e-8
        y_scaled = (y - self._y_mean) / self._y_std

        n_features = X_scaled.shape[1]

        # Build PyMC model
        with pm.Model() as model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=1)
            beta = pm.Normal('beta', mu=0, sigma=1, shape=n_features)
            sigma = pm.HalfNormal('sigma', sigma=1)

            # Linear model
            mu = alpha + pm.math.dot(X_scaled, beta)

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_scaled)

            # Sample from posterior
            self.trace = pm.sample(
                samples,
                tune=tune,
                return_inferencedata=True,
                progressbar=False,
                **kwargs
            )

        self.model = model
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        hdi_prob: float = 0.94
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with uncertainty quantification.

        Args:
            X: Feature matrix (n_samples, n_features)
            hdi_prob: Probability mass for highest density interval

        Returns:
            mean: Posterior mean predictions
            hdi: Tuple of (lower, upper) HDI bounds
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before prediction")

        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Standardize using training statistics
        X_scaled = (X - self._X_mean) / self._X_std

        # Extract posterior samples
        alpha_samples = self.trace.posterior['alpha'].values.flatten()
        beta_samples = self.trace.posterior['beta'].values.reshape(-1, X_scaled.shape[1])

        # Compute predictions for each posterior sample
        predictions = alpha_samples[:, None] + X_scaled @ beta_samples.T

        # Unstandardize predictions
        predictions = predictions * self._y_std + self._y_mean

        # Compute mean and HDI
        mean = predictions.mean(axis=1)
        hdi_lower = np.percentile(predictions, (1 - hdi_prob) / 2 * 100, axis=1)
        hdi_upper = np.percentile(predictions, (1 + hdi_prob) / 2 * 100, axis=1)

        return mean, (hdi_lower, hdi_upper)

    def summary(self) -> Dict[str, pd.DataFrame]:
        """
        Get summary statistics of posterior distributions.

        Returns:
            Dictionary with posterior statistics for each parameter
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before summary")

        summary_df = az.summary(self.trace)
        return summary_df.to_dict()

    def plot_posterior(self, param_name: str = None):
        """
        Plot posterior distributions.

        Args:
            param_name: Specific parameter to plot, or None for all

        Returns:
            matplotlib figure object
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before plotting")

        if param_name:
            return az.plot_posterior(self.trace, var_names=[param_name])
        else:
            return az.plot_posterior(self.trace)


def bayesian_sharpe(
    returns: Union[np.ndarray, pd.Series],
    samples: int = 5000,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate Bayesian Sharpe ratio with uncertainty quantification.

    Uses a Student's t-distribution to model returns for robustness
    to outliers and fat tails.

    Args:
        returns: Array of returns (not percentages)
        samples: Number of posterior samples
        risk_free_rate: Risk-free rate (annualized if returns are daily)

    Returns:
        Dictionary with:
            - mean: Posterior mean Sharpe ratio
            - hdi_low: Lower 94% HDI bound
            - hdi_high: Upper 94% HDI bound
            - prob_positive: Probability Sharpe > 0
            - std: Posterior standard deviation
    """
    _check_pymc()

    if isinstance(returns, pd.Series):
        returns = returns.values

    excess_returns = returns - risk_free_rate
    n = len(excess_returns)

    with pm.Model() as model:
        # Prior on mean excess return
        mu = pm.Normal('mu', mu=0, sigma=excess_returns.std() * 2)

        # Prior on standard deviation
        sigma = pm.HalfNormal('sigma', sigma=excess_returns.std() * 2)

        # Prior on degrees of freedom (nu > 2 ensures finite variance)
        nu = pm.Exponential('nu', lam=1/10) + 2

        # Likelihood: returns follow Student's t-distribution
        returns_obs = pm.StudentT('returns_obs', nu=nu, mu=mu, sigma=sigma, observed=excess_returns)

        # Sharpe ratio as deterministic variable
        sharpe = pm.Deterministic('sharpe', mu / sigma * np.sqrt(252))  # Annualized

        # Sample
        trace = pm.sample(
            samples,
            tune=1000,
            return_inferencedata=True,
            progressbar=False
        )

    # Extract Sharpe ratio samples
    sharpe_samples = trace.posterior['sharpe'].values.flatten()

    # Compute statistics
    hdi = az.hdi(trace, var_names=['sharpe'], hdi_prob=0.94)['sharpe'].values

    return {
        'mean': float(sharpe_samples.mean()),
        'std': float(sharpe_samples.std()),
        'hdi_low': float(hdi[0]),
        'hdi_high': float(hdi[1]),
        'prob_positive': float((sharpe_samples > 0).mean())
    }


def compare_strategies_bayesian(
    returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
    samples: int = 5000
) -> pd.DataFrame:
    """
    Compare multiple strategies using Bayesian Sharpe ratio estimation.

    Args:
        returns_dict: Dictionary mapping strategy names to return series
        samples: Number of posterior samples per strategy

    Returns:
        DataFrame with strategies ranked by posterior mean Sharpe ratio,
        including credible intervals and probability of positive Sharpe

    Example:
        >>> results = compare_strategies_bayesian({
        ...     'momentum': momentum_returns,
        ...     'mean_reversion': mean_rev_returns,
        ...     'ml_strategy': ml_returns
        ... })
    """
    _check_pymc()

    results = []

    for name, returns in returns_dict.items():
        try:
            sharpe_stats = bayesian_sharpe(returns, samples=samples)
            results.append({
                'strategy': name,
                'sharpe_mean': sharpe_stats['mean'],
                'sharpe_std': sharpe_stats['std'],
                'hdi_low': sharpe_stats['hdi_low'],
                'hdi_high': sharpe_stats['hdi_high'],
                'prob_positive': sharpe_stats['prob_positive']
            })
        except Exception as e:
            warnings.warn(f"Failed to compute Sharpe for {name}: {e}")

    df = pd.DataFrame(results)

    # Rank by mean Sharpe ratio
    df = df.sort_values('sharpe_mean', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df[['rank', 'strategy', 'sharpe_mean', 'sharpe_std',
               'hdi_low', 'hdi_high', 'prob_positive']]


class BayesianPairsTrading:
    """
    Bayesian pairs trading with dynamic hedge ratio estimation.

    Uses online Bayesian updating (similar to Kalman filter) to
    estimate time-varying hedge ratios for pairs trading.

    Example:
        >>> pairs = BayesianPairsTrading()
        >>> hedge_ratios = pairs.fit_dynamic_hedge(y, x, window=60)
        >>> signals = pairs.generate_signals(spread, entry_threshold=2.0)
    """

    def __init__(self):
        """Initialize Bayesian pairs trading model."""
        _check_pymc()
        self.hedge_ratios = None
        self.spread_stats = None

    def fit_dynamic_hedge(
        self,
        y: Union[np.ndarray, pd.Series],
        x: Union[np.ndarray, pd.Series],
        window: int = 60
    ) -> pd.DataFrame:
        """
        Estimate dynamic hedge ratios using rolling Bayesian regression.

        Args:
            y: Dependent price series
            x: Independent price series
            window: Rolling window size for estimation

        Returns:
            DataFrame with columns:
                - hedge_ratio_mean: Posterior mean hedge ratio
                - hedge_ratio_std: Posterior std of hedge ratio
                - spread: Calculated spread (y - hedge_ratio * x)
        """
        if isinstance(y, pd.Series):
            index = y.index
            y = y.values
        else:
            index = pd.RangeIndex(len(y))

        if isinstance(x, pd.Series):
            x = x.values

        n = len(y)
        hedge_mean = np.full(n, np.nan)
        hedge_std = np.full(n, np.nan)

        # Rolling window estimation
        for i in range(window, n):
            y_window = y[i-window:i]
            x_window = x[i-window:i]

            try:
                with pm.Model():
                    # Prior on hedge ratio (centered around OLS estimate)
                    ols_estimate = np.cov(y_window, x_window)[0, 1] / np.var(x_window)
                    beta = pm.Normal('beta', mu=ols_estimate, sigma=abs(ols_estimate) + 0.1)

                    # Intercept
                    alpha = pm.Normal('alpha', mu=0, sigma=y_window.std())

                    # Noise
                    sigma = pm.HalfNormal('sigma', sigma=y_window.std())

                    # Likelihood
                    mu = alpha + beta * x_window
                    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_window)

                    # Sample (fewer samples for speed in rolling window)
                    trace = pm.sample(
                        500,
                        tune=200,
                        return_inferencedata=True,
                        progressbar=False,
                        cores=1
                    )

                beta_samples = trace.posterior['beta'].values.flatten()
                hedge_mean[i] = beta_samples.mean()
                hedge_std[i] = beta_samples.std()

            except Exception as e:
                warnings.warn(f"Failed to estimate hedge ratio at index {i}: {e}")
                # Use previous value or OLS estimate
                if i > window:
                    hedge_mean[i] = hedge_mean[i-1]
                    hedge_std[i] = hedge_std[i-1]
                else:
                    hedge_mean[i] = np.cov(y_window, x_window)[0, 1] / np.var(x_window)
                    hedge_std[i] = 0.1

        # Calculate spread
        spread = y - hedge_mean * x

        self.hedge_ratios = pd.DataFrame({
            'hedge_ratio_mean': hedge_mean,
            'hedge_ratio_std': hedge_std,
            'spread': spread
        }, index=index)

        return self.hedge_ratios

    def generate_signals(
        self,
        spread: Optional[Union[np.ndarray, pd.Series]] = None,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ) -> pd.Series:
        """
        Generate trading signals based on spread z-score.

        Args:
            spread: Spread series (uses fitted spread if None)
            entry_threshold: Z-score threshold for entry (in std devs)
            exit_threshold: Z-score threshold for exit (in std devs)

        Returns:
            Series with signals:
                1: Long spread (short y, long x)
                -1: Short spread (long y, short x)
                0: No position
        """
        if spread is None:
            if self.hedge_ratios is None:
                raise ValueError("Must fit model or provide spread")
            spread = self.hedge_ratios['spread']

        if isinstance(spread, np.ndarray):
            spread = pd.Series(spread)

        # Calculate rolling z-score
        window = 20  # Short window for z-score
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        z_score = (spread - mean) / (std + 1e-8)

        # Generate signals
        signals = pd.Series(0, index=spread.index)

        # Entry signals
        signals[z_score > entry_threshold] = -1  # Spread too high, short it
        signals[z_score < -entry_threshold] = 1  # Spread too low, long it

        # Exit signals (mean reversion)
        signals[abs(z_score) < exit_threshold] = 0

        # Forward fill to maintain positions
        signals = signals.replace(0, np.nan).ffill().fillna(0)

        return signals
