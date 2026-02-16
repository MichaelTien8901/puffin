"""
Risk Parity Portfolio Optimization

Implements risk parity and equal risk contribution portfolio strategies.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


def risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate risk contribution of each asset to portfolio risk.

    The risk contribution of asset i is defined as:
    RC_i = w_i * (Σw)_i / σ_p
    where (Σw)_i is the i-th element of the covariance matrix times weights vector,
    and σ_p is the portfolio volatility.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix of asset returns

    Returns
    -------
    np.ndarray
        Array of risk contributions per asset (sums to portfolio variance)
    """
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

    # Marginal contribution to risk (partial derivative of portfolio variance w.r.t. weights)
    marginal_contrib = np.dot(cov_matrix, weights)

    # Risk contribution = weight * marginal contribution
    risk_contrib = weights * marginal_contrib

    return risk_contrib


def risk_parity_weights(
    returns: pd.DataFrame,
    risk_target: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate risk parity portfolio weights.

    Risk parity portfolios allocate capital such that each asset contributes
    equally to the total portfolio risk. This is also known as equal risk
    contribution (ERC).

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset (rows: time, columns: assets)
    risk_target : np.ndarray, optional
        Target risk contribution for each asset. If None, uses equal risk
        contribution (1/n for each asset).

    Returns
    -------
    np.ndarray
        Optimal risk parity weights
    """
    cov_matrix = returns.cov().values
    n_assets = len(returns.columns)

    if risk_target is None:
        risk_target = np.ones(n_assets) / n_assets

    # Initial guess (equal weights)
    x0 = np.ones(n_assets) / n_assets

    # Constraints (weights sum to 1)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    ]

    # Bounds (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Objective function: minimize sum of squared deviations from target risk contributions
    def objective(x):
        portfolio_variance = np.dot(x, np.dot(cov_matrix, x))
        if portfolio_variance == 0:
            return np.inf

        # Calculate actual risk contributions (as fractions of total risk)
        marginal_contrib = np.dot(cov_matrix, x)
        risk_contrib = x * marginal_contrib
        risk_contrib_pct = risk_contrib / portfolio_variance

        # Minimize squared deviations from target
        deviations = risk_contrib_pct - risk_target
        return np.sum(deviations ** 2)

    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


def inverse_volatility_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Calculate inverse volatility portfolio weights.

    Weights are proportional to the inverse of each asset's volatility,
    a simple heuristic for risk-based allocation.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset

    Returns
    -------
    np.ndarray
        Inverse volatility weights
    """
    volatilities = returns.std().values

    # Inverse volatility
    inv_vol = 1.0 / volatilities

    # Normalize to sum to 1
    weights = inv_vol / np.sum(inv_vol)

    return weights


def equal_risk_contribution_analytical(returns: pd.DataFrame) -> np.ndarray:
    """
    Calculate equal risk contribution weights using analytical approximation.

    This is a faster approximation that works well when correlations are low.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset

    Returns
    -------
    np.ndarray
        Approximate ERC weights
    """
    cov_matrix = returns.cov().values
    n_assets = len(returns.columns)

    # Get volatilities
    volatilities = np.sqrt(np.diag(cov_matrix))

    # Initial weights: inverse volatility
    weights = 1.0 / volatilities
    weights = weights / np.sum(weights)

    # Iterative refinement (a few iterations usually suffice)
    for _ in range(10):
        # Calculate risk contributions
        risk_contrib = risk_contribution(weights, cov_matrix)

        # Adjust weights inversely to their current risk contribution
        weights = weights / risk_contrib
        weights = weights / np.sum(weights)

    return weights


def minimum_correlation_algorithm(returns: pd.DataFrame) -> np.ndarray:
    """
    Calculate weights using the Minimum Correlation Algorithm.

    This approach finds weights that minimize the average correlation
    between the portfolio and its constituents.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset

    Returns
    -------
    np.ndarray
        Minimum correlation weights
    """
    corr_matrix = returns.corr().values
    n_assets = len(returns.columns)

    # Initial guess (equal weights)
    x0 = np.ones(n_assets) / n_assets

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    ]

    # Bounds (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Objective: minimize average correlation
    def objective(x):
        # Portfolio correlations with each asset
        portfolio_corr = np.dot(corr_matrix, x)

        # Average correlation weighted by portfolio weights
        avg_corr = np.dot(x, portfolio_corr)

        return avg_corr

    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x


def diversification_ratio(
    weights: np.ndarray,
    returns: pd.DataFrame
) -> float:
    """
    Calculate the diversification ratio of a portfolio.

    The diversification ratio is defined as:
    DR = (weighted average volatility) / (portfolio volatility)

    A higher ratio indicates better diversification.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    returns : pd.DataFrame
        Historical returns for each asset

    Returns
    -------
    float
        Diversification ratio
    """
    volatilities = returns.std().values
    cov_matrix = returns.cov().values

    # Weighted average volatility
    weighted_vol = np.dot(weights, volatilities)

    # Portfolio volatility
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

    if portfolio_vol == 0:
        return 0.0

    return weighted_vol / portfolio_vol


def maximum_diversification_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Calculate weights that maximize the diversification ratio.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset

    Returns
    -------
    np.ndarray
        Maximum diversification weights
    """
    volatilities = returns.std().values
    cov_matrix = returns.cov().values
    n_assets = len(returns.columns)

    # Initial guess (equal weights)
    x0 = np.ones(n_assets) / n_assets

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    ]

    # Bounds (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))

    # Objective: maximize diversification ratio = minimize negative DR
    def objective(x):
        weighted_vol = np.dot(x, volatilities)
        portfolio_vol = np.sqrt(np.dot(x, np.dot(cov_matrix, x)))

        if portfolio_vol == 0:
            return np.inf

        div_ratio = weighted_vol / portfolio_vol
        return -div_ratio

    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x
