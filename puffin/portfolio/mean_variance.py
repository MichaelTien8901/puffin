"""
Mean-Variance Portfolio Optimization

Implements Markowitz mean-variance optimization for portfolio construction.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Dict, Tuple


class MeanVarianceOptimizer:
    """
    Mean-variance portfolio optimizer based on Markowitz portfolio theory.

    Solves portfolio optimization problems using quadratic programming to
    find optimal asset weights that maximize return for a given risk level
    or minimize risk for a given return level.
    """

    def __init__(self):
        """Initialize the mean-variance optimizer."""
        pass

    def _compute_portfolio_stats(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute portfolio return and risk (volatility).

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        mean_returns : np.ndarray
            Expected returns for each asset
        cov_matrix : np.ndarray
            Covariance matrix of returns

        Returns
        -------
        tuple of (float, float)
            Portfolio return and volatility
        """
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_risk

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns for each asset (rows: time, columns: assets)
        n_points : int, optional
            Number of points to compute on the frontier, by default 50

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: risk, return, and weights for each asset
        """
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(mean_returns)

        # Find min and max returns
        min_var_result = self.min_variance(returns)
        min_return = min_var_result['return']

        max_sharpe_result = self.max_sharpe(returns)
        max_return = max_sharpe_result['return']

        # Generate target returns
        target_returns = np.linspace(min_return, max_return * 1.5, n_points)

        frontier_data = []
        for target_return in target_returns:
            result = self.optimize(
                returns,
                target_return=target_return
            )

            if result is not None:
                row_data = {
                    'return': result['return'],
                    'risk': result['risk']
                }
                for i, asset in enumerate(returns.columns):
                    row_data[asset] = result['weights'][i]
                frontier_data.append(row_data)

        return pd.DataFrame(frontier_data)

    def optimize(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        risk_free_rate: float = 0.0
    ) -> Optional[Dict]:
        """
        Optimize portfolio for a target return or target risk.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns for each asset
        target_return : float, optional
            Target portfolio return
        target_risk : float, optional
            Target portfolio risk (volatility)
        risk_free_rate : float, optional
            Risk-free rate for Sharpe ratio, by default 0.0

        Returns
        -------
        dict or None
            Dictionary containing:
            - weights: optimal portfolio weights
            - return: portfolio expected return
            - risk: portfolio volatility
            - sharpe: Sharpe ratio
            Returns None if optimization fails
        """
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(mean_returns)

        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, mean_returns) - target_return
            })

        if target_risk is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(np.dot(x, np.dot(cov_matrix, x))) - target_risk
            })

        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Objective function (minimize variance)
        def objective(x):
            return np.dot(x, np.dot(cov_matrix, x))

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            return None

        weights = result.x
        portfolio_return, portfolio_risk = self._compute_portfolio_stats(
            weights, mean_returns, cov_matrix
        )
        sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0.0

        return {
            'weights': weights,
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe': sharpe
        }

    def max_sharpe(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0
    ) -> Dict:
        """
        Find portfolio weights that maximize the Sharpe ratio.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns for each asset
        risk_free_rate : float, optional
            Risk-free rate, by default 0.0

        Returns
        -------
        dict
            Dictionary containing:
            - weights: optimal portfolio weights
            - return: portfolio expected return
            - risk: portfolio volatility
            - sharpe: Sharpe ratio
        """
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(mean_returns)

        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]

        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Objective function (minimize negative Sharpe ratio)
        def objective(x):
            portfolio_return, portfolio_risk = self._compute_portfolio_stats(
                x, mean_returns, cov_matrix
            )
            if portfolio_risk == 0:
                return np.inf
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
            return -sharpe  # minimize negative = maximize positive

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        portfolio_return, portfolio_risk = self._compute_portfolio_stats(
            weights, mean_returns, cov_matrix
        )
        sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0.0

        return {
            'weights': weights,
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe': sharpe
        }

    def min_variance(self, returns: pd.DataFrame) -> Dict:
        """
        Find portfolio weights that minimize variance.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns for each asset

        Returns
        -------
        dict
            Dictionary containing:
            - weights: optimal portfolio weights
            - return: portfolio expected return
            - risk: portfolio volatility
            - sharpe: Sharpe ratio (with risk_free_rate=0)
        """
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(mean_returns)

        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]

        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Objective function (minimize variance)
        def objective(x):
            return np.dot(x, np.dot(cov_matrix, x))

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        portfolio_return, portfolio_risk = self._compute_portfolio_stats(
            weights, mean_returns, cov_matrix
        )
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0.0

        return {
            'weights': weights,
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe': sharpe
        }
