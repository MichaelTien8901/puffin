"""Portfolio-level risk management and monitoring."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class Position:
    """Position information for portfolio risk calculations."""
    ticker: str
    quantity: float
    current_price: float
    market_value: float
    weight: float  # Portfolio weight


class PortfolioRiskManager:
    """Comprehensive portfolio risk management system."""

    def __init__(self):
        """Initialize portfolio risk manager."""
        self.initial_equity = None
        self.peak_equity = None
        self.trading_halted = False

    def check_drawdown(
        self,
        equity_curve: pd.Series,
        max_dd: float = 0.1
    ) -> Tuple[bool, float]:
        """
        Check if current drawdown exceeds maximum allowed.

        Parameters
        ----------
        equity_curve : pd.Series
            Time series of equity values
        max_dd : float, default=0.1
            Maximum allowed drawdown (e.g., 0.1 for 10%)

        Returns
        -------
        tuple
            (ok, current_dd) where ok is True if within limits,
            current_dd is the current drawdown percentage

        Examples
        --------
        >>> equity = pd.Series([100, 105, 110, 95, 100])
        >>> rm = PortfolioRiskManager()
        >>> ok, dd = rm.check_drawdown(equity, max_dd=0.2)
        >>> ok
        True
        """
        if len(equity_curve) == 0:
            return True, 0.0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        current_dd = abs(drawdown.iloc[-1])

        # Check if within limits
        ok = current_dd <= max_dd

        return ok, current_dd

    def check_exposure(
        self,
        positions: List[Position],
        max_exposure: float
    ) -> Tuple[bool, float]:
        """
        Check if total portfolio exposure exceeds maximum allowed.

        Parameters
        ----------
        positions : list of Position
            Current positions in the portfolio
        max_exposure : float
            Maximum allowed exposure as fraction of equity (e.g., 1.0 for 100%)

        Returns
        -------
        tuple
            (ok, current_exposure) where ok is True if within limits,
            current_exposure is total exposure as fraction of equity

        Examples
        --------
        >>> positions = [
        ...     Position('AAPL', 100, 150, 15000, 0.5),
        ...     Position('GOOGL', 50, 200, 10000, 0.33)
        ... ]
        >>> rm = PortfolioRiskManager()
        >>> ok, exposure = rm.check_exposure(positions, max_exposure=1.0)
        """
        if len(positions) == 0:
            return True, 0.0

        # Calculate total exposure (sum of absolute position values)
        total_exposure = sum(abs(pos.market_value) for pos in positions)

        # Get total portfolio value (sum of weights should equal 1.0)
        # Market value = weight * portfolio_value, so portfolio_value = market_value / weight
        if positions[0].weight > 0:
            portfolio_value = positions[0].market_value / positions[0].weight
        else:
            portfolio_value = sum(pos.market_value for pos in positions)

        current_exposure = total_exposure / portfolio_value if portfolio_value > 0 else 0.0

        ok = current_exposure <= max_exposure

        return ok, current_exposure

    def circuit_breaker(
        self,
        equity_curve: pd.Series,
        threshold: float
    ) -> bool:
        """
        Check if circuit breaker should halt trading.

        Circuit breaker triggers when drawdown exceeds threshold,
        halting all trading until manually reset.

        Parameters
        ----------
        equity_curve : pd.Series
            Time series of equity values
        threshold : float
            Drawdown threshold to trigger circuit breaker (e.g., 0.15 for 15%)

        Returns
        -------
        bool
            True if circuit breaker triggered (halt trading)

        Examples
        --------
        >>> equity = pd.Series([100, 105, 110, 80, 85])
        >>> rm = PortfolioRiskManager()
        >>> rm.circuit_breaker(equity, threshold=0.20)
        True
        """
        if self.trading_halted:
            return True

        if len(equity_curve) == 0:
            return False

        # Initialize peak tracking
        if self.initial_equity is None:
            self.initial_equity = equity_curve.iloc[0]
            self.peak_equity = equity_curve.iloc[0]

        # Update peak
        current_equity = equity_curve.iloc[-1]
        self.peak_equity = max(self.peak_equity, current_equity)

        # Calculate drawdown from peak
        drawdown = (current_equity - self.peak_equity) / self.peak_equity

        # Trigger circuit breaker if threshold exceeded
        if abs(drawdown) >= threshold:
            self.trading_halted = True
            return True

        return False

    def reset_circuit_breaker(self):
        """Reset circuit breaker to resume trading."""
        self.trading_halted = False

    def compute_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Compute Value at Risk (VaR).

        Parameters
        ----------
        returns : pd.Series
            Series of portfolio returns
        confidence : float, default=0.95
            Confidence level (e.g., 0.95 for 95%)
        method : str, default='historical'
            Method to use: 'historical' or 'parametric'

        Returns
        -------
        float
            VaR value (negative value indicating potential loss)

        Examples
        --------
        >>> returns = pd.Series(np.random.randn(1000) * 0.01)
        >>> rm = PortfolioRiskManager()
        >>> var = rm.compute_var(returns, confidence=0.95)
        """
        if len(returns) == 0:
            return 0.0

        if method == 'historical':
            # Historical VaR: percentile of actual returns
            var = np.percentile(returns, (1 - confidence) * 100)

        elif method == 'parametric':
            # Parametric VaR: assumes normal distribution
            mean = returns.mean()
            std = returns.std()
            # Z-score for confidence level
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence)
            var = mean + z_score * std

        else:
            raise ValueError(f"Unknown method: {method}. Use 'historical' or 'parametric'")

        return var

    def compute_expected_shortfall(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Compute Expected Shortfall (ES) / Conditional Value at Risk (CVaR).

        ES is the expected loss given that VaR has been exceeded.

        Parameters
        ----------
        returns : pd.Series
            Series of portfolio returns
        confidence : float, default=0.95
            Confidence level (e.g., 0.95 for 95%)

        Returns
        -------
        float
            Expected shortfall value

        Examples
        --------
        >>> returns = pd.Series(np.random.randn(1000) * 0.01)
        >>> rm = PortfolioRiskManager()
        >>> es = rm.compute_expected_shortfall(returns, confidence=0.95)
        """
        if len(returns) == 0:
            return 0.0

        # Compute VaR threshold
        var_threshold = np.percentile(returns, (1 - confidence) * 100)

        # Expected shortfall is mean of returns below VaR
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return var_threshold

        es = tail_returns.mean()

        return es

    def concentration_metrics(
        self,
        positions: List[Position]
    ) -> Dict[str, float]:
        """
        Compute portfolio concentration metrics.

        Parameters
        ----------
        positions : list of Position
            Current positions in the portfolio

        Returns
        -------
        dict
            Dictionary with concentration metrics:
            - hhi: Herfindahl-Hirschman Index (0 to 1, higher = more concentrated)
            - max_weight: Maximum single position weight
            - top5_weight: Combined weight of top 5 positions

        Examples
        --------
        >>> positions = [
        ...     Position('AAPL', 100, 150, 15000, 0.5),
        ...     Position('GOOGL', 50, 200, 10000, 0.33),
        ...     Position('MSFT', 30, 200, 6000, 0.17)
        ... ]
        >>> rm = PortfolioRiskManager()
        >>> metrics = rm.concentration_metrics(positions)
        >>> metrics['hhi']  # doctest: +SKIP
        0.4028
        """
        if len(positions) == 0:
            return {
                'hhi': 0.0,
                'max_weight': 0.0,
                'top5_weight': 0.0,
                'num_positions': 0
            }

        # Get absolute weights
        weights = [abs(pos.weight) for pos in positions]

        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Herfindahl-Hirschman Index (sum of squared weights)
        hhi = sum(w ** 2 for w in weights)

        # Maximum weight
        max_weight = max(weights) if weights else 0.0

        # Top 5 weight
        sorted_weights = sorted(weights, reverse=True)
        top5_weight = sum(sorted_weights[:5])

        return {
            'hhi': hhi,
            'max_weight': max_weight,
            'top5_weight': top5_weight,
            'num_positions': len(positions)
        }
