"""Tests for portfolio risk management."""

import pytest
import pandas as pd
import numpy as np
from puffin.risk.portfolio_risk import PortfolioRiskManager, Position


class TestPortfolioRiskManager:
    """Tests for portfolio risk manager."""

    def test_check_drawdown_within_limit(self):
        """Test drawdown within limit."""
        rm = PortfolioRiskManager()
        equity = pd.Series([100, 105, 110, 108, 107])

        ok, dd = rm.check_drawdown(equity, max_dd=0.1)

        assert ok == True
        assert dd < 0.1

    def test_check_drawdown_exceeded(self):
        """Test drawdown exceeded."""
        rm = PortfolioRiskManager()
        equity = pd.Series([100, 105, 110, 95, 90])

        ok, dd = rm.check_drawdown(equity, max_dd=0.1)

        assert ok == False
        assert dd > 0.1

    def test_check_exposure_within_limit(self):
        """Test exposure within limit."""
        rm = PortfolioRiskManager()
        positions = [
            Position('AAPL', 100, 150, 15000, 0.5),
            Position('GOOGL', 50, 200, 10000, 0.33)
        ]

        ok, exposure = rm.check_exposure(positions, max_exposure=1.0)

        assert ok == True
        assert exposure <= 1.0

    def test_check_exposure_exceeded(self):
        """Test exposure exceeded."""
        rm = PortfolioRiskManager()
        positions = [
            Position('AAPL', 100, 150, 15000, 0.6),
            Position('GOOGL', 50, 200, 10000, 0.4),
            Position('MSFT', 30, 200, 6000, 0.24)
        ]

        ok, exposure = rm.check_exposure(positions, max_exposure=1.0)

        # Total exposure > 100% of portfolio
        assert exposure > 1.0

    def test_circuit_breaker_not_triggered(self):
        """Test circuit breaker not triggered."""
        rm = PortfolioRiskManager()
        equity = pd.Series([100, 105, 110, 108, 107])

        triggered = rm.circuit_breaker(equity, threshold=0.2)

        assert triggered == False
        assert rm.trading_halted == False

    def test_circuit_breaker_triggered(self):
        """Test circuit breaker triggered."""
        rm = PortfolioRiskManager()
        equity = pd.Series([100, 105, 110, 85, 80])

        triggered = rm.circuit_breaker(equity, threshold=0.2)

        assert triggered == True
        assert rm.trading_halted == True

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset."""
        rm = PortfolioRiskManager()
        equity = pd.Series([100, 105, 110, 85, 80])

        rm.circuit_breaker(equity, threshold=0.2)
        assert rm.trading_halted == True

        rm.reset_circuit_breaker()
        assert rm.trading_halted == False

    def test_compute_var_historical(self):
        """Test historical VaR calculation."""
        rm = PortfolioRiskManager()
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)

        var = rm.compute_var(returns, confidence=0.95, method='historical')

        # VaR should be negative (loss)
        assert var < 0

        # About 5% of returns should be worse than VaR
        worse_count = (returns <= var).sum()
        assert 40 <= worse_count <= 60  # Around 50 out of 1000

    def test_compute_var_parametric(self):
        """Test parametric VaR calculation."""
        rm = PortfolioRiskManager()
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)

        var = rm.compute_var(returns, confidence=0.95, method='parametric')

        # VaR should be negative (loss)
        assert var < 0

    def test_compute_expected_shortfall(self):
        """Test expected shortfall calculation."""
        rm = PortfolioRiskManager()
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)

        var = rm.compute_var(returns, confidence=0.95, method='historical')
        es = rm.compute_expected_shortfall(returns, confidence=0.95)

        # ES should be worse (more negative) than VaR
        assert es < var

    def test_concentration_metrics(self):
        """Test concentration metrics."""
        rm = PortfolioRiskManager()
        positions = [
            Position('AAPL', 100, 150, 15000, 0.5),
            Position('GOOGL', 50, 200, 10000, 0.33),
            Position('MSFT', 30, 200, 5100, 0.17)
        ]

        metrics = rm.concentration_metrics(positions)

        assert 'hhi' in metrics
        assert 'max_weight' in metrics
        assert 'top5_weight' in metrics
        assert 'num_positions' in metrics

        assert metrics['num_positions'] == 3
        assert 0 < metrics['hhi'] <= 1
        assert metrics['max_weight'] > 0
        assert metrics['top5_weight'] <= 1

    def test_concentration_empty_portfolio(self):
        """Test concentration with empty portfolio."""
        rm = PortfolioRiskManager()
        positions = []

        metrics = rm.concentration_metrics(positions)

        assert metrics['hhi'] == 0.0
        assert metrics['max_weight'] == 0.0
        assert metrics['top5_weight'] == 0.0
        assert metrics['num_positions'] == 0

    def test_concentration_single_position(self):
        """Test concentration with single position."""
        rm = PortfolioRiskManager()
        positions = [
            Position('AAPL', 100, 150, 15000, 1.0)
        ]

        metrics = rm.concentration_metrics(positions)

        # Single position -> max concentration
        assert metrics['hhi'] == 1.0
        assert metrics['max_weight'] == 1.0
        assert metrics['num_positions'] == 1
