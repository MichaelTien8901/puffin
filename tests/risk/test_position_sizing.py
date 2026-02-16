"""Tests for position sizing algorithms."""

import pytest
import numpy as np
from puffin.risk.position_sizing import (
    fixed_fractional,
    kelly_criterion,
    volatility_based,
)


class TestFixedFractional:
    """Tests for fixed fractional position sizing."""

    def test_basic_calculation(self):
        """Test basic fixed fractional calculation."""
        position_size = fixed_fractional(
            equity=100000,
            risk_pct=0.02,
            stop_distance=5.0
        )
        assert position_size == 400.0

    def test_higher_risk(self):
        """Test with higher risk percentage."""
        position_size = fixed_fractional(
            equity=100000,
            risk_pct=0.05,
            stop_distance=5.0
        )
        assert position_size == 1000.0

    def test_wider_stop(self):
        """Test with wider stop distance."""
        position_size = fixed_fractional(
            equity=100000,
            risk_pct=0.02,
            stop_distance=10.0
        )
        assert position_size == 200.0

    def test_invalid_equity(self):
        """Test with invalid equity."""
        with pytest.raises(ValueError):
            fixed_fractional(equity=-100, risk_pct=0.02, stop_distance=5.0)

    def test_invalid_risk_pct(self):
        """Test with invalid risk percentage."""
        with pytest.raises(ValueError):
            fixed_fractional(equity=100000, risk_pct=1.5, stop_distance=5.0)

        with pytest.raises(ValueError):
            fixed_fractional(equity=100000, risk_pct=-0.02, stop_distance=5.0)

    def test_invalid_stop_distance(self):
        """Test with invalid stop distance."""
        with pytest.raises(ValueError):
            fixed_fractional(equity=100000, risk_pct=0.02, stop_distance=-5.0)


class TestKellyCriterion:
    """Tests for Kelly Criterion position sizing."""

    def test_basic_calculation(self):
        """Test basic Kelly calculation."""
        kelly_pct = kelly_criterion(
            win_rate=0.55,
            win_loss_ratio=1.5,
            fraction=1.0
        )
        # Kelly = (0.55 * 1.5 - 0.45) / 1.5 = 0.25
        assert abs(kelly_pct - 0.25) < 0.01

    def test_half_kelly(self):
        """Test half Kelly (conservative)."""
        kelly_pct = kelly_criterion(
            win_rate=0.55,
            win_loss_ratio=1.5,
            fraction=0.5
        )
        # Half Kelly = 0.25 / 2 = 0.125
        assert abs(kelly_pct - 0.125) < 0.01

    def test_negative_kelly(self):
        """Test negative Kelly (losing strategy)."""
        kelly_pct = kelly_criterion(
            win_rate=0.40,
            win_loss_ratio=1.0,
            fraction=1.0
        )
        # Should return 0 for losing strategy
        assert kelly_pct == 0.0

    def test_invalid_win_rate(self):
        """Test with invalid win rate."""
        with pytest.raises(ValueError):
            kelly_criterion(win_rate=1.5, win_loss_ratio=1.5, fraction=0.5)

        with pytest.raises(ValueError):
            kelly_criterion(win_rate=-0.1, win_loss_ratio=1.5, fraction=0.5)

    def test_invalid_win_loss_ratio(self):
        """Test with invalid win/loss ratio."""
        with pytest.raises(ValueError):
            kelly_criterion(win_rate=0.55, win_loss_ratio=-1.0, fraction=0.5)

    def test_invalid_fraction(self):
        """Test with invalid fraction."""
        with pytest.raises(ValueError):
            kelly_criterion(win_rate=0.55, win_loss_ratio=1.5, fraction=1.5)


class TestVolatilityBased:
    """Tests for volatility-based position sizing."""

    def test_basic_calculation(self):
        """Test basic volatility-based calculation."""
        position_size = volatility_based(
            equity=100000,
            atr=3.0,
            risk_pct=0.02,
            multiplier=2.0
        )
        # Position = (100000 * 0.02) / (3.0 * 2.0) = 333.33
        assert abs(position_size - 333.33) < 0.01

    def test_high_volatility(self):
        """Test with high volatility (smaller position)."""
        position_size = volatility_based(
            equity=100000,
            atr=6.0,
            risk_pct=0.02,
            multiplier=2.0
        )
        # Position = (100000 * 0.02) / (6.0 * 2.0) = 166.67
        assert abs(position_size - 166.67) < 0.01

    def test_low_volatility(self):
        """Test with low volatility (larger position)."""
        position_size = volatility_based(
            equity=100000,
            atr=1.5,
            risk_pct=0.02,
            multiplier=2.0
        )
        # Position = (100000 * 0.02) / (1.5 * 2.0) = 666.67
        assert abs(position_size - 666.67) < 0.01

    def test_invalid_equity(self):
        """Test with invalid equity."""
        with pytest.raises(ValueError):
            volatility_based(equity=-100, atr=3.0, risk_pct=0.02, multiplier=2.0)

    def test_invalid_atr(self):
        """Test with invalid ATR."""
        with pytest.raises(ValueError):
            volatility_based(equity=100000, atr=-3.0, risk_pct=0.02, multiplier=2.0)

    def test_invalid_risk_pct(self):
        """Test with invalid risk percentage."""
        with pytest.raises(ValueError):
            volatility_based(equity=100000, atr=3.0, risk_pct=1.5, multiplier=2.0)

    def test_invalid_multiplier(self):
        """Test with invalid multiplier."""
        with pytest.raises(ValueError):
            volatility_based(equity=100000, atr=3.0, risk_pct=0.02, multiplier=-2.0)
