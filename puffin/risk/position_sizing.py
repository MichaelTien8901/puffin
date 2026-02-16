"""Position sizing algorithms for risk management."""

import numpy as np
from typing import Optional


def fixed_fractional(
    equity: float,
    risk_pct: float,
    stop_distance: float
) -> float:
    """
    Calculate position size using fixed fractional method.

    Position size = (Equity * Risk%) / Stop Distance

    Parameters
    ----------
    equity : float
        Current account equity
    risk_pct : float
        Percentage of equity to risk (e.g., 0.02 for 2%)
    stop_distance : float
        Distance from entry to stop loss in price units

    Returns
    -------
    float
        Position size (number of shares/contracts)

    Examples
    --------
    >>> fixed_fractional(100000, 0.02, 5.0)
    400.0
    """
    if equity <= 0:
        raise ValueError("Equity must be positive")
    if risk_pct <= 0 or risk_pct > 1:
        raise ValueError("Risk percentage must be between 0 and 1")
    if stop_distance <= 0:
        raise ValueError("Stop distance must be positive")

    risk_amount = equity * risk_pct
    position_size = risk_amount / stop_distance

    return position_size


def kelly_criterion(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 0.5
) -> float:
    """
    Calculate optimal position size using Kelly Criterion.

    Kelly % = (Win Rate * Win/Loss Ratio - (1 - Win Rate)) / Win/Loss Ratio

    Parameters
    ----------
    win_rate : float
        Historical win rate (e.g., 0.55 for 55%)
    win_loss_ratio : float
        Ratio of average win to average loss (e.g., 1.5)
    fraction : float, default=0.5
        Fraction of Kelly to use (0.5 = half Kelly, conservative)

    Returns
    -------
    float
        Optimal position size as fraction of equity (e.g., 0.15 for 15%)

    Examples
    --------
    >>> kelly_criterion(0.55, 1.5, 0.5)
    0.1...

    Notes
    -----
    Full Kelly can be very aggressive and lead to large drawdowns.
    Half Kelly (fraction=0.5) or quarter Kelly (fraction=0.25) is recommended.
    """
    if not 0 <= win_rate <= 1:
        raise ValueError("Win rate must be between 0 and 1")
    if win_loss_ratio <= 0:
        raise ValueError("Win/loss ratio must be positive")
    if fraction <= 0 or fraction > 1:
        raise ValueError("Fraction must be between 0 and 1")

    # Kelly formula
    kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

    # Apply fractional Kelly
    kelly_pct = kelly_pct * fraction

    # Don't allow negative positions or over-leveraging
    kelly_pct = max(0.0, min(kelly_pct, 1.0))

    return kelly_pct


def volatility_based(
    equity: float,
    atr: float,
    risk_pct: float,
    multiplier: float = 2.0
) -> float:
    """
    Calculate position size based on volatility (ATR).

    Position size = (Equity * Risk%) / (ATR * Multiplier)

    Parameters
    ----------
    equity : float
        Current account equity
    atr : float
        Average True Range (volatility measure)
    risk_pct : float
        Percentage of equity to risk (e.g., 0.02 for 2%)
    multiplier : float, default=2.0
        ATR multiplier for stop distance (e.g., 2.0 for 2x ATR)

    Returns
    -------
    float
        Position size (number of shares/contracts)

    Examples
    --------
    >>> volatility_based(100000, 3.0, 0.02, 2.0)
    333.33...

    Notes
    -----
    This method automatically adjusts position size based on volatility:
    - High volatility (large ATR) -> smaller position
    - Low volatility (small ATR) -> larger position
    """
    if equity <= 0:
        raise ValueError("Equity must be positive")
    if atr <= 0:
        raise ValueError("ATR must be positive")
    if risk_pct <= 0 or risk_pct > 1:
        raise ValueError("Risk percentage must be between 0 and 1")
    if multiplier <= 0:
        raise ValueError("Multiplier must be positive")

    risk_amount = equity * risk_pct
    stop_distance = atr * multiplier
    position_size = risk_amount / stop_distance

    return position_size
