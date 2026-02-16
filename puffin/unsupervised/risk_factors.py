"""Data-driven risk factor extraction using PCA."""

import numpy as np
import pandas as pd
from typing import Optional
from .pca import MarketPCA


def extract_risk_factors(
    returns: pd.DataFrame,
    n_factors: int = 5
) -> pd.DataFrame:
    """Extract data-driven risk factors from asset returns using PCA.

    Args:
        returns: DataFrame of asset returns (rows=dates, cols=assets).
        n_factors: Number of risk factors to extract.

    Returns:
        DataFrame of factor returns (rows=dates, cols=factors).
    """
    pca = MarketPCA(n_components=n_factors)
    factor_returns = pca.fit_transform(returns)

    # Create interpretable factor names
    factor_names = [f"Factor_{i+1}" for i in range(n_factors)]

    return pd.DataFrame(
        factor_returns,
        index=returns.dropna().index,
        columns=factor_names
    )


def factor_exposures(
    returns: pd.DataFrame,
    factors: pd.DataFrame
) -> pd.DataFrame:
    """Compute factor exposures (loadings) for each asset.

    Args:
        returns: DataFrame of asset returns.
        factors: DataFrame of factor returns from extract_risk_factors().

    Returns:
        DataFrame of factor loadings (rows=assets, cols=factors).
    """
    returns_clean = returns.dropna()
    factors_clean = factors.loc[returns_clean.index]

    # Compute loadings using regression (beta coefficients)
    loadings = {}

    for asset in returns_clean.columns:
        asset_returns = returns_clean[asset].values.reshape(-1, 1)
        factor_values = factors_clean.values

        # Ordinary least squares: beta = (X'X)^-1 X'y
        XtX_inv = np.linalg.pinv(factor_values.T @ factor_values)
        beta = XtX_inv @ factor_values.T @ asset_returns
        loadings[asset] = beta.flatten()

    return pd.DataFrame(loadings, index=factors_clean.columns).T


def factor_attribution(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    loadings: pd.DataFrame
) -> pd.DataFrame:
    """Attribute asset returns to risk factors.

    Args:
        returns: DataFrame of asset returns.
        factors: DataFrame of factor returns.
        loadings: DataFrame of factor loadings from factor_exposures().

    Returns:
        DataFrame of factor contributions (rows=dates, cols=assets).
    """
    returns_clean = returns.dropna()
    factors_clean = factors.loc[returns_clean.index]

    # Factor contribution = factor_return * loading
    contributions = {}

    for asset in returns_clean.columns:
        asset_loadings = loadings.loc[asset]
        # Matrix multiply: (T x F) @ (F x 1) = (T x 1)
        contrib = factors_clean.values @ asset_loadings.values
        contributions[asset] = contrib

    return pd.DataFrame(contributions, index=factors_clean.index)


def specific_risk(
    returns: pd.DataFrame,
    factor_attribution: pd.DataFrame
) -> pd.Series:
    """Compute asset-specific (idiosyncratic) risk.

    Args:
        returns: DataFrame of asset returns.
        factor_attribution: DataFrame from factor_attribution().

    Returns:
        Series of specific risk (volatility) for each asset.
    """
    returns_clean = returns.dropna()
    factor_attr_clean = factor_attribution.loc[returns_clean.index]

    # Specific returns = total returns - factor returns
    specific_returns = returns_clean - factor_attr_clean

    # Specific risk = standard deviation of specific returns
    specific_risk = specific_returns.std() * np.sqrt(252)  # Annualized

    return specific_risk


def factor_variance_decomposition(
    returns: pd.DataFrame,
    n_factors: int = 5
) -> pd.DataFrame:
    """Decompose asset variance into factor and specific components.

    Args:
        returns: DataFrame of asset returns.
        n_factors: Number of risk factors.

    Returns:
        DataFrame with total, factor, and specific variance for each asset.
    """
    # Extract factors
    factors = extract_risk_factors(returns, n_factors)
    loadings = factor_exposures(returns, factors)
    factor_attr = factor_attribution(returns, factors, loadings)

    returns_clean = returns.dropna()

    # Compute variances
    total_var = returns_clean.var() * 252  # Annualized
    factor_var = factor_attr.loc[returns_clean.index].var() * 252
    specific_var = (returns_clean - factor_attr.loc[returns_clean.index]).var() * 252

    return pd.DataFrame({
        "total_variance": total_var,
        "factor_variance": factor_var,
        "specific_variance": specific_var,
        "pct_factor": factor_var / total_var * 100,
        "pct_specific": specific_var / total_var * 100
    })


def factor_mimicking_portfolio(
    returns: pd.DataFrame,
    target_factor_idx: int = 0,
    n_factors: int = 5
) -> pd.Series:
    """Create a portfolio that mimics a specific risk factor.

    Args:
        returns: DataFrame of asset returns.
        target_factor_idx: Index of target factor (0-indexed).
        n_factors: Total number of factors to extract.

    Returns:
        Series of portfolio weights that replicate the target factor.
    """
    pca = MarketPCA(n_components=n_factors)
    pca.fit(returns)

    # Get component weights
    component = pca.components[target_factor_idx]

    # Normalize to sum to 1 (long-only)
    weights_abs = np.abs(component)
    weights = weights_abs / weights_abs.sum()

    return pd.Series(weights, index=returns.columns, name=f"Factor_{target_factor_idx+1}")


def dynamic_factor_exposure(
    returns: pd.DataFrame,
    window: int = 252,
    n_factors: int = 3
) -> dict[str, pd.DataFrame]:
    """Compute rolling factor exposures over time.

    Args:
        returns: DataFrame of asset returns.
        window: Rolling window size (trading days).
        n_factors: Number of risk factors.

    Returns:
        Dictionary mapping factor names to DataFrames of rolling exposures.
    """
    returns_clean = returns.dropna()

    # Initialize storage for each factor
    exposures = {f"Factor_{i+1}": [] for i in range(n_factors)}
    dates = []

    # Rolling window
    for i in range(window, len(returns_clean)):
        window_returns = returns_clean.iloc[i-window:i]

        # Extract factors for this window
        factors = extract_risk_factors(window_returns, n_factors)
        loadings = factor_exposures(window_returns, factors)

        # Store loadings
        for j in range(n_factors):
            factor_name = f"Factor_{j+1}"
            exposures[factor_name].append(loadings[factor_name].values)

        dates.append(returns_clean.index[i])

    # Convert to DataFrames
    result = {}
    for factor_name, exposure_list in exposures.items():
        result[factor_name] = pd.DataFrame(
            exposure_list,
            index=dates,
            columns=returns.columns
        )

    return result


def factor_timing_signal(
    returns: pd.DataFrame,
    factor_idx: int = 0,
    n_factors: int = 5,
    lookback: int = 21
) -> pd.Series:
    """Generate timing signal based on factor momentum.

    Args:
        returns: DataFrame of asset returns.
        factor_idx: Index of factor to time (0-indexed).
        n_factors: Total number of factors.
        lookback: Lookback period for momentum calculation.

    Returns:
        Series of timing signals (+1 = long factor, -1 = short factor, 0 = neutral).
    """
    factors = extract_risk_factors(returns, n_factors)
    target_factor = factors.iloc[:, factor_idx]

    # Compute momentum
    momentum = target_factor.rolling(lookback).mean()

    # Generate signals
    signals = pd.Series(0, index=momentum.index)
    signals[momentum > 0] = 1
    signals[momentum < 0] = -1

    return signals
