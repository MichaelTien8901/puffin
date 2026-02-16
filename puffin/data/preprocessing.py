"""Data preprocessing and validation utilities."""

import numpy as np
import pandas as pd


def preprocess(
    data: pd.DataFrame,
    fill_method: str = "ffill",
    remove_outliers: bool = True,
    outlier_std: float = 5.0,
) -> pd.DataFrame:
    """Preprocess market data: handle missing values, outliers, and validate.

    Args:
        data: DataFrame with OHLCV columns.
        fill_method: How to handle missing values ('ffill', 'interpolate', 'drop').
        remove_outliers: Whether to clip extreme return outliers.
        outlier_std: Number of standard deviations for outlier detection.

    Returns:
        Cleaned DataFrame.
    """
    df = data.copy()
    df = _handle_missing(df, fill_method)
    if remove_outliers:
        df = _handle_outliers(df, outlier_std)
    df = _validate(df)
    return df


def _handle_missing(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "ffill":
        df = df.ffill()
    elif method == "interpolate":
        df = df.interpolate(method="time")
    elif method == "drop":
        df = df.dropna()
    # Back-fill any remaining NaN at the start
    df = df.bfill()
    return df


def _handle_outliers(df: pd.DataFrame, std_threshold: float) -> pd.DataFrame:
    if "Close" not in df.columns:
        return df

    returns = df["Close"].pct_change()
    mean = returns.mean()
    std = returns.std()

    if std == 0 or pd.isna(std):
        return df

    lower = mean - std_threshold * std
    upper = mean + std_threshold * std

    # Flag outlier returns but don't remove rows — clip instead
    outlier_mask = (returns < lower) | (returns > upper)
    if outlier_mask.any():
        clipped_returns = returns.clip(lower, upper)
        # Reconstruct prices from clipped returns
        df.loc[outlier_mask.index[1:], "Close"] = (
            df["Close"].iloc[0] * (1 + clipped_returns).cumprod()
        ).iloc[1:]

    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic validation checks."""
    # Ensure no negative prices
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Ensure High >= Low
    if "High" in df.columns and "Low" in df.columns:
        invalid = df["High"] < df["Low"]
        if invalid.any():
            df.loc[invalid, "High"] = df.loc[invalid, ["High", "Low"]].max(axis=1)
            df.loc[invalid, "Low"] = df.loc[invalid, ["High", "Low"]].min(axis=1)

    # Ensure Volume >= 0
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].clip(lower=0)

    return df


def adjust_splits(df: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """Apply split adjustments to historical data.

    Args:
        df: OHLCV DataFrame.
        splits: Series with split ratios indexed by date (e.g., 4.0 for a 4:1 split).

    Returns:
        Split-adjusted DataFrame.
    """
    adjusted = df.copy()
    cumulative_factor = 1.0

    for date in sorted(splits.index, reverse=True):
        ratio = splits[date]
        if ratio != 0 and ratio != 1:
            mask = adjusted.index < date if not isinstance(adjusted.index, pd.MultiIndex) else True
            for col in ["Open", "High", "Low", "Close"]:
                if col in adjusted.columns:
                    adjusted.loc[mask, col] = adjusted.loc[mask, col] / ratio
            if "Volume" in adjusted.columns:
                adjusted.loc[mask, "Volume"] = adjusted.loc[mask, "Volume"] * ratio

    return adjusted
