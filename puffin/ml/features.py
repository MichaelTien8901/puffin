"""Feature engineering pipeline for ML trading models."""

import numpy as np
import pandas as pd


_custom_features: dict[str, callable] = {}


def register_feature(name: str):
    """Decorator to register a custom feature function."""
    def decorator(func):
        _custom_features[name] = func
        return func
    return decorator


def compute_features(
    data: pd.DataFrame,
    indicators: list[str] | None = None,
    lag_periods: list[int] | None = None,
    include_custom: bool = True,
) -> pd.DataFrame:
    """Compute features from OHLCV data.

    Args:
        data: DataFrame with OHLCV columns.
        indicators: List of indicator names to compute. None = all defaults.
        lag_periods: List of lag periods for return features.
        include_custom: Whether to include registered custom features.

    Returns:
        DataFrame with feature columns.
    """
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    features = pd.DataFrame(index=data.index)

    default_indicators = indicators or [
        "rsi", "macd", "bollinger", "atr", "volume_profile", "returns"
    ]

    if "rsi" in default_indicators:
        features["rsi_14"] = _rsi(close, 14)

    if "macd" in default_indicators:
        macd_line, signal_line, histogram = _macd(close)
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_hist"] = histogram

    if "bollinger" in default_indicators:
        upper, middle, lower = _bollinger_bands(close, 20, 2)
        features["bb_upper"] = upper
        features["bb_middle"] = middle
        features["bb_lower"] = lower
        features["bb_pct"] = (close - lower) / (upper - lower)

    if "atr" in default_indicators:
        features["atr_14"] = _atr(high, low, close, 14)

    if "volume_profile" in default_indicators:
        features["volume_sma_20"] = volume.rolling(20).mean()
        features["volume_ratio"] = volume / volume.rolling(20).mean()

    if "returns" in default_indicators:
        lags = lag_periods or [1, 5, 10, 21]
        for lag in lags:
            features[f"return_{lag}d"] = close.pct_change(lag)

    if include_custom:
        for name, func in _custom_features.items():
            features[name] = func(data)

    return features


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()
