"""Tests for ML feature engineering."""

import numpy as np
import pandas as pd
import pytest

from puffin.ml.features import compute_features, register_feature


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame(
        {
            "Open": prices - 0.2,
            "High": prices + abs(np.random.randn(100)),
            "Low": prices - abs(np.random.randn(100)),
            "Close": prices,
            "Volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )


def test_compute_all_features(sample_data):
    features = compute_features(sample_data)
    assert "rsi_14" in features.columns
    assert "macd" in features.columns
    assert "bb_upper" in features.columns
    assert "atr_14" in features.columns
    assert "volume_ratio" in features.columns
    assert "return_1d" in features.columns
    assert len(features) == len(sample_data)


def test_selective_features(sample_data):
    features = compute_features(sample_data, indicators=["rsi", "macd"])
    assert "rsi_14" in features.columns
    assert "macd" in features.columns
    assert "atr_14" not in features.columns


def test_custom_feature(sample_data):
    @register_feature("test_custom")
    def my_feature(data):
        return data["Close"].rolling(5).mean()

    features = compute_features(sample_data)
    assert "test_custom" in features.columns


def test_rsi_bounds(sample_data):
    features = compute_features(sample_data, indicators=["rsi"])
    rsi = features["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()
