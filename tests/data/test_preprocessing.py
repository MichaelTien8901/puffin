"""Tests for data preprocessing."""

import numpy as np
import pandas as pd
import pytest

from puffin.data.preprocessing import preprocess, _validate


@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-02", periods=10, freq="B")
    return pd.DataFrame(
        {
            "Open": [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            "High": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "Low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "Close": [100.5, 101.5, np.nan, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        },
        index=dates,
    )


def test_ffill_missing(sample_data):
    result = preprocess(sample_data, fill_method="ffill", remove_outliers=False)
    assert not result.isna().any().any()
    # Forward-filled value should equal previous
    assert result.loc[result.index[2], "Open"] == 101.0


def test_drop_missing(sample_data):
    result = preprocess(sample_data, fill_method="drop", remove_outliers=False)
    assert not result.isna().any().any()
    assert len(result) == 8


def test_interpolate_missing(sample_data):
    result = preprocess(sample_data, fill_method="interpolate", remove_outliers=False)
    assert not result.isna().any().any()


def test_validate_negative_prices():
    dates = pd.date_range("2023-01-02", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100, -5, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000, -100, 1200],
        },
        index=dates,
    )
    result = _validate(df)
    assert (result["Open"] >= 0).all()
    assert (result["Volume"] >= 0).all()


def test_validate_high_low_consistency():
    dates = pd.date_range("2023-01-02", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [99, 102, 103],  # First row: High < Low
            "Low": [101, 100, 101],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200],
        },
        index=dates,
    )
    result = _validate(df)
    assert (result["High"] >= result["Low"]).all()
