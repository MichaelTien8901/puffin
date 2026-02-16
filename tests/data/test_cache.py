"""Tests for SQLite data cache."""

import os
import tempfile

import pandas as pd
import pytest

from puffin.data.cache import DataCache


@pytest.fixture
def cache(tmp_path):
    db_path = str(tmp_path / "test.db")
    return DataCache(db_path=db_path)


@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=dates,
    )


def test_cache_miss(cache):
    result = cache.get("AAPL", "2023-01-01", "2023-12-31")
    assert result is None


def test_cache_put_and_get(cache, sample_data):
    cache.put("AAPL", sample_data)
    result = cache.get("AAPL", "2023-01-01", "2023-12-31")
    assert result is not None
    assert len(result) == 5
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_cache_different_symbols(cache, sample_data):
    cache.put("AAPL", sample_data)
    result = cache.get("MSFT", "2023-01-01", "2023-12-31")
    assert result is None


def test_cache_clear_symbol(cache, sample_data):
    cache.put("AAPL", sample_data)
    cache.put("MSFT", sample_data)
    cache.clear("AAPL")
    assert cache.get("AAPL", "2023-01-01", "2023-12-31") is None
    assert cache.get("MSFT", "2023-01-01", "2023-12-31") is not None


def test_cache_clear_all(cache, sample_data):
    cache.put("AAPL", sample_data)
    cache.put("MSFT", sample_data)
    cache.clear()
    assert cache.get("AAPL", "2023-01-01", "2023-12-31") is None
    assert cache.get("MSFT", "2023-01-01", "2023-12-31") is None
