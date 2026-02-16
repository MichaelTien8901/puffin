"""Tests for market data storage."""

import pandas as pd
import pytest

from puffin.data.storage import MarketDataStore


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        },
        index=dates,
    )


@pytest.fixture
def parquet_store(tmp_path):
    return MarketDataStore(tmp_path / "parquet_store", format="parquet")


@pytest.fixture
def hdf5_store(tmp_path):
    return MarketDataStore(tmp_path / "hdf5_store", format="hdf5")


def test_save_and_load_parquet(parquet_store, sample_ohlcv):
    parquet_store.save_ohlcv("AAPL", sample_ohlcv, source="test", frequency="1d")

    loaded = parquet_store.load_ohlcv("AAPL")
    assert loaded is not None
    assert len(loaded) == 10
    assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
    pd.testing.assert_frame_equal(loaded, sample_ohlcv)


def test_save_and_load_hdf5(hdf5_store, sample_ohlcv):
    hdf5_store.save_ohlcv("AAPL", sample_ohlcv, source="test", frequency="1d")

    loaded = hdf5_store.load_ohlcv("AAPL")
    assert loaded is not None
    assert len(loaded) == 10
    assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
    pd.testing.assert_frame_equal(loaded, sample_ohlcv)


def test_load_nonexistent_symbol(parquet_store):
    result = parquet_store.load_ohlcv("NONEXISTENT")
    assert result is None


def test_list_symbols(parquet_store, sample_ohlcv):
    assert parquet_store.list_symbols() == []

    parquet_store.save_ohlcv("AAPL", sample_ohlcv, source="test", frequency="1d")
    parquet_store.save_ohlcv("MSFT", sample_ohlcv, source="test", frequency="1d")

    symbols = parquet_store.list_symbols()
    assert len(symbols) == 2
    assert "AAPL" in symbols
    assert "MSFT" in symbols


def test_delete_symbol(parquet_store, sample_ohlcv):
    parquet_store.save_ohlcv("AAPL", sample_ohlcv, source="test", frequency="1d")
    parquet_store.save_ohlcv("MSFT", sample_ohlcv, source="test", frequency="1d")

    assert "AAPL" in parquet_store.list_symbols()

    parquet_store.delete_symbol("AAPL")

    assert "AAPL" not in parquet_store.list_symbols()
    assert "MSFT" in parquet_store.list_symbols()
    assert parquet_store.load_ohlcv("AAPL") is None


def test_metadata_tracking(parquet_store, sample_ohlcv):
    parquet_store.save_ohlcv("AAPL", sample_ohlcv, source="yfinance", frequency="1d")

    metadata = parquet_store.get_metadata("AAPL")
    assert not metadata.empty
    assert metadata.iloc[0]["symbol"] == "AAPL"
    assert metadata.iloc[0]["source"] == "yfinance"
    assert metadata.iloc[0]["frequency"] == "1d"
    assert metadata.iloc[0]["rows"] == 10
    assert metadata.iloc[0]["format"] == "parquet"


def test_get_all_metadata(parquet_store, sample_ohlcv):
    parquet_store.save_ohlcv("AAPL", sample_ohlcv, source="yfinance", frequency="1d")
    parquet_store.save_ohlcv("MSFT", sample_ohlcv, source="alpaca", frequency="1h")

    metadata = parquet_store.get_metadata()
    assert len(metadata) == 2
    assert set(metadata["symbol"]) == {"AAPL", "MSFT"}
    assert set(metadata["source"]) == {"yfinance", "alpaca"}


def test_append_mode(parquet_store, sample_ohlcv):
    # Save initial data
    initial_data = sample_ohlcv.iloc[:5]
    parquet_store.save_ohlcv("AAPL", initial_data, source="test", frequency="1d")

    loaded = parquet_store.load_ohlcv("AAPL")
    assert len(loaded) == 5

    # Append more data
    new_data = sample_ohlcv.iloc[5:]
    parquet_store.save_ohlcv("AAPL", new_data, source="test", frequency="1d", append=True)

    loaded = parquet_store.load_ohlcv("AAPL")
    assert len(loaded) == 10
    pd.testing.assert_frame_equal(loaded, sample_ohlcv)


def test_append_ohlcv_convenience(parquet_store, sample_ohlcv):
    initial_data = sample_ohlcv.iloc[:5]
    parquet_store.save_ohlcv("AAPL", initial_data, source="test", frequency="1d")

    new_data = sample_ohlcv.iloc[5:]
    parquet_store.append_ohlcv("AAPL", new_data, source="test", frequency="1d")

    loaded = parquet_store.load_ohlcv("AAPL")
    assert len(loaded) == 10


def test_append_deduplication(parquet_store, sample_ohlcv):
    # Save initial data
    parquet_store.save_ohlcv("AAPL", sample_ohlcv, source="test", frequency="1d")

    # Append overlapping data (last 5 rows)
    overlap_data = sample_ohlcv.iloc[5:]
    parquet_store.append_ohlcv("AAPL", overlap_data, source="test", frequency="1d")

    loaded = parquet_store.load_ohlcv("AAPL")
    # Should still be 10 rows (deduplicated)
    assert len(loaded) == 10


def test_overwrite_mode(parquet_store, sample_ohlcv):
    # Save initial data
    parquet_store.save_ohlcv("AAPL", sample_ohlcv, source="test", frequency="1d")
    assert len(parquet_store.load_ohlcv("AAPL")) == 10

    # Overwrite with less data
    new_data = sample_ohlcv.iloc[:3]
    parquet_store.save_ohlcv("AAPL", new_data, source="test", frequency="1d", append=False)

    loaded = parquet_store.load_ohlcv("AAPL")
    assert len(loaded) == 3


def test_migrate_parquet_to_hdf5(tmp_path, sample_ohlcv):
    store = MarketDataStore(tmp_path / "store", format="parquet")
    store.save_ohlcv("AAPL", sample_ohlcv, source="test", frequency="1d")

    # Verify parquet file exists
    assert (tmp_path / "store" / "AAPL.parquet").exists()

    # Migrate to HDF5
    store.migrate_format("AAPL", "hdf5")

    # Verify HDF5 file exists and parquet is gone
    assert (tmp_path / "store" / "AAPL.h5").exists()
    assert not (tmp_path / "store" / "AAPL.parquet").exists()

    # Verify data integrity
    loaded = store.load_ohlcv("AAPL", format="hdf5")
    pd.testing.assert_frame_equal(loaded, sample_ohlcv)


def test_migrate_hdf5_to_parquet(tmp_path, sample_ohlcv):
    store = MarketDataStore(tmp_path / "store", format="hdf5")
    store.save_ohlcv("AAPL", sample_ohlcv, source="test", frequency="1d")

    # Verify HDF5 file exists
    assert (tmp_path / "store" / "AAPL.h5").exists()

    # Migrate to Parquet
    store.migrate_format("AAPL", "parquet")

    # Verify Parquet file exists and HDF5 is gone
    assert (tmp_path / "store" / "AAPL.parquet").exists()
    assert not (tmp_path / "store" / "AAPL.h5").exists()

    # Verify data integrity
    loaded = store.load_ohlcv("AAPL", format="parquet")
    pd.testing.assert_frame_equal(loaded, sample_ohlcv)


def test_migrate_nonexistent_symbol(parquet_store):
    with pytest.raises(ValueError, match="not found in storage"):
        parquet_store.migrate_format("NONEXISTENT", "hdf5")


def test_format_auto_detection(tmp_path, sample_ohlcv):
    store = MarketDataStore(tmp_path / "store", format="parquet")
    store.save_ohlcv("AAPL", sample_ohlcv, source="test", frequency="1d")

    # Load without specifying format
    loaded = store.load_ohlcv("AAPL")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, sample_ohlcv)


def test_metadata_persistence(tmp_path, sample_ohlcv):
    # Create store and save data
    store1 = MarketDataStore(tmp_path / "store", format="parquet")
    store1.save_ohlcv("AAPL", sample_ohlcv, source="test", frequency="1d")

    # Create new store instance pointing to same directory
    store2 = MarketDataStore(tmp_path / "store", format="parquet")

    # Metadata should be loaded from disk
    metadata = store2.get_metadata("AAPL")
    assert not metadata.empty
    assert metadata.iloc[0]["symbol"] == "AAPL"
