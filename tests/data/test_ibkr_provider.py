"""Tests for IBKRDataProvider."""

from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

import pandas as pd
import pytest

from puffin.data.ibkr_provider import IBKRDataProvider


# --- Mock helpers ---


def _mock_ib():
    """Create a mock IB client."""
    ib = Mock()
    ib.isConnected.return_value = True
    ib.connect = Mock()
    ib.qualifyContracts = Mock()
    ib.reqHistoricalData = Mock(return_value=[])
    ib.reqRealTimeBars = Mock()
    ib.run = Mock()
    ib.disconnect = Mock()
    return ib


def _mock_bar(date="2026-01-15", open_=100.0, high=105.0, low=99.0, close=103.0, volume=1000):
    bar = Mock()
    bar.date = date
    bar.open = open_
    bar.high = high
    bar.low = low
    bar.close = close
    bar.volume = volume
    return bar


# --- Fixtures ---


@pytest.fixture
def provider():
    p = IBKRDataProvider(host="127.0.0.1", port=7497, client_id=10)
    p._ib = _mock_ib()
    return p


# --- Connection tests ---


class TestConnection:
    def test_init_defaults(self):
        p = IBKRDataProvider()
        assert p.host == "127.0.0.1"
        assert p.port == 4002
        assert p.client_id == 10
        assert p._ib is None

    def test_connect_reuses_existing(self, provider):
        ib = provider._connect()
        assert ib is provider._ib

    def test_connect_failure(self):
        p = IBKRDataProvider()
        with patch("puffin.data.ibkr_provider.IBKRDataProvider._connect") as mock:
            mock.side_effect = ConnectionError("Connection failed")
            with pytest.raises(ConnectionError):
                mock()


# --- Contract creation ---


class TestContractCreation:
    def test_stock_contract(self, provider):
        with patch("ib_async.Stock") as mock_stock:
            mock_stock.return_value = Mock()
            provider._make_contract("AAPL", "STK")
            mock_stock.assert_called_once_with("AAPL", "SMART", "USD")

    def test_futures_contract(self, provider):
        with patch("ib_async.Future") as mock_fut:
            mock_fut.return_value = Mock()
            provider._make_contract("ES", "FUT")
            mock_fut.assert_called_once_with("ES", exchange="CME", currency="USD")

    def test_forex_contract(self, provider):
        with patch("ib_async.Forex") as mock_fx:
            mock_fx.return_value = Mock()
            provider._make_contract("EURUSD", "CASH")
            mock_fx.assert_called_once_with("EURUSD")

    def test_unsupported_type(self, provider):
        with pytest.raises(ValueError, match="Unsupported asset type"):
            provider._make_contract("X", "BOND")


# --- fetch_historical tests ---


class TestFetchHistorical:
    def test_single_symbol(self, provider):
        mock_bars = [_mock_bar(), _mock_bar(date="2026-01-16", close=104.0)]
        provider._ib.reqHistoricalData.return_value = mock_bars

        with patch("ib_async.util") as mock_util:
            mock_util.df.return_value = pd.DataFrame({
                "date": ["2026-01-15", "2026-01-16"],
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [99.0, 100.0],
                "close": [103.0, 104.0],
                "volume": [1000, 1100],
            })
            with patch.object(provider, "_make_contract", return_value=Mock()):
                df = provider.fetch_historical("AAPL", "2026-01-15", "2026-01-16")

        assert not df.empty
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_multiple_symbols(self, provider):
        mock_bars = [_mock_bar()]
        provider._ib.reqHistoricalData.return_value = mock_bars

        with patch("ib_async.util") as mock_util:
            mock_util.df.return_value = pd.DataFrame({
                "date": ["2026-01-15"],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [103.0],
                "volume": [1000],
            })
            with patch.object(provider, "_make_contract", return_value=Mock()):
                df = provider.fetch_historical(
                    ["AAPL", "MSFT"], "2026-01-15", "2026-01-16"
                )

        assert not df.empty

    def test_empty_result(self, provider):
        provider._ib.reqHistoricalData.return_value = []

        with patch.object(provider, "_make_contract", return_value=Mock()):
            df = provider.fetch_historical("AAPL", "2026-01-15", "2026-01-16")

        assert df.empty
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_interval_mapping(self, provider):
        provider._ib.reqHistoricalData.return_value = []

        with patch.object(provider, "_make_contract", return_value=Mock()):
            for interval in ["1m", "5m", "15m", "30m", "1h", "1d", "1w"]:
                provider.fetch_historical("AAPL", "2026-01-15", "2026-01-16", interval=interval)

        assert provider._ib.reqHistoricalData.call_count == 7

    def test_forex_uses_midpoint(self, provider):
        provider._ib.reqHistoricalData.return_value = []

        with patch.object(provider, "_make_contract", return_value=Mock()):
            provider.fetch_historical(
                "EURUSD", "2026-01-15", "2026-01-16", asset_type="CASH"
            )

        call_kwargs = provider._ib.reqHistoricalData.call_args
        assert call_kwargs[1]["whatToShow"] == "MIDPOINT"

    def test_error_resilience(self, provider):
        provider._ib.reqHistoricalData.side_effect = Exception("API timeout")

        with patch.object(provider, "_make_contract", return_value=Mock()):
            with pytest.raises(Exception, match="API timeout"):
                provider.fetch_historical("AAPL", "2026-01-15", "2026-01-16")


# --- Supported assets ---


class TestSupportedAssets:
    def test_supported_assets(self, provider):
        assets = provider.get_supported_assets()
        assert "equity" in assets
        assert "futures" in assets
        assert "forex" in assets
        assert "options" in assets
        assert "etf" in assets


# --- Streaming ---


class TestStreaming:
    def test_stream_realtime(self, provider):
        mock_bars = Mock()
        mock_bars.updateEvent = Mock()
        mock_bars.updateEvent.__iadd__ = Mock(return_value=mock_bars.updateEvent)
        provider._ib.reqRealTimeBars.return_value = mock_bars

        callback = Mock()

        with patch.object(provider, "_make_contract", return_value=Mock()):
            with patch("puffin.data.ibkr_provider.threading") as mock_threading:
                mock_thread = Mock()
                mock_threading.Thread.return_value = mock_thread
                thread = provider.stream_realtime(["AAPL"], callback)

        assert thread is mock_thread
        mock_thread.start.assert_called_once()
        provider._ib.reqRealTimeBars.assert_called_once()


# --- Disconnect ---


class TestDisconnect:
    def test_disconnect(self, provider):
        provider.disconnect()
        provider._ib.disconnect.assert_called_once()

    def test_disconnect_not_connected(self, provider):
        provider._ib.isConnected.return_value = False
        provider.disconnect()
        provider._ib.disconnect.assert_not_called()

    def test_disconnect_no_client(self):
        p = IBKRDataProvider()
        p.disconnect()  # Should not raise
