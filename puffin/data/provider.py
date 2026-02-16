"""Abstract data provider interface."""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataProvider(ABC):
    """Base class for market data providers.

    All data providers must implement this interface so they can be
    swapped without changing consuming code.
    """

    @abstractmethod
    def fetch_historical(
        self,
        symbols: str | list[str],
        start: str | datetime,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Args:
            symbols: Ticker symbol(s) to fetch.
            start: Start date.
            end: End date (defaults to today).
            interval: Bar interval (e.g., '1m', '5m', '1h', '1d').

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume.
            MultiIndex (Date, Symbol) for multiple symbols.
        """

    @abstractmethod
    def get_supported_assets(self) -> list[str]:
        """Return list of supported asset types (e.g., ['equity', 'etf', 'crypto'])."""

    def stream_realtime(self, symbols: list[str], callback):
        """Stream real-time market data. Optional — not all providers support this.

        Args:
            symbols: Tickers to subscribe to.
            callback: Function called with each update (symbol, price, volume, timestamp).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support real-time streaming")
