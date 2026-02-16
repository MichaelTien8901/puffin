"""Alpaca data provider for historical and real-time market data."""

import os
import time
import threading
from datetime import datetime

import pandas as pd

from puffin.data.provider import DataProvider


class AlpacaProvider(DataProvider):
    """Data provider using Alpaca Markets API.

    Supports historical data and real-time WebSocket streaming.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url = base_url or os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
        self._client = None
        self._stream = None

    def _get_client(self):
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self._client

    def fetch_historical(
        self,
        symbols: str | list[str],
        start: str | datetime,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        if isinstance(symbols, str):
            symbols = [symbols]

        timeframe_map = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame(5, "Min"),
            "15m": TimeFrame(15, "Min"),
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day,
        }
        tf = timeframe_map.get(interval, TimeFrame.Day)

        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
        )

        client = self._get_client()
        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        df.index.names = ["Symbol", "Date"]
        df = df.swaplevel().sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_supported_assets(self) -> list[str]:
        return ["equity", "etf"]

    def stream_realtime(self, symbols: list[str], callback):
        """Stream real-time data via Alpaca WebSocket with auto-reconnect."""
        from alpaca.data.live import StockDataStream

        max_retries = 5
        retry_delay = 5

        def _run_stream():
            for attempt in range(max_retries):
                try:
                    self._stream = StockDataStream(self.api_key, self.secret_key)

                    async def handle_bar(bar):
                        callback(bar.symbol, float(bar.close), int(bar.volume), bar.timestamp)

                    self._stream.subscribe_bars(handle_bar, *symbols)
                    self._stream.run()
                except Exception:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise

        thread = threading.Thread(target=_run_stream, daemon=True)
        thread.start()
        return thread
