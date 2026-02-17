"""Interactive Brokers data provider using ib_async."""

import logging
import threading
from datetime import datetime

import pandas as pd

from puffin.data.provider import DataProvider

logger = logging.getLogger(__name__)


class IBKRDataProvider(DataProvider):
    """Data provider using Interactive Brokers API.

    Supports stocks, futures, forex, and options historical data
    plus real-time bar streaming.

    Requires IB Gateway or TWS running locally.
    Uses client_id=10 by default to avoid collision with IBKRBroker (client_id=1).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 10,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None

    def _connect(self):
        """Lazy-connect to IB Gateway/TWS."""
        if self._ib is not None and self._ib.isConnected():
            return self._ib

        from ib_async import IB

        self._ib = IB()
        try:
            self._ib.connect(self.host, self.port, clientId=self.client_id)
        except Exception as e:
            self._ib = None
            raise ConnectionError(
                f"Failed to connect to IB Gateway at {self.host}:{self.port}: {e}"
            )
        return self._ib

    def _make_contract(self, symbol: str, asset_type: str = "STK"):
        """Create an IB contract for data requests."""
        asset_type = asset_type.upper()

        if asset_type == "STK":
            from ib_async import Stock
            return Stock(symbol, "SMART", "USD")
        elif asset_type == "FUT":
            from ib_async import Future
            return Future(symbol, exchange="CME", currency="USD")
        elif asset_type == "CASH":
            from ib_async import Forex
            return Forex(symbol)
        elif asset_type == "OPT":
            from ib_async import Stock
            # For options data, use underlying stock contract
            return Stock(symbol, "SMART", "USD")
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")

    def fetch_historical(
        self,
        symbols: str | list[str],
        start: str | datetime,
        end: str | datetime | None = None,
        interval: str = "1d",
        asset_type: str = "STK",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from IB.

        Args:
            symbols: Ticker symbol(s).
            start: Start date.
            end: End date (defaults to now).
            interval: Bar interval ('1m', '5m', '15m', '30m', '1h', '1d', '1w').
            asset_type: Contract type ('STK', 'FUT', 'CASH', 'OPT').
        """
        from ib_async import util

        ib = self._connect()

        if isinstance(symbols, str):
            symbols = [symbols]
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        if end is None:
            end = datetime.now()

        bar_size_map = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "1d": "1 day",
            "1w": "1 week",
        }
        bar_size = bar_size_map.get(interval, "1 day")

        # Calculate duration from start to end
        delta = end - start
        days = max(delta.days, 1)
        if days <= 1:
            duration = "1 D"
        elif days <= 30:
            duration = f"{days} D"
        elif days <= 365:
            months = (days // 30) + 1
            duration = f"{months} M"
        else:
            years = (days // 365) + 1
            duration = f"{years} Y"

        # Use MIDPOINT for forex, TRADES for everything else
        what_to_show = "MIDPOINT" if asset_type.upper() == "CASH" else "TRADES"

        all_frames = []
        for symbol in symbols:
            contract = self._make_contract(symbol, asset_type)
            ib.qualifyContracts(contract)

            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
            )

            if not bars:
                continue

            df = util.df(bars)
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "date": "Date",
            })
            df["Symbol"] = symbol
            df = df.set_index(["Date", "Symbol"])
            all_frames.append(df[["Open", "High", "Low", "Close", "Volume"]])

        if not all_frames:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        result = pd.concat(all_frames)
        result = result.sort_index()
        return result

    def get_supported_assets(self) -> list[str]:
        return ["equity", "etf", "futures", "options", "forex"]

    def stream_realtime(self, symbols: list[str], callback, asset_type: str = "STK"):
        """Stream real-time 5-second bars via IB.

        Args:
            symbols: Tickers to subscribe to.
            callback: Function called with (symbol, price, volume, timestamp).
            asset_type: Contract type for the symbols.
        """
        ib = self._connect()

        what_to_show = "MIDPOINT" if asset_type.upper() == "CASH" else "TRADES"

        def _on_bar(bars, has_new_bar):
            if has_new_bar and bars:
                bar = bars[-1]
                callback(
                    bars.contract.symbol,
                    float(bar.close),
                    int(bar.volume) if hasattr(bar, "volume") else 0,
                    bar.time,
                )

        contracts = []
        for symbol in symbols:
            contract = self._make_contract(symbol, asset_type)
            ib.qualifyContracts(contract)
            contracts.append(contract)

            bars = ib.reqRealTimeBars(
                contract, barSize=5, whatToShow=what_to_show, useRTH=True
            )
            bars.updateEvent += _on_bar

        def _run():
            ib.run()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread

    def disconnect(self):
        """Disconnect from IB Gateway/TWS."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("IBKRDataProvider disconnected")
