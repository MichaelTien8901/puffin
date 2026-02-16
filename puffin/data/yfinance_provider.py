"""Yahoo Finance data provider using yfinance."""

from datetime import datetime

import pandas as pd
import yfinance as yf

from puffin.data.provider import DataProvider


class YFinanceProvider(DataProvider):
    """Data provider using Yahoo Finance (via yfinance).

    Free historical data for equities, ETFs, and crypto.
    No API key required.
    """

    def fetch_historical(
        self,
        symbols: str | list[str],
        start: str | datetime,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if isinstance(symbols, str):
            symbols = [symbols]

        if len(symbols) == 1:
            ticker = yf.Ticker(symbols[0])
            df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
            df["Symbol"] = symbols[0]
            df.index.name = "Date"
            df = df.reset_index().set_index(["Date", "Symbol"])
            return df[["Open", "High", "Low", "Close", "Volume"]]

        # Multi-ticker download
        raw = yf.download(symbols, start=start, end=end, interval=interval, auto_adjust=True)

        frames = []
        for symbol in symbols:
            try:
                if len(symbols) > 1:
                    df = raw.xs(symbol, level=1, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw
                else:
                    df = raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df["Symbol"] = symbol
                df.index.name = "Date"
                df = df.reset_index().set_index(["Date", "Symbol"])
                frames.append(df)
            except (KeyError, TypeError):
                continue

        if not frames:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        return pd.concat(frames).sort_index()

    def get_supported_assets(self) -> list[str]:
        return ["equity", "etf", "crypto", "forex", "futures"]
