"""SQLite-based data cache for market data."""

import os
import sqlite3
from datetime import datetime

import pandas as pd


class DataCache:
    """Local SQLite cache for market data to avoid redundant API calls."""

    def __init__(self, db_path: str | None = None):
        data_dir = os.environ.get("PUFFIN_DATA_DIR", "./data")
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = db_path or os.path.join(data_dir, "market_data.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date, interval)
                )
            """)

    def get(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Retrieve cached data. Returns None on cache miss."""
        start_str = str(start)[:10]
        end_str = str(end)[:10]

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT date, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND interval = ? AND date >= ? AND date <= ?
                ORDER BY date
                """,
                conn,
                params=(symbol, interval, start_str, end_str),
            )

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    def put(self, symbol: str, data: pd.DataFrame, interval: str = "1d"):
        """Store data in cache."""
        if data.empty:
            return

        records = []
        for date, row in data.iterrows():
            date_str = str(date)[:10]
            records.append((
                symbol, date_str, interval,
                float(row["Open"]), float(row["High"]),
                float(row["Low"]), float(row["Close"]),
                int(row["Volume"]),
            ))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO ohlcv
                (symbol, date, interval, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )

    def clear(self, symbol: str | None = None):
        """Clear cache, optionally for a specific symbol."""
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                conn.execute("DELETE FROM ohlcv WHERE symbol = ?", (symbol,))
            else:
                conn.execute("DELETE FROM ohlcv")
