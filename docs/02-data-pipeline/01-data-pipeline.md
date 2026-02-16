---
layout: default
title: "Building the Data Pipeline"
parent: "Part 2: Data Pipeline"
nav_order: 1
---

# Building the Data Pipeline

## Overview

Every trading system starts with data. In this chapter, you'll build a data pipeline that fetches historical market data, caches it locally, and preprocesses it for analysis and strategy development.

## The DataProvider Interface

We define an abstract interface so we can swap data sources without changing any consuming code:

```python
from puffin.data.provider import DataProvider

class DataProvider(ABC):
    @abstractmethod
    def fetch_historical(self, symbols, start, end, interval="1d") -> pd.DataFrame:
        ...

    @abstractmethod
    def get_supported_assets(self) -> list[str]:
        ...

    def stream_realtime(self, symbols, callback):
        raise NotImplementedError
```

{: .note }
This pattern is called the **Strategy pattern** (not to be confused with trading strategies). It lets us add new data sources (Polygon.io, Binance, etc.) without modifying existing code.

## Fetching Data with yfinance

The simplest provider uses yfinance — free, no API key required:

```python
from puffin.data import YFinanceProvider

provider = YFinanceProvider()

# Single ticker
aapl = provider.fetch_historical("AAPL", start="2023-01-01", end="2024-01-01")
print(aapl.head())

# Multiple tickers
data = provider.fetch_historical(
    ["AAPL", "MSFT", "GOOGL"],
    start="2023-01-01",
    end="2024-01-01"
)
```

## Caching with SQLite

API calls are slow and rate-limited. We cache data locally:

```python
from puffin.data.cache import DataCache

cache = DataCache()  # Defaults to ./data/market_data.db

# Check cache first, fetch if missing
cached = cache.get("AAPL", "2023-01-01", "2024-01-01")
if cached is None:
    data = provider.fetch_historical("AAPL", "2023-01-01", "2024-01-01")
    cache.put("AAPL", data)
else:
    data = cached
```

{: .tip }
The cache uses SQLite — no server setup required. Your data persists between sessions automatically.

## Preprocessing

Raw data often has issues: missing values, outliers, and inconsistencies. The preprocessing module handles this:

```python
from puffin.data.preprocessing import preprocess

# Clean the data
clean_data = preprocess(
    raw_data,
    fill_method="ffill",       # Forward-fill missing values
    remove_outliers=True,      # Clip extreme returns
    outlier_std=5.0,           # 5 standard deviations
)
```

### Handling Missing Data

| Method | Behavior | Best For |
|--------|----------|----------|
| `ffill` | Forward-fill from last known value | Most cases |
| `interpolate` | Linear interpolation | Smooth data |
| `drop` | Remove rows with missing data | When accuracy matters more than completeness |

{: .warning }
Always be cautious with forward-filling. In a live system, forward-filling means using stale data, which can lead to incorrect signals.

## Real-Time Data with Alpaca

For paper and live trading, you'll need real-time data:

```python
from puffin.data.alpaca_provider import AlpacaProvider

provider = AlpacaProvider()  # Reads API keys from .env

# Stream real-time bars
def on_update(symbol, price, volume, timestamp):
    print(f"{symbol}: ${price:.2f} ({volume} shares) at {timestamp}")

provider.stream_realtime(["AAPL", "MSFT"], callback=on_update)
```

## Exercises

1. Fetch 5 years of daily SPY data and examine the data quality (missing days, outliers)
2. Compare the cache performance: time a fresh fetch vs. a cached fetch
3. Experiment with different preprocessing fill methods and observe the differences

{: .note }
> **Modern Alternatives:** This tutorial uses pandas and SQLite, which are the industry standard for getting started. If you're working with large datasets (millions of rows) or need faster performance, consider [Polars](https://pola.rs/) as a drop-in DataFrame alternative (often 10-100x faster than pandas) and [DuckDB](https://duckdb.org/) as a local analytics database that can query CSV/Parquet files directly. Both integrate well with the pandas ecosystem and are increasingly popular in quantitative finance workflows.

## Summary

- The `DataProvider` interface abstracts data sources for easy swapping
- `YFinanceProvider` gives free historical data with no setup
- `DataCache` stores data in SQLite to avoid redundant API calls
- Preprocessing handles missing values, outliers, and data validation
- `AlpacaProvider` adds real-time streaming for live trading

## Next Steps

With data flowing, we're ready to build **trading strategies** in Part 3.
