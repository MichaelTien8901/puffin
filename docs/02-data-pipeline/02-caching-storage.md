---
layout: default
title: "Caching & Storage"
parent: "Part 2: Data Pipeline"
nav_order: 2
---

# Caching & Storage

## Overview

API calls are slow and rate-limited. Every time you fetch AAPL daily data from yfinance, you are making an HTTP request, waiting for a response, and parsing JSON -- even if you fetched the exact same data five minutes ago. In a research workflow where you run dozens of backtests per hour, this adds up fast.

The solution is a two-tier storage architecture:
- **SQLite cache** (`DataCache`) for fast, transparent lookups that sit between your code and the data provider
- **File-based storage** (`MarketDataStore`) using HDF5 or Parquet for long-term, high-performance data persistence

## SQLite Cache with DataCache

The `DataCache` class wraps a local SQLite database. It stores OHLCV data keyed by symbol, date, and interval, and serves as a transparent caching layer:

```python
from puffin.data.cache import DataCache

cache = DataCache()  # Defaults to ./data/market_data.db
```

{: .tip }
The cache uses SQLite -- no server setup required. Your data persists between sessions automatically. The database file is created in the directory specified by the `PUFFIN_DATA_DIR` environment variable, or `./data/` by default.

### Basic Cache Operations

The typical workflow is check-then-fetch: look in the cache first, and only call the API if the data is missing:

```python
from puffin.data.cache import DataCache
from puffin.data import YFinanceProvider

cache = DataCache()
provider = YFinanceProvider()

# Check cache first, fetch if missing
cached = cache.get("AAPL", "2023-01-01", "2024-01-01")
if cached is None:
    data = provider.fetch_historical("AAPL", "2023-01-01", "2024-01-01")
    cache.put("AAPL", data)
else:
    data = cached
```

### The Cache Schema

Under the hood, the SQLite database uses a single table:

```sql
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
```

The composite primary key `(symbol, date, interval)` ensures that each bar is stored exactly once. The `put` method uses `INSERT OR REPLACE`, so re-caching the same data is safe and idempotent.

### Cache Management

You can clear the cache entirely or for a specific symbol:

```python
# Clear cache for a single symbol
cache.clear(symbol="AAPL")

# Clear the entire cache
cache.clear()
```

{: .warning }
Clearing the cache means your next data request will hit the API again. Be mindful of rate limits, especially with yfinance.

### Performance Comparison

To see the benefit of caching, time a fresh fetch versus a cached fetch:

```python
import time

# Fresh fetch (API call)
start = time.time()
data = provider.fetch_historical("AAPL", "2020-01-01", "2024-01-01")
cache.put("AAPL", data)
print(f"API fetch: {time.time() - start:.2f}s")

# Cached fetch (SQLite lookup)
start = time.time()
data = cache.get("AAPL", "2020-01-01", "2024-01-01")
print(f"Cache fetch: {time.time() - start:.4f}s")
```

You will typically see a 100x or greater speedup from the cache.

## Long-Term Storage with MarketDataStore

While `DataCache` is excellent for transparent caching during research, you may want a more structured storage layer for larger datasets. The `MarketDataStore` class stores data in HDF5 or Parquet format, with metadata tracking:

```python
from puffin.data.storage import MarketDataStore

# Create a Parquet-based store
store = MarketDataStore("./data/store", format="parquet")

# Save OHLCV data
store.save_ohlcv("AAPL", data, source="yfinance", frequency="1d")

# Load it back
aapl = store.load_ohlcv("AAPL")
```

### Choosing a Format

`MarketDataStore` supports two storage formats:

| Feature | HDF5 | Parquet |
|---------|------|---------|
| File extension | `.h5` | `.parquet` |
| Compression | Built-in (zlib, lzo) | Built-in (snappy, gzip) |
| Columnar access | No (row-oriented) | Yes |
| Pandas integration | `to_hdf` / `read_hdf` | `to_parquet` / `read_parquet` |
| Best for | Time-series append | Columnar analytics |
| Ecosystem | PyTables, h5py | Apache Arrow, DuckDB |

{: .tip }
Parquet is the default and recommended format. It offers better compression, faster columnar reads, and integrates with modern tools like DuckDB and Polars. Use HDF5 only if you have specific requirements for hierarchical data or frequent appends.

### Metadata Tracking

Every `save_ohlcv` call records metadata -- the data source, frequency, last update time, and row count:

```python
# View metadata for all symbols
print(store.get_metadata())

# View metadata for a specific symbol
print(store.get_metadata("AAPL"))
```

The metadata DataFrame includes:

| Column | Description |
|--------|-------------|
| `symbol` | Ticker symbol |
| `source` | Data provider name |
| `frequency` | Bar interval (e.g., `1d`, `1h`) |
| `last_updated` | ISO timestamp of last save |
| `format` | Storage format (`hdf5` or `parquet`) |
| `rows` | Number of rows stored |

### Appending Data

For incremental updates (e.g., adding today's data to an existing history), use `append_ohlcv`:

```python
# Fetch the latest data
new_data = provider.fetch_historical("AAPL", start="2024-01-01", end="2024-02-01")

# Append to existing store (deduplicates automatically)
store.append_ohlcv("AAPL", new_data, source="yfinance", frequency="1d")
```

The append operation loads existing data, concatenates the new rows, removes duplicates (keeping the latest version), sorts by date, and saves the result. This makes it safe to call repeatedly without worrying about double-counting.

### Listing and Deleting Symbols

```python
# List all stored symbols
symbols = store.list_symbols()
print(symbols)  # ['AAPL', 'MSFT', 'GOOGL', ...]

# Delete a symbol entirely
store.delete_symbol("AAPL")
```

### Format Migration

If you need to switch from HDF5 to Parquet (or vice versa), `MarketDataStore` provides a migration method:

```python
# Migrate AAPL from HDF5 to Parquet
store.migrate_format("AAPL", target_format="parquet")
```

This loads the data in the old format, saves it in the new format, and deletes the old file.

## Cache vs. Store: When to Use Each

| Use Case | DataCache (SQLite) | MarketDataStore (Parquet/HDF5) |
|----------|-------------------|-------------------------------|
| Quick research iterations | Best choice | Overkill |
| Long-term data archival | Not designed for this | Best choice |
| Incremental daily updates | Works but limited | Built-in `append_ohlcv` |
| Query by date range | Yes (SQL WHERE) | Load full file, then slice |
| Multi-format support | No (SQLite only) | HDF5 and Parquet |
| Metadata tracking | No | Yes (source, frequency, etc.) |

{: .note }
In practice, many workflows use both: `DataCache` as a transparent API cache during development, and `MarketDataStore` for curated datasets that feed backtests and model training.

## Exercises

1. Compare the cache performance: time a fresh `fetch_historical` call versus a `cache.get` call for 5 years of SPY data. What speedup do you observe?
2. Create a `MarketDataStore` in Parquet format, save 3 tickers, then inspect the directory structure and metadata. What files were created?
3. Write a helper function that combines `DataCache` and `MarketDataStore`: check the cache first, then the store, and only call the API as a last resort. Save new data to both layers.
4. Migrate a symbol from Parquet to HDF5 using `migrate_format`. Compare the file sizes.

## Summary

- `DataCache` provides a SQLite-backed transparent cache that sits between your code and data providers, delivering 100x+ speedups on repeated queries
- `MarketDataStore` offers structured, metadata-tracked storage in HDF5 or Parquet format for long-term data management
- Parquet is the recommended storage format for its columnar access, compression, and ecosystem compatibility
- Both layers can be combined: `DataCache` for fast iteration, `MarketDataStore` for curated datasets

## Next Steps

Your data is fetched and stored. But raw market data is messy -- missing bars, outlier returns, and inconsistencies can corrupt your backtests and models. In the next chapter, you will build a preprocessing pipeline to clean and validate your data.
