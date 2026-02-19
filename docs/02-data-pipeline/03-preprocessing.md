---
layout: default
title: "Preprocessing"
parent: "Part 2: Data Pipeline"
nav_order: 3
---

# Preprocessing

## Overview

Raw market data is never clean. Exchanges close for holidays, data providers return incomplete bars, stock splits create artificial price jumps, and fat-finger trades produce outlier returns that can destroy a backtest. The preprocessing module handles all of these problems so your strategies and models operate on reliable data.

The `preprocess` function in `puffin.data.preprocessing` is the primary entry point. It chains together three operations: missing data handling, outlier detection, and validation.

## The Preprocessing Pipeline

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

Under the hood, `preprocess` calls three internal functions in sequence:

1. **`_handle_missing`** -- fills or removes gaps in the data
2. **`_handle_outliers`** -- detects and clips extreme returns
3. **`_validate`** -- enforces structural constraints (no negative prices, High >= Low, Volume >= 0)

This pipeline design means each step is independent and testable, but the `preprocess` function gives you a single, convenient entry point.

## Handling Missing Data

Missing data in market time series typically comes from:
- Exchange holidays (market closed, no data)
- Data provider gaps (incomplete API responses)
- Ticker changes or delistings
- Network issues during data collection

The `fill_method` parameter controls how missing values are handled:

| Method | Behavior | Best For |
|--------|----------|----------|
| `ffill` | Forward-fill from last known value | Most cases |
| `interpolate` | Linear interpolation | Smooth data |
| `drop` | Remove rows with missing data | When accuracy matters more than completeness |

{: .warning }
Always be cautious with forward-filling. In a live system, forward-filling means using stale data, which can lead to incorrect signals.

### Forward-Fill Details

Forward-fill (`ffill`) carries the last known value forward until new data arrives. After forward-filling, any remaining NaN values at the start of the series (where there is no prior value to carry forward) are back-filled:

```python
# Forward-fill with back-fill for leading NaNs
clean = preprocess(raw_data, fill_method="ffill")
```

{: .note }
The back-fill step only applies to NaN values at the very beginning of the series. It prevents the common pitfall of having a DataFrame that starts with missing rows and produces NaN-based signals on the first few bars.

### Interpolation

Linear interpolation estimates missing values based on surrounding data points. It uses time-based interpolation (`method="time"`) to account for irregular spacing between bars:

```python
# Time-based linear interpolation
clean = preprocess(raw_data, fill_method="interpolate")
```

{: .tip }
Interpolation works well for smooth, trending data (e.g., bond prices), but can produce misleading fills during volatile periods or across weekends. Use it with care.

### Drop Missing Rows

The most conservative approach simply removes any row with a missing value:

```python
# Remove incomplete rows
clean = preprocess(raw_data, fill_method="drop")
```

This preserves data integrity at the cost of completeness. It is the best choice when you need exact values and cannot tolerate any estimation -- for example, when computing precise transaction costs.

## Outlier Detection and Clipping

Extreme returns can distort statistical measures (mean, standard deviation, Sharpe ratio) and cause models to overfit to noise. The preprocessing module detects outliers by computing how many standard deviations each return is from the mean, then clips returns that exceed the threshold:

```python
# Clip returns beyond 3 standard deviations
clean = preprocess(raw_data, remove_outliers=True, outlier_std=3.0)

# More permissive: 5 standard deviations (default)
clean = preprocess(raw_data, remove_outliers=True, outlier_std=5.0)
```

The outlier detection works as follows:
1. Compute daily returns from the `Close` column
2. Calculate the mean and standard deviation of returns
3. Flag any return outside the range `[mean - threshold*std, mean + threshold*std]`
4. Clip flagged returns to the threshold boundary
5. Reconstruct prices from the clipped returns

{: .important }
Outlier clipping modifies the `Close` column but leaves `Open`, `High`, `Low`, and `Volume` unchanged. If you need consistent OHLCV data after clipping, consider applying additional adjustments to the other price columns.

### Choosing the Threshold

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 3.0 | Aggressive clipping (removes ~0.3% of normal data) | Short-term strategies sensitive to noise |
| 5.0 | Moderate clipping (default) | General purpose |
| 10.0 | Minimal clipping | When you want to preserve most extreme moves |

## Data Validation

The final step in the pipeline enforces structural constraints that all OHLCV data must satisfy:

- **No negative prices**: `Open`, `High`, `Low`, and `Close` are clipped to a minimum of 0
- **High >= Low**: If any bar has `High < Low`, the values are swapped
- **Volume >= 0**: Negative volume values are clipped to 0

These checks catch data corruption that would otherwise produce nonsensical results downstream.

## Split Adjustments

Stock splits create artificial price discontinuities. A 4:1 split makes the stock look like it dropped 75% overnight, which would be flagged as an outlier and would corrupt any return-based analysis. The `adjust_splits` function retroactively adjusts historical prices:

```python
from puffin.data.preprocessing import adjust_splits

# Apply split adjustments
# splits is a pandas Series with split ratios indexed by date
# e.g., a 4:1 split on 2022-08-05 would be: splits["2022-08-05"] = 4.0
adjusted = adjust_splits(raw_data, splits)
```

The function walks backward through the split dates, dividing prices and multiplying volumes by the split ratio for all bars before each split date.

{: .note }
Most data providers (including yfinance) return split-adjusted data by default. You only need `adjust_splits` when working with raw, unadjusted data from sources like exchange feeds or proprietary databases.

## Putting It All Together

A typical data pipeline combines fetching, caching, and preprocessing:

```python
from puffin.data import YFinanceProvider
from puffin.data.cache import DataCache
from puffin.data.preprocessing import preprocess

provider = YFinanceProvider()
cache = DataCache()

def get_clean_data(symbol, start, end):
    """Fetch, cache, and preprocess market data."""
    # Check cache
    data = cache.get(symbol, start, end)
    if data is None:
        data = provider.fetch_historical(symbol, start, end)
        cache.put(symbol, data)

    # Preprocess
    return preprocess(data, fill_method="ffill", remove_outliers=True, outlier_std=5.0)

# Usage
clean_spy = get_clean_data("SPY", "2019-01-01", "2024-01-01")
print(f"Shape: {clean_spy.shape}")
print(f"Missing values: {clean_spy.isna().sum().sum()}")
print(f"Date range: {clean_spy.index[0]} to {clean_spy.index[-1]}")
```

{: .note }
> **Modern Alternatives:** This tutorial uses pandas and SQLite, which are the industry standard for getting started. If you're working with large datasets (millions of rows) or need faster performance, consider [Polars](https://pola.rs/) as a drop-in DataFrame alternative (often 10-100x faster than pandas) and [DuckDB](https://duckdb.org/) as a local analytics database that can query CSV/Parquet files directly. Both integrate well with the pandas ecosystem and are increasingly popular in quantitative finance workflows.

## Exercises

1. Fetch 5 years of daily SPY data and examine the data quality: how many missing days are there? Are there any outlier returns beyond 5 standard deviations?
2. Experiment with different preprocessing fill methods (`ffill`, `interpolate`, `drop`) and observe the differences in the resulting DataFrame shape and statistics.
3. Compare the daily returns distribution before and after outlier clipping with `outlier_std=3.0`. Plot histograms of both to visualize the effect.
4. Write a data quality report function that takes a DataFrame and prints: row count, date range, missing value count per column, number of outlier returns, and any High < Low violations.

## Summary

- The `preprocess` function chains missing data handling, outlier detection, and validation into a single pipeline
- Forward-fill (`ffill`) is the safest default for most use cases, but be cautious about stale data in live systems
- Outlier clipping at 5 standard deviations removes extreme moves that distort statistics without being overly aggressive
- Data validation enforces structural constraints (no negative prices, High >= Low, Volume >= 0)
- The `adjust_splits` function handles stock split adjustments for raw, unadjusted data
- Combining providers, caching, and preprocessing gives you a complete, reliable data pipeline

{: .tip }
For a deep dive into real-time streaming -- WebSocket protocols, tick-to-bar aggregation, order book depth, and auto-reconnection -- see [Part 26: Real-Time Market Data]({{ site.baseurl }}/26-realtime-data/).

## Next Steps

With clean, reliable data flowing through your pipeline, you are ready to build **trading strategies** in Part 3.
