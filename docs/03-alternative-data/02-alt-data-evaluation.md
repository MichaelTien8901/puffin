---
layout: default
title: "Alternative Data Evaluation"
parent: "Part 3: Alternative Data"
nav_order: 2
---

# Alternative Data Evaluation

Not all alternative data is useful. A dataset might look promising in a pitch deck but contain no real predictive power. This chapter introduces a rigorous evaluation framework using `AltDataEvaluator` to measure signal quality, data quality, and backtested performance before you commit capital.

## The Evaluation Framework

Before deploying any alternative data signal, it must pass three gates:

1. **Signal quality** -- Does the signal predict future returns?
2. **Data quality** -- Is the data complete, timely, and well-structured?
3. **Backtest performance** -- Does a simple strategy built on the signal make money?

{: .warning }
Skipping any of these steps is a common source of live trading losses. A signal with strong IC but poor data quality will fail in production when missing values cause position errors.

## Signal Quality Metrics

The `AltDataEvaluator` measures the predictive power of a signal against forward returns. The core metric is the **Information Coefficient (IC)**, the rank correlation between your signal and subsequent returns.

```python
from puffin.factors.alt_data_eval import AltDataEvaluator
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range("2023-01-01", periods=100, freq="D")
signal = pd.Series(np.random.randn(100), index=dates)
returns = pd.Series(np.random.randn(100) / 100, index=dates)

evaluator = AltDataEvaluator(risk_free_rate=0.02)

# Evaluate signal quality
quality = evaluator.evaluate_signal_quality(signal, returns)

print(f"Information Coefficient (IC): {quality['ic']:.3f}")
print(f"Information Ratio (IR): {quality['ir']:.3f}")
print(f"T-Statistic: {quality['t_stat']:.3f}")
print(f"P-Value: {quality['p_value']:.3f}")
print(f"Decay Half-Life: {quality['decay_half_life']:.1f} days")
```

Output:
```
Information Coefficient (IC): 0.042
Information Ratio (IR): 0.418
T-Statistic: 0.414
P-Value: 0.680
Decay Half-Life: 8.0 days
```

### Key Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **IC (Information Coefficient)** | Correlation between signal and forward returns | > 0.05 |
| **IR (Information Ratio)** | IC divided by its standard deviation | > 0.5 |
| **T-Statistic** | Statistical significance of IC | > 2.0 |
| **Decay Half-Life** | Days until signal predictive power halves | 5-20 days |

{: .warning }
A signal with IC < 0.02 or p-value > 0.05 is likely noise. Don't trade on it without further validation.

### Understanding IC Decay

The **decay half-life** tells you how quickly the signal's predictive power fades. A half-life of 8 days means the IC drops by 50% after 8 days. This directly affects your trading frequency:

- **Short half-life (1-3 days)**: Requires high-frequency rebalancing and low transaction costs.
- **Medium half-life (5-20 days)**: Suitable for weekly rebalancing strategies.
- **Long half-life (20+ days)**: Can support monthly or quarterly strategies.

{: .tip }
If your signal decays faster than your rebalancing frequency, you are systematically trading on stale information.

## Data Quality Assessment

Even a predictive signal is useless if the underlying data is unreliable. The `evaluate_data_quality` method checks completeness, coverage, and update frequency:

```python
import pandas as pd
import numpy as np
from puffin.factors.alt_data_eval import AltDataEvaluator

# Sample alternative data
alt_data = pd.DataFrame({
    "web_traffic": np.random.randint(1000, 5000, 60),
    "sentiment": np.random.randn(60),
    "mentions": np.random.randint(0, 100, 60),
}, index=pd.date_range("2023-01-01", periods=60, freq="D"))

# Add some missing values (realistic scenario)
alt_data.iloc[5:8, 0] = np.nan

evaluator = AltDataEvaluator(risk_free_rate=0.02)
data_quality = evaluator.evaluate_data_quality(alt_data)

print(f"Completeness: {data_quality['completeness']:.1f}%")
print(f"Missing: {data_quality['missing_pct']:.1f}%")
print(f"Coverage: {data_quality['coverage_days']} days")
print(f"Update Frequency: {data_quality['update_frequency']:.1f} days")
print(f"Data Points: {data_quality['data_points']}")
```

Output:
```
Completeness: 98.3%
Missing: 1.7%
Coverage: 59 days
Update Frequency: 1.0 days
Data Points: 60
```

### Data Quality Thresholds

Use these guidelines when assessing whether a dataset is production-ready:

| Metric | Minimum for Production | Notes |
|--------|----------------------|-------|
| Completeness | > 95% | Below this, imputation introduces too much noise |
| Coverage | > 252 trading days | At least one year of history for seasonal effects |
| Update frequency | <= strategy rebalance period | Data must arrive before you need to trade |

{: .note }
Missing data is not always random. Gaps in web traffic data often occur on weekends and holidays. Gaps in earnings data cluster around reporting seasons. Understand the pattern before choosing an imputation method.

## Technical Evaluation

The technical evaluation assesses storage requirements and data structure, which matter at scale:

```python
import pandas as pd
import numpy as np
from puffin.factors.alt_data_eval import AltDataEvaluator

alt_data = pd.DataFrame({
    "web_traffic": np.random.randint(1000, 5000, 60),
    "sentiment": np.random.randn(60),
    "mentions": np.random.randint(0, 100, 60),
}, index=pd.date_range("2023-01-01", periods=60, freq="D"))

evaluator = AltDataEvaluator(risk_free_rate=0.02)
technical = evaluator.evaluate_technical(alt_data)

print(f"Storage: {technical['storage_mb']:.2f} MB")
print(f"Memory per row: {technical['memory_per_row_kb']:.2f} KB")
print(f"Rows: {technical['row_count']}")
print(f"Columns: {technical['column_count']}")
print(f"DateTime Index: {technical['has_datetime_index']}")
print(f"Numeric Columns: {technical['numeric_columns']}")
```

Output:
```
Storage: 0.02 MB
Memory per row: 0.31 KB
Rows: 60
Columns: 3
DateTime Index: True
Numeric Columns: 3
```

{: .tip }
For datasets with millions of rows, switch from pandas DataFrames to Parquet files or HDF5 stores. The `MarketDataStore` from Part 2 supports both formats.

## Backtesting the Signal

The ultimate test: does a simple long-short strategy built on the signal make money? The `backtest_signal` method sorts observations into quantile buckets and measures return spreads:

```python
import pandas as pd
import numpy as np
from puffin.factors.alt_data_eval import AltDataEvaluator

dates = pd.date_range("2023-01-01", periods=100, freq="D")
signal = pd.Series(np.random.randn(100), index=dates)
returns = pd.Series(np.random.randn(100) / 100, index=dates)

evaluator = AltDataEvaluator(risk_free_rate=0.02)
backtest = evaluator.backtest_signal(signal, returns, quantiles=5)

print("\nQuantile Returns:")
for q_name, q_stats in backtest["quantile_returns"].items():
    print(f"{q_name}: {q_stats['mean_return']*100:.3f}% "
          f"(Sharpe: {q_stats['sharpe']:.2f})")

print(f"\nLong-Short Return: {backtest['long_short_return']*100:.3f}%")
print(f"Long-Short Sharpe: {backtest['long_short_sharpe']:.2f}")
```

Output:
```
Quantile Returns:
Q1: -0.012% (Sharpe: -0.15)
Q2: 0.023% (Sharpe: 0.31)
Q3: -0.008% (Sharpe: -0.09)
Q4: 0.015% (Sharpe: 0.18)
Q5: 0.031% (Sharpe: 0.42)

Long-Short Return: 0.043%
Long-Short Sharpe: 0.28
```

{: .tip }
A good alternative data signal should show monotonic returns across quantiles (Q1 < Q2 < Q3 < Q4 < Q5) and positive long-short Sharpe ratio.

### Interpreting Backtest Results

- **Monotonic quantile returns**: The strongest evidence of a real signal. If Q5 consistently outperforms Q1, the signal has predictive power.
- **Long-short Sharpe > 0.5**: Suggests the signal is tradable after transaction costs.
- **Long-short Sharpe < 0.3**: Marginal at best. Transaction costs will likely eliminate the edge.
- **Non-monotonic quantiles**: The signal may have a nonlinear relationship with returns. Consider transformations (log, rank, winsorize) before discarding.

## Complete Example: Earnings Sentiment Signal

Let's build a complete pipeline from transcript to tradable signal, combining `WebScraper`, `TranscriptParser`, and `AltDataEvaluator`:

```python
from puffin.data.scraper import WebScraper
from puffin.data.transcript_parser import TranscriptParser
from puffin.factors.alt_data_eval import AltDataEvaluator
import pandas as pd

# Step 1: Scrape earnings transcripts (mock for example)
scraper = WebScraper()
parser = TranscriptParser()

tickers = ["AAPL", "MSFT", "GOOGL"]
sentiment_data = []

for ticker in tickers:
    # In production, fetch real transcripts
    transcript_text = f"""
    CEO: We're pleased with {ticker}'s strong performance this quarter.
    Revenue growth was excellent, and we remain optimistic about the future.
    """

    parsed = parser.parse(transcript_text)
    sentiment = parser.sentiment_sections(parsed)

    sentiment_data.append({
        "ticker": ticker,
        "date": "2024-01-15",
        "sentiment_score": sentiment["prepared_remarks"]["score"],
    })

sentiment_df = pd.DataFrame(sentiment_data)
print(sentiment_df)

# Step 2: Evaluate signal quality
# (In production, align with actual forward returns)
dates = pd.date_range("2024-01-15", periods=len(sentiment_df), freq="D")
signal = pd.Series(sentiment_df["sentiment_score"].values, index=dates)
returns = pd.Series([0.01, 0.02, 0.015], index=dates)

evaluator = AltDataEvaluator()
quality = evaluator.evaluate_signal_quality(signal, returns)

print(f"\nSignal IC: {quality['ic']:.3f}")
print(f"Signal IR: {quality['ir']:.3f}")

# Step 3: Deploy if quality metrics pass thresholds
if abs(quality['ic']) > 0.05 and quality['p_value'] < 0.05:
    print("\nSignal passes quality check - ready for trading!")
else:
    print("\nSignal quality insufficient - needs refinement")
```

This pipeline demonstrates the full workflow: source data, parse it, extract a signal, and evaluate whether it meets your quality bar before deploying capital.

## Exercises

1. Create a sentiment signal and evaluate its IC against real stock returns using `yfinance` data
2. Build a data quality dashboard comparing multiple alternative data sources
3. Backtest a web traffic signal: does increasing traffic for a company predict positive returns?
4. Experiment with different quantile counts (3, 5, 10) and observe how the long-short Sharpe changes

## Source Code

- [`puffin/factors/alt_data_eval.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/factors/alt_data_eval.py) -- AltDataEvaluator implementation
