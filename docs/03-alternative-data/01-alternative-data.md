---
layout: default
title: "Chapter 1: Alternative Data Sources"
parent: "Part 3: Alternative Data"
nav_order: 1
---

# Chapter 1: Alternative Data Sources

## Overview

Traditional financial data (prices, volumes, fundamentals) is widely available and heavily analyzed. Alternative data provides an edge by revealing information before it appears in conventional sources. In this chapter, you'll learn to source, parse, and evaluate alternative data for trading signals.

## What is Alternative Data?

Alternative data refers to non-traditional information sources that can provide trading signals:

- **Web data**: Earnings transcripts, news sentiment, social media
- **Satellite imagery**: Retail parking lots, shipping traffic, agricultural output
- **Transaction data**: Credit card purchases, app downloads, web traffic
- **Sensor data**: Weather patterns, foot traffic, supply chain monitoring

{: .note }
The key advantage of alternative data is **information asymmetry**. You gain insights before the market broadly recognizes them.

## Types of Alternative Data

### 1. Textual Data
- Earnings call transcripts
- SEC filings (10-K, 10-Q, 8-K)
- News articles and press releases
- Social media sentiment

### 2. Behavioral Data
- Web traffic and search trends
- App downloads and usage
- Credit card transactions
- Geolocation data

### 3. Sensor and Satellite Data
- Satellite imagery (parking lots, ships, crops)
- Weather data
- IoT sensor readings

### 4. Market Microstructure
- Order book depth
- Trade-level data
- Dark pool activity

{: .warning }
Always verify data licensing and ensure compliance with regulations. Some alternative data sources have strict usage terms.

## Sourcing Strategies

### Web Scraping

The most accessible alternative data comes from public websites. Our `WebScraper` class handles the technical details:

```python
from puffin.data.scraper import WebScraper

scraper = WebScraper(
    rate_limit=1.0,      # Wait 1 second between requests
    max_retries=3,       # Retry failed requests 3 times
    timeout=10,          # 10 second timeout
)

# Scrape SEC filings
filings = scraper.scrape_sec_filings(
    cik="0000320193",    # Apple's CIK
    form_type="10-K"     # Annual reports
)

print(filings.head())
```

Output:
```
  filing_date form_type                                        url accession_number
0  2024-01-15      10-K  https://www.sec.gov/Archives/edgar/...  0000320193240001
1  2023-01-16      10-K  https://www.sec.gov/Archives/edgar/...  0000320193230001
```

### Scraping Financial Tables

Many websites display financial data in HTML tables:

```python
# Scrape financial tables from a webpage
url = "https://example.com/company/financials"
data = scraper.scrape_financial_tables(url)

print(data)
```

The scraper automatically:
- Rotates user agents to avoid detection
- Implements exponential backoff for retries
- Respects rate limits to avoid blocking
- Parses HTML tables into pandas DataFrames

{: .tip }
For production use, consider commercial data APIs (Quandl, Alpha Vantage, IEX Cloud) which provide cleaner data and better reliability than scraping.

## Earnings Call Analysis

Earnings calls reveal management sentiment and forward guidance before it appears in financial statements.

### Parsing Transcripts

```python
from puffin.data.transcript_parser import TranscriptParser

parser = TranscriptParser()

# Sample earnings call transcript
transcript_text = """
John Smith - CEO: Good morning. I'm pleased to report strong revenue
growth of 15% year-over-year. Our revenue was $1.2 billion this quarter,
exceeding guidance. We remain optimistic about future growth.

Operator: We'll now begin the question-and-answer session.

Jane Analyst: Can you discuss the margin outlook?

John Smith - CEO: We expect margins to improve as we scale operations.
"""

# Parse into sections
parsed = parser.parse(transcript_text)

print("Participants:", parsed["participants"])
print("\nPrepared Remarks Length:", len(parsed["prepared_remarks"]))
print("Q&A Session Length:", len(parsed["qa_session"]))
```

Output:
```
Participants: ['Jane Analyst', 'John Smith']

Prepared Remarks Length: 245
Q&A Session Length: 178
```

### Extracting Financial Metrics

The parser identifies mentioned metrics automatically:

```python
metrics = parser.extract_metrics(parsed)

for metric in metrics:
    print(f"{metric['metric_type']}: {metric['value']} {metric['unit']}")
```

Output:
```
revenue: 1.2 billion
growth: 15.0 None
```

### Sentiment Analysis

Analyze management tone across sections:

```python
sentiment = parser.sentiment_sections(parsed)

for section, scores in sentiment.items():
    print(f"\n{section}:")
    print(f"  Score: {scores['score']:.2f} (-1 negative, +1 positive)")
    print(f"  Positive words: {scores['positive_count']}")
    print(f"  Negative words: {scores['negative_count']}")
```

Output:
```
prepared_remarks:
  Score: 0.60 (-1 negative, +1 positive)
  Positive words: 4
  Negative words: 1

qa_session:
  Score: 0.33 (-1 negative, +1 positive)
  Positive words: 1
  Negative words: 0
```

{: .note }
This is basic sentiment analysis. For production, consider NLP libraries (spaCy, transformers) or sentiment models trained on financial text.

## Alternative Data Evaluation Framework

Not all alternative data is useful. We need rigorous evaluation before deploying signals.

### Signal Quality Metrics

The `AltDataEvaluator` measures predictive power:

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

#### Key Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **IC (Information Coefficient)** | Correlation between signal and forward returns | > 0.05 |
| **IR (Information Ratio)** | IC divided by its standard deviation | > 0.5 |
| **T-Statistic** | Statistical significance of IC | > 2.0 |
| **Decay Half-Life** | Days until signal predictive power halves | 5-20 days |

{: .warning }
A signal with IC < 0.02 or p-value > 0.05 is likely noise. Don't trade on it without further validation.

### Data Quality Assessment

Evaluate the raw data quality:

```python
# Sample alternative data
alt_data = pd.DataFrame({
    "web_traffic": np.random.randint(1000, 5000, 60),
    "sentiment": np.random.randn(60),
    "mentions": np.random.randint(0, 100, 60),
}, index=pd.date_range("2023-01-01", periods=60, freq="D"))

# Add some missing values (realistic scenario)
alt_data.iloc[5:8, 0] = np.nan

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

### Technical Evaluation

Assess storage and format:

```python
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

### Backtesting the Signal

Test signal performance with quantile portfolios:

```python
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

## Complete Example: Earnings Sentiment Signal

Let's build a complete pipeline from transcript to tradable signal:

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

## Best Practices

1. **Start Simple**: Web scraping and sentiment analysis are accessible starting points
2. **Validate Rigorously**: Use IC, IR, and backtests before deploying
3. **Monitor Decay**: Alternative data edges erode as they become known
4. **Respect Terms**: Always check website terms of service and robots.txt
5. **Version Data**: Track data provenance and changes over time

## Exercises

1. Scrape SEC filings for 3 companies and compare filing frequencies
2. Parse an earnings transcript and extract all mentioned financial metrics
3. Create a sentiment signal and evaluate its IC against real stock returns
4. Build a data quality dashboard comparing multiple alternative data sources

## Summary

- Alternative data provides information asymmetry and early signals
- Web scraping with `WebScraper` accesses public financial data
- `TranscriptParser` extracts structure and sentiment from earnings calls
- `AltDataEvaluator` rigorously tests signal and data quality
- IC, IR, and decay metrics distinguish useful signals from noise
- Always validate alternative data before deploying capital

## Next Steps

In Part 4, we'll use alternative data to construct **alpha factors** and combine them into multi-factor models.
