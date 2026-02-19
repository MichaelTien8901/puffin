---
layout: default
title: "Part 2: Data Pipeline"
nav_order: 3
has_children: true
permalink: /02-data-pipeline/
---

# Part 2: Data Pipeline

Every trading system starts with data. Before you can build strategies, train models, or execute trades, you need a reliable pipeline that fetches market data, stores it efficiently, and cleans it for downstream consumption. A poorly built data pipeline leads to stale prices, missing bars, and ultimately wrong trading decisions.

This section walks you through building a production-quality data pipeline from scratch. You will implement a pluggable provider interface that abstracts away the data source, a caching layer that eliminates redundant API calls, and a preprocessing module that handles the messy realities of real-world market data.

## Data Pipeline Architecture

```mermaid
flowchart LR
    A[Market Source] --> B{DataProvider}
    B --> C[YFinanceProvider]
    B --> D[AlpacaProvider]

    C --> E[SQLite Cache]
    D --> E

    E --> F[Preprocessing]
    F --> G[Missing Values]
    F --> H[Outlier Detection]
    F --> I[Validation]

    G --> J[Clean DataFrame]
    H --> J
    I --> J

    J --> K[Strategies & Models]

    classDef source fill:#1a3a5c,stroke:#0d2137,color:#e8e0d4
    classDef provider fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4
    classDef cache fill:#6b2d5b,stroke:#4a1e3f,color:#e8e0d4
    classDef process fill:#8b4513,stroke:#5c2e0d,color:#e8e0d4
    classDef output fill:#2d5050,stroke:#1a3a3a,color:#e8e0d4

    class A source
    class B,C,D provider
    class E cache
    class F,G,H,I process
    class J,K output
```

## Chapters

1. [Data Providers](01-data-providers) -- The DataProvider interface, yfinance for free historical data, and Alpaca for real-time streaming
2. [Caching & Storage](02-caching-storage) -- SQLite cache for fast lookups, HDF5 and Parquet for long-term storage with MarketDataStore
3. [Preprocessing](03-preprocessing) -- Handling missing values, outlier detection, data validation, and the preprocessing pipeline

{: .tip }
> **Notebook**: Run the examples interactively in [`data_pipeline.ipynb`](https://github.com/MichaelTien8901/puffin/blob/main/notebooks/data_pipeline.ipynb)

## Related Chapters

- [Part 3: Alternative Data]({{ site.baseurl }}/03-alternative-data/) -- Alternative data feeds into the pipeline as a non-traditional data source
- [Part 4: Alpha Factors]({{ site.baseurl }}/04-alpha-factors/) -- Alpha factors consume clean data produced by the pipeline
- [Part 7: Backtesting]({{ site.baseurl }}/07-backtesting/) -- The backtesting engine relies on historical data delivered by the pipeline

## Source Code

Browse the implementation: [`puffin/data/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/data)
