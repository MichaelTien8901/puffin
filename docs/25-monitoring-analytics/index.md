---
layout: default
title: "Part 25: Monitoring & Analytics"
nav_order: 26
has_children: true
permalink: /25-monitoring-analytics/
---

# Part 25: Monitoring & Analytics

Comprehensive monitoring and analytics are essential for understanding strategy performance and making informed trading decisions. The Puffin framework provides tools for trade logging, P&L tracking, benchmark comparison, system health monitoring, and real-time dashboards.

## Monitoring Pipeline

```mermaid
flowchart LR
    subgraph Execution
        A[Order Engine]
        B[Broker API]
    end

    subgraph Logging
        C[TradeRecord]
        D[TradeLog]
        E[CSV / JSON Export]
    end

    subgraph PnL
        F[PnLTracker]
        G[Position Book]
        H[Daily P&L Series]
        I[Attribution]
    end

    subgraph Benchmark
        J[BenchmarkComparison]
        K[Rolling Metrics]
    end

    subgraph Dashboard
        L[Streamlit App]
        M[Equity Curve]
        N[Drawdown Chart]
    end

    A --> C
    B --> C
    C --> D
    D --> E
    D --> F
    F --> G
    G --> H
    G --> I
    H --> J
    J --> K
    H --> L
    I --> L
    K --> L
    L --> M
    L --> N

    classDef exec fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4
    classDef logging fill:#1a3a5c,stroke:#0d2137,color:#e8e0d4
    classDef pnl fill:#6b2d5b,stroke:#4a1d40,color:#e8e0d4
    classDef bench fill:#8b4513,stroke:#5a2d0a,color:#e8e0d4
    classDef dash fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4

    class A,B exec
    class C,D,E logging
    class F,G,H,I pnl
    class J,K bench
    class L,M,N dash

    linkStyle default stroke:#4a5568,stroke-width:2px
```

## Chapters

1. [Trade Logging & P&L](01-trade-logging-pnl) - Record every execution with `TradeLog` and `TradeRecord`, track realized and unrealized profit with `PnLTracker`, generate attribution reports by strategy and asset
2. [Benchmark Comparison](02-benchmark-comparison) - Compute alpha, beta, information ratio, and tracking error with `BenchmarkComparison`, visualize cumulative and rolling performance against SPY or custom benchmarks
3. [Streamlit Dashboard](03-streamlit-dashboard) - Build a real-time monitoring dashboard with portfolio overview, equity curve, drawdown analysis, position allocation, and filterable trade history

## Key Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `TradeRecord` | `puffin.monitor.trade_log` | Immutable record of a single execution |
| `TradeLog` | `puffin.monitor.trade_log` | Collection of trades with filter, export, and summary |
| `PnLTracker` | `puffin.monitor.pnl` | Real-time P&L with position book and history |
| `Position` | `puffin.monitor.pnl` | Per-ticker position with market value and unrealized P&L |
| `BenchmarkComparison` | `puffin.monitor.benchmark` | Strategy-vs-benchmark analytics |
| `SystemHealth` | `puffin.monitor.health` | Data feed and broker health checks with alerting |
| `create_dashboard` | `puffin.monitor.dashboard` | Streamlit multi-page dashboard builder |

{: .note }
All monitoring components are designed to work independently or together. You can use `TradeLog` on its own for audit purposes, or wire everything into the Streamlit dashboard for a unified view.

## Quick Start

```python
from puffin.monitor.trade_log import TradeLog, TradeRecord
from puffin.monitor.pnl import PnLTracker
from puffin.monitor.benchmark import BenchmarkComparison
from puffin.monitor.health import SystemHealth

# Initialize the monitoring stack
log = TradeLog()
tracker = PnLTracker(initial_cash=100_000.0)
bc = BenchmarkComparison()
health = SystemHealth()
```

## Related Chapters

- [Part 24: Risk Management]({{ site.baseurl }}/24-risk-management/) -- Drawdown, VaR, and exposure metrics feed directly into monitoring dashboards
- [Part 23: Live Trading Execution]({{ site.baseurl }}/23-live-trading/) -- Live order fills and position updates generate the data that monitoring tracks
- [Part 7: Backtesting]({{ site.baseurl }}/07-backtesting/) -- Backtest results use the same analytics pipeline for equity curves and benchmark comparison
- [Part 22: AI-Assisted Trading]({{ site.baseurl }}/22-ai-assisted-trading/) -- AI-generated market reports and sentiment summaries surface in monitoring views

## Next Steps

- See [Risk Management]({{ site.baseurl }}/24-risk-management/) for risk controls that feed into monitoring
- See [Live Trading]({{ site.baseurl }}/23-live-trading/01-live-trading) for production deployment patterns
