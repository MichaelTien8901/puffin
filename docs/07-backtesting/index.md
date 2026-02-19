---
layout: default
title: "Part 7: Backtesting"
nav_order: 8
has_children: true
permalink: /07-backtesting/
---

# Part 7: Backtesting

Backtesting lets you test a trading strategy against historical data to estimate how it would have performed. This part covers building an event-driven backtester that processes data bar-by-bar, simulates order execution with realistic slippage and commission models, and calculates performance metrics. We also introduce walk-forward analysis to guard against overfitting.

## Chapters

1. [Event-Driven Engine](01-event-driven-engine) -- Event-driven vs vectorized approaches, execution flow, and basic usage
2. [Execution Models](02-execution-models) -- Slippage models, commission models, and performance metrics
3. [Walk-Forward Analysis](03-walk-forward-analysis) -- Rolling train/test splits, visualization, and overfitting pitfalls

## Execution Flow

The diagram below shows how Puffin's backtester processes each bar. Orders generated on one bar execute on the next, ensuring the strategy never sees future data.

```mermaid
graph TD
    A[Start: Bar 0] --> B[Get data up to current bar]
    B --> C[Execute pending orders from previous bar]
    C --> D[Strategy generates signals]
    D --> E[Convert signals to orders]
    E --> F[Record portfolio value]
    F --> G{More bars?}
    G -->|Yes| B
    G -->|No| H[Calculate metrics]

    classDef dark fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4
    classDef accent fill:#1a3a5c,stroke:#0d2137,color:#e8e0d4
    classDef decision fill:#6b2d5b,stroke:#4a1a4a,color:#e8e0d4
    classDef endpoint fill:#8b4513,stroke:#5c2d0e,color:#e8e0d4

    class A,H endpoint
    class B,C,F dark
    class D,E accent
    class G decision
```

{: .tip }
> **Notebook**: Run the examples interactively in [`operational.ipynb`](https://github.com/MichaelTien8901/puffin/blob/main/notebooks/operational.ipynb)

## Related Chapters

- [Part 2: Data Pipeline]({{ site.baseurl }}/02-data-pipeline/) -- Provides the market data feeds that the backtester consumes bar-by-bar
- [Part 6: Strategy Modules]({{ site.baseurl }}/06-trading-strategies/) -- Defines the strategies that backtesting evaluates against historical data
- [Part 24: Risk Management]({{ site.baseurl }}/24-risk-management/) -- Position sizing and stop-loss rules applied during backtest execution
- [Part 8: Linear Models]({{ site.baseurl }}/08-linear-models/) -- Evaluate linear model predictions in a realistic backtest setting
- [Part 25: Monitoring & Analytics]({{ site.baseurl }}/25-monitoring-analytics/) -- Performance dashboards that visualize backtest results

## Source Code

Browse the implementation: [`puffin/backtest/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/backtest)
