---
layout: default
title: "Part 24: Risk Management"
nav_order: 25
has_children: true
permalink: /24-risk-management/
---

# Risk Management

Risk management is critical for long-term trading success. No matter how strong your alpha signal is, poor risk controls can destroy an account. The Puffin framework provides comprehensive tools for position sizing, stop loss management, and portfolio-level risk controls through the `puffin.risk` package.

## Risk Management Pipeline

```mermaid
flowchart TB
    A[Trade Signal] --> B[Position Sizer]
    B --> C{Sizing Method}
    C --> D[Fixed Fractional]
    C --> E[Kelly Criterion]
    C --> F[Volatility-Based]

    D --> G[Sized Order]
    E --> G
    F --> G

    G --> H[Portfolio Risk Check]
    H --> I[Exposure Limits]
    H --> J[Drawdown Monitor]
    H --> K[Circuit Breaker]

    I --> L{Pass?}
    J --> L
    K --> L

    L -->|Yes| M[Execute Trade]
    L -->|No| N[Reject / Reduce]

    M --> O[Stop Loss Manager]
    O --> P[Fixed Stop]
    O --> Q[Trailing Stop]
    O --> R[ATR Stop]

    classDef signal fill:#1a3a5c,stroke:#0d2137,color:#e8e0d4
    classDef sizer fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4
    classDef risk fill:#6b2d5b,stroke:#4a1e3f,color:#e8e0d4
    classDef check fill:#8b4513,stroke:#5c2e0d,color:#e8e0d4
    classDef output fill:#2d5050,stroke:#1a3a3a,color:#e8e0d4

    class A signal
    class B,C,D,E,F sizer
    class G,H,I,J,K risk
    class L,M,N check
    class O,P,Q,R output
```

This chapter is divided into three sections:

1. **[Position Sizing]({{ site.baseurl }}/24-risk-management/01-position-sizing)** -- Fixed fractional, Kelly criterion, and volatility-based methods for determining how much capital to allocate to each trade.
2. **[Stop Losses]({{ site.baseurl }}/24-risk-management/02-stop-losses)** -- Fixed, trailing, ATR, and time-based stops, plus a manager that combines multiple stop types per position.
3. **[Portfolio Risk Controls]({{ site.baseurl }}/24-risk-management/03-portfolio-risk-controls)** -- Drawdown monitoring, exposure limits, circuit breakers, Value at Risk, and concentration metrics.

---

## Best Practices

1. **Position Sizing**
   - Never risk more than 1-2% per trade
   - Use volatility-adjusted sizing
   - Scale position size with confidence

2. **Stop Losses**
   - Always use stops
   - Place stops beyond normal volatility
   - Consider using multiple stop types
   - Trail stops on profitable trades

3. **Portfolio Risk**
   - Monitor drawdown continuously
   - Limit total exposure
   - Implement circuit breakers
   - Track concentration metrics
   - Calculate VaR regularly

4. **Risk/Reward**
   - Target at least 2:1 reward/risk ratio
   - Use wider stops in volatile markets
   - Tighten stops in trending markets

## Related Chapters

- [Part 5: Portfolio Optimization]({{ site.baseurl }}/05-portfolio-optimization/) -- Mean-variance and risk-parity constraints that define the allocation boundaries risk controls enforce
- [Part 7: Backtesting]({{ site.baseurl }}/07-backtesting/) -- Walk-forward analysis reveals drawdown and risk profiles before going live
- [Part 23: Live Trading Execution]({{ site.baseurl }}/23-live-trading/) -- Position sizing and stop losses are applied in real time during order execution
- [Part 25: Monitoring & Analytics]({{ site.baseurl }}/25-monitoring-analytics/) -- Drawdown, VaR, and exposure metrics flow into dashboards for continuous risk monitoring

## Source Code

Browse the implementation: [`puffin/risk/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/risk)

## Next Steps

- See [Monitoring & Analytics]({{ site.baseurl }}/25-monitoring-analytics/) for performance tracking
- See [Live Trading]({{ site.baseurl }}/23-live-trading/01-live-trading) for integrating risk management
- Check `examples/` for complete implementations
