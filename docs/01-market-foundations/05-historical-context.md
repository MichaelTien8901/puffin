---
layout: default
title: "Historical Context"
parent: "Part 1: Market Foundations"
nav_order: 5
---

# Historical Context

## Overview

Understanding past market events is essential for building robust trading systems. These case studies illustrate how extreme events can break assumptions that work in "normal" markets.

## Case Study 1: The 2010 Flash Crash

**Date**: May 6, 2010
**What happened**: The Dow Jones dropped nearly 1,000 points (about 9%) in under 5 minutes, then recovered most of the loss over the next ~20 minutes. Some stocks traded at $0.01 while others traded at $100,000.

**Cause**: A large institutional sell order (Waddell & Reed) executed via an algorithm that sold E-mini S&P 500 futures based purely on volume, without regard to price or time. This triggered a cascade of selling by HFT firms and other algorithms.

### Implications for Algorithmic Trading

- **Stop-loss orders can be devastating**: Stop orders triggered during the crash sold at absurd prices
- **Liquidity can evaporate instantly**: Order books emptied as market makers pulled quotes
- **Market orders are dangerous in volatile conditions**: Use limit orders for protection
- **Circuit breakers matter**: Exchanges now have "limit up/limit down" bands that halt trading when prices move too fast

{: .warning }
Always use limit orders or stop-limit orders rather than market/stop orders for automated systems. A flash crash can fill your stop order at a price far worse than intended.

## Case Study 2: The August 2015 ETF Dislocation

**Date**: August 24, 2015
**What happened**: Many ETFs traded at significant discounts to their net asset value (NAV) at the market open. SPY, the most liquid ETF in the world, opened 5% below its fair value.

**Cause**: Uncertainty from China's market crash combined with a flood of market-on-open sell orders. Market makers couldn't properly price ETFs because many underlying stocks hadn't opened yet.

### Implications for Algorithmic Trading

- **Market opens are dangerous**: Avoid trading in the first 15-30 minutes when prices are most volatile
- **ETF pricing depends on underlying liquidity**: Even liquid ETFs can misprice when underlying stocks aren't trading
- **Mean reversion opportunities exist**: Dislocations like this can be profitable if you can identify them in real-time
- **Gap risk is real**: Overnight positions can gap significantly at the open

## Case Study 3: The COVID-19 Crash (March 2020)

**Date**: February–March 2020
**What happened**: The S&P 500 fell 34% in 23 trading days — the fastest bear market in history. Volatility (VIX) spiked to 82, and multiple circuit breakers were triggered. This was followed by one of the fastest recoveries, with markets reaching new highs by August.

### Implications for Algorithmic Trading

- **Regime changes break models**: Strategies trained on low-volatility data fail when volatility spikes 5-10x
- **Correlation spikes in crises**: "Everything sells off together" — diversification benefits disappear when you need them most
- **Circuit breakers halt trading**: Your algorithm must handle market halts gracefully
- **Recovery can be as fast as the crash**: Staying out of the market after a crash has its own risk (missing the recovery)
- **Volatility-adjusted position sizing is critical**: Fixed-size positions in a high-volatility regime lead to outsized losses

{: .important }
The COVID crash is an excellent stress test for any backtesting system. If your strategy survives March 2020, it has passed a meaningful robustness check.

## Case Study 4: The GameStop Short Squeeze (January 2021)

**Date**: January 2021
**What happened**: GameStop (GME) rose from $17 to $483 in two weeks, driven by retail traders coordinating on Reddit's WallStreetBets forum. Several hedge funds with large short positions lost billions. Brokers restricted buying, causing widespread controversy.

### Implications for Algorithmic Trading

- **Short squeezes can be extreme and irrational**: Mean reversion strategies on heavily shorted stocks can produce catastrophic losses
- **Social media is a market-moving force**: Sentiment analysis must include Reddit, Twitter/X, and other social platforms
- **Broker risk is real**: Your broker may restrict trading during extreme volatility
- **Short interest data is valuable**: Monitoring short interest can help avoid or exploit squeeze situations

## Lessons for System Design

These events inform several design decisions in Puffin:

| Lesson | System Feature |
|--------|---------------|
| Liquidity can vanish | Use limit orders, not market orders |
| Volatility regimes change | Volatility-adjusted position sizing |
| Markets can halt | Graceful handling of trading halts |
| Correlations spike in crises | Portfolio-level risk controls |
| Flash crashes happen | Maximum drawdown circuit breaker |
| Social media moves markets | AI-assisted sentiment analysis |
| Overnight gaps | Risk management for overnight positions |

## Summary

- Market crashes reveal the fragility of assumptions built into most trading models
- Flash crashes, ETF dislocations, pandemic selloffs, and short squeezes each teach different lessons
- Robust trading systems must handle regime changes, liquidity crises, and extreme volatility
- Historical stress testing against these events is essential for validating strategy robustness

## Next Steps

You now have a solid foundation in how markets work. In **Part 2**, we'll start building the **data pipeline** — fetching, caching, and preprocessing market data for analysis and trading.
