---
layout: default
title: "Asset Classes"
parent: "Part 1: Market Foundations"
nav_order: 2
---

# Asset Classes

## Overview

An **asset class** is a group of financial instruments with similar characteristics. Puffin focuses on three primary asset classes: equities, ETFs, and cryptocurrency. Each has distinct properties that affect how you design and deploy trading strategies.

## Equities (Stocks)

A **stock** represents fractional ownership in a company. When you buy shares of AAPL, you own a small piece of Apple Inc.

### Key Characteristics

| Property | Details |
|----------|---------|
| Trading Hours | 9:30 AM – 4:00 PM ET (regular session) |
| Settlement | T+1 (trade date + 1 business day) |
| Minimum Tick | $0.01 for stocks above $1.00 |
| Commissions | $0 at most retail brokers (Alpaca, Robinhood) |
| Regulation | SEC (US), with pattern day trader rules |

### Corporate Actions

Stocks are subject to **corporate actions** that affect price and position:

- **Stock splits**: A 4:1 split converts 100 shares at $400 into 400 shares at $100
- **Dividends**: Cash payments to shareholders (ex-dividend date matters for strategies)
- **Mergers/Acquisitions**: Can cause sudden price changes and delistings

{: .warning }
Your backtesting data must be **split-adjusted** to avoid false signals. A 2:1 split would look like a 50% price drop in unadjusted data.

## ETFs (Exchange-Traded Funds)

An **ETF** is a fund that trades on an exchange like a stock but holds a basket of underlying assets. ETFs are popular for algorithmic trading due to their liquidity and diversification.

### Popular ETFs for Algorithmic Trading

| ETF | Tracks | Avg Daily Volume | Typical Spread |
|-----|--------|-----------------|----------------|
| SPY | S&P 500 | ~80M shares | $0.01 |
| QQQ | NASDAQ-100 | ~50M shares | $0.01 |
| IWM | Russell 2000 | ~25M shares | $0.01 |
| GLD | Gold | ~8M shares | $0.01 |
| TLT | 20+ Year Treasuries | ~15M shares | $0.01 |

### Advantages for Algo Trading

- **High liquidity**: Tight spreads and deep order books
- **Diversification**: Trade an entire index with one instrument
- **No single-stock risk**: Less susceptible to individual company events
- **Options available**: Most major ETFs have liquid options markets

## Cryptocurrency

Cryptocurrencies are digital assets traded on dedicated exchanges. They offer unique opportunities and risks for algorithmic trading.

### Key Characteristics

| Property | Details |
|----------|---------|
| Trading Hours | 24/7/365 |
| Settlement | Varies (minutes for most chains) |
| Minimum Tick | Varies by exchange and pair |
| Commissions | 0.1%–0.5% maker/taker fees |
| Regulation | Evolving, varies by jurisdiction |

### Crypto vs. Traditional Markets

| Factor | Equities/ETFs | Crypto |
|--------|--------------|--------|
| Market hours | 6.5h/day, weekdays | 24/7 |
| Volatility | Moderate (1-2% daily) | High (3-10% daily) |
| Regulation | Well-established | Emerging |
| Data availability | Excellent | Good but fragmented |
| Manipulation risk | Low (regulated) | Higher (less regulated) |
| Fragmentation | Few major exchanges | Many exchanges, price differences |

{: .tip }
The 24/7 nature of crypto markets means your trading system must handle continuous operation, weekend data, and no market open/close patterns.

## Comparing Asset Classes

```python
import pandas as pd

# Typical characteristics for strategy design
asset_comparison = pd.DataFrame({
    'Asset': ['Large Cap Equity', 'ETF (SPY)', 'Bitcoin'],
    'Avg Daily Volatility': ['1.5%', '1.0%', '4.0%'],
    'Spread Cost': ['$0.01', '$0.01', '0.05%'],
    'Trading Hours': ['6.5h', '6.5h', '24h'],
    'Data Cost': ['Free (yfinance)', 'Free (yfinance)', 'Free (many APIs)'],
    'Min Capital': ['$100', '$100', '$10'],
})
print(asset_comparison.to_string(index=False))
```

## Which Asset Class to Start With?

For learning algorithmic trading, we recommend starting with **US equities and ETFs**:

1. **Free data**: yfinance provides free historical data
2. **Well-understood**: Extensive research and literature available
3. **Regular hours**: Easier to reason about (no 24/7 complexity)
4. **Paper trading**: Alpaca offers free paper trading with real market data
5. **Low costs**: Commission-free trading at most brokers

{: .note }
Puffin tutorials use equities and ETFs for most examples, with crypto-specific guidance where applicable.

## Summary

- **Equities** offer ownership in companies with well-regulated, liquid markets
- **ETFs** provide diversified exposure and are ideal for algorithmic trading due to high liquidity
- **Cryptocurrency** trades 24/7 with higher volatility and fewer regulatory guardrails
- Start with equities/ETFs for learning — the concepts transfer to any asset class
- Always account for asset-specific characteristics (trading hours, settlement, corporate actions) in your strategies

## Next Steps

Now that you understand what you can trade, the next chapter covers **how** trades work — order types, execution mechanics, and costs.
