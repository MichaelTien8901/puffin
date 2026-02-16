---
layout: default
title: "Market Structure"
parent: "Part 1: Market Foundations"
nav_order: 1
---

# Market Structure

## Overview

Before building an algorithmic trading system, you need to understand how financial markets actually work. This chapter covers the infrastructure that enables trading: exchanges, order books, market participants, and the mechanics of price discovery.

## Exchanges

An **exchange** is a regulated marketplace where buyers and sellers meet to trade financial instruments. Major exchanges include:

| Exchange | Location | Asset Types | Trading Hours (ET) |
|----------|----------|-------------|-------------------|
| NYSE | New York | Equities, ETFs | 9:30 AM – 4:00 PM |
| NASDAQ | New York | Equities, ETFs | 9:30 AM – 4:00 PM |
| CME | Chicago | Futures, Options | Nearly 24h (Sun–Fri) |
| Binance | Global | Cryptocurrency | 24/7 |

{: .note }
US equity markets also have **pre-market** (4:00–9:30 AM) and **after-hours** (4:00–8:00 PM) sessions with lower liquidity.

### Electronic vs. Floor Trading

Modern exchanges are almost entirely electronic. Orders are matched by a **matching engine** — a software system that pairs buy and sell orders based on price-time priority.

## The Order Book

The **order book** is the core data structure of any exchange. It maintains all outstanding buy and sell orders for a given instrument.

{: .tip }
> **Plain English:** Think of the order book as a queue at a store, but instead of waiting in line, people can offer *different prices* to cut ahead. Buyers line up on one side saying "I'll pay this much," sellers line up on the other saying "I want at least this much," and whenever the prices meet, a trade happens.

```
         Order Book: AAPL
    ┌─────────────────────────┐
    │   ASK (Sell Orders)     │
    │   $150.05  ×  200       │
    │   $150.04  ×  500       │
    │   $150.03  ×  1,200     │ ← Best Ask (Lowest)
    │─────────────────────────│
    │   $150.02  ×  800       │ ← Best Bid (Highest)
    │   $150.01  ×  1,500     │
    │   $150.00  ×  3,000     │
    │   BID (Buy Orders)      │
    └─────────────────────────┘
         Spread: $0.01
```

### Key Concepts

- **Bid**: The highest price a buyer is willing to pay
- **Ask** (Offer): The lowest price a seller is willing to accept
- **Spread**: The difference between the best ask and best bid (`ask - bid`)
- **Depth**: The total quantity of orders at each price level
- **Mid-price**: The midpoint between bid and ask: `(bid + ask) / 2`

{: .tip }
> **Plain English — Liquidity:** Liquidity is how easy it is to sell something without lowering the price. Cash is perfectly liquid — you can spend it instantly. A house is illiquid — selling it takes time and you might have to accept a lower price. Stocks with tight spreads (like AAPL) are very liquid.

{: .tip }
A **tight spread** (small bid-ask gap) indicates high liquidity. Liquid instruments like AAPL or SPY typically have spreads of $0.01. Less liquid stocks may have spreads of $0.10 or more.

### Price-Time Priority

Most exchanges use **price-time priority** (also called FIFO) for order matching:

1. **Price priority**: Better-priced orders execute first (higher bids, lower asks)
2. **Time priority**: Among orders at the same price, earlier orders execute first

## Market Participants

Understanding who you're trading against is crucial for strategy design.

| Participant | Role | Time Horizon |
|-------------|------|-------------|
| **Market Makers** | Provide liquidity by quoting bid/ask | Seconds to minutes |

{: .tip }
> **Plain English:** A market maker is like a grocery store that always has milk in stock. Without the store, you'd have to find a farmer every time you wanted milk. Market makers "stock the shelves" of the market so there's always someone to buy from or sell to.


| **Institutional Investors** | Mutual funds, pension funds, insurance | Months to years |
| **Hedge Funds** | Diverse strategies, often quantitative | Minutes to months |
| **Retail Traders** | Individual investors | Days to years |
| **High-Frequency Traders** | Ultra-fast automated strategies | Microseconds to seconds |
| **Algorithmic Traders** | Systematic rule-based strategies | Minutes to weeks |

{: .important }
As an algorithmic trader, you're competing with all of these participants. Understanding their behavior helps you avoid strategies where you're at a structural disadvantage (e.g., competing on speed with HFT firms).

## Price Discovery

**Price discovery** is the process by which the market determines the price of an asset. It emerges from the interaction of supply and demand in the order book.

Prices move when:
- New information arrives (earnings, news, economic data)
- Large orders create supply/demand imbalance
- Market makers adjust quotes based on inventory and risk

```python
# Simple example: mid-price calculation
best_bid = 150.02
best_ask = 150.03
spread = best_ask - best_bid  # $0.01
mid_price = (best_bid + best_ask) / 2  # $150.025
```

## Dark Pools and Alternative Venues

Not all trading happens on public exchanges. **Dark pools** are private exchanges where large orders can execute without revealing size to the public order book. This is relevant because:

- Large institutional orders in dark pools can move prices when they eventually print to the tape
- Dark pool activity can signal institutional interest in a stock
- As retail/algorithmic traders, we primarily interact with lit (public) exchanges

## Summary

- Exchanges match buyers and sellers using electronic matching engines
- The order book shows all outstanding bids and asks with price-time priority
- The bid-ask spread reflects liquidity — tighter spreads mean more liquid markets
- Different market participants have different goals, time horizons, and advantages
- Price discovery emerges from the continuous interaction of orders in the book

## Next Steps

In the next chapter, we'll explore the different **asset classes** you can trade and their unique characteristics.
