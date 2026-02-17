---
layout: default
title: "Broker Comparison"
parent: "Part 23: Live Trading"
nav_order: 3
---

# Alpaca vs Interactive Brokers

## Feature Comparison

| Feature | Alpaca | Interactive Brokers |
|---------|--------|-------------------|
| **Asset classes** | US stocks, ETFs | Stocks, options, futures, forex, bonds, global markets |
| **Markets** | US only | 150+ markets in 33 countries |
| **Commission** | $0 stocks | Tiered (low per-share) |
| **API style** | REST + WebSocket | Socket-based (TWS API / ib_async) |
| **Latency** | ~50-100ms | ~1-10ms (local gateway) |
| **Paper trading** | Built-in | Requires paper account setup |
| **Data included** | Free real-time US equities | Requires market data subscriptions |
| **Setup complexity** | API keys only | Gateway/TWS installation + config |
| **Minimum account** | $0 | $0 (varies by product) |

## When to Use Each

### Choose Alpaca When

- You trade **US stocks and ETFs only**
- You want the **simplest setup** (API keys, no gateway)
- You need **free real-time data**
- You're **getting started** with algorithmic trading

### Choose IBKR When

- You need **options, futures, or forex**
- You trade **international markets**
- You need **lowest latency** (local gateway)
- You want **the widest product selection**
- You need **advanced order types** (bracket, OCA, etc.)

## Side-by-Side Code

Both brokers share the same `Broker` interface. Swapping is a one-line change:

### Setup

```python
# Alpaca
from puffin.broker import AlpacaBroker
broker = AlpacaBroker(paper=True)

# IBKR
from puffin.broker import IBKRBroker
broker = IBKRBroker(port=4002, paper=True)
```

### Submit an Order

```python
from puffin.broker import Order, OrderSide, OrderType

order = Order(symbol="AAPL", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)

# Identical call for both brokers
order_id = broker.submit_order(order)
```

### Check Positions

```python
# Identical for both
positions = broker.get_positions()
for symbol, pos in positions.items():
    print(f"{symbol}: {pos.qty} shares @ ${pos.avg_price:.2f}")
```

### Account Info

```python
# Identical for both
account = broker.get_account()
print(f"Equity: ${account.equity:,.2f}")
print(f"Buying power: ${account.buying_power:,.2f}")
```

## Data Provider Comparison

```python
# Alpaca
from puffin.data import AlpacaProvider
data = AlpacaProvider()

# IBKR
from puffin.data import IBKRDataProvider
data = IBKRDataProvider(port=4002)

# Same interface
df = data.fetch_historical("AAPL", start="2025-01-01", interval="1d")
```

| Feature | AlpacaProvider | IBKRDataProvider |
|---------|---------------|-----------------|
| Assets | US stocks, ETFs | Stocks, futures, forex, options |
| Real-time | WebSocket bars | 5-second real-time bars |
| Auth | API key env vars | Local gateway connection |
| Free data | Yes | Requires subscriptions |

## Migration Guide

Thanks to the shared `Broker` and `DataProvider` interfaces, migrating between brokers requires minimal changes:

1. **Change the import and constructor** — one line each
2. **Environment setup** — Alpaca needs API keys; IBKR needs Gateway running
3. **Everything else stays the same** — orders, positions, account info all use identical methods

```python
# Before (Alpaca)
from puffin.broker import AlpacaBroker
broker = AlpacaBroker(paper=True)

# After (IBKR) — only this line changes
from puffin.broker import IBKRBroker
broker = IBKRBroker(port=4002, paper=True)

# All downstream code is unchanged
order_id = broker.submit_order(order)
positions = broker.get_positions()
account = broker.get_account()
```

For IBKR-specific features (options, futures, forex), use `submit_order_with_spec()` with a `ContractSpec` — see [IBKR Advanced Trading](02-ibkr-advanced).
