---
layout: default
title: "Trading Strategies"
parent: "Part 6: Trading Strategies"
nav_order: 1
---

# Trading Strategies

## Overview

This chapter covers four classical trading strategies, the theory behind each, and their implementation in Puffin. All strategies share a common interface, making them interchangeable for backtesting and live trading.

## The Strategy Interface

Every strategy in Puffin implements the same contract:

```python
from puffin.strategies.base import Strategy, SignalFrame

class Strategy(ABC):
    def generate_signals(self, data: pd.DataFrame) -> SignalFrame:
        """Returns DataFrame with 'signal' (-1 to 1) and 'confidence' (0 to 1)."""

    def get_parameters(self) -> dict:
        """Returns current parameter values."""
```

This means you can swap any strategy into the backtester or live engine without code changes.

## 1. Momentum Strategy

**Theory**: Assets that have been rising tend to continue rising (and vice versa). This is one of the most well-documented anomalies in finance.

**Implementation**: Moving average crossover. When the short-term MA crosses above the long-term MA, the trend is up → buy. When it crosses below → sell.

```python
from puffin.strategies.momentum import MomentumStrategy

strategy = MomentumStrategy(
    short_window=20,    # 20-day fast MA
    long_window=50,     # 50-day slow MA
    ma_type="sma",      # Simple moving average (or "ema")
)

signals = strategy.generate_signals(data)
```

{: .tip }
EMA (exponential moving average) reacts faster to recent price changes than SMA. Try both and compare in your backtests.

## 2. Mean Reversion Strategy

**Theory**: Prices that deviate significantly from their historical mean tend to revert back. This works best in range-bound markets.

**Implementation**: Bollinger Bands with z-score. Buy when price is oversold (low z-score), sell when overbought (high z-score), exit at the mean.

```python
from puffin.strategies.mean_reversion import MeanReversionStrategy

strategy = MeanReversionStrategy(
    window=20,            # Lookback period
    num_std=2.0,          # Bollinger Band width
    zscore_entry=-2.0,    # Buy below this z-score
    zscore_exit=0.0,      # Exit at the mean
)
```

{: .warning }
Mean reversion strategies can suffer catastrophic losses in trending markets. Always pair with stop losses and regime detection.

## 3. Statistical Arbitrage (Pairs Trading)

**Theory**: Two historically correlated assets occasionally diverge. When they do, bet on convergence by going long the underperformer and short the outperformer.

**Implementation**: Cointegration testing to find pairs, then trade the spread.

```python
from puffin.strategies.stat_arb import StatArbStrategy

# Step 1: Find cointegrated pairs
pairs = StatArbStrategy.find_cointegrated_pairs(
    price_data,  # DataFrame with tickers as columns
    pvalue_threshold=0.05
)

# Step 2: Trade a pair
strategy = StatArbStrategy(
    lookback=60,
    entry_zscore=2.0,
    exit_zscore=0.5,
)
```

{: .note }
Cointegration is not the same as correlation. Two assets can be highly correlated but not cointegrated. Cointegration means they share a long-run equilibrium relationship.

## 4. Market Making

**Theory**: Profit from the bid-ask spread by providing liquidity. Place buy orders below and sell orders above the current price. Earn the spread on each round trip.

**Implementation**: Quote placement around the mid-price with volatility-adjusted spreads.

```python
from puffin.strategies.market_making import MarketMakingStrategy

strategy = MarketMakingStrategy(
    spread_bps=10,          # 10 basis points spread
    max_inventory=100,      # Maximum position size
    volatility_window=20,   # Lookback for vol calculation
)

quotes = strategy.get_quote_prices(mid_price=150.0)
# {'bid': 149.925, 'ask': 150.075, 'spread_bps': 10.0}
```

## Using the Strategy Registry

All built-in strategies are registered and can be instantiated by name:

```python
from puffin.strategies.registry import get_strategy, list_strategies

print(list_strategies())
# ['momentum', 'mean_reversion', 'stat_arb', 'market_making']

strategy = get_strategy("momentum", short_window=10, long_window=30)
```

## Strategy Comparison

| Strategy | Market Regime | Holding Period | Complexity |
|----------|--------------|----------------|------------|
| Momentum | Trending | Days–Weeks | Low |
| Mean Reversion | Range-bound | Hours–Days | Low |
| Stat Arb | Any (pair-dependent) | Days–Weeks | Medium |
| Market Making | Any (prefers calm) | Seconds–Minutes | High |

## Exercises

1. Run the momentum strategy on SPY with different window sizes (10/30, 20/50, 50/200). Which performs best?
2. Apply mean reversion to a range-bound stock. How does it perform on a trending stock?
3. Use `find_cointegrated_pairs` on a set of bank stocks (JPM, BAC, WFC, C, GS). Which pairs are cointegrated?

## Summary

- All strategies share the `Strategy` interface for interoperability
- Momentum follows trends; mean reversion bets against them
- Stat arb exploits relative mispricings between correlated assets
- Market making profits from the spread but requires careful inventory management
- No strategy works in all market conditions — regime awareness is critical

## Next Steps

In Part 4, we'll build the **backtesting engine** to test these strategies against historical data.
