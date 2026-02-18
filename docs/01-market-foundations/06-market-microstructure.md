---
layout: default
title: "Market Microstructure"
parent: "Part 1: Market Foundations"
nav_order: 6
---

# Market Microstructure

## Overview

Market microstructure studies the mechanics of how markets actually function at the lowest level: how orders are transmitted, how prices form, and how liquidity providers interact with liquidity takers. Understanding microstructure is critical for high-frequency trading, execution algorithms, and even for longer-term strategies that need optimal order execution.

## ITCH Data Format

NASDAQ ITCH is a raw market data feed that provides every message sent to the exchange matching engine. Unlike traditional trade/quote data, ITCH gives you the full order book reconstruction capability.

### Key ITCH Message Types

| Message Type | Code | Description |
|-------------|------|-------------|
| System Event | `S` | Market open/close, trading halts |
| Stock Directory | `R` | Stock information and trading status |
| Add Order | `A` | New limit order added to book |
| Add Order with MPID | `F` | New limit order with market participant ID |
| Execute Order | `E` | Full or partial execution |
| Cancel Order | `X` | Order cancellation |
| Delete Order | `D` | Order deletion |
| Replace Order | `U` | Order replacement (price/size change) |
| Trade | `P` | Non-displayed trade |

### Parsing ITCH Messages

ITCH messages are binary and fixed-length. Each message type has a specific structure defined in the NASDAQ specification.

```python
import struct
from dataclasses import dataclass
from typing import BinaryIO

@dataclass
class AddOrderMessage:
    timestamp: int
    order_ref: int
    side: str  # 'B' or 'S'
    shares: int
    symbol: str
    price: int  # Price in 1/10000 dollars

    @classmethod
    def parse(cls, data: bytes) -> "AddOrderMessage":
        # ITCH Add Order message is 36 bytes
        # Message type (1) + Stock locate (2) + Tracking (2) + Timestamp (6) +
        # Order ref (8) + Side (1) + Shares (4) + Symbol (8) + Price (4)
        unpacked = struct.unpack(">HHQ QcI 8s I", data[1:])

        return cls(
            timestamp=unpacked[2],
            order_ref=unpacked[3],
            side=unpacked[4].decode("ascii"),
            shares=unpacked[5],
            symbol=unpacked[6].decode("ascii").strip(),
            price=unpacked[7],
        )

@dataclass
class ExecuteOrderMessage:
    timestamp: int
    order_ref: int
    shares: int
    match_number: int

    @classmethod
    def parse(cls, data: bytes) -> "ExecuteOrderMessage":
        unpacked = struct.unpack(">HHQ QIQ", data[1:])
        return cls(
            timestamp=unpacked[2],
            order_ref=unpacked[3],
            shares=unpacked[4],
            match_number=unpacked[5],
        )

class ITCHParser:
    def __init__(self):
        self.order_book: dict[int, AddOrderMessage] = {}
        self.trades: list[tuple[int, int, int]] = []  # (timestamp, price, shares)

    def process_message(self, msg: bytes):
        msg_type = chr(msg[0])

        if msg_type == "A":
            order = AddOrderMessage.parse(msg)
            self.order_book[order.order_ref] = order

        elif msg_type == "E":
            execution = ExecuteOrderMessage.parse(msg)
            if execution.order_ref in self.order_book:
                order = self.order_book[execution.order_ref]
                self.trades.append((execution.timestamp, order.price, execution.shares))

                # Update or remove order
                order.shares -= execution.shares
                if order.shares == 0:
                    del self.order_book[execution.order_ref]

        elif msg_type in ("X", "D"):
            # Cancel or Delete
            order_ref = struct.unpack(">Q", msg[11:19])[0]
            self.order_book.pop(order_ref, None)

    def parse_file(self, file_path: str):
        with open(file_path, "rb") as f:
            while True:
                # Read message length (first 2 bytes)
                length_bytes = f.read(2)
                if not length_bytes:
                    break

                length = struct.unpack(">H", length_bytes)[0]
                msg = f.read(length)
                self.process_message(msg)
```

{: .note }
Real ITCH parsing requires handling all message types and proper byte alignment. Libraries like `itch-parser` can handle the complexity for you.

## FIX Protocol Basics

Financial Information eXchange (FIX) is the industry-standard protocol for electronic trading communications. It's text-based (unlike ITCH's binary format) and used for order routing, execution, and post-trade messaging.

### FIX Message Structure

FIX messages are composed of tag-value pairs separated by ASCII SOH (Start of Header, `\x01`) characters.

```
8=FIX.4.4|9=154|35=D|49=SENDER|56=TARGET|34=1|52=20260216-12:30:00|
11=ORDER123|21=1|55=AAPL|54=1|60=20260216-12:30:00|40=2|44=150.00|38=100|10=123|
```

(Where `|` represents the SOH character)

### Common FIX Tags

| Tag | Name | Description |
|-----|------|-------------|
| 8 | BeginString | FIX version (e.g., FIX.4.4) |
| 35 | MsgType | Message type (D=New Order, 8=Execution Report) |
| 49 | SenderCompID | Sender identification |
| 55 | Symbol | Ticker symbol |
| 54 | Side | 1=Buy, 2=Sell |
| 38 | OrderQty | Order quantity |
| 40 | OrdType | 1=Market, 2=Limit |
| 44 | Price | Limit price |
| 150 | ExecType | 0=New, 1=Partial Fill, 2=Fill, 4=Canceled |

### Basic FIX Message Parser

```python
from datetime import datetime

class FIXMessage:
    def __init__(self, msg_string: str, delimiter: str = "\x01"):
        self.fields: dict[int, str] = {}
        self.delimiter = delimiter
        self._parse(msg_string)

    def _parse(self, msg_string: str):
        pairs = msg_string.split(self.delimiter)
        for pair in pairs:
            if "=" in pair:
                tag, value = pair.split("=", 1)
                self.fields[int(tag)] = value

    def get(self, tag: int, default=None) -> str | None:
        return self.fields.get(tag, default)

    def msg_type(self) -> str:
        return self.get(35, "")

    def symbol(self) -> str:
        return self.get(55, "")

    def side(self) -> str:
        side_code = self.get(54, "")
        return "BUY" if side_code == "1" else "SELL" if side_code == "2" else "UNKNOWN"

    def price(self) -> float | None:
        price_str = self.get(44)
        return float(price_str) if price_str else None

    def quantity(self) -> int | None:
        qty_str = self.get(38)
        return int(qty_str) if qty_str else None

# Example usage
fix_msg = FIXMessage(
    "8=FIX.4.4\x0135=D\x0155=AAPL\x0154=1\x0138=100\x0140=2\x0144=150.00"
)
print(f"Symbol: {fix_msg.symbol()}")  # AAPL
print(f"Side: {fix_msg.side()}")      # BUY
print(f"Qty: {fix_msg.quantity()}")   # 100
print(f"Price: {fix_msg.price()}")    # 150.0
```

## Tick-to-Bar Conversion

Raw tick data (individual trades/quotes) is too granular for most strategies. We aggregate ticks into "bars" (OHLCV candles). The choice of bar type significantly impacts strategy performance.

### Time Bars

Traditional bars based on fixed time intervals (1 min, 5 min, 1 hour, 1 day).

```python
import pandas as pd

def create_time_bars(ticks: pd.DataFrame, interval: str = "5min") -> pd.DataFrame:
    """
    Convert tick data to time-based OHLCV bars.

    Args:
        ticks: DataFrame with columns [timestamp, price, volume]
        interval: Pandas resample frequency string

    Returns:
        OHLCV bars
    """
    ticks = ticks.set_index("timestamp")

    bars = pd.DataFrame({
        "Open": ticks["price"].resample(interval).first(),
        "High": ticks["price"].resample(interval).max(),
        "Low": ticks["price"].resample(interval).min(),
        "Close": ticks["price"].resample(interval).last(),
        "Volume": ticks["volume"].resample(interval).sum(),
    })

    return bars.dropna()
```

**Problem with time bars**: Market activity is not uniform. Pre-market is quiet, mid-day is active, close is volatile. Fixed time bars don't adapt to information flow.

### Volume Bars

Bars formed after a fixed number of shares/contracts trade. More bars during high activity, fewer during low activity.

```python
def create_volume_bars(ticks: pd.DataFrame, volume_threshold: int) -> pd.DataFrame:
    """
    Create bars based on cumulative volume.

    Args:
        ticks: DataFrame with columns [timestamp, price, volume]
        volume_threshold: Number of shares per bar

    Returns:
        OHLCV bars
    """
    bars = []
    cumulative_volume = 0
    bar_data: dict = {}

    for _, tick in ticks.iterrows():
        if not bar_data:
            bar_data = {
                "timestamp": tick["timestamp"],
                "Open": tick["price"],
                "High": tick["price"],
                "Low": tick["price"],
                "Close": tick["price"],
                "Volume": 0,
            }

        bar_data["High"] = max(bar_data["High"], tick["price"])
        bar_data["Low"] = min(bar_data["Low"], tick["price"])
        bar_data["Close"] = tick["price"]
        bar_data["Volume"] += tick["volume"]
        cumulative_volume += tick["volume"]

        if cumulative_volume >= volume_threshold:
            bars.append(bar_data)
            bar_data = {}
            cumulative_volume = 0

    df = pd.DataFrame(bars)
    if not df.empty:
        df = df.set_index("timestamp")
    return df
```

**Advantage**: Bars reflect actual trading activity. Liquid periods get more bars, illiquid periods get fewer.

### Dollar Bars

Bars formed after a fixed dollar volume (price × volume) trades. This normalizes for both volume and price changes.

```python
def create_dollar_bars(ticks: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
    """
    Create bars based on cumulative dollar volume.

    Args:
        ticks: DataFrame with columns [timestamp, price, volume]
        dollar_threshold: Dollar volume per bar

    Returns:
        OHLCV bars
    """
    bars = []
    cumulative_dollars = 0.0
    bar_data: dict = {}

    for _, tick in ticks.iterrows():
        if not bar_data:
            bar_data = {
                "timestamp": tick["timestamp"],
                "Open": tick["price"],
                "High": tick["price"],
                "Low": tick["price"],
                "Close": tick["price"],
                "Volume": 0,
            }

        bar_data["High"] = max(bar_data["High"], tick["price"])
        bar_data["Low"] = min(bar_data["Low"], tick["price"])
        bar_data["Close"] = tick["price"]
        bar_data["Volume"] += tick["volume"]

        dollar_volume = tick["price"] * tick["volume"]
        cumulative_dollars += dollar_volume

        if cumulative_dollars >= dollar_threshold:
            bars.append(bar_data)
            bar_data = {}
            cumulative_dollars = 0.0

    df = pd.DataFrame(bars)
    if not df.empty:
        df = df.set_index("timestamp")
    return df
```

**Advantage**: Best for portfolios with different price ranges. $1M in AAPL trades vs $1M in SPY trades represents similar market impact.

### Comparison of Bar Types

```python
# Example: Process the same tick data with all three methods
ticks = pd.DataFrame({
    "timestamp": pd.date_range("2026-02-16 09:30", periods=10000, freq="1s"),
    "price": 150 + (pd.Series(range(10000)) * 0.01).cumsum().apply(lambda x: x % 10 - 5),
    "volume": pd.Series([100] * 10000) + pd.Series(range(10000)).apply(lambda x: int(x % 500)),
})

time_bars = create_time_bars(ticks.copy(), interval="5min")
volume_bars = create_volume_bars(ticks.copy(), volume_threshold=50000)
dollar_bars = create_dollar_bars(ticks.copy(), dollar_threshold=7500000)

print(f"Time bars: {len(time_bars)}")
print(f"Volume bars: {len(volume_bars)}")
print(f"Dollar bars: {len(dollar_bars)}")
```

## Order Flow Analysis

Order flow examines the imbalance between buy and sell orders to predict short-term price movements.

### Volume-Weighted Average Price (VWAP)

VWAP is the average price weighted by volume — the "fair price" for a trading period.

```python
def calculate_vwap(bars: pd.DataFrame) -> pd.Series:
    """Calculate volume-weighted average price."""
    typical_price = (bars["High"] + bars["Low"] + bars["Close"]) / 3
    return (typical_price * bars["Volume"]).cumsum() / bars["Volume"].cumsum()
```

### Order Imbalance

The difference between buy volume and sell volume. Requires tick-level data with trade direction (aggressor side).

```python
def calculate_order_imbalance(ticks: pd.DataFrame) -> pd.Series:
    """
    Calculate order imbalance from tick data.

    Requires 'side' column: 'buy' for buy-side aggressor, 'sell' for sell-side.
    """
    ticks = ticks.copy()
    ticks["signed_volume"] = ticks["volume"] * ticks["side"].map({"buy": 1, "sell": -1})
    return ticks["signed_volume"].cumsum()
```

### Bid-Ask Spread

The difference between best bid and best ask prices. Wide spreads indicate low liquidity.

```python
def calculate_spread_metrics(quotes: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate spread metrics from quote data.

    Args:
        quotes: DataFrame with columns [timestamp, bid, ask, bid_size, ask_size]
    """
    quotes = quotes.copy()
    quotes["spread"] = quotes["ask"] - quotes["bid"]
    quotes["spread_bps"] = (quotes["spread"] / quotes["bid"]) * 10000
    quotes["mid_price"] = (quotes["bid"] + quotes["ask"]) / 2
    return quotes
```

## Market Impact Models

Market impact measures how much your order moves the price. Critical for large orders.

### Square-Root Model (Almgren-Chriss)

Price impact grows with the square root of order size:

```
Impact = σ × √(Q / V)
```

Where:
- σ = volatility
- Q = order size
- V = average daily volume

```python
def estimate_market_impact(
    order_size: int,
    daily_volume: int,
    volatility: float,
    participation_rate: float = 0.1,
) -> float:
    """
    Estimate market impact using square-root model.

    Args:
        order_size: Shares to trade
        daily_volume: Average daily volume
        volatility: Daily volatility (standard deviation of returns)
        participation_rate: Fraction of volume you're taking

    Returns:
        Estimated price impact in basis points
    """
    impact_bps = volatility * (order_size / daily_volume) ** 0.5 * participation_rate * 10000
    return impact_bps
```

### Example: Evaluating Execution Quality

```python
# You want to buy 50,000 shares of AAPL
# AAPL trades 50M shares/day on average
# Volatility: 2% daily

order_size = 50_000
daily_volume = 50_000_000
volatility = 0.02

impact = estimate_market_impact(order_size, daily_volume, volatility)
print(f"Expected impact: {impact:.2f} bps")

# If current price is $150, expected slippage:
price = 150
slippage_dollars = price * (impact / 10000)
total_slippage = slippage_dollars * order_size
print(f"Total slippage cost: ${total_slippage:.2f}")
```

{: .important }
Market impact is why institutional traders use execution algorithms (TWAP, VWAP, POV) to slice large orders over time. Dumping a 50K share order as a market order would likely cost 10-20 bps in slippage.

## Summary

- ITCH provides raw order book data for full market depth reconstruction
- FIX protocol is the standard for order routing and execution communication
- Bar types matter: time bars are simple, volume/dollar bars adapt to market activity
- Order flow analysis reveals short-term supply/demand imbalances
- Market impact models help optimize execution for large orders

## Next Steps

With a solid understanding of market foundations, we're ready to move to **Part 2: Data Pipeline**, where we'll build systems to fetch, store, and preprocess market data for analysis and trading.
