---
layout: default
title: "Chapter 1: Live Trading"
parent: "Part 23: Live Trading"
nav_order: 1
---

# Chapter 1: Live Trading

## Overview

You've built strategies, backtested them, and now you're ready to go live. This chapter covers everything you need to safely transition from backtesting to paper trading to live execution. We'll use Alpaca as our broker, but the same principles apply to any broker.

The `puffin.broker` module provides:
- **Broker abstraction**: Unified interface for any broker
- **Order management**: Track orders from submission to fill
- **Trading session management**: Handle market hours and calendars
- **Safety controls**: Prevent costly mistakes with validation and circuit breakers

{: .warning }
Live trading involves real money and real risk. Always start with paper trading, use safety controls, and never automate trading without extensive testing.

## Broker Abstraction

Puffin defines an abstract `Broker` interface so you can swap brokers without changing your trading code:

```python
from abc import ABC, abstractmethod
from puffin.broker import Broker, Order, Position, AccountInfo, OrderStatusInfo

class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order and return order ID."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Get all current positions."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Get account information."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatusInfo:
        """Get order status."""
```

All broker implementations follow this interface, making it easy to switch between paper and live trading, or between different brokers.

## Setting Up Alpaca

Alpaca offers commission-free trading with a modern API. Sign up at [alpaca.markets](https://alpaca.markets) and get your API keys.

### Installation

```bash
pip install alpaca-py
```

### Paper Trading Setup

Start with paper trading (no real money):

```python
from puffin.broker import AlpacaBroker

# Paper trading (default)
broker = AlpacaBroker(
    api_key="your-paper-api-key",
    secret_key="your-paper-secret-key",
    paper=True,  # Paper trading mode
)

# Check account
account = broker.get_account()
print(f"Equity: ${account.equity:,.2f}")
print(f"Cash: ${account.cash:,.2f}")
print(f"Buying Power: ${account.buying_power:,.2f}")
```

### Environment Variables

Store API keys in environment variables:

```bash
# .env file
ALPACA_API_KEY=your-api-key
ALPACA_SECRET_KEY=your-secret-key
ALPACA_PAPER=true
```

```python
import os
from puffin.broker import AlpacaBroker

broker = AlpacaBroker(
    api_key=os.environ["ALPACA_API_KEY"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
    paper=os.environ.get("ALPACA_PAPER", "true").lower() == "true",
)
```

## Setting Up Interactive Brokers (IBKR)

Interactive Brokers offers professional-grade execution, global market access, and low commissions. Puffin supports IBKR via the `ib_async` library which connects to TWS (Trader Workstation) or IB Gateway.

### Prerequisites

1. **Install IB Gateway or TWS** from [interactivebrokers.com](https://www.interactivebrokers.com/en/trading/tws.php)
2. **Enable API connections** in TWS/Gateway: File > Global Configuration > API > Settings:
   - Check "Enable ActiveX and Socket Clients"
   - Check "Allow connections from localhost only" (recommended)
   - Note the socket port (default: 7497 for TWS paper, 4002 for Gateway paper)

### Installation

```bash
pip install puffin[ibkr]
```

### Paper Trading Setup

```python
from puffin.broker import IBKRBroker

# Paper trading via IB Gateway (default)
broker = IBKRBroker(
    host="127.0.0.1",
    port=4002,       # Gateway paper trading port
    client_id=1,
    paper=True,
)

# Or via TWS
broker = IBKRBroker(port=7497)  # TWS paper trading port

# Check account
account = broker.get_account()
print(f"Equity: ${account.equity:,.2f}")
```

### Live Trading

```python
# IB Gateway live
broker = IBKRBroker(port=4001, paper=False)

# TWS live
broker = IBKRBroker(port=7496, paper=False)
```

### Environment Variables

```bash
# .env file
IBKR_HOST=127.0.0.1
IBKR_PORT=4002
IBKR_CLIENT_ID=1
```

```python
import os
from puffin.broker import IBKRBroker

broker = IBKRBroker(
    host=os.environ.get("IBKR_HOST", "127.0.0.1"),
    port=int(os.environ.get("IBKR_PORT", "4002")),
    client_id=int(os.environ.get("IBKR_CLIENT_ID", "1")),
)
```

### Common Ports Reference

| Application   | Paper Trading | Live Trading |
|--------------|:------------:|:------------:|
| IB Gateway   | 4002         | 4001         |
| TWS          | 7497         | 7496         |

### Disconnecting

Always disconnect when done to free the client ID slot:

```python
broker.disconnect()
```

## Placing Orders

### Order Types

Puffin supports all major order types:

```python
from puffin.broker import Order, OrderSide, OrderType, TimeInForce

# Market order - execute immediately at best available price
market_order = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    qty=100,
    type=OrderType.MARKET,
)

# Limit order - only execute at specified price or better
limit_order = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    qty=100,
    type=OrderType.LIMIT,
    limit_price=175.50,
    time_in_force=TimeInForce.GTC,  # Good til canceled
)

# Stop order - trigger market order when price reached
stop_order = Order(
    symbol="AAPL",
    side=OrderSide.SELL,
    qty=100,
    type=OrderType.STOP,
    stop_price=170.00,
)

# Stop-limit order - trigger limit order when stop price reached
stop_limit_order = Order(
    symbol="AAPL",
    side=OrderSide.SELL,
    qty=100,
    type=OrderType.STOP_LIMIT,
    stop_price=170.00,
    limit_price=169.50,
)
```

### Submitting Orders

```python
# Submit a market order
order_id = broker.submit_order(market_order)
print(f"Order submitted: {order_id}")

# Check order status
status = broker.get_order_status(order_id)
print(f"Status: {status.status}")
print(f"Filled: {status.filled_qty}/{status.qty}")
if status.avg_fill_price:
    print(f"Avg Fill Price: ${status.avg_fill_price:.2f}")
```

### Canceling Orders

```python
# Cancel a specific order
success = broker.cancel_order(order_id)
if success:
    print("Order canceled")
else:
    print("Failed to cancel order")
```

## Managing Positions

### Viewing Positions

```python
# Get all positions
positions = broker.get_positions()

for symbol, position in positions.items():
    print(f"\n{symbol}:")
    print(f"  Quantity: {position.qty}")
    print(f"  Avg Price: ${position.avg_price:.2f}")
    print(f"  Market Value: ${position.market_value:,.2f}")
    print(f"  P&L: ${position.unrealized_pnl:,.2f} ({position.side})")

# Get specific position
aapl_position = broker.get_position("AAPL")
if aapl_position:
    print(f"AAPL position: {aapl_position.qty} shares")
```

### Closing Positions

```python
# Close entire position
order_id = broker.close_position("AAPL")
print(f"Close order submitted: {order_id}")

# Close partial position
order_id = broker.close_position("AAPL", qty=50)
print(f"Closing 50 shares")

# Close all positions
results = broker.close_all_positions()
for symbol, order_id in results.items():
    if order_id:
        print(f"Closed {symbol}: order {order_id}")
    else:
        print(f"Failed to close {symbol}")
```

## Order Management

The `OrderManager` provides advanced order tracking and callbacks:

```python
from puffin.broker import OrderManager

# Create order manager
manager = OrderManager(broker)

# Submit order through manager
order = Order(symbol="AAPL", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)
order_id = manager.submit(order)

# Track order status
status = manager.track_order(order_id)
print(f"Order status: {status.status}")

# Track all pending orders
pending = manager.track_all_pending()
for order_id, status in pending.items():
    print(f"{order_id}: {status.status}")
```

### Order Callbacks

Register callbacks to react to order events:

```python
# Callback when order is filled
def on_fill(status):
    print(f"Order {status.order_id} filled!")
    print(f"  {status.symbol}: {status.filled_qty} @ ${status.avg_fill_price:.2f}")

# Callback when order is canceled
def on_cancel(status):
    print(f"Order {status.order_id} canceled")

# Callback when order is rejected
def on_reject(order, reason):
    print(f"Order rejected: {reason}")
    print(f"  {order.symbol} {order.side.value} {order.qty}")

# Register callbacks
manager.on_fill(on_fill)
manager.on_cancel(on_cancel)
manager.on_reject(on_reject)

# Now submit orders - callbacks will be triggered automatically
order_id = manager.submit(order)
manager.track_order(order_id)  # Triggers on_fill if filled
```

### Order History and Statistics

```python
# Get order history
history = manager.get_order_history("AAPL")  # Filter by symbol
for order in history[:10]:  # Last 10 orders
    print(f"{order.submitted_at}: {order.symbol} {order.side.value} {order.filled_qty}/{order.qty}")

# Get statistics
stats = manager.get_stats()
print(f"Total orders: {stats['total_orders']}")
print(f"Filled: {stats['filled']}")
print(f"Pending: {stats['pending']}")
print(f"Canceled: {stats['canceled']}")
print(f"Fill rate: {stats['fill_rate']:.1%}")

# Get pending and filled orders
pending_orders = manager.pending_orders()
filled_orders = manager.filled_orders()

print(f"\nPending: {len(pending_orders)}")
print(f"Filled: {len(filled_orders)}")
```

### Position Reconciliation

Ensure your tracked orders match actual broker positions:

```python
# Reconcile positions
discrepancies = manager.reconcile_positions()

if discrepancies:
    print("Position discrepancies found:")
    for disc in discrepancies:
        print(f"  {disc['symbol']}: expected {disc['expected_qty']}, "
              f"actual {disc['actual_qty']} (diff: {disc['diff']})")
else:
    print("All positions reconciled successfully")
```

## Trading Session Management

The `TradingSession` class handles market hours and calendars:

```python
from puffin.broker import TradingSession

# Create session
session = TradingSession(
    timezone="America/New_York",
    extended_hours=False,  # Regular hours only
)

# Check if market is open
if session.is_market_open():
    print("Market is open!")
else:
    print("Market is closed")
    next_open = session.next_open()
    print(f"Next open: {next_open}")

# Get time until open/close
time_until_open = session.time_until_open()
time_until_close = session.time_until_close()

print(f"Opens in: {time_until_open.total_seconds()/3600:.1f} hours")
print(f"Closes in: {time_until_close.total_seconds()/3600:.1f} hours")
```

### Market Calendar

```python
from datetime import datetime, timedelta

# Get trading days
start = datetime.now()
end = start + timedelta(days=30)
trading_days = session.get_trading_days(start, end)

print(f"Trading days in next 30 days: {len(trading_days)}")

# Check if today is a trading day
if session.is_trading_day():
    print("Today is a trading day")

# Get today's session schedule
schedule = session.get_session_schedule()
if schedule["trading_day"]:
    print(f"Regular open: {schedule['regular_open']}")
    print(f"Regular close: {schedule['regular_close']}")
```

### Extended Hours Trading

```python
# Enable extended hours (pre-market 4am-9:30am, after-hours 4pm-8pm ET)
extended_session = TradingSession(extended_hours=True)

if extended_session.supports_extended_hours():
    print("Extended hours enabled")

# Wait for market to open (async)
import asyncio

async def trade_when_open():
    await session.wait_for_open()
    print("Market is now open, placing orders...")
    # Place orders here

asyncio.run(trade_when_open())

# Or use synchronous version
session.wait_for_open_sync()
```

## Safety Controls

The `SafetyController` prevents costly mistakes:

```python
from puffin.broker import SafetyController

# Create safety controller
safety = SafetyController(
    broker=broker,
    max_order_size=1000,  # Max 1000 shares per order
    max_position_size=5000,  # Max 5000 shares per position
    max_daily_loss=5000.0,  # Max $5000 loss per day
    max_total_position_value=100000.0,  # Max $100k total positions
    require_confirmation=True,  # Require explicit confirmation
)

# Confirm live trading (required before first order)
safety.confirm_live_trading()
# Will prompt: Type 'CONFIRM' to proceed

# Or provide confirmation code programmatically
safety.confirm_live_trading("CONFIRM")
```

### Order Validation

```python
# Validate order before submitting
order = Order(symbol="AAPL", side=OrderSide.BUY, qty=100, type=OrderType.MARKET)

is_valid, reason = safety.validate_order(order)
if is_valid:
    order_id = broker.submit_order(order)
    print(f"Order submitted: {order_id}")
else:
    print(f"Order rejected: {reason}")

# Example rejections:
# - "Order size 2000 exceeds max_order_size 1000"
# - "Order would result in position size 6000, exceeding max_position_size 5000"
# - "Daily loss $-5500 exceeds limit $-5000. Circuit breaker activated."
```

### Circuit Breaker

The circuit breaker stops all trading when daily loss limit is reached:

```python
# Check daily P&L
daily_pnl = safety.check_daily_pnl()
print(f"Today's P&L: ${daily_pnl:,.2f}")

# Check circuit breaker status
if safety.is_circuit_breaker_active():
    print("Circuit breaker active - trading disabled")

    # Admin override (use with caution!)
    safety.reset_circuit_breaker()
```

### Custom Validators

Add your own validation logic:

```python
# Define custom validator
def min_price_validator(order: Order) -> tuple[bool, str]:
    """Reject orders for stocks under $10."""
    # You'd fetch real price here
    price = 5.0  # Example
    if price < 10.0:
        return False, f"Price ${price:.2f} below minimum $10"
    return True, ""

# Add to safety controller
safety.add_validator(min_price_validator)

# Now all orders will be checked by custom validator
```

### Pre-built Validators

Puffin includes common validators:

```python
from puffin.broker import (
    PositionSizingValidator,
    TradingHoursValidator,
    SymbolWhitelistValidator,
)

# Position sizing (max 25% of portfolio per position)
position_validator = PositionSizingValidator(max_position_pct=0.25)

# Only trade during regular hours
hours_validator = TradingHoursValidator(allow_extended_hours=False)

# Only trade specific symbols
whitelist_validator = SymbolWhitelistValidator(
    allowed_symbols=["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]
)

# Add all validators
for validator in [position_validator, hours_validator, whitelist_validator]:
    safety.add_validator(lambda order: validator(broker, order))
```

## Complete Trading Bot Example

Here's a complete example that ties everything together:

```python
import os
from datetime import datetime
from puffin.broker import (
    AlpacaBroker,
    OrderManager,
    SafetyController,
    TradingSession,
    Order,
    OrderSide,
    OrderType,
)
from puffin.strategies import MomentumStrategy  # Your strategy

# Initialize components
broker = AlpacaBroker(
    api_key=os.environ["ALPACA_API_KEY"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
    paper=True,  # Start with paper trading
)

manager = OrderManager(broker)
session = TradingSession()
safety = SafetyController(
    broker=broker,
    max_order_size=500,
    max_position_size=2000,
    max_daily_loss=1000.0,
    require_confirmation=True,
)

# Confirm live trading
safety.confirm_live_trading()

# Register callbacks
def on_fill(status):
    print(f"✓ Filled: {status.symbol} {status.side.value} {status.filled_qty} @ ${status.avg_fill_price:.2f}")

manager.on_fill(on_fill)

# Initialize strategy
strategy = MomentumStrategy()
watchlist = ["AAPL", "TSLA", "NVDA", "MSFT"]

# Trading loop
print("Starting trading bot...")

while True:
    # Wait for market to open
    if not session.is_market_open():
        print(f"Market closed. Next open: {session.next_open()}")
        session.wait_for_open_sync(check_interval=300)  # Check every 5 min

    # Check circuit breaker
    daily_pnl = safety.check_daily_pnl()
    if safety.is_circuit_breaker_active():
        print(f"Circuit breaker active (P&L: ${daily_pnl:,.2f}). Stopping.")
        break

    # Generate signals
    signals = strategy.generate_signals(watchlist)

    # Execute trades
    for symbol, signal in signals.items():
        if signal > 0:  # Buy signal
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                qty=100,
                type=OrderType.MARKET,
            )

            # Validate before submitting
            is_valid, reason = safety.validate_order(order)
            if is_valid:
                try:
                    order_id = manager.submit(order)
                    print(f"→ Submitted buy order for {symbol}: {order_id}")
                except Exception as e:
                    print(f"✗ Failed to submit order: {e}")
            else:
                print(f"✗ Order validation failed: {reason}")

        elif signal < 0:  # Sell signal
            position = broker.get_position(symbol)
            if position and position.qty > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=position.qty,
                    type=OrderType.MARKET,
                )
                order_id = manager.submit(order)
                print(f"→ Submitted sell order for {symbol}: {order_id}")

    # Track pending orders
    pending = manager.track_all_pending()
    print(f"Tracking {len(pending)} pending orders")

    # Print daily stats
    stats = manager.get_stats()
    print(f"\nDaily Stats:")
    print(f"  Total orders: {stats['total_orders']}")
    print(f"  Fill rate: {stats['fill_rate']:.1%}")
    print(f"  P&L: ${daily_pnl:,.2f}")

    # Wait before next iteration
    import time
    time.sleep(60)  # Check every minute
```

## Best Practices

### 1. Always Start with Paper Trading

Never go straight to live trading:

```python
# Start here
broker = AlpacaBroker(api_key=key, secret_key=secret, paper=True)

# Run for weeks/months, verify everything works

# Only then switch to live
broker = AlpacaBroker(api_key=key, secret_key=secret, paper=False)
```

### 2. Use Safety Controls

Always use a `SafetyController`:

```python
safety = SafetyController(
    broker=broker,
    max_order_size=1000,
    max_position_size=5000,
    max_daily_loss=5000.0,
    require_confirmation=True,
)

# Validate EVERY order
is_valid, reason = safety.validate_order(order)
if is_valid:
    broker.submit_order(order)
```

### 3. Start Small

Begin with tiny positions:

```python
# Start with 1-10 shares
order = Order(symbol="AAPL", side=OrderSide.BUY, qty=1, type=OrderType.MARKET)

# Gradually increase after proving consistency
```

### 4. Log Everything

Maintain detailed logs:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
logger.info(f"Submitted order: {order_id}")
```

### 5. Monitor Actively

Don't set and forget:

```python
# Check positions regularly
positions = broker.get_positions()
account = broker.get_account()

print(f"Equity: ${account.equity:,.2f}")
print(f"Positions: {len(positions)}")

# Send alerts for important events
def on_fill(status):
    send_alert(f"Order filled: {status.symbol}")

def on_reject(order, reason):
    send_alert(f"Order rejected: {reason}")
```

### 6. Handle Errors Gracefully

```python
try:
    order_id = broker.submit_order(order)
except InsufficientFundsError:
    logger.error("Insufficient funds - skipping order")
except OrderRejectedError as e:
    logger.error(f"Order rejected: {e}")
except BrokerError as e:
    logger.error(f"Broker error: {e}")
    # Maybe retry, or exit safely
```

### 7. Test Order Execution

Test all order types before going live:

```python
# Test market orders
test_market_order()

# Test limit orders
test_limit_order()

# Test stop orders
test_stop_order()

# Test cancellations
test_cancel_order()

# Test position closing
test_close_position()
```

## Transitioning from Backtest to Live

### 1. Adapt Your Strategy

Your backtest strategy needs minor changes for live trading:

```python
class LiveMomentumStrategy(MomentumStrategy):
    def __init__(self, broker, safety):
        super().__init__()
        self.broker = broker
        self.safety = safety

    def execute_signals(self, signals):
        """Execute signals with safety checks."""
        for symbol, signal in signals.items():
            if abs(signal) < 0.5:  # Ignore weak signals
                continue

            order = self._build_order(symbol, signal)

            # Validate before submitting
            is_valid, reason = self.safety.validate_order(order)
            if is_valid:
                self.broker.submit_order(order)
            else:
                logger.warning(f"Order rejected: {reason}")

    def _build_order(self, symbol, signal):
        # Your order construction logic
        ...
```

### 2. Handle Real-Time Data

Use streaming data instead of historical:

```python
from puffin.data import AlpacaStreamProvider

stream = AlpacaStreamProvider(api_key=key, secret_key=secret)

def on_bar(bar):
    """Handle real-time bar data."""
    signals = strategy.generate_signals([bar.symbol])
    strategy.execute_signals(signals)

stream.subscribe_bars(watchlist, on_bar)
stream.start()
```

### 3. Account for Slippage

Live execution has slippage:

```python
# In backtest: assume fill at exact price
# In live: expect slippage

# Use limit orders to control slippage
order = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    qty=100,
    type=OrderType.LIMIT,
    limit_price=current_price * 1.001,  # 0.1% slippage tolerance
)
```

## Next Steps

Now that you understand live trading, you're ready to add risk management:

- [Part 24: Risk Management](../../24-risk-management/) - Position sizing, stop losses, portfolio controls

## Further Reading

- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [Trading System Best Practices](https://www.investopedia.com/articles/trading/11/automated-trading-systems.asp)
- [Risk Management for Algo Trading](https://www.quantstart.com/articles/Risk-Management-for-Algorithmic-Trading/)
