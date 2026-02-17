## 1. Streaming Data Types & Provider Extensions

- [x] 1.1 Create `puffin/data/realtime.py` with `Trade`, `Quote`, and `OrderBookUpdate` dataclasses
- [x] 1.2 Add `OrderBook` class with bid/ask tracking, `best_bid`, `best_ask`, `spread` properties, and level add/remove logic
- [x] 1.3 Add `stream_trades()`, `stream_quotes()`, and `stream_orderbook()` optional methods to `DataProvider` base class (default `NotImplementedError`)

## 2. RealtimeEngine Core

- [x] 2.1 Implement `RealtimeEngine.__init__(provider)` with callback registries and state tracking
- [x] 2.2 Implement `start(symbols)` and `stop()` lifecycle methods with thread management
- [x] 2.3 Implement `on_trade()`, `on_quote()`, `on_book_update()`, and `on_bar()` callback registration
- [x] 2.4 Wire engine to call provider streaming methods and dispatch typed messages to registered callbacks

## 3. Tick-to-Bar Aggregation

- [x] 3.1 Implement `set_bar_interval(seconds)` and per-symbol aggregation buffers
- [x] 3.2 Implement time-window bar completion logic (open/high/low/close/volume from trades)
- [x] 3.3 Implement bar emission via `on_bar` callbacks with `Bar` dataclass

## 4. Order Book & Reconnection

- [x] 4.1 Integrate `OrderBook` into engine — apply `OrderBookUpdate` messages, expose per-symbol book access
- [x] 4.2 Implement snapshot reset on reconnection (clear book, rebuild from fresh snapshot)
- [x] 4.3 Implement auto-reconnection with exponential backoff (1s base, 60s max, 10 retries)
- [x] 4.4 Add `on_reconnect()` and `on_error()` callback hooks

## 5. Tests

- [x] 5.1 Unit tests for `Trade`, `Quote`, `OrderBookUpdate` dataclasses
- [x] 5.2 Unit tests for `OrderBook` (add/remove levels, best bid/ask, spread, snapshot reset)
- [x] 5.3 Unit tests for `RealtimeEngine` lifecycle (start/stop, callback dispatch)
- [x] 5.4 Unit tests for tick-to-bar aggregation (single symbol, multi-symbol, empty intervals)
- [x] 5.5 Unit tests for reconnection logic (backoff timing, max retries, callback invocation)

## 6. Tutorial — WebSocket Fundamentals

- [x] 6.1 Create `docs/26-realtime-data/index.md` section index page
- [x] 6.2 Create `docs/26-realtime-data/01-websocket-fundamentals.md` — protocol lifecycle, heartbeats, binary vs text frames, market data WebSocket patterns
- [x] 6.3 Add provider comparison table (Alpaca vs IBKR vs Polygon.io WebSocket APIs)

## 7. Tutorial — RealtimeEngine

- [x] 7.1 Create `docs/26-realtime-data/02-realtime-engine.md` — engine setup, callback registration, streaming trades and quotes
- [x] 7.2 Add tick-to-bar aggregation section with code examples and diagrams
- [x] 7.3 Add exercises: stream-to-cache pipeline, tick anomaly detection

## 8. Tutorial — Order Book Streaming

- [x] 8.1 Create `docs/26-realtime-data/03-orderbook-streaming.md` — L1/L2 concepts, local book maintenance, spread analysis
- [x] 8.2 Add order book visualization example and exercise
- [x] 8.3 Add cross-references to `docs/02-data-pipeline` and `docs/23-live-trading`
