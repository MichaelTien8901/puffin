### Requirement: Typed streaming message dataclasses
The system SHALL define `Trade`, `Quote`, and `OrderBookUpdate` dataclasses in `puffin/data/realtime.py` to represent the three stream message types. Each dataclass SHALL include `symbol: str` and `timestamp: datetime` fields. `Trade` SHALL include `price: float` and `volume: int`. `Quote` SHALL include `bid: float`, `ask: float`, `bid_size: int`, and `ask_size: int`. `OrderBookUpdate` SHALL include `side: str` ("bid" or "ask"), `price: float`, and `size: int` (size of 0 means remove level).

#### Scenario: Trade message creation
- **WHEN** a raw trade tick arrives from a provider
- **THEN** the system creates a `Trade` dataclass with symbol, price, volume, and timestamp populated

#### Scenario: Quote message creation
- **WHEN** a raw quote update arrives from a provider
- **THEN** the system creates a `Quote` dataclass with symbol, bid, ask, bid_size, ask_size, and timestamp populated

#### Scenario: Order book update message creation
- **WHEN** a raw L2 depth update arrives from a provider
- **THEN** the system creates an `OrderBookUpdate` dataclass with symbol, side, price, size, and timestamp populated

### Requirement: RealtimeEngine lifecycle management
The system SHALL provide a `RealtimeEngine` class that accepts a `DataProvider` instance and manages the streaming lifecycle. The engine SHALL expose `start(symbols)` to begin streaming and `stop()` to cleanly shut down all streams and threads. The engine SHALL NOT start streaming until `start()` is explicitly called.

#### Scenario: Start streaming
- **WHEN** `engine.start(["AAPL", "MSFT"])` is called
- **THEN** the engine begins streaming data from the underlying provider for the specified symbols

#### Scenario: Stop streaming
- **WHEN** `engine.stop()` is called while streaming is active
- **THEN** all streaming threads are shut down and resources are released

#### Scenario: Engine not started
- **WHEN** no call to `start()` has been made
- **THEN** the engine SHALL NOT have any active streaming threads or connections

### Requirement: Callback registration for stream types
The engine SHALL allow registering callbacks via `on_trade(callback)`, `on_quote(callback)`, and `on_book_update(callback)` methods. Each callback SHALL receive the corresponding typed dataclass. Multiple callbacks MAY be registered for the same stream type and all SHALL be invoked.

#### Scenario: Register and receive trade callback
- **WHEN** a callback is registered via `engine.on_trade(my_handler)` and a trade arrives
- **THEN** `my_handler` is called with a `Trade` instance

#### Scenario: Multiple callbacks for same stream type
- **WHEN** two callbacks are registered via `on_trade()` and a trade arrives
- **THEN** both callbacks are invoked with the same `Trade` instance

### Requirement: Tick-to-bar aggregation
The engine SHALL aggregate raw `Trade` messages into OHLCV bars at a configurable time interval. The interval SHALL be set via `engine.set_bar_interval(seconds)` accepting values such as 1, 5, 60, and 300. Completed bars SHALL be emitted via callbacks registered with `on_bar(callback)`. Each bar SHALL contain `symbol`, `open`, `high`, `low`, `close`, `volume`, `timestamp` (bar open time), and `interval` fields.

#### Scenario: One-minute bar aggregation
- **WHEN** the bar interval is set to 60 seconds and trades arrive for AAPL over a 60-second window
- **THEN** at the end of the window, the engine emits a bar with open=first trade price, high=max price, low=min price, close=last trade price, volume=sum of volumes

#### Scenario: No trades in interval
- **WHEN** a bar interval elapses with no trades for a symbol
- **THEN** no bar is emitted for that symbol for that interval

#### Scenario: Multiple symbols aggregated independently
- **WHEN** trades arrive for both AAPL and MSFT
- **THEN** each symbol's bar is aggregated and emitted independently

### Requirement: Local order book maintenance
The engine SHALL maintain a local `OrderBook` object per symbol that tracks bid and ask price levels. The order book SHALL be updated incrementally from `OrderBookUpdate` messages. An update with `size=0` SHALL remove the price level. The `OrderBook` SHALL expose `best_bid`, `best_ask`, `spread`, `bids` (sorted descending by price), and `asks` (sorted ascending by price) properties.

#### Scenario: Apply bid update
- **WHEN** an `OrderBookUpdate` with side="bid", price=150.00, size=100 arrives
- **THEN** the order book adds or updates the bid level at 150.00 with size 100

#### Scenario: Remove price level
- **WHEN** an `OrderBookUpdate` with side="ask", price=151.00, size=0 arrives
- **THEN** the price level at 151.00 is removed from the ask side

#### Scenario: Best bid/ask and spread
- **WHEN** the order book has bids at [150.00, 149.50] and asks at [150.50, 151.00]
- **THEN** `best_bid` returns 150.00, `best_ask` returns 150.50, and `spread` returns 0.50

#### Scenario: Snapshot reset after reconnection
- **WHEN** the engine reconnects after a disconnection
- **THEN** the order book SHALL be cleared and rebuilt from a fresh snapshot before applying incremental updates

### Requirement: Auto-reconnection with exponential backoff
The engine SHALL automatically reconnect when the underlying stream disconnects unexpectedly. Reconnection attempts SHALL use exponential backoff starting at 1 second, doubling each attempt, with a maximum delay of 60 seconds and a maximum of 10 attempts. After successful reconnection, the engine SHALL resume streaming and notify via an optional `on_reconnect(callback)`.

#### Scenario: Automatic reconnection on disconnect
- **WHEN** the stream connection drops unexpectedly
- **THEN** the engine attempts to reconnect with delays of 1s, 2s, 4s, 8s, etc.

#### Scenario: Max retries exceeded
- **WHEN** 10 consecutive reconnection attempts fail
- **THEN** the engine stops retrying and invokes `on_error(callback)` with the failure reason

#### Scenario: Successful reconnection
- **WHEN** a reconnection attempt succeeds
- **THEN** streaming resumes for all previously subscribed symbols and `on_reconnect` callbacks are invoked

### Requirement: DataProvider streaming method extensions
The `DataProvider` base class SHALL add optional methods `stream_trades(symbols, callback)`, `stream_quotes(symbols, callback)`, and `stream_orderbook(symbols, callback)` with default implementations that raise `NotImplementedError`. The existing `stream_realtime()` method SHALL remain unchanged for backward compatibility.

#### Scenario: Provider supports trades streaming
- **WHEN** a provider implements `stream_trades()`
- **THEN** `RealtimeEngine` uses it to receive `Trade` messages

#### Scenario: Provider does not support order book streaming
- **WHEN** a provider has not overridden `stream_orderbook()`
- **THEN** calling `stream_orderbook()` raises `NotImplementedError` and the engine skips order book streaming for that provider

### Requirement: Tutorial documentation
The system SHALL include 3 tutorial pages in `docs/26-realtime-data/`: (1) WebSocket fundamentals covering protocol lifecycle, heartbeats, and market data patterns; (2) RealtimeEngine usage covering setup, callbacks, tick-to-bar aggregation, and multi-provider configuration; (3) Order book streaming covering L1/L2 concepts, local book maintenance, and spread analysis. Each page SHALL include practical exercises.

#### Scenario: Tutorial page structure
- **WHEN** a user navigates to the real-time data section
- **THEN** they see three pages in order: WebSocket Fundamentals, Real-Time Engine, Order Book Streaming

#### Scenario: Exercises included
- **WHEN** a user reads any tutorial page in the section
- **THEN** the page includes at least one hands-on exercise
