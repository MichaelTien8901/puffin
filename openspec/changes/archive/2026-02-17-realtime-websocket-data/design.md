## Context

The existing `DataProvider.stream_realtime()` interface is minimal — a simple callback receiving `(symbol, price, volume, timestamp)`. Both `AlpacaProvider` and `IBKRDataProvider` implement streaming but in isolation, with basic retry logic (Alpaca) or no reconnection handling (IBKR). There is no unified streaming abstraction, no tick-to-bar aggregation, no order book support, and no tutorial coverage beyond a 15-line code snippet.

The tutorial site has 25 sections (`docs/01-*` through `docs/25-*`). Real-time data will be section 26.

## Goals / Non-Goals

**Goals:**
- Teach WebSocket fundamentals in the context of market data (protocol lifecycle, message types, heartbeats)
- Build a `RealtimeEngine` class that provides unified streaming across providers with proper lifecycle management
- Support three stream types: trades, quotes (L1), and order book depth (L2)
- Implement tick-to-bar aggregation (build OHLCV bars from raw trade ticks at configurable intervals)
- Implement robust reconnection with exponential backoff and sequence gap detection
- Compare Alpaca, IBKR, and Polygon.io WebSocket APIs side-by-side
- Provide practical exercises (real-time dashboard, anomaly detection, stream-to-cache)

**Non-Goals:**
- Building a full order matching engine or exchange simulator
- Sub-millisecond latency optimization (HFT-grade infrastructure)
- Implementing a custom WebSocket server — we only consume provider streams
- Crypto exchange WebSocket support (Binance, Coinbase) — can be added later
- Real-time options chain streaming (complex, separate topic)

## Decisions

### 1. Unified `RealtimeEngine` orchestrator vs. enhancing individual providers

**Decision**: Create a new `RealtimeEngine` class in `puffin/data/realtime.py` that wraps any `DataProvider` and adds streaming orchestration.

**Rationale**: The existing providers already have streaming implementations. Rather than duplicating reconnection/aggregation logic in each provider, `RealtimeEngine` sits on top and adds cross-cutting concerns (lifecycle, aggregation, buffering). Individual providers keep their `stream_realtime()` but gain a richer callback protocol.

**Alternatives considered**:
- Enhance `DataProvider` base class directly — rejected because it would force all providers (including yfinance, which has no streaming) to deal with streaming complexity
- Separate streaming classes per provider — rejected because it duplicates reconnection and aggregation logic

### 2. Callback-based vs. async iterator streaming interface

**Decision**: Use an event-based callback system with typed dataclasses (`Trade`, `Quote`, `OrderBookUpdate`) rather than async generators.

**Rationale**: The existing codebase uses callbacks (both Alpaca and IBKR providers). Callbacks integrate naturally with the threading model already in use. The underlying libraries (alpaca-py, ib_async) also use event/callback patterns.

**Alternatives considered**:
- Async generators with `async for` — cleaner API but would require rewriting existing providers and introducing asyncio throughout the codebase
- Queue-based with `queue.Queue` — considered as internal implementation detail; `RealtimeEngine` may use queues internally but exposes callbacks externally

### 3. Extend `stream_realtime()` signature vs. new methods

**Decision**: Add new optional methods to `DataProvider` (`stream_quotes()`, `stream_trades()`, `stream_orderbook()`) while keeping the existing `stream_realtime()` for backward compatibility. `RealtimeEngine` calls whichever methods a provider supports.

**Rationale**: The current `stream_realtime()` signature `(symbols, callback)` is too simple for quotes and order book data. Adding separate methods keeps each stream type's callback signature clear and doesn't break existing code.

### 4. Tick-to-bar aggregation approach

**Decision**: Time-based aggregation with configurable intervals (1s, 5s, 1m, 5m). Aggregation runs in the `RealtimeEngine`, not in individual providers.

**Rationale**: Different strategies need different bar intervals. Centralizing aggregation means any provider's raw ticks can be aggregated uniformly. The engine maintains a rolling window per symbol and emits completed bars via a separate `on_bar` callback.

### 5. Order book representation

**Decision**: Maintain a local `OrderBook` object per symbol with sorted bid/ask levels, updated incrementally from L2 stream messages. Expose top-of-book (L1) as a convenience property.

**Rationale**: Most providers send incremental updates (price level changes), not full snapshots. A local order book that applies deltas is the standard approach. Keeping it per-symbol and in-memory is sufficient for the tutorial scope.

### 6. Tutorial structure

**Decision**: 3 pages in `docs/26-realtime-data/`:
1. `01-websocket-fundamentals.md` — Protocol basics, connection lifecycle, market data WebSocket patterns
2. `02-realtime-engine.md` — `RealtimeEngine` usage, tick-to-bar aggregation, multi-provider setup
3. `03-orderbook-streaming.md` — Order book concepts, L1/L2 data, maintaining local book state

**Rationale**: Separates conceptual (WebSocket protocol) from practical (engine usage) from advanced (order book). Each page is self-contained enough to read independently but builds naturally in sequence.

## Risks / Trade-offs

**[Provider API differences]** → Each provider has different WebSocket message formats and capabilities (IBKR has no native WebSocket, uses TWS socket protocol). Mitigation: Abstract behind the `DataProvider` streaming methods; tutorial acknowledges differences in the comparison section.

**[Thread safety of tick aggregation]** → Multiple provider threads pushing ticks while the engine aggregates. Mitigation: Use `threading.Lock` on the aggregation buffer; keep critical sections minimal.

**[Order book accuracy]** → Incremental L2 updates can drift if messages are dropped during reconnection. Mitigation: Request a full snapshot after reconnection before applying incremental updates. Document this pattern in the tutorial.

**[Tutorial complexity]** → WebSocket + order book + aggregation is a lot of ground. Mitigation: Three separate pages with progressive complexity. Page 1 is conceptual with no code to write; pages 2-3 are hands-on.

## Open Questions

- Should `RealtimeEngine` support pluggable message serialization (e.g., Protocol Buffers) or keep it simple with Python dataclasses only? Leaning toward dataclasses-only for tutorial scope.
- Whether to include a minimal Streamlit dashboard exercise or defer that to the monitoring section (`docs/25-monitoring-analytics/`). Leaning toward a lightweight exercise here with a pointer to the monitoring section for production dashboards.
