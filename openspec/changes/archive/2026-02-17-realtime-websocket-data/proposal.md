## Why

The current data pipeline tutorial covers real-time streaming in just ~15 lines, treating it as an afterthought. But real-time market data is a critical pillar of any live trading system — WebSocket protocols, message handling, reconnection logic, tick-to-bar aggregation, and order book streaming all require dedicated coverage. Traders moving from backtesting to live need to understand how streaming data works, how it differs from historical data, and how to build robust real-time pipelines.

## What Changes

- Add a new tutorial section `docs/26-realtime-data/` dedicated to real-time WebSocket market data
- Cover WebSocket protocol fundamentals (connection lifecycle, heartbeats, binary vs text frames)
- Explain streaming data architecture: pub/sub patterns, message queues, backpressure handling
- Tutorial on tick-to-bar aggregation (building OHLCV bars from raw ticks in real time)
- Cover order book streaming (L1 quotes, L2 depth, maintaining local order book state)
- Robust connection management: auto-reconnect with exponential backoff, sequence gap detection
- Multi-provider streaming: Alpaca, IBKR, and Polygon.io WebSocket APIs compared
- Add a `RealtimeEngine` class to `puffin/data/` that orchestrates streaming with proper lifecycle management
- Add practical exercises: build a real-time price dashboard, detect tick anomalies, stream-to-cache pipeline

## Capabilities

### New Capabilities
- `realtime-streaming`: Core real-time data engine — WebSocket management, tick-to-bar aggregation, order book streaming, reconnection logic, and multi-provider support

### Modified Capabilities

## Impact

- **New docs section**: `docs/26-realtime-data/` (3-4 tutorial pages)
- **New code**: `puffin/data/realtime.py` — `RealtimeEngine` with tick aggregation, order book state, connection lifecycle
- **Modified code**: `puffin/data/provider.py` — extend `stream_realtime()` interface to support richer streaming (quotes, trades, order book) beyond the current simple callback
- **Dependencies**: No new external dependencies (uses existing alpaca-py, ib_insync)
- **Existing tutorials**: Minor cross-references added to `02-data-pipeline` and `23-live-trading`
