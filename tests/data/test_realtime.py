"""Tests for puffin.data.realtime module."""

import threading
import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from puffin.data.realtime import (
    Bar,
    OrderBook,
    OrderBookUpdate,
    Quote,
    RealtimeEngine,
    Trade,
    _BarBuffer,
)
from puffin.data.provider import DataProvider


# ---------------------------------------------------------------------------
# 5.1 — Dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_trade_creation(self):
        t = Trade(symbol="AAPL", price=150.0, volume=100, timestamp=datetime(2024, 1, 1))
        assert t.symbol == "AAPL"
        assert t.price == 150.0
        assert t.volume == 100
        assert t.timestamp == datetime(2024, 1, 1)

    def test_trade_is_frozen(self):
        t = Trade(symbol="AAPL", price=150.0, volume=100, timestamp=datetime(2024, 1, 1))
        with pytest.raises(AttributeError):
            t.price = 200.0

    def test_quote_creation(self):
        q = Quote(symbol="AAPL", bid=149.9, ask=150.1, bid_size=200, ask_size=300,
                  timestamp=datetime(2024, 1, 1))
        assert q.bid == 149.9
        assert q.ask == 150.1
        assert q.bid_size == 200
        assert q.ask_size == 300

    def test_order_book_update_creation(self):
        u = OrderBookUpdate(symbol="AAPL", side="bid", price=150.0, size=100,
                            timestamp=datetime(2024, 1, 1))
        assert u.side == "bid"
        assert u.size == 100

    def test_bar_creation(self):
        b = Bar(symbol="AAPL", open=150.0, high=152.0, low=149.0, close=151.0,
                volume=1000, timestamp=datetime(2024, 1, 1), interval=60)
        assert b.open == 150.0
        assert b.interval == 60


# ---------------------------------------------------------------------------
# 5.2 — OrderBook tests
# ---------------------------------------------------------------------------

class TestOrderBook:
    def test_add_bid_level(self):
        book = OrderBook()
        book.update(OrderBookUpdate("AAPL", "bid", 150.0, 100, datetime.now()))
        assert book.best_bid == 150.0
        assert book.bids == [(150.0, 100)]

    def test_add_ask_level(self):
        book = OrderBook()
        book.update(OrderBookUpdate("AAPL", "ask", 151.0, 200, datetime.now()))
        assert book.best_ask == 151.0
        assert book.asks == [(151.0, 200)]

    def test_remove_level_with_size_zero(self):
        book = OrderBook()
        now = datetime.now()
        book.update(OrderBookUpdate("AAPL", "ask", 151.0, 200, now))
        book.update(OrderBookUpdate("AAPL", "ask", 151.0, 0, now))
        assert book.best_ask is None
        assert book.asks == []

    def test_best_bid_ask_spread(self):
        book = OrderBook()
        now = datetime.now()
        book.update(OrderBookUpdate("AAPL", "bid", 150.0, 100, now))
        book.update(OrderBookUpdate("AAPL", "bid", 149.5, 200, now))
        book.update(OrderBookUpdate("AAPL", "ask", 150.5, 150, now))
        book.update(OrderBookUpdate("AAPL", "ask", 151.0, 300, now))
        assert book.best_bid == 150.0
        assert book.best_ask == 150.5
        assert book.spread == pytest.approx(0.5)

    def test_bids_sorted_descending(self):
        book = OrderBook()
        now = datetime.now()
        book.update(OrderBookUpdate("AAPL", "bid", 149.0, 100, now))
        book.update(OrderBookUpdate("AAPL", "bid", 150.0, 200, now))
        book.update(OrderBookUpdate("AAPL", "bid", 148.0, 50, now))
        prices = [p for p, _ in book.bids]
        assert prices == [150.0, 149.0, 148.0]

    def test_asks_sorted_ascending(self):
        book = OrderBook()
        now = datetime.now()
        book.update(OrderBookUpdate("AAPL", "ask", 152.0, 100, now))
        book.update(OrderBookUpdate("AAPL", "ask", 150.5, 200, now))
        book.update(OrderBookUpdate("AAPL", "ask", 151.0, 50, now))
        prices = [p for p, _ in book.asks]
        assert prices == [150.5, 151.0, 152.0]

    def test_empty_book_properties(self):
        book = OrderBook()
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.spread is None

    def test_clear(self):
        book = OrderBook()
        now = datetime.now()
        book.update(OrderBookUpdate("AAPL", "bid", 150.0, 100, now))
        book.update(OrderBookUpdate("AAPL", "ask", 151.0, 200, now))
        book.clear()
        assert book.best_bid is None
        assert book.best_ask is None


# ---------------------------------------------------------------------------
# 5.3 — RealtimeEngine lifecycle tests
# ---------------------------------------------------------------------------

class _MockProvider(DataProvider):
    """Mock provider that records calls but doesn't actually stream."""

    def __init__(self):
        self.stream_trades_called = False
        self.stream_quotes_called = False
        self.stream_orderbook_called = False
        self._trade_callback = None

    def fetch_historical(self, symbols, start, end=None, interval="1d"):
        import pandas as pd
        return pd.DataFrame()

    def get_supported_assets(self):
        return ["equity"]

    def stream_trades(self, symbols, callback):
        self.stream_trades_called = True
        self._trade_callback = callback

    def stream_quotes(self, symbols, callback):
        self.stream_quotes_called = True

    def stream_orderbook(self, symbols, callback):
        self.stream_orderbook_called = True


class TestRealtimeEngineLifecycle:
    def test_not_running_before_start(self):
        provider = _MockProvider()
        engine = RealtimeEngine(provider)
        assert engine._running is False
        assert engine._threads == []

    def test_start_sets_running(self):
        provider = _MockProvider()
        engine = RealtimeEngine(provider)
        engine.start(["AAPL"])
        assert engine._running is True
        time.sleep(0.1)
        engine.stop()

    def test_stop_clears_running(self):
        provider = _MockProvider()
        engine = RealtimeEngine(provider)
        engine.start(["AAPL"])
        time.sleep(0.1)
        engine.stop()
        assert engine._running is False

    def test_callback_registration_and_dispatch(self):
        provider = _MockProvider()
        engine = RealtimeEngine(provider)
        received = []
        engine.on_trade(lambda t: received.append(t))

        trade = Trade("AAPL", 150.0, 100, datetime.now())
        engine._dispatch_trade(trade)
        assert len(received) == 1
        assert received[0] is trade

    def test_multiple_callbacks(self):
        provider = _MockProvider()
        engine = RealtimeEngine(provider)
        r1, r2 = [], []
        engine.on_trade(lambda t: r1.append(t))
        engine.on_trade(lambda t: r2.append(t))

        trade = Trade("AAPL", 150.0, 100, datetime.now())
        engine._dispatch_trade(trade)
        assert len(r1) == 1
        assert len(r2) == 1

    def test_quote_dispatch(self):
        provider = _MockProvider()
        engine = RealtimeEngine(provider)
        received = []
        engine.on_quote(lambda q: received.append(q))

        quote = Quote("AAPL", 149.9, 150.1, 200, 300, datetime.now())
        engine._dispatch_quote(quote)
        assert len(received) == 1

    def test_book_update_dispatch_updates_order_book(self):
        provider = _MockProvider()
        engine = RealtimeEngine(provider)
        engine._books["AAPL"] = OrderBook()

        update = OrderBookUpdate("AAPL", "bid", 150.0, 100, datetime.now())
        engine._dispatch_book_update(update)
        assert engine.get_order_book("AAPL").best_bid == 150.0


# ---------------------------------------------------------------------------
# 5.4 — Tick-to-bar aggregation tests
# ---------------------------------------------------------------------------

class TestBarAggregation:
    def test_bar_buffer_single_trade(self):
        buf = _BarBuffer()
        t = Trade("AAPL", 150.0, 100, datetime(2024, 1, 1, 10, 0, 0))
        buf.add(t)
        bar = buf.to_bar("AAPL", 60)
        assert bar.open == 150.0
        assert bar.high == 150.0
        assert bar.low == 150.0
        assert bar.close == 150.0
        assert bar.volume == 100

    def test_bar_buffer_multiple_trades(self):
        buf = _BarBuffer()
        buf.add(Trade("AAPL", 150.0, 100, datetime(2024, 1, 1, 10, 0, 0)))
        buf.add(Trade("AAPL", 152.0, 200, datetime(2024, 1, 1, 10, 0, 30)))
        buf.add(Trade("AAPL", 149.0, 50, datetime(2024, 1, 1, 10, 0, 45)))
        buf.add(Trade("AAPL", 151.0, 150, datetime(2024, 1, 1, 10, 0, 59)))
        bar = buf.to_bar("AAPL", 60)
        assert bar.open == 150.0
        assert bar.high == 152.0
        assert bar.low == 149.0
        assert bar.close == 151.0
        assert bar.volume == 500

    def test_bar_buffer_empty(self):
        buf = _BarBuffer()
        assert buf.empty is True

    def test_flush_emits_bars(self):
        provider = _MockProvider()
        engine = RealtimeEngine(provider)
        engine.set_bar_interval(1)
        engine._symbols = ["AAPL"]
        engine._bar_buffers = {"AAPL": _BarBuffer()}
        engine._running = True

        bars_received = []
        engine.on_bar(lambda b: bars_received.append(b))

        # Add trades directly to buffer
        engine._dispatch_trade(Trade("AAPL", 150.0, 100, datetime(2024, 1, 1, 10, 0, 0)))
        engine._dispatch_trade(Trade("AAPL", 152.0, 200, datetime(2024, 1, 1, 10, 0, 30)))

        # Manually flush (normally done by timer)
        engine._running = False  # prevent timer restart
        engine._flush_bars()

        assert len(bars_received) == 1
        assert bars_received[0].open == 150.0
        assert bars_received[0].close == 152.0
        assert bars_received[0].volume == 300

    def test_no_bar_emitted_for_empty_interval(self):
        provider = _MockProvider()
        engine = RealtimeEngine(provider)
        engine.set_bar_interval(1)
        engine._symbols = ["AAPL"]
        engine._bar_buffers = {"AAPL": _BarBuffer()}
        engine._running = False

        bars_received = []
        engine.on_bar(lambda b: bars_received.append(b))
        engine._flush_bars()

        assert len(bars_received) == 0

    def test_multi_symbol_independent_bars(self):
        provider = _MockProvider()
        engine = RealtimeEngine(provider)
        engine.set_bar_interval(1)
        engine._symbols = ["AAPL", "MSFT"]
        engine._bar_buffers = {"AAPL": _BarBuffer(), "MSFT": _BarBuffer()}
        engine._running = False

        bars_received = []
        engine.on_bar(lambda b: bars_received.append(b))

        engine._dispatch_trade(Trade("AAPL", 150.0, 100, datetime.now()))
        engine._dispatch_trade(Trade("MSFT", 300.0, 200, datetime.now()))
        engine._flush_bars()

        assert len(bars_received) == 2
        symbols = {b.symbol for b in bars_received}
        assert symbols == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# 5.5 — Reconnection logic tests
# ---------------------------------------------------------------------------

class _FailingProvider(DataProvider):
    """Provider that raises on stream to test reconnection.

    After fail_count failures, succeeds once then sets engine._running = False
    to prevent infinite looping in tests.
    """

    def __init__(self, fail_count=3, engine=None):
        self._fail_count = fail_count
        self._calls = 0
        self._engine = engine

    def fetch_historical(self, symbols, start, end=None, interval="1d"):
        import pandas as pd
        return pd.DataFrame()

    def get_supported_assets(self):
        return ["equity"]

    def stream_trades(self, symbols, callback):
        self._calls += 1
        if self._calls <= self._fail_count:
            raise ConnectionError("Simulated disconnect")
        # On success, stop the engine so the loop exits
        if self._engine:
            self._engine._running = False


class TestReconnection:
    def test_backoff_calculation(self):
        engine = RealtimeEngine(_MockProvider())
        base = engine._reconnect_base_delay
        max_delay = engine._reconnect_max_delay
        # attempt 1: 1s, attempt 2: 2s, attempt 3: 4s, ...
        for attempt in range(1, 8):
            delay = min(base * (2 ** (attempt - 1)), max_delay)
            assert delay <= max_delay
        # attempt 7: min(64, 60) = 60
        assert min(base * (2 ** 6), max_delay) == 60.0

    def test_max_retries_triggers_error(self):
        provider = _FailingProvider(fail_count=100)
        engine = RealtimeEngine(provider)
        engine._reconnect_base_delay = 0.01
        engine._reconnect_max_delay = 0.01
        engine._reconnect_max_retries = 3

        errors = []
        engine.on_error(lambda msg: errors.append(msg))

        engine._running = True
        engine._symbols = ["AAPL"]
        engine._books = {"AAPL": OrderBook()}
        engine._stream_loop()

        assert len(errors) == 1
        assert "Max reconnection" in errors[0]

    def test_reconnect_callback_invoked(self):
        provider = _FailingProvider(fail_count=2)
        engine = RealtimeEngine(provider)
        provider._engine = engine
        engine._reconnect_base_delay = 0.01
        engine._reconnect_max_delay = 0.01

        reconnects = []
        engine.on_reconnect(lambda: reconnects.append(True))

        engine._running = True
        engine._symbols = ["AAPL"]
        engine._books = {"AAPL": OrderBook()}
        engine._stream_loop()

        assert len(reconnects) == 2  # failed twice, reconnected twice

    def test_order_book_cleared_on_reconnect(self):
        provider = _FailingProvider(fail_count=1)
        engine = RealtimeEngine(provider)
        provider._engine = engine
        engine._reconnect_base_delay = 0.01
        engine._reconnect_max_delay = 0.01

        book = OrderBook()
        book.update(OrderBookUpdate("AAPL", "bid", 150.0, 100, datetime.now()))
        engine._books = {"AAPL": book}
        engine._running = True
        engine._symbols = ["AAPL"]

        engine._stream_loop()

        assert provider._calls == 2  # 1 fail + 1 success
