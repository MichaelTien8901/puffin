"""Real-time streaming engine for market data."""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from puffin.data.provider import DataProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed streaming messages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Trade:
    """A single trade tick."""
    symbol: str
    price: float
    volume: int
    timestamp: datetime


@dataclass(frozen=True)
class Quote:
    """A top-of-book quote update (L1)."""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime


@dataclass(frozen=True)
class OrderBookUpdate:
    """An incremental order book level change (L2)."""
    symbol: str
    side: str  # "bid" or "ask"
    price: float
    size: int  # 0 means remove level
    timestamp: datetime


@dataclass(frozen=True)
class Bar:
    """An aggregated OHLCV bar built from trade ticks."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime  # bar open time
    interval: int  # seconds


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------

class OrderBook:
    """Local order book maintained from incremental L2 updates.

    Bids are sorted descending by price, asks ascending.
    """

    def __init__(self):
        self._bids: dict[float, int] = {}  # price -> size
        self._asks: dict[float, int] = {}  # price -> size

    def update(self, update: OrderBookUpdate) -> None:
        book = self._bids if update.side == "bid" else self._asks
        if update.size == 0:
            book.pop(update.price, None)
        else:
            book[update.price] = update.size

    @property
    def best_bid(self) -> float | None:
        return max(self._bids) if self._bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self._asks) if self._asks else None

    @property
    def spread(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    @property
    def bids(self) -> list[tuple[float, int]]:
        """Bid levels sorted descending by price."""
        return sorted(self._bids.items(), key=lambda x: -x[0])

    @property
    def asks(self) -> list[tuple[float, int]]:
        """Ask levels sorted ascending by price."""
        return sorted(self._asks.items(), key=lambda x: x[0])

    def clear(self) -> None:
        self._bids.clear()
        self._asks.clear()


# ---------------------------------------------------------------------------
# Bar aggregation buffer
# ---------------------------------------------------------------------------

class _BarBuffer:
    """Accumulates trades for a single symbol over a time window."""

    def __init__(self):
        self.open: float | None = None
        self.high: float = float("-inf")
        self.low: float = float("inf")
        self.close: float | None = None
        self.volume: int = 0
        self.timestamp: datetime | None = None

    def add(self, trade: Trade) -> None:
        if self.open is None:
            self.open = trade.price
            self.timestamp = trade.timestamp
        self.high = max(self.high, trade.price)
        self.low = min(self.low, trade.price)
        self.close = trade.price
        self.volume += trade.volume

    @property
    def empty(self) -> bool:
        return self.open is None

    def to_bar(self, symbol: str, interval: int) -> Bar:
        return Bar(
            symbol=symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            timestamp=self.timestamp,
            interval=interval,
        )


# ---------------------------------------------------------------------------
# Realtime engine
# ---------------------------------------------------------------------------

class RealtimeEngine:
    """Orchestrates real-time streaming across any DataProvider.

    Provides callback registration, tick-to-bar aggregation,
    local order book maintenance, and auto-reconnection.
    """

    def __init__(self, provider: DataProvider):
        self._provider = provider
        self._symbols: list[str] = []
        self._running = False
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

        # Callback registries
        self._on_trade: list = []
        self._on_quote: list = []
        self._on_book_update: list = []
        self._on_bar: list = []
        self._on_reconnect: list = []
        self._on_error: list = []

        # Bar aggregation
        self._bar_interval: int | None = None
        self._bar_buffers: dict[str, _BarBuffer] = {}
        self._bar_timer: threading.Timer | None = None

        # Order books
        self._books: dict[str, OrderBook] = {}

        # Reconnection config
        self._reconnect_base_delay = 1.0
        self._reconnect_max_delay = 60.0
        self._reconnect_max_retries = 10

    # -- Callback registration -----------------------------------------------

    def on_trade(self, callback) -> None:
        self._on_trade.append(callback)

    def on_quote(self, callback) -> None:
        self._on_quote.append(callback)

    def on_book_update(self, callback) -> None:
        self._on_book_update.append(callback)

    def on_bar(self, callback) -> None:
        self._on_bar.append(callback)

    def on_reconnect(self, callback) -> None:
        self._on_reconnect.append(callback)

    def on_error(self, callback) -> None:
        self._on_error.append(callback)

    # -- Bar aggregation -----------------------------------------------------

    def set_bar_interval(self, seconds: int) -> None:
        self._bar_interval = seconds

    def _flush_bars(self) -> None:
        """Emit completed bars and reset buffers."""
        with self._lock:
            for symbol, buf in self._bar_buffers.items():
                if not buf.empty:
                    bar = buf.to_bar(symbol, self._bar_interval)
                    for cb in self._on_bar:
                        try:
                            cb(bar)
                        except Exception:
                            logger.exception("Error in on_bar callback")
            self._bar_buffers = {s: _BarBuffer() for s in self._symbols}

        if self._running and self._bar_interval:
            self._bar_timer = threading.Timer(self._bar_interval, self._flush_bars)
            self._bar_timer.daemon = True
            self._bar_timer.start()

    # -- Internal dispatch ---------------------------------------------------

    def _dispatch_trade(self, trade: Trade) -> None:
        for cb in self._on_trade:
            try:
                cb(trade)
            except Exception:
                logger.exception("Error in on_trade callback")
        if self._bar_interval:
            with self._lock:
                if trade.symbol not in self._bar_buffers:
                    self._bar_buffers[trade.symbol] = _BarBuffer()
                self._bar_buffers[trade.symbol].add(trade)

    def _dispatch_quote(self, quote: Quote) -> None:
        for cb in self._on_quote:
            try:
                cb(quote)
            except Exception:
                logger.exception("Error in on_quote callback")

    def _dispatch_book_update(self, update: OrderBookUpdate) -> None:
        if update.symbol not in self._books:
            self._books[update.symbol] = OrderBook()
        self._books[update.symbol].update(update)
        for cb in self._on_book_update:
            try:
                cb(update)
            except Exception:
                logger.exception("Error in on_book_update callback")

    # -- Order book access ---------------------------------------------------

    def get_order_book(self, symbol: str) -> OrderBook | None:
        return self._books.get(symbol)

    # -- Lifecycle -----------------------------------------------------------

    def start(self, symbols: list[str]) -> None:
        if self._running:
            return
        self._symbols = list(symbols)
        self._running = True
        self._bar_buffers = {s: _BarBuffer() for s in symbols}
        self._books = {s: OrderBook() for s in symbols}

        # Start bar flush timer
        if self._bar_interval:
            self._bar_timer = threading.Timer(self._bar_interval, self._flush_bars)
            self._bar_timer.daemon = True
            self._bar_timer.start()

        # Start streaming with reconnection
        thread = threading.Thread(target=self._stream_loop, daemon=True)
        thread.start()
        self._threads.append(thread)

    def stop(self) -> None:
        self._running = False
        if self._bar_timer:
            self._bar_timer.cancel()
            self._bar_timer = None
        self._threads.clear()

    def _stream_loop(self) -> None:
        """Main streaming loop with auto-reconnection."""
        attempt = 0
        while self._running:
            try:
                self._connect_and_stream()
                attempt = 0  # reset on clean exit
            except Exception as exc:
                if not self._running:
                    break
                attempt += 1
                if attempt > self._reconnect_max_retries:
                    logger.error("Max reconnection attempts (%d) exceeded", self._reconnect_max_retries)
                    for cb in self._on_error:
                        try:
                            cb(f"Max reconnection attempts exceeded: {exc}")
                        except Exception:
                            logger.exception("Error in on_error callback")
                    break
                delay = min(
                    self._reconnect_base_delay * (2 ** (attempt - 1)),
                    self._reconnect_max_delay,
                )
                logger.warning("Stream disconnected (attempt %d/%d), retrying in %.1fs",
                               attempt, self._reconnect_max_retries, delay)
                time.sleep(delay)

                # Reset order books on reconnection
                for book in self._books.values():
                    book.clear()
                for cb in self._on_reconnect:
                    try:
                        cb()
                    except Exception:
                        logger.exception("Error in on_reconnect callback")

    def _connect_and_stream(self) -> None:
        """Connect to provider streams. Blocks until disconnected."""
        # Try rich streaming methods first, fall back to stream_realtime
        try:
            self._provider.stream_trades(
                self._symbols,
                lambda sym, price, vol, ts: self._dispatch_trade(
                    Trade(symbol=sym, price=price, volume=vol, timestamp=ts)
                ),
            )
        except NotImplementedError:
            # Fall back to legacy stream_realtime
            self._provider.stream_realtime(
                self._symbols,
                lambda sym, price, vol, ts: self._dispatch_trade(
                    Trade(symbol=sym, price=price, volume=vol, timestamp=ts)
                ),
            )

        try:
            self._provider.stream_quotes(
                self._symbols,
                lambda sym, bid, ask, bs, as_, ts: self._dispatch_quote(
                    Quote(symbol=sym, bid=bid, ask=ask, bid_size=bs, ask_size=as_, timestamp=ts)
                ),
            )
        except NotImplementedError:
            pass

        try:
            self._provider.stream_orderbook(
                self._symbols,
                lambda sym, side, price, size, ts: self._dispatch_book_update(
                    OrderBookUpdate(symbol=sym, side=side, price=price, size=size, timestamp=ts)
                ),
            )
        except NotImplementedError:
            pass
