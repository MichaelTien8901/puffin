"""Tests for trade logging."""

import pytest
import tempfile
import json
from datetime import datetime
from pathlib import Path
from puffin.monitor.trade_log import TradeLog, TradeRecord


class TestTradeRecord:
    """Tests for trade record."""

    def test_create_trade_record(self):
        """Test creating trade record."""
        record = TradeRecord(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            ticker='AAPL',
            side='buy',
            qty=100,
            price=150.0,
            commission=1.0,
            slippage=0.05,
            strategy='momentum'
        )

        assert record.ticker == 'AAPL'
        assert record.side == 'buy'
        assert record.qty == 100
        assert record.price == 150.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        record = TradeRecord(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            ticker='AAPL',
            side='buy',
            qty=100,
            price=150.0,
            commission=1.0,
            slippage=0.05,
            strategy='momentum'
        )

        d = record.to_dict()

        assert d['ticker'] == 'AAPL'
        assert d['side'] == 'buy'
        assert isinstance(d['timestamp'], str)

    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {
            'timestamp': '2024-01-01T10:00:00',
            'ticker': 'AAPL',
            'side': 'buy',
            'qty': 100,
            'price': 150.0,
            'commission': 1.0,
            'slippage': 0.05,
            'strategy': 'momentum',
            'metadata': {}
        }

        record = TradeRecord.from_dict(d)

        assert record.ticker == 'AAPL'
        assert record.side == 'buy'
        assert isinstance(record.timestamp, datetime)


class TestTradeLog:
    """Tests for trade log."""

    def test_record_trade(self):
        """Test recording trades."""
        log = TradeLog()

        record = TradeRecord(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            ticker='AAPL',
            side='buy',
            qty=100,
            price=150.0,
            commission=1.0,
            slippage=0.05,
            strategy='momentum'
        )

        log.record(record)

        assert len(log.trades) == 1
        assert log.trades[0].ticker == 'AAPL'

    def test_export_import_json(self):
        """Test exporting and importing JSON."""
        log = TradeLog()

        for i in range(3):
            record = TradeRecord(
                timestamp=datetime(2024, 1, i+1, 10, 0, 0),
                ticker='AAPL',
                side='buy' if i % 2 == 0 else 'sell',
                qty=100,
                price=150.0 + i,
                commission=1.0,
                slippage=0.05,
                strategy='momentum'
            )
            log.record(record)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            log.export_json(temp_path)

            # Load back
            log2 = TradeLog()
            log2.load_json(temp_path)

            assert len(log2.trades) == 3
            assert log2.trades[0].ticker == 'AAPL'

        finally:
            Path(temp_path).unlink()

    def test_export_import_csv(self):
        """Test exporting and importing CSV."""
        log = TradeLog()

        for i in range(3):
            record = TradeRecord(
                timestamp=datetime(2024, 1, i+1, 10, 0, 0),
                ticker='AAPL',
                side='buy' if i % 2 == 0 else 'sell',
                qty=100,
                price=150.0 + i,
                commission=1.0,
                slippage=0.05,
                strategy='momentum'
            )
            log.record(record)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            log.export_csv(temp_path)

            # Load back
            log2 = TradeLog()
            log2.load_csv(temp_path)

            assert len(log2.trades) == 3
            assert log2.trades[0].ticker == 'AAPL'

        finally:
            Path(temp_path).unlink()

    def test_filter_by_ticker(self):
        """Test filtering by ticker."""
        log = TradeLog()

        for ticker in ['AAPL', 'GOOGL', 'AAPL']:
            record = TradeRecord(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                ticker=ticker,
                side='buy',
                qty=100,
                price=150.0,
                commission=1.0,
                slippage=0.05,
                strategy='momentum'
            )
            log.record(record)

        filtered = log.filter(ticker='AAPL')

        assert len(filtered) == 2
        assert all(t.ticker == 'AAPL' for t in filtered)

    def test_filter_by_strategy(self):
        """Test filtering by strategy."""
        log = TradeLog()

        for strategy in ['momentum', 'mean_reversion', 'momentum']:
            record = TradeRecord(
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                ticker='AAPL',
                side='buy',
                qty=100,
                price=150.0,
                commission=1.0,
                slippage=0.05,
                strategy=strategy
            )
            log.record(record)

        filtered = log.filter(strategy='momentum')

        assert len(filtered) == 2
        assert all(t.strategy == 'momentum' for t in filtered)

    def test_filter_by_date_range(self):
        """Test filtering by date range."""
        log = TradeLog()

        for i in range(5):
            record = TradeRecord(
                timestamp=datetime(2024, 1, i+1, 10, 0, 0),
                ticker='AAPL',
                side='buy',
                qty=100,
                price=150.0,
                commission=1.0,
                slippage=0.05,
                strategy='momentum'
            )
            log.record(record)

        filtered = log.filter(
            date_range=(datetime(2024, 1, 2), datetime(2024, 1, 4))
        )

        assert len(filtered) == 3

    def test_summary(self):
        """Test trade summary."""
        log = TradeLog()

        for i in range(5):
            record = TradeRecord(
                timestamp=datetime(2024, 1, i+1, 10, 0, 0),
                ticker='AAPL',
                side='buy' if i % 2 == 0 else 'sell',
                qty=100,
                price=150.0,
                commission=1.0,
                slippage=0.05,
                strategy='momentum'
            )
            log.record(record)

        summary = log.summary()

        assert summary['total_trades'] == 5
        assert summary['total_buy'] == 3
        assert summary['total_sell'] == 2
        assert summary['total_commission'] == 5.0
        assert summary['total_slippage'] == 0.25
        assert 'AAPL' in summary['tickers']
        assert 'momentum' in summary['strategies']
