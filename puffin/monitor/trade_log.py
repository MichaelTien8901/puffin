"""Trade logging and record keeping."""

import json
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class TradeRecord:
    """Record of a single trade execution."""
    timestamp: datetime
    ticker: str
    side: str  # 'buy' or 'sell'
    qty: float
    price: float
    commission: float
    slippage: float
    strategy: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade record to dictionary."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TradeRecord':
        """Create trade record from dictionary."""
        d = d.copy()
        if isinstance(d['timestamp'], str):
            d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        return cls(**d)


class TradeLog:
    """Comprehensive trade logging system."""

    def __init__(self):
        """Initialize trade log."""
        self.trades: List[TradeRecord] = []

    def record(self, trade_record: TradeRecord) -> None:
        """
        Record a trade execution.

        Parameters
        ----------
        trade_record : TradeRecord
            Trade record to log
        """
        self.trades.append(trade_record)

    def export_csv(self, path: str) -> None:
        """
        Export trade log to CSV file.

        Parameters
        ----------
        path : str
            Path to save CSV file
        """
        if len(self.trades) == 0:
            return

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', newline='') as f:
            # Get all possible fields from first record
            fieldnames = ['timestamp', 'ticker', 'side', 'qty', 'price',
                         'commission', 'slippage', 'strategy']

            # Add metadata keys from first record
            if self.trades[0].metadata:
                for key in self.trades[0].metadata.keys():
                    if key not in fieldnames:
                        fieldnames.append(f'metadata_{key}')

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trade in self.trades:
                row = {
                    'timestamp': trade.timestamp.isoformat(),
                    'ticker': trade.ticker,
                    'side': trade.side,
                    'qty': trade.qty,
                    'price': trade.price,
                    'commission': trade.commission,
                    'slippage': trade.slippage,
                    'strategy': trade.strategy,
                }

                # Add metadata fields
                if trade.metadata:
                    for key, value in trade.metadata.items():
                        row[f'metadata_{key}'] = value

                writer.writerow(row)

    def export_json(self, path: str) -> None:
        """
        Export trade log to JSON file.

        Parameters
        ----------
        path : str
            Path to save JSON file
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            trades_dict = [trade.to_dict() for trade in self.trades]
            json.dump(trades_dict, f, indent=2)

    def filter(
        self,
        ticker: Optional[str] = None,
        strategy: Optional[str] = None,
        date_range: Optional[tuple] = None
    ) -> List[TradeRecord]:
        """
        Filter trades by various criteria.

        Parameters
        ----------
        ticker : str, optional
            Filter by ticker symbol
        strategy : str, optional
            Filter by strategy name
        date_range : tuple, optional
            Filter by date range (start_date, end_date)

        Returns
        -------
        list of TradeRecord
            Filtered trade records

        Examples
        --------
        >>> log = TradeLog()
        >>> log.record(TradeRecord(
        ...     timestamp=datetime(2024, 1, 1),
        ...     ticker='AAPL',
        ...     side='buy',
        ...     qty=100,
        ...     price=150.0,
        ...     commission=1.0,
        ...     slippage=0.05,
        ...     strategy='momentum'
        ... ))
        >>> filtered = log.filter(ticker='AAPL')
        >>> len(filtered)
        1
        """
        filtered_trades = self.trades

        if ticker is not None:
            filtered_trades = [t for t in filtered_trades if t.ticker == ticker]

        if strategy is not None:
            filtered_trades = [t for t in filtered_trades if t.strategy == strategy]

        if date_range is not None:
            start_date, end_date = date_range
            filtered_trades = [
                t for t in filtered_trades
                if start_date.date() <= t.timestamp.date() <= end_date.date()
            ]

        return filtered_trades

    def load_csv(self, path: str) -> None:
        """
        Load trades from CSV file.

        Parameters
        ----------
        path : str
            Path to CSV file
        """
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract metadata fields
                metadata = {}
                for key in list(row.keys()):
                    if key.startswith('metadata_'):
                        metadata_key = key.replace('metadata_', '')
                        metadata[metadata_key] = row.pop(key)

                trade = TradeRecord(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    ticker=row['ticker'],
                    side=row['side'],
                    qty=float(row['qty']),
                    price=float(row['price']),
                    commission=float(row['commission']),
                    slippage=float(row['slippage']),
                    strategy=row['strategy'],
                    metadata=metadata
                )
                self.trades.append(trade)

    def load_json(self, path: str) -> None:
        """
        Load trades from JSON file.

        Parameters
        ----------
        path : str
            Path to JSON file
        """
        with open(path, 'r') as f:
            trades_dict = json.load(f)
            for trade_dict in trades_dict:
                trade = TradeRecord.from_dict(trade_dict)
                self.trades.append(trade)

    def summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics for trade log.

        Returns
        -------
        dict
            Summary statistics including:
            - total_trades: Total number of trades
            - total_buy: Number of buy trades
            - total_sell: Number of sell trades
            - total_commission: Total commission paid
            - total_slippage: Total slippage cost
            - avg_trade_size: Average trade size
            - strategies: List of unique strategies
            - tickers: List of unique tickers
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'total_buy': 0,
                'total_sell': 0,
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'avg_trade_size': 0.0,
                'strategies': [],
                'tickers': []
            }

        total_buy = sum(1 for t in self.trades if t.side == 'buy')
        total_sell = sum(1 for t in self.trades if t.side == 'sell')
        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        avg_trade_size = sum(t.qty * t.price for t in self.trades) / len(self.trades)

        strategies = list(set(t.strategy for t in self.trades))
        tickers = list(set(t.ticker for t in self.trades))

        return {
            'total_trades': len(self.trades),
            'total_buy': total_buy,
            'total_sell': total_sell,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'avg_trade_size': avg_trade_size,
            'strategies': strategies,
            'tickers': tickers
        }
