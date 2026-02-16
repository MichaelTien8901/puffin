"""Market data storage with HDF5 and Parquet support."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd


class MarketDataStore:
    """Store and retrieve market data in HDF5 or Parquet format.

    Supports both formats with automatic detection based on file extension.
    Metadata tracking includes source, frequency, and last update time.
    """

    def __init__(self, storage_path: str | Path, format: Literal["hdf5", "parquet"] = "parquet"):
        self.storage_path = Path(storage_path)
        self.format = format
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Metadata storage (always use parquet for metadata)
        self.metadata_path = self.storage_path / "metadata.parquet"
        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_path.exists():
            self.metadata = pd.read_parquet(self.metadata_path)
        else:
            self.metadata = pd.DataFrame(
                columns=["symbol", "source", "frequency", "last_updated", "format", "rows"]
            )

    def _save_metadata(self):
        self.metadata.to_parquet(self.metadata_path, index=False)

    def _get_file_path(self, symbol: str, format: str | None = None) -> Path:
        if format is None:
            format = self.format

        ext = "h5" if format == "hdf5" else "parquet"
        return self.storage_path / f"{symbol}.{ext}"

    def _detect_format(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in (".h5", ".hdf5"):
            return "hdf5"
        elif suffix in (".parquet", ".pq"):
            return "parquet"
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def save_ohlcv(
        self,
        symbol: str,
        data: pd.DataFrame,
        source: str = "unknown",
        frequency: str = "1d",
        append: bool = False,
    ):
        """Save OHLCV data to storage.

        Args:
            symbol: Ticker symbol
            data: DataFrame with OHLCV data (indexed by date)
            source: Data source identifier
            frequency: Bar frequency (e.g., '1d', '1h', '5m')
            append: If True, append to existing data; if False, overwrite
        """
        file_path = self._get_file_path(symbol)

        # Ensure data has proper index
        if not isinstance(data.index, pd.DatetimeIndex):
            if "Date" in data.columns:
                data = data.set_index("Date")
            else:
                raise ValueError("Data must have DatetimeIndex or 'Date' column")

        # Append mode
        if append and file_path.exists():
            existing = self.load_ohlcv(symbol)
            if existing is not None and not existing.empty:
                # Combine and deduplicate
                combined = pd.concat([existing, data])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                data = combined

        # Save based on format
        if self.format == "hdf5":
            data.to_hdf(file_path, key="ohlcv", mode="w", format="table")
        else:  # parquet
            data.to_parquet(file_path, engine="pyarrow")

        # Update metadata
        metadata_row = {
            "symbol": symbol,
            "source": source,
            "frequency": frequency,
            "last_updated": datetime.now().isoformat(),
            "format": self.format,
            "rows": len(data),
        }

        # Remove old metadata for this symbol
        self.metadata = self.metadata[self.metadata["symbol"] != symbol]

        # Add new metadata
        self.metadata = pd.concat(
            [self.metadata, pd.DataFrame([metadata_row])],
            ignore_index=True,
        )
        self._save_metadata()

    def load_ohlcv(self, symbol: str, format: str | None = None) -> pd.DataFrame | None:
        """Load OHLCV data from storage.

        Args:
            symbol: Ticker symbol
            format: Override format detection (optional)

        Returns:
            DataFrame with OHLCV data or None if not found
        """
        file_path = self._get_file_path(symbol, format)

        if not file_path.exists():
            return None

        # Auto-detect format if not specified
        if format is None:
            format = self._detect_format(file_path)

        # Load based on format
        if format == "hdf5":
            return pd.read_hdf(file_path, key="ohlcv")
        else:  # parquet
            return pd.read_parquet(file_path)

    def list_symbols(self) -> list[str]:
        """List all symbols in storage.

        Returns:
            List of ticker symbols
        """
        if self.metadata.empty:
            return []
        return self.metadata["symbol"].unique().tolist()

    def delete_symbol(self, symbol: str):
        """Delete all data for a symbol.

        Args:
            symbol: Ticker symbol to delete
        """
        # Try both formats
        for fmt in ["hdf5", "parquet"]:
            file_path = self._get_file_path(symbol, fmt)
            if file_path.exists():
                file_path.unlink()

        # Remove from metadata
        self.metadata = self.metadata[self.metadata["symbol"] != symbol]
        self._save_metadata()

    def get_metadata(self, symbol: str | None = None) -> pd.DataFrame:
        """Get metadata for symbols.

        Args:
            symbol: Specific symbol or None for all symbols

        Returns:
            DataFrame with metadata
        """
        if symbol is None:
            return self.metadata.copy()
        return self.metadata[self.metadata["symbol"] == symbol].copy()

    def append_ohlcv(
        self,
        symbol: str,
        data: pd.DataFrame,
        source: str = "unknown",
        frequency: str = "1d",
    ):
        """Append new data to existing storage.

        Convenience method for incremental updates.

        Args:
            symbol: Ticker symbol
            data: New OHLCV data to append
            source: Data source identifier
            frequency: Bar frequency
        """
        self.save_ohlcv(symbol, data, source, frequency, append=True)

    def migrate_format(self, symbol: str, target_format: Literal["hdf5", "parquet"]):
        """Migrate data from one format to another.

        Args:
            symbol: Ticker symbol
            target_format: Target storage format
        """
        # Determine current format
        for current_format in ["hdf5", "parquet"]:
            file_path = self._get_file_path(symbol, current_format)
            if file_path.exists():
                break
        else:
            raise ValueError(f"Symbol {symbol} not found in storage")

        if current_format == target_format:
            return  # Already in target format

        # Load data
        data = self.load_ohlcv(symbol, current_format)
        if data is None:
            raise ValueError(f"Failed to load data for {symbol}")

        # Get metadata
        meta = self.get_metadata(symbol)
        if meta.empty:
            source = "unknown"
            frequency = "1d"
        else:
            source = meta.iloc[0]["source"]
            frequency = meta.iloc[0]["frequency"]

        # Save in new format
        old_format = self.format
        self.format = target_format
        self.save_ohlcv(symbol, data, source, frequency, append=False)
        self.format = old_format

        # Delete old file
        file_path.unlink()
