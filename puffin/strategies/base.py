"""Base strategy interface."""

from abc import ABC, abstractmethod

import pandas as pd


class SignalFrame(pd.DataFrame):
    """DataFrame of trading signals.

    Columns: signal (float, -1 to 1), confidence (float, 0 to 1).
    Index: same as input data (Date or MultiIndex).
    Positive signal = buy, negative = sell, 0 = neutral.
    """
    pass


class Strategy(ABC):
    """Base class for all trading strategies."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> SignalFrame:
        """Generate trading signals from market data.

        Args:
            data: DataFrame with OHLCV columns.

        Returns:
            SignalFrame with 'signal' and 'confidence' columns.
        """

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return current strategy parameters."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
