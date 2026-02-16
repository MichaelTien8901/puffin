"""Momentum strategy using moving average crossover."""

import pandas as pd

from puffin.strategies.base import Strategy, SignalFrame


class MomentumStrategy(Strategy):
    """Moving average crossover momentum strategy.

    Generates buy signals when the short MA crosses above the long MA,
    and sell signals when it crosses below.
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        ma_type: str = "sma",
        signal_threshold: float = 0.0,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type
        self.signal_threshold = signal_threshold

    def _compute_ma(self, series: pd.Series, window: int) -> pd.Series:
        if self.ma_type == "ema":
            return series.ewm(span=window, adjust=False).mean()
        return series.rolling(window=window).mean()

    def generate_signals(self, data: pd.DataFrame) -> SignalFrame:
        close = data["Close"]
        short_ma = self._compute_ma(close, self.short_window)
        long_ma = self._compute_ma(close, self.long_window)

        # Signal strength: normalized difference between MAs
        spread = (short_ma - long_ma) / long_ma
        signal = pd.Series(0.0, index=data.index)

        signal[spread > self.signal_threshold] = 1.0
        signal[spread < -self.signal_threshold] = -1.0

        confidence = spread.abs().clip(upper=1.0)

        result = SignalFrame({"signal": signal, "confidence": confidence})
        result.index = data.index
        return result

    def get_parameters(self) -> dict:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "ma_type": self.ma_type,
            "signal_threshold": self.signal_threshold,
        }
