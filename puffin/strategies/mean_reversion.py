"""Mean reversion strategy using Bollinger Bands and z-score."""

import numpy as np
import pandas as pd

from puffin.strategies.base import Strategy, SignalFrame


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using Bollinger Bands.

    Buys when price touches the lower band (oversold) and sells
    when price reverts to the mean or touches the upper band.
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        zscore_entry: float = -2.0,
        zscore_exit: float = 0.0,
    ):
        self.window = window
        self.num_std = num_std
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit

    def generate_signals(self, data: pd.DataFrame) -> SignalFrame:
        close = data["Close"]
        ma = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std()

        zscore = (close - ma) / std

        signal = pd.Series(0.0, index=data.index)

        # Buy when z-score drops below entry threshold (oversold)
        signal[zscore <= self.zscore_entry] = 1.0
        # Sell when z-score rises above positive entry threshold (overbought)
        signal[zscore >= -self.zscore_entry] = -1.0
        # Exit when price reverts to mean
        signal[(zscore > self.zscore_exit - 0.5) & (zscore < self.zscore_exit + 0.5)] = 0.0

        confidence = zscore.abs() / (self.num_std * 2)
        confidence = confidence.clip(0.0, 1.0)

        result = SignalFrame({"signal": signal, "confidence": confidence})
        result.index = data.index
        return result

    def get_parameters(self) -> dict:
        return {
            "window": self.window,
            "num_std": self.num_std,
            "zscore_entry": self.zscore_entry,
            "zscore_exit": self.zscore_exit,
        }
