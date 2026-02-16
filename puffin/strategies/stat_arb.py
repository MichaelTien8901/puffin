"""Statistical arbitrage / pairs trading strategy."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from puffin.strategies.base import Strategy, SignalFrame


class StatArbStrategy(Strategy):
    """Pairs trading strategy using cointegration.

    Identifies cointegrated pairs and trades the spread when it
    deviates from its historical mean.
    """

    def __init__(
        self,
        lookback: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        coint_pvalue: float = 0.05,
    ):
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.coint_pvalue = coint_pvalue

    @staticmethod
    def find_cointegrated_pairs(
        prices: pd.DataFrame, pvalue_threshold: float = 0.05
    ) -> list[tuple[str, str, float]]:
        """Screen a universe of stocks for cointegrated pairs.

        Args:
            prices: DataFrame with tickers as columns and prices as values.
            pvalue_threshold: Maximum p-value for cointegration test.

        Returns:
            List of (ticker_a, ticker_b, pvalue) tuples.
        """
        tickers = prices.columns.tolist()
        pairs = []

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                s1 = prices[tickers[i]].dropna()
                s2 = prices[tickers[j]].dropna()
                # Align the series
                common_idx = s1.index.intersection(s2.index)
                if len(common_idx) < 30:
                    continue
                try:
                    _, pvalue, _ = coint(s1[common_idx], s2[common_idx])
                    if pvalue < pvalue_threshold:
                        pairs.append((tickers[i], tickers[j], pvalue))
                except Exception:
                    continue

        return sorted(pairs, key=lambda x: x[2])

    def generate_signals(self, data: pd.DataFrame) -> SignalFrame:
        """Generate signals for a pair spread.

        Expects data with 'Close_A' and 'Close_B' columns representing
        the two legs of the pair.
        """
        if "Close_A" not in data.columns or "Close_B" not in data.columns:
            raise ValueError("Data must have 'Close_A' and 'Close_B' columns for pairs trading")

        # Compute spread as ratio
        spread = data["Close_A"] / data["Close_B"]
        spread_mean = spread.rolling(window=self.lookback).mean()
        spread_std = spread.rolling(window=self.lookback).std()
        zscore = (spread - spread_mean) / spread_std

        signal = pd.Series(0.0, index=data.index)

        # Long spread (buy A, sell B) when spread is too low
        signal[zscore <= -self.entry_zscore] = 1.0
        # Short spread (sell A, buy B) when spread is too high
        signal[zscore >= self.entry_zscore] = -1.0
        # Exit when spread reverts
        signal[zscore.abs() <= self.exit_zscore] = 0.0

        confidence = (zscore.abs() / self.entry_zscore).clip(0.0, 1.0)

        result = SignalFrame({"signal": signal, "confidence": confidence})
        result.index = data.index
        return result

    def get_parameters(self) -> dict:
        return {
            "lookback": self.lookback,
            "entry_zscore": self.entry_zscore,
            "exit_zscore": self.exit_zscore,
            "coint_pvalue": self.coint_pvalue,
        }
