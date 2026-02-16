"""Basic market making strategy."""

import pandas as pd

from puffin.strategies.base import Strategy, SignalFrame


class MarketMakingStrategy(Strategy):
    """Simple market making strategy.

    Places symmetric bid and ask orders around the mid-price.
    Generates signals based on inventory and spread conditions.
    """

    def __init__(
        self,
        spread_bps: float = 10.0,
        max_inventory: int = 100,
        volatility_window: int = 20,
    ):
        self.spread_bps = spread_bps
        self.max_inventory = max_inventory
        self.volatility_window = volatility_window
        self._inventory = 0

    def generate_signals(self, data: pd.DataFrame) -> SignalFrame:
        close = data["Close"]

        # Compute volatility for dynamic spread adjustment
        returns = close.pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()

        # Base signal: always want to be near neutral
        # Positive when we should buy more, negative when we should sell
        signal = pd.Series(0.0, index=data.index)

        # Wider spread in high volatility → less aggressive
        vol_ratio = volatility / volatility.rolling(window=100).mean()

        # Signal based on short-term mean reversion (market makers profit from mean reversion)
        short_ma = close.rolling(window=5).mean()
        deviation = (close - short_ma) / short_ma

        # Buy when price dips below short-term mean
        signal[deviation < -self.spread_bps / 10000] = 0.5
        # Sell when price rises above short-term mean
        signal[deviation > self.spread_bps / 10000] = -0.5

        # Reduce confidence in high volatility
        confidence = (1.0 / vol_ratio.clip(lower=0.5)).clip(0.0, 1.0)

        result = SignalFrame({"signal": signal, "confidence": confidence})
        result.index = data.index
        return result

    def get_quote_prices(self, mid_price: float, volatility: float = 0.0) -> dict:
        """Calculate bid and ask prices for market making.

        Args:
            mid_price: Current mid-price.
            volatility: Current volatility for dynamic spread.

        Returns:
            Dict with 'bid' and 'ask' prices.
        """
        half_spread = mid_price * (self.spread_bps / 10000) / 2
        # Widen spread in high volatility
        if volatility > 0:
            half_spread *= max(1.0, volatility * 100)

        return {
            "bid": mid_price - half_spread,
            "ask": mid_price + half_spread,
            "spread_bps": (half_spread * 2 / mid_price) * 10000,
        }

    def get_parameters(self) -> dict:
        return {
            "spread_bps": self.spread_bps,
            "max_inventory": self.max_inventory,
            "volatility_window": self.volatility_window,
        }
