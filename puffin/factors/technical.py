"""
Technical indicators module with TA-Lib integration.

This module provides a unified interface for computing technical indicators,
with automatic fallback to pure Python implementations if TA-Lib is not available.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import warnings


# Try to import TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn(
        "TA-Lib not available. Using pure Python fallback implementations. "
        "Install TA-Lib for better performance: pip install TA-Lib",
        ImportWarning
    )


class TechnicalIndicators:
    """
    Technical indicators calculator with TA-Lib integration.

    This class provides a unified interface for computing technical indicators
    across multiple categories: overlap studies, momentum indicators, volume
    indicators, and volatility indicators.

    If TA-Lib is not available, falls back to pure Python implementations.

    Parameters
    ----------
    use_talib : bool, optional
        Force use of TA-Lib if available (default: True)

    Examples
    --------
    >>> ohlcv = {
    ...     'open': pd.Series([100, 102, 101]),
    ...     'high': pd.Series([103, 104, 103]),
    ...     'low': pd.Series([99, 101, 100]),
    ...     'close': pd.Series([102, 101, 102]),
    ...     'volume': pd.Series([1000, 1100, 1050])
    ... }
    >>> ti = TechnicalIndicators()
    >>> indicators = ti.compute_all(ohlcv, categories=['overlap', 'momentum'])
    """

    def __init__(self, use_talib: bool = True):
        self.use_talib = use_talib and TALIB_AVAILABLE

    def compute_all(
        self,
        ohlcv: Dict[str, pd.Series],
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute all technical indicators for specified categories.

        Parameters
        ----------
        ohlcv : dict
            Dictionary with keys: 'open', 'high', 'low', 'close', 'volume'
            Each value should be a pandas Series with datetime index.
        categories : list of str, optional
            Categories to compute. Options: 'overlap', 'momentum', 'volume', 'volatility'
            If None, computes all categories.

        Returns
        -------
        pd.DataFrame
            DataFrame with all computed indicators as columns.
        """
        if categories is None:
            categories = ['overlap', 'momentum', 'volume', 'volatility']

        result = pd.DataFrame(index=ohlcv['close'].index)

        if 'overlap' in categories:
            overlap = self.compute_overlap(ohlcv)
            result = result.join(overlap)

        if 'momentum' in categories:
            momentum = self.compute_momentum(ohlcv)
            result = result.join(momentum)

        if 'volume' in categories:
            volume = self.compute_volume(ohlcv)
            result = result.join(volume)

        if 'volatility' in categories:
            volatility = self.compute_volatility(ohlcv)
            result = result.join(volatility)

        return result

    def compute_overlap(self, ohlcv: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute overlap studies (moving averages, bands, etc.).

        Indicators:
        - SMA: Simple Moving Average (20, 50, 200 periods)
        - EMA: Exponential Moving Average (12, 26 periods)
        - BBANDS: Bollinger Bands (20 period, 2 std dev)
        - SAR: Parabolic SAR

        Parameters
        ----------
        ohlcv : dict
            OHLCV data dictionary

        Returns
        -------
        pd.DataFrame
            DataFrame with overlap indicators
        """
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values

        indicators = pd.DataFrame(index=ohlcv['close'].index)

        if self.use_talib:
            # SMA
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['sma_200'] = talib.SMA(close, timeperiod=200)

            # EMA
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            indicators['bb_upper'] = upper
            indicators['bb_middle'] = middle
            indicators['bb_lower'] = lower
            indicators['bb_width'] = (upper - lower) / middle
            indicators['bb_position'] = (close - lower) / (upper - lower)

            # Parabolic SAR
            indicators['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

        else:
            # Pure Python implementations
            close_series = ohlcv['close']

            # SMA
            indicators['sma_20'] = close_series.rolling(20).mean()
            indicators['sma_50'] = close_series.rolling(50).mean()
            indicators['sma_200'] = close_series.rolling(200).mean()

            # EMA
            indicators['ema_12'] = close_series.ewm(span=12, adjust=False).mean()
            indicators['ema_26'] = close_series.ewm(span=26, adjust=False).mean()

            # Bollinger Bands
            rolling_mean = close_series.rolling(20).mean()
            rolling_std = close_series.rolling(20).std()
            indicators['bb_upper'] = rolling_mean + (2 * rolling_std)
            indicators['bb_middle'] = rolling_mean
            indicators['bb_lower'] = rolling_mean - (2 * rolling_std)
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / rolling_mean
            indicators['bb_position'] = (
                (close_series - indicators['bb_lower']) /
                (indicators['bb_upper'] - indicators['bb_lower'])
            )

            # Parabolic SAR (simplified)
            indicators['sar'] = self._calculate_sar_python(ohlcv['high'], ohlcv['low'])

        return indicators

    def compute_momentum(self, ohlcv: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute momentum indicators.

        Indicators:
        - RSI: Relative Strength Index (14 period)
        - MACD: Moving Average Convergence Divergence
        - STOCH: Stochastic Oscillator
        - ADX: Average Directional Index
        - CCI: Commodity Channel Index
        - MFI: Money Flow Index

        Parameters
        ----------
        ohlcv : dict
            OHLCV data dictionary

        Returns
        -------
        pd.DataFrame
            DataFrame with momentum indicators
        """
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        volume = ohlcv['volume'].values

        indicators = pd.DataFrame(index=ohlcv['close'].index)

        if self.use_talib:
            # RSI
            indicators['rsi'] = talib.RSI(close, timeperiod=14)

            # MACD
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_hist'] = hist

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd

            # ADX
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
            indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # CCI
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)

            # MFI
            indicators['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)

        else:
            # Pure Python implementations
            close_series = ohlcv['close']
            high_series = ohlcv['high']
            low_series = ohlcv['low']

            # RSI
            indicators['rsi'] = self._calculate_rsi_python(close_series, period=14)

            # MACD
            macd_data = self._calculate_macd_python(close_series)
            indicators['macd'] = macd_data['macd']
            indicators['macd_signal'] = macd_data['signal']
            indicators['macd_hist'] = macd_data['hist']

            # Stochastic
            stoch_data = self._calculate_stoch_python(high_series, low_series, close_series)
            indicators['stoch_k'] = stoch_data['k']
            indicators['stoch_d'] = stoch_data['d']

            # ADX
            adx_data = self._calculate_adx_python(high_series, low_series, close_series)
            indicators['adx'] = adx_data['adx']
            indicators['plus_di'] = adx_data['plus_di']
            indicators['minus_di'] = adx_data['minus_di']

            # CCI
            indicators['cci'] = self._calculate_cci_python(high_series, low_series, close_series)

            # MFI
            indicators['mfi'] = self._calculate_mfi_python(
                high_series, low_series, close_series, ohlcv['volume']
            )

        return indicators

    def compute_volume(self, ohlcv: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute volume indicators.

        Indicators:
        - OBV: On Balance Volume
        - AD: Accumulation/Distribution Line
        - ADOSC: Chaikin A/D Oscillator

        Parameters
        ----------
        ohlcv : dict
            OHLCV data dictionary

        Returns
        -------
        pd.DataFrame
            DataFrame with volume indicators
        """
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        volume = ohlcv['volume'].values

        indicators = pd.DataFrame(index=ohlcv['close'].index)

        if self.use_talib:
            # OBV
            indicators['obv'] = talib.OBV(close, volume)

            # Accumulation/Distribution
            indicators['ad'] = talib.AD(high, low, close, volume)

            # Chaikin A/D Oscillator
            indicators['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

        else:
            # Pure Python implementations
            close_series = ohlcv['close']
            volume_series = ohlcv['volume']

            # OBV
            indicators['obv'] = self._calculate_obv_python(close_series, volume_series)

            # Accumulation/Distribution
            indicators['ad'] = self._calculate_ad_python(
                ohlcv['high'], ohlcv['low'], close_series, volume_series
            )

            # Chaikin A/D Oscillator
            ad_series = indicators['ad']
            indicators['adosc'] = (
                ad_series.ewm(span=3, adjust=False).mean() -
                ad_series.ewm(span=10, adjust=False).mean()
            )

        return indicators

    def compute_volatility(self, ohlcv: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute volatility indicators.

        Indicators:
        - ATR: Average True Range
        - NATR: Normalized Average True Range
        - TRANGE: True Range

        Parameters
        ----------
        ohlcv : dict
            OHLCV data dictionary

        Returns
        -------
        pd.DataFrame
            DataFrame with volatility indicators
        """
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values

        indicators = pd.DataFrame(index=ohlcv['close'].index)

        if self.use_talib:
            # ATR
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)

            # NATR
            indicators['natr'] = talib.NATR(high, low, close, timeperiod=14)

            # True Range
            indicators['trange'] = talib.TRANGE(high, low, close)

        else:
            # Pure Python implementations
            high_series = ohlcv['high']
            low_series = ohlcv['low']
            close_series = ohlcv['close']

            # True Range
            tr1 = high_series - low_series
            tr2 = abs(high_series - close_series.shift(1))
            tr3 = abs(low_series - close_series.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['trange'] = true_range

            # ATR
            indicators['atr'] = true_range.rolling(14).mean()

            # NATR (normalized)
            indicators['natr'] = (indicators['atr'] / close_series) * 100

        return indicators

    # Pure Python helper methods

    @staticmethod
    def _calculate_rsi_python(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using pure Python/pandas."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_macd_python(close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD using pure Python/pandas."""
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return {'macd': macd, 'signal': signal, 'hist': hist}

    @staticmethod
    def _calculate_stoch_python(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator using pure Python/pandas."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
        d = k.rolling(window=d_period).mean()
        return {'k': k, 'd': d}

    @staticmethod
    def _calculate_adx_python(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Dict[str, pd.Series]:
        """Calculate ADX using pure Python/pandas."""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # Smoothed TR and DM
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}

    @staticmethod
    def _calculate_cci_python(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate CCI using pure Python/pandas."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: abs(x - x.mean()).mean())
        cci = (typical_price - sma) / (0.015 * mad + 1e-10)
        return cci

    @staticmethod
    def _calculate_mfi_python(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate MFI using pure Python/pandas."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        # Positive and negative money flow
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0)
        negative_flow = money_flow.where(delta < 0, 0)

        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        return mfi

    @staticmethod
    def _calculate_obv_python(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV using pure Python/pandas."""
        direction = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
        obv = (volume * direction).cumsum()
        return pd.Series(obv, index=close.index)

    @staticmethod
    def _calculate_ad_python(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Calculate Accumulation/Distribution using pure Python/pandas."""
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        ad = (clv * volume).cumsum()
        return ad

    @staticmethod
    def _calculate_sar_python(
        high: pd.Series,
        low: pd.Series,
        acceleration: float = 0.02,
        maximum: float = 0.2
    ) -> pd.Series:
        """Calculate Parabolic SAR using pure Python/pandas (simplified)."""
        # This is a simplified version
        sar = pd.Series(index=high.index, dtype=float)
        sar.iloc[0] = low.iloc[0]

        for i in range(1, len(high)):
            # Simplified: just use lowest low as SAR approximation
            sar.iloc[i] = low.iloc[:i].min()

        return sar
