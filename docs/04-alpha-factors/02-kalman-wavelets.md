---
layout: default
title: "Kalman Filters & Wavelets"
parent: "Part 4: Alpha Factors"
nav_order: 2
---

# Kalman Filters & Wavelets

Financial data is inherently noisy. Raw price series contain a mixture of genuine signal (trend, mean-reversion, momentum) and random noise (microstructure effects, stale quotes, transient liquidity shocks). Before feeding factors into a model or trading strategy, denoising them can dramatically improve signal-to-noise ratio.

This page covers two powerful signal-processing techniques available in Puffin:

1. **Kalman Filters** -- optimal recursive estimators for extracting trends and dynamic relationships from noisy observations.
2. **Wavelet Transforms** -- multi-resolution decomposition that separates a signal into components at different time scales.

---

## Signal Denoising with Kalman Filters

A Kalman filter is a Bayesian state-space model that recursively updates its estimate of a hidden state (the "true" signal) as new noisy observations arrive. It balances two sources of uncertainty:

- **Process covariance** (`Q`): how much the true signal is expected to change between observations.
- **Observation covariance** (`R`): how noisy the measurements are.

A small `Q` relative to `R` produces a very smooth output; a large `Q` tracks the observations more closely.

{: .tip }
> **Plain English:** Imagine you are trying to follow someone walking through fog. The Kalman filter combines what you *expect* them to do next (keep walking forward) with what you *see* through the fog (a noisy glimpse). The result is a smoother, more accurate path than either estimate alone.

```python
from puffin.factors import KalmanFilter, extract_trend, dynamic_hedge_ratio
import pandas as pd

# Basic Kalman filtering
prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])

kf = KalmanFilter(
    process_covariance=1e-5,  # How much we expect signal to change
    observation_covariance=1e-2  # Noise level in observations
)

# Filter signal (forward pass only)
filtered = kf.filter(prices)

# Smooth signal (forward-backward pass - better estimates)
smoothed = kf.smooth(prices)

print("Original:", prices.iloc[-1])
print("Filtered:", filtered.iloc[-1])
print("Smoothed:", smoothed.iloc[-1])
```

{: .note }
> **Filter vs. Smooth:** The `filter` method processes data causally (only past observations), making it suitable for live trading. The `smooth` method uses both past and future data (forward-backward pass), so it produces better estimates but can only be used offline (e.g., for research and backtesting).

### Trend Extraction

A common use case is extracting a slow-moving trend from a noisy price series for trend-following strategies:

```python
from puffin.factors import extract_trend
import pandas as pd

prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])

# Extract trend from noisy price series
trend = extract_trend(
    prices,
    process_variance=1e-5,  # Lower = smoother trend
    observation_variance=1e-2  # Higher = more smoothing
)

# Use for trend-following strategy
signal = (prices > trend).astype(int)  # 1 when above trend, 0 when below
```

{: .tip }
> Tuning `process_variance` and `observation_variance` is an art. Start with `process_variance=1e-5` and `observation_variance=1e-2`, then adjust: lower `process_variance` for smoother trends (position trading), higher for faster-reacting trends (swing trading).

### Dynamic Hedge Ratio for Pairs Trading

In pairs trading, the hedge ratio between two co-moving assets drifts over time. A Kalman filter provides time-varying estimates that adapt to structural changes in the relationship:

```python
from puffin.factors import dynamic_hedge_ratio
import pandas as pd

# Calculate time-varying hedge ratio for pairs trading
stock1 = pd.Series([100, 101, 102, 103, 104])
stock2 = pd.Series([50, 51, 50, 52, 53])

hedge_ratio = dynamic_hedge_ratio(stock1, stock2, delta=1e-5)

# Construct spread
spread = stock1 - hedge_ratio * stock2

# Trade when spread deviates from mean
```

{: .warning }
> A static OLS hedge ratio assumes the relationship between two assets is constant. In practice, correlations and betas shift over months. Using a Kalman-filtered dynamic hedge ratio avoids the "structural break" problem that plagues static pairs-trading strategies.

---

## Signal Denoising with Wavelets

Wavelets provide multi-resolution decomposition of signals, allowing you to separate different time scales. Unlike Fourier transforms (which lose time information), wavelets retain both frequency *and* time localization, making them ideal for financial signals that change character over time.

The key idea: decompose a signal into an **approximation** (low-frequency trend) and multiple **detail** layers (high-frequency components at progressively coarser scales). You can then threshold the detail coefficients to remove noise while preserving the underlying structure.

```python
from puffin.factors import (
    wavelet_denoise,
    wavelet_decompose,
    multiscale_decomposition
)
import pandas as pd

# Denoise signal using wavelet thresholding
noisy_signal = pd.Series([1, 2, 1.5, 3, 2.5, 4, 3.5, 5])

denoised = wavelet_denoise(
    noisy_signal,
    wavelet='db4',  # Daubechies 4 wavelet
    level=3,  # Decomposition depth
    threshold_method='soft'  # Soft thresholding
)
```

### Wavelet Decomposition

Decomposing a price series into multiple time scales reveals structure that is invisible in the raw data:

```python
from puffin.factors import wavelet_decompose
import pandas as pd

prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

components = wavelet_decompose(prices, wavelet='db4', level=3)

print("Trend:", components['approximation'])
print("High-frequency noise:", components['detail_1'])
print("Medium-frequency:", components['detail_2'])
```

### Multi-Scale Trading

Different wavelet scales correspond to different trading horizons. You can build separate strategies for each time scale:

```python
from puffin.factors import multiscale_decomposition
import pandas as pd

prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

# Multi-scale trading
scales = multiscale_decomposition(prices, level=4)

# Trade on different time scales
# scales['trend'] - Long-term position trading
# scales['D4'] - Swing trading (days to weeks)
# scales['D1'] - Day trading (intraday)
```

{: .tip }
> **Choosing a wavelet:** `db4` (Daubechies 4) is a good default for financial data. It balances smoothness and compactness. For very noisy intraday data, try `sym8` (Symlet 8) which is more symmetric. For simple trend extraction, `haar` (the simplest wavelet) can work well.

### Kalman vs. Wavelets -- When to Use Which?

| Criterion | Kalman Filter | Wavelet Transform |
|:----------|:-------------|:------------------|
| **Real-time capable** | Yes (causal filter) | Partial (boundary effects) |
| **Multi-scale analysis** | No (single scale) | Yes (multiple detail levels) |
| **Adaptive to regime changes** | Yes (recursive update) | Limited |
| **Pairs trading hedge ratio** | Ideal | Not applicable |
| **Trend extraction** | Good | Good |
| **Parameter tuning** | 2 params (Q, R) | wavelet type + level + threshold |

{: .note }
> In practice, Kalman filters and wavelets are complementary. A common pipeline is to first wavelet-denoise the raw prices, then apply a Kalman filter for trend extraction or dynamic hedge-ratio estimation.

---

## Source Code

Browse the implementation: [`puffin/factors/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/factors)
