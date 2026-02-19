---
layout: default
title: "1D CNNs for Time Series"
parent: "Part 17: CNNs for Trading"
nav_order: 1
permalink: /17-cnns-for-trading/01-1d-cnn-time-series
---

# 1D CNNs for Time Series

1D Convolutional Neural Networks apply convolutional filters along the time dimension, making them a natural fit for autoregressive prediction tasks where we forecast future values based on historical sequences. Compared to recurrent architectures (LSTMs, GRUs), 1D CNNs train faster, parallelize better on GPUs, and often achieve competitive accuracy on fixed-length lookback windows.

## Concept

A 1D convolution slides a kernel of length _k_ across an input sequence of length _T_, producing a feature map that highlights local patterns of width _k_. Stacking multiple convolutional layers creates a hierarchy: early layers detect simple motifs (e.g., short-term momentum), while deeper layers compose them into complex patterns (e.g., head-and-shoulders formations).

{: .note }
> Unlike 2D CNNs that scan height and width (images), 1D CNNs scan only along one axis (time). Each filter produces a single feature map that captures a specific temporal pattern.

## Architecture

The `Conv1DNet` model in Puffin uses the following building blocks:

- **Conv1D layers**: Sliding filters to detect temporal patterns at each scale
- **Max pooling**: Dimension reduction that provides local translation invariance
- **Batch normalization**: Stabilizes training by normalizing layer activations
- **Dropout**: Randomly zeros activations during training to reduce overfitting
- **Fully connected layers**: Maps extracted features to the final prediction

The architecture is configurable through `n_filters` and `kernel_sizes` lists, so you can experiment with shallow versus deep networks without modifying the model code.

## Sequence Preparation

Before feeding data into a 1D CNN, raw price or return series must be sliced into fixed-length input/target pairs. The `TradingCNN.prepare_sequences` static method handles this:

```python
from puffin.deep.cnn import TradingCNN
import numpy as np

# Create synthetic price data (random walk)
np.random.seed(42)
prices = np.cumsum(np.random.randn(1000)) + 100

# Prepare overlapping sequences
X_seq, y = TradingCNN.prepare_sequences(
    prices.reshape(-1, 1),   # Shape: (T, n_features)
    lookback=20              # Each input window is 20 time steps
)

print(f"Input shape:  {X_seq.shape}")   # (980, 1, 20)
print(f"Target shape: {y.shape}")       # (980, 1)
```

{: .tip }
> The `prepare_sequences` method returns inputs in channels-first format `(N, C, T)` as expected by PyTorch's `nn.Conv1d`. If your data has multiple features (e.g., price + volume), set `input_channels` accordingly.

## Training a 1D CNN

### Basic Training Loop

```python
from puffin.deep.cnn import TradingCNN

# Initialize model
model = TradingCNN(
    input_channels=1,      # Single feature (price)
    seq_length=20,          # Lookback window
    n_filters=[32, 64],    # Two conv layers: 32 then 64 filters
    kernel_sizes=[3, 3],   # Kernel size 3 for both layers
    output_dim=1,          # Regression: predict next value
    device='cpu'
)

# Train with validation split
history = model.fit(
    X_seq,
    y,
    epochs=50,
    lr=0.001,
    batch_size=32,
    validation_split=0.2
)

# Inspect training history
print(f"Final train loss: {history['train_loss'][-1]:.6f}")
print(f"Final val loss:   {history['val_loss'][-1]:.6f}")
```

### Multi-Feature Input

When multiple features are available (OHLCV, technical indicators), each feature becomes a separate input channel:

```python
from puffin.deep.cnn import TradingCNN
import numpy as np
import pandas as pd

# Simulate multi-feature data: price, volume, RSI
np.random.seed(42)
n_samples = 1000
data = np.column_stack([
    np.cumsum(np.random.randn(n_samples)) + 100,   # Price
    np.abs(np.random.randn(n_samples)) * 1e6,       # Volume
    50 + 20 * np.random.randn(n_samples),            # RSI-like
])

# Prepare sequences with 3 input channels
X_seq, y = TradingCNN.prepare_sequences(data, lookback=20)

model = TradingCNN(
    input_channels=3,       # Three features
    seq_length=20,
    n_filters=[32, 64, 128],
    kernel_sizes=[5, 3, 3],
    output_dim=1,
    device='cpu'
)

history = model.fit(X_seq, y, epochs=30, lr=0.001, batch_size=64)
```

{: .warning }
> Always normalize each feature channel independently before training. Raw prices and volumes can differ by orders of magnitude, which destabilizes gradient descent.

## Making Predictions

```python
from puffin.deep.cnn import TradingCNN
import numpy as np

# Predict on the last 10 sequences
predictions = model.predict(X_seq[-10:])
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions: {predictions.flatten()}")

# Compare to actuals
actuals = y[-10:].flatten()
mae = np.mean(np.abs(predictions.flatten() - actuals))
print(f"Mean Absolute Error: {mae:.4f}")
```

## Use Cases

1D CNNs are well suited for several trading applications:

- **Price forecasting**: Predicting the next period's price or return
- **Volatility prediction**: Forecasting future realized or implied volatility
- **Multi-step ahead forecasting**: Predicting multiple future periods by setting `output_dim > 1`
- **Regime detection**: Classifying the current market state (trending, mean-reverting, volatile)

## Best Practices

### Data Preparation

1. **Normalization**: Always normalize inputs before training.

   ```python
   # Percentage returns (stationary)
   returns = np.diff(prices) / prices[:-1]

   # Or min-max scaling per window
   normalized = (window - window.min()) / (window.max() - window.min() + 1e-8)
   ```

2. **Sequence length**: Choose an appropriate lookback period.
   - Too short (< 5): misses important patterns
   - Too long (> 100): overfitting, slow training
   - Typical range: 10-50 periods for daily data

3. **Stationarity**: Prefer returns or log-returns over raw prices to avoid non-stationarity issues.

### Model Training

1. **Start simple**: Begin with one or two conv layers and 32 filters. Add complexity only if validation performance plateaus.
2. **Use validation**: Monitor validation loss every epoch to detect overfitting.
3. **Early stopping**: Use `EarlyStopping` from `puffin.deep.training` to halt training when validation loss stops improving.
4. **Learning rate**: Start with 0.001; reduce by a factor of 10 if training is unstable.

```python
from puffin.deep.training import EarlyStopping, LRScheduler

early_stop = EarlyStopping(patience=10, min_delta=1e-5)
scheduler = LRScheduler(optimizer, patience=5, factor=0.5)
```

### Avoiding Overfitting

{: .warning }
> Financial time series have low signal-to-noise ratios. CNNs can easily memorize noise instead of learning genuine patterns. Regularization is essential.

1. **Dropout**: Add dropout layers with rates between 0.3 and 0.5
2. **Data augmentation**: Add small Gaussian noise or time shifts to training sequences
3. **Cross-validation**: Use time-series aware cross-validation (expanding or sliding window) -- never shuffle temporal data
4. **Regularization**: Apply L2 weight decay through the optimizer

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

## Performance Evaluation

For regression tasks, evaluate with standard error metrics. For classification tasks (up/down/flat), use precision, recall, and trading-specific metrics:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

predictions = model.predict(X_test)
actuals = y_test.flatten()

mae = mean_absolute_error(actuals, predictions.flatten())
rmse = np.sqrt(mean_squared_error(actuals, predictions.flatten()))

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Directional accuracy: did we predict the correct sign?
pred_direction = np.sign(predictions.flatten())
actual_direction = np.sign(actuals)
directional_accuracy = np.mean(pred_direction == actual_direction)
print(f"Directional Accuracy: {directional_accuracy:.2%}")
```

## Source Code

The 1D CNN implementation lives in the following modules:

- **Model architecture**: [`puffin/deep/cnn.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/cnn.py) -- `Conv1DNet` (PyTorch module) and `TradingCNN` (training wrapper)
- **Training utilities**: [`puffin/deep/training.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/training.py) -- `EarlyStopping`, `LRScheduler`, `create_dataloaders`, `set_seed`
