---
layout: default
title: "LSTM Fundamentals"
parent: "Part 18: RNNs for Trading"
nav_order: 1
permalink: /18-rnns-for-trading/01-lstm-fundamentals
---

# LSTM Fundamentals

Long Short-Term Memory (LSTM) networks were designed by Hochreiter and Schmidhuber (1997) to solve the vanishing gradient problem that plagues vanilla RNNs. By introducing a gated cell structure, LSTMs can learn to retain or discard information over long sequences, making them well-suited for financial time series where patterns may span dozens or hundreds of time steps.

## LSTM Architecture

An LSTM cell contains two state vectors and three gates that regulate the flow of information:

1. **Cell State** (C_t): The long-term memory that runs through the entire chain
2. **Hidden State** (h_t): The short-term output passed to the next time step
3. **Three Gates**:
   - **Forget Gate**: Decides what to remove from cell state
   - **Input Gate**: Decides what new information to add
   - **Output Gate**: Decides what to output from the cell state

### Mathematical Formulation

```
Forget Gate:    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:     i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate:      C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell State:     C_t = f_t * C_{t-1} + i_t * C̃_t
Output Gate:    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State:   h_t = o_t * tanh(C_t)
```

Where:
- σ is the sigmoid function (output between 0 and 1)
- \* denotes element-wise multiplication
- \[h\_{t-1}, x\_t\] is the concatenation of the previous hidden state and the current input

{: .note }
> The **cell state** is the key innovation. Because it flows through the chain with only linear interactions (element-wise multiply and add), gradients can propagate backward through many time steps without vanishing or exploding.

## Implementing LSTM for Price Prediction

The `LSTMNet` class in Puffin provides a configurable LSTM network, and `TradingLSTM` wraps it with data preparation, training, and prediction utilities.

### The LSTMNet Module

```python
import torch
import torch.nn as nn
from puffin.deep.rnn import LSTMNet

# LSTMNet is a configurable LSTM network
model = LSTMNet(
    input_dim=1,       # single feature (e.g., closing price)
    hidden_dim=64,     # 64 hidden units per layer
    num_layers=2,      # 2 stacked LSTM layers
    output_dim=1,      # predict a single value
    dropout=0.2        # dropout between layers
)

# Forward pass with dummy data
x = torch.randn(16, 20, 1)   # batch=16, seq_len=20, features=1
output = model(x)
print(f"Output shape: {output.shape}")  # torch.Size([16, 1])
```

### Training with TradingLSTM

The `TradingLSTM` class handles sequence windowing, train/validation splitting, normalization, and the training loop internally.

```python
import yfinance as yf
from puffin.deep.rnn import TradingLSTM

# Download stock data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="2y")
prices = df['Close'].values

# Create and train LSTM
lstm = TradingLSTM()
history = lstm.fit(
    prices,
    lookback=20,      # use 20 days of history per sample
    epochs=50,
    lr=0.001,
    batch_size=32
)

# Make predictions
predictions = lstm.predict(prices, steps=5)
print(f"Next 5 predictions: {predictions}")
```

{: .tip }
> Start with a lookback of 20 trading days (approximately one month). Shorter windows respond faster to regime changes but capture less context; longer windows are more stable but slower to adapt.

### Visualizing Training Progress

Monitoring the gap between training and validation loss is essential for detecting overfitting.

```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Training Progress')
plt.legend()
plt.grid(True)
plt.show()
```

## Multivariate Time Series Prediction

Real trading models typically use multiple input features -- price, volume, technical indicators -- rather than a single closing price. The `MultivariateLSTM` class extends `TradingLSTM` to handle DataFrames with multiple columns.

### Preparing Multivariate Features

```python
import pandas as pd
import numpy as np
from puffin.deep.rnn import MultivariateLSTM

# Download data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="2y")

# Create features
features = pd.DataFrame({
    'close': df['Close'],
    'volume': df['Volume'],
    'high': df['High'],
    'low': df['Low'],
    'returns': df['Close'].pct_change()
})

# Add technical indicators
features['sma_20'] = df['Close'].rolling(20).mean()
features['volatility'] = features['returns'].rolling(20).std()

# Create target (next day's return)
features['target'] = features['returns'].shift(-1)
features = features.dropna()
```

### Training the Multivariate Model

```python
# Train multivariate LSTM
lstm = MultivariateLSTM()
history = lstm.fit(
    features,
    target_col='target',
    lookback=20,
    epochs=50,
    lr=0.001,
    hidden_dims=[128, 64, 32]
)

# Predict using all features
prediction = lstm.predict(features.iloc[-20:])
print(f"Predicted next return: {prediction[0]:.4f}")
```

### Comprehensive Feature Engineering

A richer feature set often improves predictive power. The function below builds a broad set of price-derived features suitable for a multivariate LSTM.

```python
def create_trading_features(df):
    """Create comprehensive feature set for trading."""
    features = pd.DataFrame(index=df.index)

    # Price features
    features['close'] = df['Close']
    features['high'] = df['High']
    features['low'] = df['Low']
    features['volume'] = df['Volume']

    # Returns
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close']).diff()

    # Volatility
    features['volatility'] = features['returns'].rolling(20).std()

    # Moving averages and ratios
    for window in [5, 10, 20, 50]:
        features[f'sma_{window}'] = df['Close'].rolling(window).mean()
        features[f'sma_{window}_ratio'] = df['Close'] / features[f'sma_{window}']

    # Volume indicators
    features['volume_sma'] = df['Volume'].rolling(20).mean()
    features['volume_ratio'] = df['Volume'] / features['volume_sma']

    return features.dropna()

# Build features and train
features = create_trading_features(df)
features['target'] = features['returns'].shift(-1)
features = features.dropna()

lstm = MultivariateLSTM()
history = lstm.fit(features, target_col='target', lookback=30, epochs=50)
```

{: .warning }
> Always normalize or standardize features before feeding them to an LSTM. The `TradingLSTM` and `MultivariateLSTM` classes handle this internally, but if you build a custom pipeline, apply `StandardScaler` or min-max scaling to each feature independently.

## Practical Considerations

### Choosing the Lookback Window

The lookback (sequence length) is one of the most impactful hyperparameters. Too short and the model misses longer-term patterns; too long and training slows with diminishing returns.

```python
from puffin.deep.rnn import TradingLSTM

# Experiment with different lookback periods
lookbacks = [10, 20, 30, 50]
results = {}

for lookback in lookbacks:
    lstm = TradingLSTM()
    history = lstm.fit(prices, lookback=lookback, epochs=30)
    results[lookback] = history['val_loss'][-1]

# Find best lookback
best_lookback = min(results, key=results.get)
print(f"Best lookback period: {best_lookback} (val_loss: {results[best_lookback]:.4f})")
```

{: .tip }
> For daily equity data, lookback values between 15 and 60 tend to work well. For intraday data with more observations, consider longer windows (60-200 bars).

## Source Code

- **LSTMNet** and **TradingLSTM**: [`puffin/deep/rnn.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/rnn.py)
- **MultivariateLSTM**: [`puffin/deep/rnn.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/rnn.py)
- **Training utilities**: [`puffin/deep/training.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/training.py)
