---
layout: default
title: "Stacked LSTM & GRU"
parent: "Part 18: RNNs for Trading"
nav_order: 2
permalink: /18-rnns-for-trading/02-stacked-lstm-gru
---

# Stacked LSTM & GRU

Stacking multiple recurrent layers enables a model to learn hierarchical representations of sequential data. Lower layers capture short-term, fine-grained patterns (e.g., intraday momentum), while upper layers learn higher-level structure (e.g., multi-day trends). This section also introduces the Gated Recurrent Unit (GRU) -- a simpler alternative to LSTM -- and covers the practical aspects of regularization, hyperparameter tuning, and evaluation.

## Stacked LSTM Architectures

### Why Stack Layers?

A single LSTM layer maps an input sequence directly to a hidden representation. When you stack multiple layers, each layer receives the hidden-state sequence from the layer below, allowing the network to build progressively more abstract features.

- **Hierarchical Features**: Lower layers learn simple patterns, upper layers learn complex patterns
- **Better Abstraction**: Each layer transforms representations into more useful forms
- **Improved Performance**: Often (but not always) better results on complex tasks

{: .note }
> Diminishing returns set in quickly. Two or three stacked layers is typical for financial time series. Going deeper usually increases overfitting risk without meaningful accuracy gains.

### The StackedLSTM Class

The `StackedLSTM` class in Puffin allows specifying different hidden dimensions at each layer, creating a funnel-shaped architecture that progressively compresses information.

```python
import torch
import torch.nn as nn
from puffin.deep.rnn import StackedLSTM

# Create a 3-layer stacked LSTM with decreasing hidden dims
model = StackedLSTM(
    input_dim=1,
    hidden_dims=[128, 64, 32],
    output_dim=1,
    dropout=0.2
)

# Test forward pass
x = torch.randn(16, 20, 1)  # batch_size=16, seq_len=20, features=1
output = model(x)
print(f"Output shape: {output.shape}")  # torch.Size([16, 1])
```

### Building a Custom Stacked Architecture

For full control, you can compose individual LSTM layers manually with inter-layer dropout:

```python
from puffin.deep.rnn import LSTMNet

class DeepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1):
        super(DeepLSTM, self).__init__()

        # Create multiple LSTM layers with different hidden dims
        self.lstm_layers = nn.ModuleList()

        for i, hidden_dim in enumerate(hidden_dims):
            layer_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.lstm_layers.append(
                nn.LSTM(layer_input_dim, hidden_dim, 1, batch_first=True)
            )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        out = x
        for lstm in self.lstm_layers:
            out, _ = lstm(out)
            out = self.dropout(out)

        out = self.fc(out[:, -1, :])
        return out

# Instantiate and test
model = DeepLSTM(input_dim=5, hidden_dims=[128, 64, 32])
x = torch.randn(16, 30, 5)  # 5 input features
output = model(x)
print(f"Output shape: {output.shape}")  # torch.Size([16, 1])
```

## GRU: Gated Recurrent Unit

The GRU, introduced by Cho et al. (2014), simplifies the LSTM by combining the forget and input gates into a single **update gate** and merging the cell state and hidden state into one vector. This reduces the parameter count and often speeds up training.

### GRU Architecture

**Gates:**
1. **Update Gate** (z_t): Controls how much of the past information to keep
2. **Reset Gate** (r_t): Controls how much of the past information to forget when computing the candidate

**Mathematical Formulation:**

```
Update Gate:    z_t = σ(W_z · [h_{t-1}, x_t])
Reset Gate:     r_t = σ(W_r · [h_{t-1}, x_t])
Candidate:      h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
Hidden State:   h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

{: .tip }
> The GRU update gate serves double duty. When z_t is close to 1, the cell copies forward the candidate (like an LSTM input gate). When z_t is close to 0, the cell copies forward the previous hidden state (like an LSTM forget gate keeping everything).

### LSTM vs GRU: When to Use Which?

| Feature | LSTM | GRU |
|---------|------|-----|
| **Parameters** | More (3 gates + cell state) | Fewer (2 gates) |
| **Training Speed** | Slower | Faster |
| **Memory Usage** | Higher | Lower |
| **Performance** | Better on complex tasks | Comparable on many tasks |
| **When to Use** | Long sequences, complex patterns | Shorter sequences, faster training needed |

### Using GRUNet and TradingGRU

Puffin provides `GRUNet` (the raw module) and `TradingGRU` (the high-level wrapper), mirroring the LSTM API exactly.

```python
from puffin.deep.rnn import GRUNet, TradingGRU

# Low-level GRU module
gru_model = GRUNet(input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2)
x = torch.randn(16, 20, 1)
output = gru_model(x)
print(f"GRUNet output shape: {output.shape}")  # torch.Size([16, 1])

# High-level TradingGRU wrapper
import yfinance as yf
ticker = yf.Ticker("AAPL")
df = ticker.history(period="2y")
prices = df['Close'].values

gru = TradingGRU()
history = gru.fit(
    prices,
    lookback=20,
    epochs=50,
    lr=0.001
)

# Make predictions
gru_predictions = gru.predict(prices, steps=5)
print(f"GRU predictions: {gru_predictions}")
```

### Comparing LSTM and GRU Side by Side

```python
from puffin.deep.rnn import TradingLSTM, TradingGRU

# Train both on the same data
lstm = TradingLSTM()
lstm_history = lstm.fit(prices, lookback=20, epochs=50, lr=0.001)

gru = TradingGRU()
gru_history = gru.fit(prices, lookback=20, epochs=50, lr=0.001)

# Compare final validation losses
print(f"LSTM val_loss: {lstm_history['val_loss'][-1]:.6f}")
print(f"GRU  val_loss: {gru_history['val_loss'][-1]:.6f}")

# Compare predictions
lstm_preds = lstm.predict(prices, steps=5)
gru_preds = gru.predict(prices, steps=5)
print(f"LSTM predictions: {lstm_preds}")
print(f"GRU  predictions: {gru_preds}")
```

## Preventing Overfitting

Financial time series are noisy and non-stationary, making overfitting one of the biggest practical challenges. The following techniques help:

### Dropout

Dropout randomly zeroes hidden units during training, preventing co-adaptation.

```python
from puffin.deep.rnn import LSTMNet

# Apply dropout between LSTM layers
model = LSTMNet(hidden_dim=64, dropout=0.3)
```

### Early Stopping

Stop training when validation loss stops improving. The `EarlyStopping` callback in Puffin handles this automatically.

```python
from puffin.deep.training import EarlyStopping

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# In a manual training loop:
for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    if early_stop(val_loss, model):
        print(f"Early stopping at epoch {epoch}")
        break
```

### Weight Decay (L2 Regularization)

Add a penalty proportional to the squared magnitude of model weights:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

### Data Augmentation for Time Series

Adding small Gaussian noise to training sequences creates augmented samples that improve generalization:

```python
import numpy as np

def add_noise(series, noise_level=0.01):
    """Add small noise to create augmented samples."""
    noise = np.random.randn(len(series)) * noise_level * series.std()
    return series + noise
```

{: .warning }
> Be careful with time series augmentation. Unlike images, even small perturbations can change the direction of returns and corrupt labels. Keep noise levels very low (0.5-1% of standard deviation).

## Hyperparameter Tuning

Systematic hyperparameter search helps find the best architecture for your data.

```python
from sklearn.model_selection import ParameterGrid
from puffin.deep.rnn import TradingLSTM

# Define hyperparameter grid
param_grid = {
    'hidden_dim': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3],
    'lr': [0.001, 0.0001],
    'lookback': [10, 20, 30]
}

best_params = None
best_val_loss = float('inf')

for params in ParameterGrid(param_grid):
    lstm = TradingLSTM()
    history = lstm.fit(
        prices,
        lookback=params['lookback'],
        epochs=30,
        lr=params['lr']
    )

    val_loss = history['val_loss'][-1]

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = params

print(f"Best parameters: {best_params}")
print(f"Best validation loss: {best_val_loss:.4f}")
```

{: .tip }
> For large grids, consider random search or Bayesian optimization (e.g., Optuna) instead of exhaustive grid search. This reduces computation while still covering the important regions of hyperparameter space.

## Evaluation and Backtesting

### Walk-Forward Evaluation

Standard train/test splits can be misleading for time series. Walk-forward evaluation retrains the model on expanding windows to simulate realistic deployment.

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from puffin.deep.rnn import TradingLSTM

# Train model on first 800 observations
lstm = TradingLSTM()
lstm.fit(prices[:800], lookback=20, epochs=50)

# Walk-forward prediction on test set
test_prices = prices[800:]
predictions = []
actuals = []

for i in range(20, len(test_prices)):
    pred = lstm.predict(test_prices[:i], steps=1)[0]
    predictions.append(pred)
    actuals.append(test_prices[i])

predictions = np.array(predictions)
actuals = np.array(actuals)

# Regression metrics
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# Direction accuracy (more important for trading)
direction_correct = np.sign(predictions[1:] - actuals[:-1]) == np.sign(actuals[1:] - actuals[:-1])
direction_accuracy = direction_correct.mean()
print(f"Direction Accuracy: {direction_accuracy:.2%}")
```

{: .note }
> Direction accuracy above 52-53% can be profitable after transaction costs, depending on the magnitude of correct and incorrect predictions. Always compute both regression metrics and directional metrics when evaluating a trading model.

### Simple Backtesting Strategy

```python
def backtest_lstm_strategy(prices, lstm, initial_capital=10000):
    """Backtest a simple LSTM-based trading strategy."""
    capital = initial_capital
    position = 0
    trades = []

    for i in range(100, len(prices) - 1):
        pred = lstm.predict(prices[i - 100:i], steps=1)[0]
        current_price = prices[i]

        predicted_return = (pred - current_price) / current_price

        if predicted_return > 0.005 and position == 0:
            position = capital / current_price
            capital = 0
            trades.append(('BUY', i, current_price, position))

        elif predicted_return < -0.005 and position > 0:
            capital = position * current_price
            trades.append(('SELL', i, current_price, capital))
            position = 0

    # Close position if still open
    if position > 0:
        capital = position * prices[-1]

    total_return = (capital - initial_capital) / initial_capital

    return {
        'final_capital': capital,
        'total_return': total_return,
        'trades': trades
    }

results = backtest_lstm_strategy(prices, lstm)
print(f"Final Capital: ${results['final_capital']:.2f}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Number of Trades: {len(results['trades'])}")
```

## Source Code

- **StackedLSTM**: [`puffin/deep/rnn.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/rnn.py)
- **GRUNet** and **TradingGRU**: [`puffin/deep/rnn.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/rnn.py)
- **EarlyStopping** and **training_loop**: [`puffin/deep/training.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/training.py)
