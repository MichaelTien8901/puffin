---
layout: default
title: "Feedforward Neural Networks"
parent: "Part 16: Deep Learning Fundamentals"
nav_order: 1
permalink: /16-deep-learning/01-feedforward-nns
---

# Feedforward Neural Networks

A feedforward network (also called multi-layer perceptron) is the simplest neural network architecture. Data flows in one direction: input to hidden layers to output. These networks form the foundation for all deep learning architectures used in trading.

## Neural Network Basics

### The Neuron

A neural network is composed of layers of artificial neurons. Each neuron:
1. Takes multiple inputs (x1, x2, ..., xn)
2. Multiplies each by a weight (w1, w2, ..., wn)
3. Adds a bias term (b)
4. Applies an activation function (sigma)

```
output = sigma(w1*x1 + w2*x2 + ... + wn*xn + b)
```

{: .note }
> The universal approximation theorem states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function. In practice, however, deeper networks with fewer neurons per layer tend to generalize better.

### Activation Functions

Common activation functions used in trading networks:

| Function | Formula | Use Case |
|----------|---------|----------|
| ReLU | max(0, x) | Hidden layers (most common) |
| Tanh | (ex - e-x)/(ex + e-x) | Hidden layers (centered output) |
| Leaky ReLU | max(0.01x, x) | Prevents dying neurons |
| ELU | x if x > 0 else alpha(ex - 1) | Smooth, self-normalizing |
| Sigmoid | 1/(1 + e-x) | Binary classification output |
| Softmax | exi / Sum(exj) | Multi-class classification |

For regression tasks (predicting returns), we typically use no activation on the output layer.

{: .tip }
> ReLU is the default choice for hidden layers. Only switch to Leaky ReLU or ELU if you observe "dying neurons" (many neurons outputting zero). For output layers, use no activation for regression, sigmoid for binary classification, and softmax for multi-class.

## PyTorch vs TensorFlow

Both frameworks are widely used in finance. The `puffin.deep` module provides implementations in both.

### PyTorch

**Pros:**
- More Pythonic and intuitive
- Dynamic computational graphs
- Easier debugging
- Popular in research

**Cons:**
- Smaller production ecosystem
- Less enterprise tooling

### TensorFlow/Keras

**Pros:**
- Larger production ecosystem
- TensorFlow Serving for deployment
- Better mobile/edge device support
- High-level Keras API

**Cons:**
- Steeper learning curve
- More verbose for research

## Architecture for Return Prediction

The `TradingFFN` class provides a high-level interface for building and training feedforward networks:

```python
from puffin.deep import TradingFFN
import numpy as np

# Generate features
X = np.random.randn(1000, 50)  # 1000 samples, 50 features
y = np.random.randn(1000)      # Target returns

# Create model with 2 hidden layers
model = TradingFFN(
    input_dim=50,
    hidden_dims=[64, 32],      # 2 hidden layers
    output_dim=1,              # Single output (return prediction)
    dropout=0.3,               # 30% dropout
    activation='relu'          # ReLU activation
)

# Train model
history = model.fit(
    X, y,
    epochs=100,
    lr=0.001,                  # Learning rate
    batch_size=64,
    validation_split=0.2,
    verbose=True
)

# Make predictions
predictions = model.predict(X)
```

{: .warning }
> Always use time-series aware train/test splits when working with financial data. Random splits introduce look-ahead bias, where future information leaks into the training set. Use chronological splits or walk-forward validation instead.

## PyTorch Implementation Details

The `FeedforwardNet` class provides full PyTorch flexibility for custom workflows:

```python
from puffin.deep import FeedforwardNet
import torch

# Create raw PyTorch model
model = FeedforwardNet(
    input_dim=50,
    hidden_dims=[64, 32, 16],
    output_dim=1,
    dropout=0.3,
    activation='relu'
)

# Manual forward pass
x = torch.randn(32, 50)  # Batch of 32
output = model(x)        # Returns (32, 1)

# Access network layers
print(model.network)
```

The `FeedforwardNet` internally constructs a sequential stack of layers: `Linear -> BatchNorm1d -> Activation -> Dropout` for each hidden layer, followed by a final `Linear` output layer.

## TensorFlow Implementation

For teams preferring TensorFlow/Keras, the `puffin.deep.feedforward_tf` module provides an equivalent API:

```python
from puffin.deep.feedforward_tf import TradingFFN_TF, check_tensorflow_available

# Check if TensorFlow is installed
if check_tensorflow_available():
    model = TradingFFN_TF(
        input_dim=50,
        hidden_dims=[64, 32],
        output_dim=1,
        dropout=0.3,
        activation='relu'
    )

    history = model.fit(
        X, y,
        epochs=100,
        lr=0.001,
        batch_size=64,
        validation_split=0.2
    )

    predictions = model.predict(X)
else:
    print("TensorFlow not installed")
```

{: .note }
> The TensorFlow implementation gracefully falls back if TensorFlow is not installed, making the codebase more portable. All `puffin.deep` classes default to PyTorch, with TensorFlow offered as an optional alternative.

## Complete Example: Predicting Stock Returns

Here is a full pipeline for training a neural network to predict next-day returns:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from puffin.data import YFinanceProvider
from puffin.ml import compute_features
from puffin.deep import TradingFFN
from puffin.deep.training import set_seed

# Set random seed for reproducibility
set_seed(42)

# Load data
data = YFinanceProvider().fetch('AAPL', start='2020-01-01', end='2023-12-31')

# Compute features
features = compute_features(data)

# Create target: next-day return
target = data['Close'].pct_change().shift(-1)

# Remove NaNs
mask = features.notna().all(axis=1) & target.notna()
X = features[mask].values
y = target[mask].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split (time-series aware)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Create and train model
model = TradingFFN(
    input_dim=X_train.shape[1],
    hidden_dims=[128, 64, 32],
    output_dim=1,
    dropout=0.4,
    activation='relu'
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    lr=0.001,
    batch_size=64,
    validation_split=0.2,
    verbose=True
)

# Evaluate on test set
test_predictions = model.predict(X_test)

# Compute test metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

test_mse = mean_squared_error(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f"Test MSE: {test_mse:.6f}")
print(f"Test MAE: {test_mae:.6f}")
print(f"Test R2: {test_r2:.4f}")

# Save model
model.save('models/aapl_return_predictor')

# Load later
loaded_model = TradingFFN.load('models/aapl_return_predictor')
```

{: .tip }
> When evaluating return prediction models, R-squared values are typically very low (often below 0.05) for daily returns. This is normal for financial data. Even a small positive R-squared can be economically significant when combined with proper position sizing and risk management.

## Architecture Design Guidelines

When designing feedforward networks for trading:

| Guideline | Recommendation |
|-----------|---------------|
| Depth | Start with 2-3 hidden layers |
| Width | First hidden layer 1-2x the input dimension |
| Narrowing | Each successive layer should be narrower |
| Dropout | 0.3-0.5 for financial data |
| Activation | ReLU for hidden layers, none for regression output |
| Output | Single neuron for return prediction, 2-3 for classification |

## Source Code

- Feedforward network (PyTorch): [`puffin/deep/feedforward.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/feedforward.py)
- Feedforward network (TensorFlow): [`puffin/deep/feedforward_tf.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/feedforward_tf.py)
- Module init: [`puffin/deep/__init__.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/__init__.py)
