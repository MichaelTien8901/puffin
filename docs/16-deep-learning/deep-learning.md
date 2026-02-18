---
layout: default
title: "Part 16: Deep Learning Fundamentals"
nav_order: 17
permalink: /16-deep-learning/
---

# Deep Learning Fundamentals for Trading

## Overview

This chapter introduces deep learning for algorithmic trading, covering feedforward neural networks, training techniques, and practical implementation using both PyTorch and TensorFlow. Deep learning models can capture complex non-linear relationships in market data that traditional ML models might miss.

## Why Deep Learning for Trading?

Deep learning offers several advantages for trading applications:

- **Non-linear Pattern Recognition**: Captures complex relationships between features
- **Feature Learning**: Automatically learns relevant representations from raw data
- **Time Series Modeling**: RNNs and LSTMs excel at sequential data
- **Multi-modal Data**: Can combine price data, news, and sentiment
- **Scalability**: Handles high-dimensional feature spaces effectively

However, deep learning also has challenges:
- Requires more data than traditional ML
- Prone to overfitting on financial data
- Computationally expensive
- Less interpretable than linear models

## Neural Network Basics

### The Neuron

A neural network is composed of layers of artificial neurons. Each neuron:
1. Takes multiple inputs (x₁, x₂, ..., xₙ)
2. Multiplies each by a weight (w₁, w₂, ..., wₙ)
3. Adds a bias term (b)
4. Applies an activation function (σ)

```
output = σ(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

### Activation Functions

Common activation functions used in trading networks:

| Function | Formula | Use Case |
|----------|---------|----------|
| ReLU | max(0, x) | Hidden layers (most common) |
| Tanh | (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ) | Hidden layers (centered output) |
| Leaky ReLU | max(0.01x, x) | Prevents dying neurons |
| ELU | x if x > 0 else α(eˣ - 1) | Smooth, self-normalizing |
| Sigmoid | 1/(1 + e⁻ˣ) | Binary classification output |
| Softmax | eˣⁱ/Σeˣʲ | Multi-class classification |

For regression tasks (predicting returns), we typically use no activation on the output layer.

## PyTorch vs TensorFlow

Both frameworks are widely used in finance:

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

## Feedforward Neural Networks

A feedforward network (also called multi-layer perceptron) is the simplest neural network architecture. Data flows in one direction: input → hidden layers → output.

### Architecture for Return Prediction

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

### PyTorch Implementation Details

The `FeedforwardNet` class provides full PyTorch flexibility:

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

### TensorFlow Implementation

For teams preferring TensorFlow/Keras:

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

The TensorFlow implementation gracefully falls back if TensorFlow is not installed, making the codebase more portable.

## Training Techniques

### Early Stopping

Early stopping prevents overfitting by monitoring validation loss and stopping when it stops improving:

```python
from puffin.deep.training import EarlyStopping, training_loop, create_dataloaders
import torch
import torch.nn as nn

# Create model and data loaders
model = FeedforwardNet(input_dim=50, hidden_dims=[64, 32], output_dim=1)
train_loader, val_loader = create_dataloaders(X, y, batch_size=64, val_split=0.2)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Create early stopping callback
early_stop = EarlyStopping(
    patience=10,              # Stop after 10 epochs without improvement
    min_delta=0.0001,         # Minimum change to qualify as improvement
    restore_best_weights=True # Restore weights from best epoch
)

# Training loop with early stopping
def early_stop_callback(epoch, train_loss, val_loss, model):
    return early_stop(val_loss, model)

history = training_loop(
    model, train_loader, val_loader,
    epochs=200,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=[early_stop_callback]
)
```

### Learning Rate Scheduling

Learning rate schedules help training converge better:

```python
from puffin.deep.training import LRScheduler

# Step schedule: reduce LR every N epochs
scheduler = LRScheduler(
    optimizer,
    schedule_type='step',
    step_size=30,     # Reduce every 30 epochs
    gamma=0.1         # Multiply by 0.1
)

# Cosine annealing: smooth decay
scheduler = LRScheduler(
    optimizer,
    schedule_type='cosine',
    T_max=100,        # Total epochs
    eta_min=0.00001   # Minimum LR
)

# Warmup + cosine: start slow, then decay
scheduler = LRScheduler(
    optimizer,
    schedule_type='warmup_cosine',
    warmup_epochs=10,  # Linear warmup for 10 epochs
    T_max=100,
    eta_min=0.00001
)

# Use in training loop
for epoch in range(100):
    # ... training code ...
    scheduler.step()
    current_lr = scheduler.get_last_lr()
    print(f"Epoch {epoch}, LR: {current_lr}")
```

### Dropout

Dropout randomly deactivates neurons during training, forcing the network to learn robust features:

```python
# Dropout is included in model architecture
model = TradingFFN(
    input_dim=50,
    hidden_dims=[64, 32],
    output_dim=1,
    dropout=0.3  # 30% of neurons dropped during training
)
```

Higher dropout rates (0.3-0.5) can help with overfitting, but too high (>0.6) may underfit.

### Batch Normalization

Batch normalization standardizes activations between layers, improving training stability:

```python
# Batch norm is automatically included in our models
# It normalizes each layer's output to have mean=0, std=1

# Already included in FeedforwardNet:
# layers.append(nn.Linear(prev_dim, hidden_dim))
# layers.append(nn.BatchNorm1d(hidden_dim))  # <-- Batch norm
# layers.append(activation)
```

Benefits:
- Allows higher learning rates
- Reduces sensitivity to initialization
- Acts as mild regularization
- Speeds up convergence

## TensorBoard Logging

TensorBoard provides visual monitoring of training:

```python
from puffin.deep.logging import TrainingLogger
import torch

# Create logger
logger = TrainingLogger(
    log_dir='runs',
    experiment_name='ffn_return_prediction'
)

# Log hyperparameters
hparams = {
    'hidden_dims': '[64, 32]',
    'dropout': 0.3,
    'lr': 0.001,
    'batch_size': 64
}
logger.log_hyperparameters(hparams)

# Log model graph (once at start)
sample_input = torch.randn(1, 50)
logger.log_model_graph(model, sample_input)

# During training: log metrics
for epoch in range(epochs):
    # ... training code ...

    logger.log_scalars(
        epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=optimizer.param_groups[0]['lr']
    )

    # Log weights and gradients periodically
    if epoch % 10 == 0:
        logger.log_weights(epoch, model)
        logger.log_gradients(epoch, model)

# Close logger
logger.close()
```

View results in TensorBoard:
```bash
tensorboard --logdir=runs
# Open browser to http://localhost:6006
```

### Monitoring Key Metrics

Track these metrics during training:

1. **Loss curves**: Train vs validation loss over time
2. **Learning rate**: Current LR (especially with scheduling)
3. **Weight distributions**: Check for vanishing/exploding gradients
4. **Gradient norms**: Should be stable, not too large/small
5. **Prediction distributions**: Ensure not collapsing to single value

## Complete Example: Predicting Stock Returns

Here's a full pipeline for training a neural network to predict next-day returns:

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
print(f"Test R²: {test_r2:.4f}")

# Save model
model.save('models/aapl_return_predictor')

# Load later
loaded_model = TradingFFN.load('models/aapl_return_predictor')
```

## Advanced Training: Custom Callbacks

Create custom callbacks for advanced training control:

```python
from puffin.deep.training import training_loop, create_dataloaders
from puffin.deep.logging import TrainingLogger, MetricsTracker

# Setup
model = FeedforwardNet(input_dim=50, hidden_dims=[64, 32], output_dim=1)
train_loader, val_loader = create_dataloaders(X_train, y_train, batch_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Create logger and metrics tracker
logger = TrainingLogger(experiment_name='custom_training')
tracker = MetricsTracker()

# Custom callback: log to TensorBoard and track best model
best_val_loss = float('inf')

def logging_callback(epoch, train_loss, val_loss, model):
    global best_val_loss

    # Log to TensorBoard
    logger.log_scalars(
        epoch,
        train_loss=train_loss,
        val_loss=val_loss
    )

    # Track metrics
    tracker.update(train_loss=train_loss, val_loss=val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"New best model saved at epoch {epoch}")

    return False  # Don't stop training

# Train with custom callback
history = training_loop(
    model, train_loader, val_loader,
    epochs=100,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=[logging_callback]
)

logger.close()
```

## Best Practices for Financial Deep Learning

### 1. Data Preprocessing
- Always standardize/normalize features
- Handle missing data carefully (forward fill vs. drop)
- Use time-series aware train/test splits
- Be cautious of look-ahead bias

### 2. Architecture Design
- Start simple (2-3 layers) before going deep
- Use dropout (0.3-0.5) to prevent overfitting
- Batch normalization for stability
- ReLU or Leaky ReLU for hidden layers

### 3. Training Strategy
- Use early stopping (patience 10-20 epochs)
- Learning rate scheduling (cosine or step)
- Monitor both train and validation loss
- Save checkpoints regularly

### 4. Regularization
- L2 weight decay in optimizer
- Dropout in hidden layers
- Early stopping
- Data augmentation (if applicable)

### 5. Evaluation
- Use walk-forward testing
- Check for overfitting (train vs. val loss)
- Evaluate on out-of-sample data
- Consider transaction costs in backtesting

## Common Pitfalls

### Overfitting
**Problem**: Model memorizes training data, fails on new data.

**Solutions**:
- More dropout
- Early stopping
- Simpler architecture
- More training data
- Regularization

### Underfitting
**Problem**: Model can't learn the patterns.

**Solutions**:
- Deeper or wider network
- Lower dropout
- More training epochs
- Better features
- Lower learning rate

### Vanishing/Exploding Gradients
**Problem**: Gradients become too small or too large.

**Solutions**:
- Use batch normalization
- Gradient clipping
- Better weight initialization
- ReLU activation
- Skip connections (ResNets)

### Predicting Noise
**Problem**: Financial returns are noisy; model may learn noise.

**Solutions**:
- Focus on feature quality over model complexity
- Use ensemble methods
- Predict direction rather than exact returns
- Incorporate market regime detection

## Summary

Deep learning for trading requires:
1. **Solid foundations**: Understanding neural networks, activation functions, and backpropagation
2. **Framework choice**: PyTorch for research/flexibility, TensorFlow for production
3. **Training techniques**: Early stopping, LR scheduling, dropout, batch norm
4. **Monitoring**: TensorBoard for visualizing training progress
5. **Best practices**: Time-series aware splits, regularization, walk-forward testing

The `puffin.deep` module provides all these tools with both PyTorch and TensorFlow implementations, making it easy to experiment with deep learning for algorithmic trading.

## Source Code

Browse the implementation: [`puffin/deep/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep)

## Next Steps

- **CNNs for Trading**: Apply convolutional networks to time series
- **RNNs and LSTMs**: Model sequential dependencies in market data
- **Autoencoders**: Dimensionality reduction and anomaly detection
- **GANs**: Generate synthetic market data for training

## References

- Goodfellow et al. (2016): [*Deep Learning*](https://www.deeplearningbook.org/)
- [Paszke et al. (2019): *PyTorch: An Imperative Style, High-Performance Deep Learning Library*](https://arxiv.org/abs/1912.01703)
- Abadi et al. (2016): *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems*
- Dixon et al. (2020): *Machine Learning in Finance*
