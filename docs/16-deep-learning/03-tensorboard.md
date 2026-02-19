---
layout: default
title: "TensorBoard Integration"
parent: "Part 16: Deep Learning Fundamentals"
nav_order: 3
permalink: /16-deep-learning/03-tensorboard
---

# TensorBoard Integration

TensorBoard provides visual monitoring of deep learning training runs. The `puffin.deep.logging` module wraps TensorBoard's `SummaryWriter` with a `TrainingLogger` class designed for trading model workflows, making it straightforward to track experiments, compare hyperparameter configurations, and diagnose training issues.

## Setting Up the TrainingLogger

The `TrainingLogger` class creates a structured logging directory and provides methods for logging all common training artifacts:

```python
from puffin.deep.logging import TrainingLogger
import torch

# Create logger
logger = TrainingLogger(
    log_dir='runs',
    experiment_name='ffn_return_prediction'
)
```

This creates a directory structure under `runs/ffn_return_prediction/` with timestamped subdirectories for each run, so you can compare multiple experiments.

{: .note }
> Each call to `TrainingLogger` with the same experiment name creates a new timestamped subdirectory. This means you can run the same experiment multiple times without overwriting previous results.

## Logging Hyperparameters

Record the hyperparameters used for each experiment so you can compare configurations:

```python
# Log hyperparameters
hparams = {
    'hidden_dims': '[64, 32]',
    'dropout': 0.3,
    'lr': 0.001,
    'batch_size': 64
}
logger.log_hyperparameters(hparams)
```

{: .tip }
> Convert list and tuple hyperparameters to strings before logging (e.g., `'[64, 32]'` instead of `[64, 32]`). TensorBoard's hyperparameter dashboard only supports scalar values and strings.

## Logging the Model Graph

Visualize the model architecture in TensorBoard by logging the computational graph:

```python
from puffin.deep import FeedforwardNet

model = FeedforwardNet(
    input_dim=50,
    hidden_dims=[64, 32],
    output_dim=1,
    dropout=0.3
)

# Log model graph (once at start of training)
sample_input = torch.randn(1, 50)
logger.log_model_graph(model, sample_input)
```

The model graph view in TensorBoard shows every layer, its input/output shapes, and the connections between layers. This is useful for verifying that the architecture matches your expectations.

## Logging Training Metrics

The core use case for TensorBoard is tracking scalar metrics over the course of training:

```python
from puffin.deep import FeedforwardNet
from puffin.deep.training import training_loop, create_dataloaders
import torch
import torch.nn as nn

# Setup model and data
model = FeedforwardNet(input_dim=50, hidden_dims=[64, 32], output_dim=1)
train_loader, val_loader = create_dataloaders(X, y, batch_size=64, val_split=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs = 100

# During training: log metrics
for epoch in range(epochs):
    # Training step (simplified)
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            pred = model(batch_X)
            loss = criterion(pred.squeeze(), batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    # Log scalars to TensorBoard
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

# Close logger when done
logger.close()
```

{: .warning }
> Always call `logger.close()` when training is complete. This flushes any buffered data to disk. If you skip this step, the last few epochs of metrics may not appear in TensorBoard.

## Launching TensorBoard

After logging training data, launch TensorBoard to visualize results:

```bash
tensorboard --logdir=runs
# Open browser to http://localhost:6006
```

You can also compare multiple experiments by pointing TensorBoard at the parent directory:

```bash
# Compare all experiments in the runs/ directory
tensorboard --logdir=runs

# Compare specific experiments
tensorboard --logdir=runs/ffn_return_prediction,runs/ffn_direction_prediction
```

## Monitoring Key Metrics

Track these metrics during training to diagnose issues and evaluate model performance:

### 1. Loss Curves

Train vs. validation loss over time is the most important diagnostic:

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Both decreasing | Healthy training | Continue |
| Train decreasing, val increasing | Overfitting | Add regularization, early stop |
| Both flat | Learning rate too low or model capacity too low | Increase LR or model size |
| Both erratic | Learning rate too high | Decrease LR |
| Large gap between train and val | Overfitting | More dropout, simpler model |

### 2. Learning Rate

Track the current learning rate, especially when using schedulers:

```python
logger.log_scalars(
    epoch,
    learning_rate=optimizer.param_groups[0]['lr']
)
```

{: .tip }
> Plotting the learning rate alongside the loss curve helps you understand how LR changes affect convergence. A sudden drop in LR (from a step scheduler) should correspond to a drop in loss.

### 3. Weight Distributions

Check for vanishing or exploding weights by logging weight histograms:

```python
# Log weight distributions every 10 epochs
if epoch % 10 == 0:
    logger.log_weights(epoch, model)
```

Healthy weight distributions should:
- Be roughly centered around zero
- Have a moderate spread (not too narrow, not too wide)
- Remain stable across epochs (not growing or shrinking)

### 4. Gradient Norms

Monitor gradient magnitudes to detect training instabilities:

```python
# Log gradient distributions every 10 epochs
if epoch % 10 == 0:
    logger.log_gradients(epoch, model)
```

| Gradient Pattern | Problem | Solution |
|-----------------|---------|----------|
| Gradients near zero | Vanishing gradients | Use ReLU, batch norm, skip connections |
| Gradients very large | Exploding gradients | Gradient clipping, lower LR |
| Gradients stable | Healthy training | Continue |

### 5. Prediction Distributions

Ensure model predictions are not collapsing to a single value, which is a common failure mode in return prediction:

```python
# Log prediction statistics periodically
if epoch % 10 == 0:
    model.eval()
    with torch.no_grad():
        sample_preds = model(sample_batch)
    logger.log_scalars(
        epoch,
        pred_mean=sample_preds.mean().item(),
        pred_std=sample_preds.std().item()
    )
```

{: .warning }
> If the standard deviation of predictions drops to near zero, the model has collapsed to predicting a constant value (usually the mean return). This means the model has given up learning patterns. Try a lower learning rate, different architecture, or better features.

## Using MetricsTracker for Programmatic Analysis

The `MetricsTracker` class provides in-memory metrics tracking for programmatic access alongside TensorBoard logging:

```python
from puffin.deep.logging import MetricsTracker

tracker = MetricsTracker()

for epoch in range(epochs):
    # ... training code ...

    # Track metrics
    tracker.update(train_loss=train_loss, val_loss=val_loss)

# Access tracked metrics as lists
print(f"Best val loss: {min(tracker.metrics['val_loss']):.6f}")
print(f"Best epoch: {tracker.metrics['val_loss'].index(min(tracker.metrics['val_loss']))}")
print(f"Final train loss: {tracker.metrics['train_loss'][-1]:.6f}")
```

{: .note }
> `MetricsTracker` stores metrics in memory as Python lists. For long training runs, TensorBoard logging to disk is more memory-efficient. Use `MetricsTracker` when you need to make programmatic decisions during training (like saving the best model) or for post-training analysis.

## Experiment Organization

For systematic hyperparameter exploration, organize experiments with descriptive names:

```python
from puffin.deep.logging import TrainingLogger

# Descriptive experiment naming
for dropout in [0.2, 0.3, 0.4, 0.5]:
    for lr in [0.01, 0.001, 0.0001]:
        experiment_name = f"ffn_dropout{dropout}_lr{lr}"
        logger = TrainingLogger(
            log_dir='runs/hyperparam_search',
            experiment_name=experiment_name
        )

        # Log hyperparameters
        logger.log_hyperparameters({
            'dropout': dropout,
            'learning_rate': lr,
            'hidden_dims': '[64, 32]'
        })

        # ... train model ...

        logger.close()
```

This creates a structured directory under `runs/hyperparam_search/` with one subdirectory per configuration, making it easy to compare results in TensorBoard's parallel coordinates view.

{: .tip }
> Use TensorBoard's "HParams" tab to compare hyperparameter configurations side by side. This requires logging hyperparameters with `log_hyperparameters()` at the start of each experiment.

## Source Code

- Training logger: [`puffin/deep/logging.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/logging.py)
- Training loop and utilities: [`puffin/deep/training.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/training.py)
