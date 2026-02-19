---
layout: default
title: "CNN-TA & Transfer Learning"
parent: "Part 17: CNNs for Trading"
nav_order: 2
permalink: /17-cnns-for-trading/02-cnn-ta-transfer-learning
---

# CNN-TA and Transfer Learning

This section covers two related approaches that leverage 2D convolutional architectures for trading: CNN-TA, which converts time series into image-like representations, and transfer learning, which adapts pretrained computer vision models for financial classification.

## CNN-TA: Time Series as Images

### Concept

The CNN-TA (CNN for Technical Analysis) approach treats multi-indicator time series data as 2D images:

- **Rows** represent different features or technical indicators
- **Columns** represent time steps
- **Pixel intensity** represents normalized indicator values

This creates a heatmap-style representation where spatial relationships between indicators at different time steps become visual patterns that standard 2D CNNs can learn to recognize.

{: .note }
> The CNN-TA approach was introduced by Sezer and Ozbayoglu (2018) and has since been adopted across multiple financial forecasting studies. It bridges the gap between time-series analysis and the rich toolkit of computer vision architectures.

### Series to Image Conversion

The `series_to_image` function computes technical indicators from OHLCV data and stacks them into a 2D array:

```python
from puffin.deep.cnn_ta import series_to_image
import yfinance as yf

# Download OHLCV data
ticker = yf.Ticker("SPY")
df = ticker.history(period="6mo")

# Convert to image representation
image = series_to_image(
    df,
    window=20,
    indicators=['sma', 'rsi', 'macd', 'bbands']
)

print(f"Image shape: {image.shape}")
# Output: (n_features, 20) -- one row per indicator, one column per time step
```

Each indicator is min-max normalized to the [0, 1] range within the window, so all rows have comparable intensity scales regardless of the original indicator magnitudes.

{: .tip }
> The choice of indicators directly affects what patterns the CNN can detect. Include indicators that capture different market dimensions: trend (SMA), momentum (RSI), volatility (Bollinger Bands), and signal crossovers (MACD).

### Training a CNN-TA Model

The `TradingCNNTA` class wraps the `CNNTA` PyTorch module with a high-level training API:

```python
from puffin.deep.cnn_ta import TradingCNNTA
import numpy as np
import yfinance as yf

ticker = yf.Ticker("SPY")
df = ticker.history(period="2y")

# Build training samples and labels
ohlcv_samples = []
labels = []

for i in range(len(df) - 35):
    window_df = df.iloc[i:i+30]
    ohlcv_samples.append(window_df)

    # Label based on 5-day forward return
    future_return = (
        df['Close'].iloc[i + 35] - df['Close'].iloc[i + 30]
    ) / df['Close'].iloc[i + 30]

    if future_return < -0.02:
        labels.append(0)   # Down / Sell
    elif future_return > 0.02:
        labels.append(2)   # Up / Buy
    else:
        labels.append(1)   # Flat / Hold

# Initialize model
model = TradingCNNTA(
    window=30,
    indicators=['sma', 'rsi', 'macd', 'bbands'],
    n_classes=3,
    device='cpu'
)

# Train
history = model.fit(
    ohlcv_samples,
    np.array(labels),
    epochs=50,
    lr=0.001,
    batch_size=16
)

# Predict
predictions = model.predict(ohlcv_samples[-5:])
probabilities = model.predict_proba(ohlcv_samples[-5:])

print(f"Predictions: {predictions}")
print(f"Probabilities:\n{probabilities}")
```

### Advantages of the Image Representation

- **Rich representation**: Combines multiple indicators into a single input tensor
- **Spatial patterns**: The CNN detects relationships _between_ indicators at the same time step (vertical patterns) and across time (horizontal patterns)
- **Visual interpretability**: Activation maps can be visualized to understand which indicator/time-step regions drive predictions
- **Architecture reuse**: Any 2D CNN architecture from the computer vision literature can be applied directly

{: .warning }
> The labeling scheme (thresholds like +/- 2%) strongly affects model behavior. Overly tight thresholds produce noisy labels; overly wide thresholds collapse most samples into the "Hold" class. Validate threshold choices with class balance analysis.

## Transfer Learning

### Concept

Transfer learning leverages pretrained models (ResNet, VGG, MobileNet) trained on millions of ImageNet images and adapts them for financial predictions. Even though financial heatmaps look nothing like natural images, the low-level features learned by these models (edges, textures, gradients) transfer surprisingly well.

### How It Works

1. **Load pretrained model**: Start with weights from ImageNet training
2. **Replace final layer**: Swap the 1000-class ImageNet head for an _n_-class financial head
3. **Resize inputs**: Scale financial images to the expected input size (typically 224 x 224)
4. **Fine-tune**: Train on financial data, optionally freezing early layers

### Implementation

```python
from puffin.deep.transfer import TransferLearningModel, prepare_financial_images
from torch.utils.data import TensorDataset, DataLoader
import torch

# Resize CNN-TA images to standard pretrained model input size
images = prepare_financial_images(
    image_array,                 # (N, C, H, W) or (N, H, W)
    target_size=(224, 224)
)

# Create model from pretrained ResNet-18
model = TransferLearningModel.from_pretrained(
    model_name='resnet18',
    n_classes=3,
    device='cpu'
)

# Split into train / validation
X_train = torch.FloatTensor(images[:800])
y_train = torch.LongTensor(labels[:800])
X_val   = torch.FloatTensor(images[800:])
y_val   = torch.LongTensor(labels[800:])

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=16,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=16
)

# Fine-tune with frozen base layers
history = model.fine_tune(
    train_loader,
    epochs=10,
    lr=0.001,
    freeze_layers=True,
    val_loader=val_loader
)

# Predict
predictions = model.predict(images[-10:])
probabilities = model.predict_proba(images[-10:])
```

### Fine-Tuning Strategies

There are two main strategies, each with different trade-offs:

**Strategy 1: Freeze Base Layers**

Only the final classification layer is trained. This is fast, requires little data, and prevents catastrophic forgetting of the pretrained features.

```python
history = model.fine_tune(
    train_loader,
    epochs=10,
    freeze_layers=True   # Only train the final layer
)
```

{: .tip }
> Freezing is the recommended starting point, especially when you have fewer than 5,000 training samples. It acts as a strong regularizer by fixing millions of pretrained weights.

**Strategy 2: Full Fine-Tuning**

All layers are trainable, allowing the model to adapt low-level features to financial data. This can improve accuracy but risks overfitting on small datasets.

```python
history = model.fine_tune(
    train_loader,
    epochs=20,
    freeze_layers=False  # Train all layers with a lower learning rate
)
```

### Available Pretrained Models

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| `resnet18` | 11M | Fast | Good | Default starting point |
| `resnet34` | 21M | Medium | Better | More capacity if needed |
| `resnet50` | 25M | Slow | Best | Large datasets with GPU |
| `vgg16` | 138M | Slow | Good | Feature extraction |
| `vgg19` | 144M | Slow | Good | Deep feature extraction |
| `mobilenet_v2` | 3.4M | Fastest | Good | Resource-constrained environments |

## Complete Trading Strategy Example

The following example combines CNN-TA with a structured trading strategy class:

```python
from puffin.deep.cnn_ta import TradingCNNTA
import yfinance as yf
import numpy as np


class CNNTradingStrategy:
    """Trading strategy using CNN-TA for signal generation."""

    def __init__(self, symbol, window=30):
        self.symbol = symbol
        self.window = window
        self.model = TradingCNNTA(
            window=window,
            indicators=['sma', 'rsi', 'macd', 'bbands'],
            n_classes=3,
            device='cpu'
        )

    def prepare_training_data(self, start_date, end_date):
        """Prepare historical data for training."""
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=start_date, end=end_date)

        samples = []
        labels = []

        for i in range(len(df) - self.window - 10):
            window_df = df.iloc[i:i + self.window]
            samples.append(window_df)

            future_return = (
                df['Close'].iloc[i + self.window + 5]
                - df['Close'].iloc[i + self.window]
            ) / df['Close'].iloc[i + self.window]

            if future_return < -0.02:
                labels.append(0)   # Sell
            elif future_return > 0.02:
                labels.append(2)   # Buy
            else:
                labels.append(1)   # Hold

        return samples, np.array(labels)

    def train(self, start_date, end_date, epochs=50):
        """Train the CNN model on historical data."""
        samples, labels = self.prepare_training_data(start_date, end_date)

        history = self.model.fit(
            samples,
            labels,
            epochs=epochs,
            lr=0.001,
            batch_size=16,
            validation_split=0.2
        )
        return history

    def generate_signal(self, recent_data):
        """Generate a trading signal from recent OHLCV data."""
        if len(recent_data) < self.window:
            return {'signal': 1, 'confidence': 0.0, 'probabilities': {}}

        prediction = self.model.predict(
            [recent_data.iloc[-self.window:]]
        )[0]
        probabilities = self.model.predict_proba(
            [recent_data.iloc[-self.window:]]
        )[0]

        return {
            'signal': prediction,           # 0=sell, 1=hold, 2=buy
            'confidence': probabilities[prediction],
            'probabilities': {
                'sell': probabilities[0],
                'hold': probabilities[1],
                'buy': probabilities[2],
            }
        }


# Usage
strategy = CNNTradingStrategy('SPY', window=30)

# Train on historical data
print("Training model...")
history = strategy.train('2020-01-01', '2023-12-31', epochs=30)

# Generate signal for recent data
ticker = yf.Ticker('SPY')
recent_data = ticker.history(period='3mo')
signal = strategy.generate_signal(recent_data)

print(f"\nCurrent Signal: {['SELL', 'HOLD', 'BUY'][signal['signal']]}")
print(f"Confidence: {signal['confidence']:.2%}")
print(f"Probabilities: {signal['probabilities']}")
```

## Performance Evaluation

Evaluate both classification accuracy and trading-specific metrics to get a complete picture of model quality:

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Classification metrics
predictions = model.predict(test_samples)

print(classification_report(
    test_labels,
    predictions,
    target_names=['Sell', 'Hold', 'Buy']
))

cm = confusion_matrix(test_labels, predictions)
print(f"\nConfusion Matrix:\n{cm}")

# Trading-specific metrics
returns = []
for i, pred in enumerate(predictions):
    if pred == 2:       # Buy signal
        returns.append(actual_returns[i])
    elif pred == 0:     # Sell signal (short)
        returns.append(-actual_returns[i])

returns = np.array(returns)
sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
max_drawdown = np.min(np.minimum.accumulate(np.cumsum(returns)) - np.cumsum(returns))

print(f"\nSharpe Ratio:  {sharpe_ratio:.2f}")
print(f"Max Drawdown:  {max_drawdown:.4f}")
print(f"Win Rate:      {np.mean(returns > 0):.2%}")
```

{: .warning }
> A high classification accuracy does not guarantee profitable trading. Always compute Sharpe ratio, maximum drawdown, and transaction cost-adjusted returns before deploying any model.

## Source Code

The CNN-TA and transfer learning implementations live in the following modules:

- **CNN-TA model**: [`puffin/deep/cnn_ta.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/cnn_ta.py) -- `series_to_image`, `CNNTA` (PyTorch module), `TradingCNNTA` (training wrapper)
- **Transfer learning**: [`puffin/deep/transfer.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/transfer.py) -- `TransferLearningModel`, `prepare_financial_images`
- **Training utilities**: [`puffin/deep/training.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/training.py) -- `EarlyStopping`, `LRScheduler`, `compute_class_weights`
