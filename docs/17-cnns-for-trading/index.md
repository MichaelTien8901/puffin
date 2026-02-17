---
layout: default
title: "Part 17: CNNs for Trading"
nav_order: 18
---

# CNNs for Trading

Convolutional Neural Networks (CNNs) are powerful tools originally designed for computer vision tasks. However, their ability to detect spatial and temporal patterns makes them surprisingly effective for financial time series analysis. In this chapter, we explore three approaches to using CNNs in algorithmic trading.

## Why CNNs for Trading?

CNNs excel at:
- **Pattern recognition**: Detecting recurring chart patterns and technical formations
- **Feature extraction**: Automatically learning relevant features from raw data
- **Translation invariance**: Recognizing patterns regardless of when they occur
- **Hierarchical learning**: Building complex representations from simple patterns

## Approach 1: 1D CNNs for Time Series

### Concept

1D CNNs apply convolutional filters along the time dimension, making them ideal for autoregressive prediction where we forecast future values based on historical sequences.

### Architecture

The `Conv1DNet` model uses:
- **Conv1D layers**: Sliding filters to detect temporal patterns
- **Max pooling**: Dimension reduction and invariance
- **Batch normalization**: Training stability
- **Fully connected layers**: Final prediction

### Implementation

```python
from puffin.deep.cnn import TradingCNN
import numpy as np

# Create synthetic price data
prices = np.cumsum(np.random.randn(1000)) + 100

# Initialize model
model = TradingCNN(
    input_channels=1,      # Single feature (price)
    seq_length=20,         # Look back 20 periods
    n_filters=[32, 64],    # Filter counts per layer
    kernel_sizes=[3, 3],   # Kernel sizes
    output_dim=1,          # Predict next value
    device='cpu'
)

# Prepare sequences
X_seq, y = TradingCNN.prepare_sequences(
    prices.reshape(-1, 1),
    lookback=20
)

# Train model
history = model.fit(
    X_seq,
    y,
    epochs=50,
    lr=0.001,
    batch_size=32,
    validation_split=0.2
)

# Make predictions
predictions = model.predict(X_seq[-10:])
print(f"Predictions: {predictions}")
```

### Use Cases

- **Price forecasting**: Predicting next period's price or return
- **Volatility prediction**: Forecasting future volatility
- **Multi-step ahead forecasting**: Predicting multiple periods into the future

## Approach 2: CNN-TA (2D Image Representation)

### Concept

This innovative approach treats time series as 2D images, where:
- **Rows** represent different features or technical indicators
- **Columns** represent time steps
- **Pixel intensity** represents normalized values

This creates a "heatmap" that CNNs can analyze for patterns.

### Series to Image Conversion

```python
from puffin.deep.cnn_ta import series_to_image, TradingCNNTA
import pandas as pd
import yfinance as yf

# Download OHLCV data
ticker = yf.Ticker("SPY")
df = ticker.history(period="6mo")

# Convert to image
image = series_to_image(
    df,
    window=20,
    indicators=['sma', 'rsi', 'macd', 'bbands']
)

print(f"Image shape: {image.shape}")
# Output: (n_features, 20)
```

### Training a CNN-TA Model

```python
# Prepare multiple samples
ohlcv_samples = []
labels = []

for i in range(len(df) - 30):
    window_df = df.iloc[i:i+30]
    ohlcv_samples.append(window_df)

    # Label: 0=down, 1=flat, 2=up based on next 5 days
    future_return = (df['close'].iloc[i+35] - df['close'].iloc[i+30]) / df['close'].iloc[i+30]
    if future_return < -0.02:
        labels.append(0)  # Down
    elif future_return > 0.02:
        labels.append(2)  # Up
    else:
        labels.append(1)  # Flat

# Initialize and train model
model = TradingCNNTA(
    window=30,
    indicators=['sma', 'rsi', 'macd', 'bbands'],
    n_classes=3,
    device='cpu'
)

history = model.fit(
    ohlcv_samples,
    np.array(labels),
    epochs=50,
    lr=0.001,
    batch_size=16
)

# Make predictions
predictions = model.predict(ohlcv_samples[-5:])
probabilities = model.predict_proba(ohlcv_samples[-5:])

print(f"Predictions: {predictions}")
print(f"Probabilities:\n{probabilities}")
```

### Advantages

- **Rich representation**: Combines multiple indicators in single image
- **Spatial patterns**: Detects relationships between indicators
- **Visual interpretability**: Can visualize what model "sees"

## Approach 3: Transfer Learning

### Concept

Transfer learning leverages pretrained models (ResNet, VGG, etc.) trained on millions of images and adapts them for financial predictions. This approach can be surprisingly effective when you have limited training data.

### How It Works

1. **Load pretrained model**: Start with weights from ImageNet
2. **Replace final layer**: Adapt output for your classification task
3. **Fine-tune**: Train on financial data with optional layer freezing

### Implementation

```python
from puffin.deep.transfer import TransferLearningModel, prepare_financial_images
from torch.utils.data import TensorDataset, DataLoader
import torch

# Prepare your image data (from CNN-TA approach)
# Resize to standard size for pretrained models
images = prepare_financial_images(
    image_array,
    target_size=(224, 224)
)

# Create model from pretrained ResNet
model = TransferLearningModel.from_pretrained(
    model_name='resnet18',
    n_classes=3,
    device='cpu'
)

# Prepare data loaders
X_train = torch.FloatTensor(images[:800])
y_train = torch.LongTensor(labels[:800])
X_val = torch.FloatTensor(images[800:])
y_val = torch.LongTensor(labels[800:])

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=16
)

# Fine-tune (freeze base layers, train only final layer)
history = model.fine_tune(
    train_loader,
    epochs=10,
    lr=0.001,
    freeze_layers=True,
    val_loader=val_loader
)

# Make predictions
predictions = model.predict(images[-10:])
probabilities = model.predict_proba(images[-10:])
```

### Fine-Tuning Strategies

**Strategy 1: Freeze Base Layers**
```python
# Fast training, prevents overfitting with small datasets
history = model.fine_tune(
    train_loader,
    epochs=10,
    freeze_layers=True  # Only train final layer
)
```

**Strategy 2: Full Fine-Tuning**
```python
# Slower, but better adaptation to financial data
history = model.fine_tune(
    train_loader,
    epochs=20,
    freeze_layers=False  # Train all layers
)
```

### Available Pretrained Models

- **ResNet**: `resnet18`, `resnet34`, `resnet50`
- **VGG**: `vgg16`, `vgg19`
- **MobileNet**: `mobilenet_v2` (lightweight, fast)

## Practical Trading Strategy Example

Let's build a complete trading strategy using CNN-TA:

```python
import yfinance as yf
import numpy as np
import pandas as pd
from puffin.deep.cnn_ta import TradingCNNTA

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

        # Create samples with labels
        for i in range(len(df) - self.window - 10):
            window_df = df.iloc[i:i+self.window]
            samples.append(window_df)

            # Label based on next 5 days return
            future_return = (
                df['Close'].iloc[i+self.window+5] -
                df['Close'].iloc[i+self.window]
            ) / df['Close'].iloc[i+self.window]

            if future_return < -0.02:
                labels.append(0)  # Sell signal
            elif future_return > 0.02:
                labels.append(2)  # Buy signal
            else:
                labels.append(1)  # Hold signal

        return samples, np.array(labels)

    def train(self, start_date, end_date, epochs=50):
        """Train the CNN model."""
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
        """Generate trading signal from recent data."""
        if len(recent_data) < self.window:
            return 1  # Hold

        # Get prediction
        prediction = self.model.predict([recent_data.iloc[-self.window:]])[0]
        probabilities = self.model.predict_proba([recent_data.iloc[-self.window:]])[0]

        # Return signal with confidence
        return {
            'signal': prediction,  # 0=sell, 1=hold, 2=buy
            'confidence': probabilities[prediction],
            'probabilities': {
                'sell': probabilities[0],
                'hold': probabilities[1],
                'buy': probabilities[2]
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

## Best Practices

### Data Preparation

1. **Normalization**: Always normalize inputs
   ```python
   # Price normalization
   normalized = prices / prices.iloc[0]

   # Or use returns
   returns = prices.pct_change()
   ```

2. **Sequence length**: Choose appropriate lookback
   - Too short: Miss important patterns
   - Too long: Overfitting, slow training
   - Typical range: 10-50 periods

3. **Feature engineering**: Include relevant indicators
   ```python
   indicators = ['sma', 'rsi', 'macd', 'bbands', 'volume']
   ```

### Model Training

1. **Start simple**: Begin with fewer layers/filters
2. **Use validation**: Monitor validation loss for overfitting
3. **Early stopping**: Stop when validation loss increases
4. **Learning rate**: Start with 0.001, adjust if needed

### Avoiding Overfitting

1. **Dropout**: Add dropout layers (0.3-0.5)
2. **Data augmentation**: Slight noise/shifts in training data
3. **Cross-validation**: Use time-series cross-validation
4. **Regularization**: L1/L2 penalties on weights

## Performance Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Get predictions
predictions = model.predict(test_samples)

# Classification report
print(classification_report(
    test_labels,
    predictions,
    target_names=['Sell', 'Hold', 'Buy']
))

# Confusion matrix
cm = confusion_matrix(test_labels, predictions)
print("\nConfusion Matrix:")
print(cm)

# Trading metrics
returns = []
for i, pred in enumerate(predictions):
    if pred == 2:  # Buy signal
        returns.append(actual_returns[i])
    elif pred == 0:  # Sell signal (short)
        returns.append(-actual_returns[i])

sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
print(f"\nSharpe Ratio: {sharpe_ratio:.2f}")
```

## Comparison of Approaches

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **1D CNN** | Fast, simple, efficient | Limited feature interaction | Single asset forecasting |
| **CNN-TA** | Rich representation, interpretable | More complex, needs OHLCV | Multi-indicator strategies |
| **Transfer** | Works with small data, powerful | Slow, needs GPU | Limited training data |

## Key Takeaways

1. **CNNs are versatile**: Can be applied to time series in multiple ways
2. **Image representation works**: Treating data as images is surprisingly effective
3. **Transfer learning helps**: Pretrained models provide strong baselines
4. **Validation is critical**: Always use out-of-sample testing
5. **Combine with traditional analysis**: CNNs complement, not replace, fundamental analysis

## Next Steps

- Experiment with different architectures
- Try ensemble methods (combine multiple CNNs)
- Explore attention mechanisms
- Integrate with other models (LSTM, transformers)
- Build complete trading systems with risk management

## References

- [LeCun et al. (1998): "Gradient-Based Learning Applied to Document Recognition"](https://doi.org/10.1109/5.726791)
- [Sezer et al. (2020): "Financial Time Series Forecasting with Deep Learning: A Systematic Literature Review"](https://doi.org/10.1016/j.asoc.2020.106181)
- Jiang et al. (2017): "Deep Neural Networks for Stock Market Prediction: A Survey"
