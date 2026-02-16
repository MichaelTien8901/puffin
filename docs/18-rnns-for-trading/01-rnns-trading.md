---
layout: default
title: "Chapter 1: RNNs for Trading"
parent: "Part 18: RNNs for Trading"
nav_order: 1
---

# Chapter 1: RNNs for Trading

Recurrent Neural Networks (RNNs) are specifically designed to handle sequential data, making them particularly well-suited for time series prediction in algorithmic trading. This chapter explores RNN architectures, their variants (LSTM and GRU), and practical applications in trading.

## 1. Understanding RNNs

### Why RNNs for Time Series?

Traditional feedforward neural networks treat each input independently, with no memory of previous inputs. RNNs, however, maintain an internal state (hidden state) that captures information about previous time steps, making them ideal for sequential data like stock prices.

**Key Characteristics:**
- **Sequential Processing**: Process one time step at a time while maintaining context
- **Parameter Sharing**: Same weights used across all time steps
- **Variable-Length Sequences**: Can handle sequences of different lengths
- **Memory**: Hidden state acts as memory of past observations

### The Vanishing Gradient Problem

One of the main challenges with basic RNNs is the **vanishing gradient problem**. During backpropagation through time (BPTT), gradients can become exponentially small as they propagate backward through many time steps.

**Why This Matters:**
- Difficult to learn long-term dependencies
- Early time steps receive almost no gradient signal
- Training becomes unstable
- Model struggles to capture patterns spanning many time steps

**Example of the Problem:**

```python
import numpy as np

# Simulate gradient flow through time
def gradient_flow(num_steps, weight=0.5):
    """Show how gradients vanish over time steps."""
    gradient = 1.0
    gradients = [gradient]

    for t in range(num_steps):
        gradient *= weight  # Gradient gets multiplied at each step
        gradients.append(gradient)

    return gradients

# After 50 time steps with weight < 1
gradients = gradient_flow(50, weight=0.5)
print(f"Initial gradient: {gradients[0]}")
print(f"Gradient after 50 steps: {gradients[-1]:.2e}")
# Output: Gradient after 50 steps: 8.88e-16 (essentially zero!)
```

## 2. LSTM: Long Short-Term Memory

LSTMs were specifically designed to solve the vanishing gradient problem by introducing a more complex cell structure with gates that control information flow.

### LSTM Architecture

An LSTM cell contains:

1. **Cell State** (C_t): The "memory" that runs through the entire chain
2. **Hidden State** (h_t): The output passed to the next time step
3. **Three Gates**:
   - **Forget Gate**: Decides what to remove from cell state
   - **Input Gate**: Decides what new information to add
   - **Output Gate**: Decides what to output

**Mathematical Formulation:**

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
- * denotes element-wise multiplication
- [h_{t-1}, x_t] is concatenation of previous hidden state and current input

### Implementing LSTM for Price Prediction

```python
import torch
import torch.nn as nn
import numpy as np
from puffin.deep import LSTMNet, TradingLSTM

# Define LSTM network
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get output from last time step
        out = self.fc(out[:, -1, :])

        return out

# Using the high-level TradingLSTM wrapper
import yfinance as yf

# Download stock data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="2y")
prices = df['Close'].values

# Create and train LSTM
lstm = TradingLSTM()
history = lstm.fit(
    prices,
    lookback=20,      # Use 20 days of history
    epochs=50,
    lr=0.001,
    batch_size=32
)

# Make predictions
predictions = lstm.predict(prices, steps=5)
print(f"Next 5 predictions: {predictions}")
```

### Visualizing Training Progress

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

## 3. GRU: Gated Recurrent Unit

GRU is a simpler alternative to LSTM that combines the forget and input gates into a single "update gate" and merges the cell state and hidden state.

### GRU Architecture

**Gates:**
1. **Update Gate** (z_t): Controls how much of the past information to keep
2. **Reset Gate** (r_t): Controls how much of the past information to forget

**Mathematical Formulation:**

```
Update Gate:    z_t = σ(W_z · [h_{t-1}, x_t])
Reset Gate:     r_t = σ(W_r · [h_{t-1}, x_t])
Candidate:      h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
Hidden State:   h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

### LSTM vs GRU: When to Use Which?

| Feature | LSTM | GRU |
|---------|------|-----|
| **Parameters** | More (3 gates + cell state) | Fewer (2 gates) |
| **Training Speed** | Slower | Faster |
| **Memory Usage** | Higher | Lower |
| **Performance** | Better on complex tasks | Comparable on many tasks |
| **When to Use** | Long sequences, complex patterns | Shorter sequences, faster training needed |

### Implementing GRU

```python
from puffin.deep import GRUNet, TradingGRU

# Using GRU instead of LSTM
gru = TradingGRU()
history = gru.fit(
    prices,
    lookback=20,
    epochs=50,
    lr=0.001
)

# Make predictions
gru_predictions = gru.predict(prices, steps=5)

# Compare with LSTM
print(f"LSTM predictions: {predictions}")
print(f"GRU predictions:  {gru_predictions}")
```

## 4. Stacked LSTM Architectures

Stacking multiple LSTM layers can help the model learn hierarchical representations of the data, where lower layers capture low-level patterns and higher layers capture high-level patterns.

### Why Stack Layers?

- **Hierarchical Features**: Lower layers learn simple patterns, upper layers learn complex patterns
- **Better Abstraction**: Each layer can transform representations into more useful forms
- **Improved Performance**: Often (but not always) better results on complex tasks

### Implementing Stacked LSTM

```python
from puffin.deep import StackedLSTM

class DeepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1):
        super(DeepLSTM, self).__init__()

        # Create multiple LSTM layers
        self.lstm_layers = nn.ModuleList()

        for i, hidden_dim in enumerate(hidden_dims):
            layer_input_dim = input_dim if i == 0 else hidden_dims[i-1]
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

# Using the StackedLSTM class
model = StackedLSTM(
    input_dim=1,
    hidden_dims=[128, 64, 32],
    output_dim=1,
    dropout=0.2
)

# Test forward pass
x = torch.randn(16, 20, 1)  # batch_size=16, seq_len=20, features=1
output = model(x)
print(f"Output shape: {output.shape}")  # Should be (16, 1)
```

## 5. Multivariate Time Series Prediction

Real trading applications often involve multiple features (price, volume, indicators, etc.). Multivariate LSTMs can handle multiple input features to predict a target variable.

### Multivariate LSTM Implementation

```python
import pandas as pd
from puffin.deep import MultivariateLSTM

# Prepare multivariate data
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
from puffin.indicators import SMA, RSI, MACD

features['sma_20'] = SMA(df['Close'], 20)
features['rsi'] = RSI(df['Close'], 14)

macd_result = MACD(df['Close'])
features['macd'] = macd_result['macd']
features['signal'] = macd_result['signal']

# Create target (next day's return)
features['target'] = features['returns'].shift(-1)
features = features.dropna()

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

# Make prediction using all features
prediction = lstm.predict(features.iloc[-20:])
print(f"Predicted next return: {prediction[0]:.4f}")
```

### Feature Engineering for Multivariate Models

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

    # Moving averages
    for window in [5, 10, 20, 50]:
        features[f'sma_{window}'] = df['Close'].rolling(window).mean()
        features[f'sma_{window}_ratio'] = df['Close'] / features[f'sma_{window}']

    # RSI
    features['rsi_14'] = RSI(df['Close'], 14)

    # MACD
    macd = MACD(df['Close'])
    features['macd'] = macd['macd']
    features['macd_signal'] = macd['signal']
    features['macd_hist'] = macd['histogram']

    # Bollinger Bands
    from puffin.indicators import BollingerBands
    bb = BollingerBands(df['Close'], 20, 2)
    features['bb_upper'] = bb['upper']
    features['bb_lower'] = bb['lower']
    features['bb_position'] = (df['Close'] - bb['lower']) / (bb['upper'] - bb['lower'])

    # Volume indicators
    features['volume_sma'] = df['Volume'].rolling(20).mean()
    features['volume_ratio'] = df['Volume'] / features['volume_sma']

    return features.dropna()

# Use in multivariate LSTM
features = create_trading_features(df)
features['target'] = features['returns'].shift(-1)
features = features.dropna()

lstm = MultivariateLSTM()
history = lstm.fit(features, target_col='target', lookback=30, epochs=50)
```

## 6. LSTM for Sentiment Analysis

Text data like news headlines, tweets, and analyst reports can provide valuable trading signals. LSTMs with word embeddings are effective for sentiment classification.

### Sentiment LSTM Architecture

```python
from puffin.deep import SentimentLSTM, SentimentClassifier

# Example financial news headlines
headlines = [
    "Company reports record profits, stock soars",
    "CEO resigns amid scandal, shares plummet",
    "Quarterly earnings meet expectations",
    "New product launch exceeds forecasts",
    "Regulatory concerns weigh on stock price",
    "Analyst upgrades rating to buy",
    "Market volatility continues amid uncertainty"
]

# Sentiment labels: 0=negative, 1=neutral, 2=positive
labels = [2, 0, 1, 2, 0, 2, 1]

# Train sentiment classifier
classifier = SentimentClassifier()
history = classifier.fit(
    headlines,
    labels,
    epochs=10,
    batch_size=4,
    max_len=20,
    embed_dim=100,
    hidden_dim=128
)

# Predict sentiment for new headlines
new_headlines = [
    "Strong growth drives stock higher",
    "Disappointing results lead to selloff"
]

predictions = classifier.predict(new_headlines)
probabilities = classifier.predict_proba(new_headlines)

print("Predictions:", predictions)  # [2, 0] (positive, negative)
print("Probabilities:", probabilities)
```

### Using Pretrained Word Embeddings

```python
import numpy as np

# Load pretrained embeddings (e.g., GloVe)
def load_pretrained_embeddings(word2idx, embedding_file, embed_dim):
    """Load pretrained word embeddings."""
    embeddings = {}

    # Read embedding file (GloVe format)
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector

    # Create embedding matrix
    vocab_size = len(word2idx)
    embedding_matrix = np.random.randn(vocab_size, embed_dim).astype(np.float32)

    # Fill with pretrained embeddings where available
    for word, idx in word2idx.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]

    return embedding_matrix

# Use pretrained embeddings
classifier = SentimentClassifier()
classifier.build_vocab(headlines, max_vocab=10000)

# Load embeddings (example with GloVe)
# pretrained = load_pretrained_embeddings(
#     classifier.word2idx,
#     'glove.6B.100d.txt',
#     embed_dim=100
# )

# Train with pretrained embeddings
# history = classifier.fit(
#     headlines,
#     labels,
#     epochs=10,
#     pretrained_embeddings=pretrained
# )
```

### Integrating Sentiment with Price Prediction

```python
def combine_sentiment_and_price(df, sentiment_scores):
    """Combine sentiment analysis with price data."""
    features = create_trading_features(df)

    # Add sentiment features
    features['sentiment'] = sentiment_scores
    features['sentiment_ma'] = features['sentiment'].rolling(5).mean()
    features['sentiment_std'] = features['sentiment'].rolling(5).std()

    # Create target
    features['target'] = features['returns'].shift(-1)
    features = features.dropna()

    return features

# Example: Analyze news sentiment and combine with prices
def analyze_daily_news(date, headlines):
    """Analyze sentiment of daily news."""
    predictions = classifier.predict_proba(headlines)
    # Use positive probability as sentiment score
    sentiment_score = predictions[:, 2].mean()
    return sentiment_score

# In practice, you'd collect news for each trading day
# sentiment_scores = [analyze_daily_news(date, news) for date, news in daily_news]
# features = combine_sentiment_and_price(df, sentiment_scores)
```

## 7. Practical Considerations

### Sequence Length and Lookback

```python
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

### Preventing Overfitting

```python
# Techniques to prevent overfitting:

# 1. Dropout
model = LSTMNet(hidden_dim=64, dropout=0.3)

# 2. Early stopping
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break

# 3. Regularization
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 4. Data augmentation (for time series)
def add_noise(series, noise_level=0.01):
    """Add small noise to create augmented samples."""
    noise = np.random.randn(len(series)) * noise_level * series.std()
    return series + noise
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import ParameterGrid

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

## 8. Evaluation and Backtesting

### Evaluating Predictions

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train model
lstm = TradingLSTM()
lstm.fit(prices[:800], lookback=20, epochs=50)

# Predict on test set
test_prices = prices[800:]
predictions = []
actuals = []

for i in range(20, len(test_prices)):
    pred = lstm.predict(test_prices[:i], steps=1)[0]
    predictions.append(pred)
    actuals.append(test_prices[i])

predictions = np.array(predictions)
actuals = np.array(actuals)

# Calculate metrics
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Direction accuracy (more important for trading)
direction_correct = np.sign(predictions[1:] - actuals[:-1]) == np.sign(actuals[1:] - actuals[:-1])
direction_accuracy = direction_correct.mean()
print(f"Direction Accuracy: {direction_accuracy:.2%}")
```

### Backtesting Trading Strategy

```python
def backtest_lstm_strategy(prices, lstm, initial_capital=10000):
    """Backtest a simple LSTM-based trading strategy."""
    capital = initial_capital
    position = 0
    trades = []

    for i in range(100, len(prices) - 1):
        # Predict next price
        pred = lstm.predict(prices[i-100:i], steps=1)[0]
        current_price = prices[i]

        # Simple strategy: buy if predicted increase > 0.5%
        predicted_return = (pred - current_price) / current_price

        if predicted_return > 0.005 and position == 0:
            # Buy
            position = capital / current_price
            capital = 0
            trades.append(('BUY', i, current_price, position))

        elif predicted_return < -0.005 and position > 0:
            # Sell
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

# Run backtest
results = backtest_lstm_strategy(prices, lstm)
print(f"Final Capital: ${results['final_capital']:.2f}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Number of Trades: {len(results['trades'])}")
```

## Summary

In this chapter, we covered:

1. **RNN Basics**: Sequential processing and the vanishing gradient problem
2. **LSTM**: Architecture with gates to maintain long-term dependencies
3. **GRU**: Simpler alternative to LSTM with fewer parameters
4. **Stacked Architectures**: Multiple layers for hierarchical feature learning
5. **Multivariate Prediction**: Using multiple features for price forecasting
6. **Sentiment Analysis**: LSTM with word embeddings for text classification
7. **Practical Considerations**: Hyperparameter tuning, regularization, evaluation

RNNs and their variants (LSTM, GRU) are powerful tools for time series prediction in trading, but they require careful tuning and validation to avoid overfitting and ensure robust performance.

## Next Steps

- Experiment with different architectures and hyperparameters
- Combine RNN predictions with other trading signals
- Implement walk-forward validation for more robust backtesting
- Explore attention mechanisms (next chapter) for better interpretability
- Consider using transformer models for even better performance

## Further Reading

- [Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"](https://doi.org/10.1162/neco.1997.9.8.1735)
- [Cho et al. (2014): "Learning Phrase Representations using RNN Encoder-Decoder"](https://arxiv.org/abs/1406.1078)
- [Colah's Blog: "Understanding LSTM Networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Sequence Models course by deeplearning.ai
