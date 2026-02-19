---
layout: default
title: "Sentiment RNN"
parent: "Part 18: RNNs for Trading"
nav_order: 3
permalink: /18-rnns-for-trading/03-sentiment-rnn
---

# Sentiment RNN

Text data -- news headlines, analyst reports, social media posts, earnings call transcripts -- often drives short-term price moves. LSTMs with word embeddings are effective at classifying financial text into sentiment categories (positive, negative, neutral), producing signals that can augment price-based models. This section covers the `SentimentLSTM` network architecture, the `SentimentClassifier` training wrapper, pretrained embeddings, and integration with price prediction.

## Sentiment LSTM Architecture

The `SentimentLSTM` class combines an embedding layer, one or more LSTM layers, and a fully connected output layer for classification. The embedding layer converts word indices to dense vectors, the LSTM captures sequential dependencies between words, and the output layer maps the final hidden state to sentiment probabilities.

### Building and Running SentimentLSTM

```python
import torch
from puffin.deep.sentiment_rnn import SentimentLSTM

# Instantiate the model directly
model = SentimentLSTM(
    vocab_size=5000,
    embed_dim=100,
    hidden_dim=128,
    output_dim=3,        # negative, neutral, positive
    dropout=0.3,
    num_layers=2
)

# Forward pass with token indices
# batch of 4 sequences, each length 15
x = torch.randint(0, 5000, (4, 15))
logits = model(x)
print(f"Logits shape: {logits.shape}")  # torch.Size([4, 3])
```

{: .note }
> The `SentimentLSTM` module uses the hidden state from the **last time step** of the top LSTM layer for classification. For variable-length inputs, the `SentimentClassifier` pads sequences to a uniform length and uses the true last hidden state based on sequence lengths.

## Training with SentimentClassifier

The `SentimentClassifier` wraps `SentimentLSTM` with vocabulary building, tokenization, padding, and the training loop, providing a scikit-learn-style `fit`/`predict` interface.

### Basic Training Example

```python
from puffin.deep.sentiment_rnn import SentimentClassifier

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

print("Predictions:", predictions)       # [2, 0] (positive, negative)
print("Probabilities:", probabilities)
```

{: .tip }
> In practice, you need hundreds or thousands of labeled examples for reliable sentiment classification. Consider using financial-specific labeled datasets such as Financial PhraseBank or labeled SEC filings rather than hand-labeling a small set.

### Understanding the Training History

The training history dictionary contains loss and accuracy for both training and validation sets at each epoch:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Sentiment LSTM Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy curves
axes[1].plot(history['train_acc'], label='Train Accuracy')
axes[1].plot(history['val_acc'], label='Val Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Sentiment LSTM Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

## Using Pretrained Word Embeddings

Randomly initialized embeddings require large training corpora to learn meaningful representations. Pretrained embeddings like GloVe or Word2Vec provide a strong starting point, especially when labeled data is limited.

### Loading GloVe Embeddings

```python
import numpy as np

def load_pretrained_embeddings(word2idx, embedding_file, embed_dim):
    """Load pretrained word embeddings into a matrix.

    Parameters
    ----------
    word2idx : dict
        Mapping from words to integer indices (from SentimentClassifier.word2idx).
    embedding_file : str
        Path to GloVe or Word2Vec text file.
    embed_dim : int
        Dimensionality of the embeddings (must match the file).

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (vocab_size, embed_dim).
    """
    embeddings = {}

    # Read embedding file (GloVe format: word dim1 dim2 ... dimN)
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector

    # Create embedding matrix with random initialization for OOV words
    vocab_size = len(word2idx)
    embedding_matrix = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.01

    # Fill with pretrained embeddings where available
    found = 0
    for word, idx in word2idx.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
            found += 1

    print(f"Found pretrained vectors for {found}/{vocab_size} words ({found/vocab_size:.1%})")
    return embedding_matrix
```

### Training with Pretrained Embeddings

```python
from puffin.deep.sentiment_rnn import SentimentClassifier

# Build vocabulary first
classifier = SentimentClassifier()
classifier.build_vocab(headlines, max_vocab=10000)

# Load pretrained embeddings
pretrained = load_pretrained_embeddings(
    classifier.word2idx,
    'glove.6B.100d.txt',
    embed_dim=100
)

# Train with pretrained embeddings
history = classifier.fit(
    headlines,
    labels,
    epochs=10,
    pretrained_embeddings=pretrained
)
```

{: .warning }
> GloVe files can be large (glove.6B.zip is ~822 MB). For production systems, consider saving only the subset of embeddings that match your vocabulary to reduce memory usage and startup time.

## Integrating Sentiment with Price Prediction

The real power of sentiment analysis for trading comes from combining text-derived signals with quantitative price data. A sentiment score computed from daily news can be added as an extra feature to a multivariate LSTM.

### Computing Daily Sentiment Scores

```python
from puffin.deep.sentiment_rnn import SentimentClassifier
import numpy as np

def analyze_daily_news(classifier, headlines):
    """Compute a single sentiment score from a list of headlines.

    Returns the mean probability of the positive class, giving a
    score between 0 (very negative) and 1 (very positive).
    """
    if not headlines:
        return 0.5  # neutral default

    probabilities = classifier.predict_proba(headlines)
    # Use positive-class probability as sentiment score
    sentiment_score = probabilities[:, 2].mean()
    return float(sentiment_score)

# Example: daily sentiment for the past week
daily_news = {
    '2024-01-15': ["Tech stocks rally on earnings beat", "Fed signals rate pause"],
    '2024-01-16': ["Market pulls back on profit taking"],
    '2024-01-17': ["Strong jobs report lifts sentiment", "IPO market heats up"],
    '2024-01-18': ["Geopolitical tensions weigh on markets"],
    '2024-01-19': ["Broad rally as inflation cools further", "Consumer spending rises"],
}

sentiment_scores = {
    date: analyze_daily_news(classifier, news)
    for date, news in daily_news.items()
}

for date, score in sentiment_scores.items():
    print(f"{date}: sentiment = {score:.3f}")
```

### Combining Sentiment with Price Features

```python
import pandas as pd
from puffin.deep.rnn import MultivariateLSTM

def combine_sentiment_and_price(df, sentiment_scores):
    """Merge sentiment time series with price-based features.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data from yfinance.
    sentiment_scores : pd.Series
        Daily sentiment scores aligned to the same index.

    Returns
    -------
    pd.DataFrame
        Combined feature set with target column.
    """
    features = pd.DataFrame(index=df.index)

    # Price features
    features['close'] = df['Close']
    features['volume'] = df['Volume']
    features['returns'] = df['Close'].pct_change()
    features['volatility'] = features['returns'].rolling(20).std()

    # Moving average ratios
    features['sma_20_ratio'] = df['Close'] / df['Close'].rolling(20).mean()

    # Sentiment features
    features['sentiment'] = sentiment_scores
    features['sentiment_ma'] = features['sentiment'].rolling(5).mean()
    features['sentiment_std'] = features['sentiment'].rolling(5).std()

    # Create target
    features['target'] = features['returns'].shift(-1)
    features = features.dropna()

    return features

# Train combined model
# features = combine_sentiment_and_price(df, sentiment_series)
# lstm = MultivariateLSTM()
# history = lstm.fit(features, target_col='target', lookback=20, epochs=50)
```

{: .tip }
> Sentiment signals tend to be most predictive over short horizons (1-5 days). For longer-horizon models, consider smoothing the sentiment series with a rolling average or using cumulative sentiment change rather than the raw score.

## End-to-End Sentiment Trading Pipeline

The following code outlines a complete pipeline that collects news, computes sentiment, combines it with price data, and generates a trading signal.

```python
from puffin.deep.sentiment_rnn import SentimentClassifier
from puffin.deep.rnn import MultivariateLSTM
import pandas as pd
import numpy as np

def sentiment_trading_pipeline(
    price_df: pd.DataFrame,
    daily_headlines: dict,
    classifier: SentimentClassifier,
    lookback: int = 20,
    threshold: float = 0.005
):
    """End-to-end sentiment + price prediction pipeline.

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV DataFrame.
    daily_headlines : dict
        Mapping of date strings to lists of headline strings.
    classifier : SentimentClassifier
        Pretrained sentiment model.
    lookback : int
        Number of past days used as input sequence.
    threshold : float
        Minimum predicted return to trigger a trade.

    Returns
    -------
    dict
        Signal, predicted return, and sentiment score.
    """
    # Step 1: Compute sentiment scores
    sentiment_series = pd.Series(
        {pd.Timestamp(date): analyze_daily_news(classifier, news)
         for date, news in daily_headlines.items()}
    )

    # Step 2: Build combined features
    features = combine_sentiment_and_price(price_df, sentiment_series)

    # Step 3: Train multivariate LSTM
    lstm = MultivariateLSTM()
    lstm.fit(features, target_col='target', lookback=lookback, epochs=30)

    # Step 4: Predict next return
    recent = features.iloc[-lookback:]
    predicted_return = lstm.predict(recent)[0]

    # Step 5: Generate signal
    if predicted_return > threshold:
        signal = 'BUY'
    elif predicted_return < -threshold:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    return {
        'signal': signal,
        'predicted_return': float(predicted_return),
        'latest_sentiment': float(sentiment_series.iloc[-1])
    }
```

{: .warning }
> This pipeline retrains the LSTM each time it runs, which is expensive. In production, retrain on a schedule (e.g., weekly) and only run inference daily. Save model weights with `torch.save()` and reload with `torch.load()` between runs.

## Source Code

- **SentimentLSTM** and **SentimentClassifier**: [`puffin/deep/sentiment_rnn.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/sentiment_rnn.py)
- **MultivariateLSTM** (for combined models): [`puffin/deep/rnn.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/rnn.py)
- **EarlyStopping** and training utilities: [`puffin/deep/training.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/training.py)
