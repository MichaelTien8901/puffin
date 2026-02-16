"""
Example: Using RNNs (LSTM and GRU) for Trading Predictions

This script demonstrates:
1. Univariate time series prediction with LSTM
2. Multivariate prediction with multiple features
3. GRU as an alternative to LSTM
4. Sentiment analysis with LSTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from puffin.deep import (
    TradingLSTM,
    TradingGRU,
    MultivariateLSTM,
    SentimentClassifier
)

# Set random seed for reproducibility
np.random.seed(42)


def example_1_univariate_lstm():
    """Example 1: Univariate time series prediction with LSTM."""
    print("\n" + "="*60)
    print("Example 1: Univariate LSTM for Price Prediction")
    print("="*60)

    # Generate synthetic price data (trending sine wave with noise)
    t = np.linspace(0, 10, 500)
    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 2)
    noise = 2 * np.random.randn(500)
    prices = 100 + trend + seasonal + noise

    # Split into train and test
    train_size = 400
    train_prices = prices[:train_size]
    test_prices = prices[train_size:]

    # Train LSTM
    print("\nTraining LSTM...")
    lstm = TradingLSTM()
    history = lstm.fit(
        train_prices,
        lookback=20,
        epochs=30,
        lr=0.001,
        batch_size=32
    )

    print(f"\nFinal Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")

    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    for i in range(len(test_prices)):
        if i == 0:
            pred = lstm.predict(train_prices, steps=1)[0]
        else:
            # Use actual prices up to current point
            pred = lstm.predict(prices[:train_size+i], steps=1)[0]
        predictions.append(pred)

    predictions = np.array(predictions)

    # Calculate metrics
    mae = np.mean(np.abs(predictions - test_prices))
    rmse = np.sqrt(np.mean((predictions - test_prices)**2))

    print(f"\nTest MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # Direction accuracy
    if len(predictions) > 1:
        pred_direction = np.sign(predictions[1:] - test_prices[:-1])
        actual_direction = np.sign(test_prices[1:] - test_prices[:-1])
        direction_accuracy = (pred_direction == actual_direction).mean()
        print(f"Direction Accuracy: {direction_accuracy:.2%}")

    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(train_prices, label='Training Data', alpha=0.7)
    plt.plot(range(train_size, train_size + len(test_prices)), test_prices,
             label='Actual Test', alpha=0.7)
    plt.plot(range(train_size, train_size + len(predictions)), predictions,
             label='LSTM Predictions', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('LSTM Price Prediction')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rnn_univariate_example.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'rnn_univariate_example.png'")


def example_2_multivariate_lstm():
    """Example 2: Multivariate LSTM with multiple features."""
    print("\n" + "="*60)
    print("Example 2: Multivariate LSTM with Multiple Features")
    print("="*60)

    # Generate synthetic multivariate data
    n_samples = 500
    t = np.linspace(0, 10, n_samples)

    # Create correlated features
    price = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 2) + 2 * np.random.randn(n_samples)
    volume = 1000000 + 50000 * np.sin(2 * np.pi * t / 3) + 100000 * np.random.randn(n_samples)
    rsi = 50 + 20 * np.sin(2 * np.pi * t / 1.5) + 5 * np.random.randn(n_samples)
    volatility = 2 + 0.5 * np.abs(np.sin(2 * np.pi * t)) + 0.2 * np.random.randn(n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'price': price,
        'volume': volume,
        'rsi': rsi,
        'volatility': volatility
    })

    # Create target (next day's return)
    df['returns'] = df['price'].pct_change()
    df['target'] = df['returns'].shift(-1)
    df = df.dropna()

    # Split into train and test
    train_size = 400
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Train multivariate LSTM
    print("\nTraining Multivariate LSTM...")
    lstm = MultivariateLSTM()
    history = lstm.fit(
        train_df,
        target_col='target',
        lookback=20,
        epochs=30,
        lr=0.001,
        hidden_dims=[128, 64, 32]
    )

    print(f"\nFinal Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")

    # Make predictions
    print("\nMaking predictions on test set...")
    test_predictions = []
    for i in range(20, len(test_df)):
        pred = lstm.predict(df.iloc[:train_size+i])
        test_predictions.append(pred[0])

    test_predictions = np.array(test_predictions)
    test_actuals = test_df['target'].values[20:]

    # Calculate metrics
    mae = np.mean(np.abs(test_predictions - test_actuals))
    rmse = np.sqrt(np.mean((test_predictions - test_actuals)**2))

    print(f"\nTest MAE: {mae:.6f}")
    print(f"Test RMSE: {rmse:.6f}")

    # Direction accuracy
    if len(test_predictions) > 0:
        pred_direction = np.sign(test_predictions)
        actual_direction = np.sign(test_actuals)
        direction_accuracy = (pred_direction == actual_direction).mean()
        print(f"Direction Accuracy: {direction_accuracy:.2%}")

    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(test_actuals, label='Actual Returns', alpha=0.7)
    plt.plot(test_predictions, label='Predicted Returns', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Return')
    plt.title('Multivariate LSTM: Predicted vs Actual Returns')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rnn_multivariate_example.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'rnn_multivariate_example.png'")


def example_3_gru_comparison():
    """Example 3: Compare GRU vs LSTM."""
    print("\n" + "="*60)
    print("Example 3: GRU vs LSTM Comparison")
    print("="*60)

    # Generate synthetic data
    t = np.linspace(0, 10, 400)
    prices = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 2) + 2 * np.random.randn(400)

    train_prices = prices[:300]
    test_prices = prices[300:]

    # Train LSTM
    print("\nTraining LSTM...")
    lstm = TradingLSTM()
    lstm_history = lstm.fit(train_prices, lookback=20, epochs=20, lr=0.001)

    # Train GRU
    print("\nTraining GRU...")
    gru = TradingGRU()
    gru_history = gru.fit(train_prices, lookback=20, epochs=20, lr=0.001)

    # Make predictions
    lstm_predictions = []
    gru_predictions = []

    for i in range(len(test_prices)):
        if i == 0:
            lstm_pred = lstm.predict(train_prices, steps=1)[0]
            gru_pred = gru.predict(train_prices, steps=1)[0]
        else:
            lstm_pred = lstm.predict(prices[:300+i], steps=1)[0]
            gru_pred = gru.predict(prices[:300+i], steps=1)[0]

        lstm_predictions.append(lstm_pred)
        gru_predictions.append(gru_pred)

    lstm_predictions = np.array(lstm_predictions)
    gru_predictions = np.array(gru_predictions)

    # Calculate metrics
    lstm_mae = np.mean(np.abs(lstm_predictions - test_prices))
    gru_mae = np.mean(np.abs(gru_predictions - test_prices))

    print(f"\nLSTM Test MAE: {lstm_mae:.4f}")
    print(f"GRU Test MAE: {gru_mae:.4f}")

    # Plot comparison
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(range(300, 300 + len(test_prices)), test_prices, label='Actual', alpha=0.7)
    plt.plot(range(300, 300 + len(lstm_predictions)), lstm_predictions,
             label='LSTM', alpha=0.7)
    plt.plot(range(300, 300 + len(gru_predictions)), gru_predictions,
             label='GRU', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('LSTM vs GRU: Predictions Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(lstm_history['train_loss'], label='LSTM Train')
    plt.plot(lstm_history['val_loss'], label='LSTM Val')
    plt.plot(gru_history['train_loss'], label='GRU Train', linestyle='--')
    plt.plot(gru_history['val_loss'], label='GRU Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    errors_lstm = np.abs(lstm_predictions - test_prices)
    errors_gru = np.abs(gru_predictions - test_prices)
    plt.plot(errors_lstm, label='LSTM Error', alpha=0.7)
    plt.plot(errors_gru, label='GRU Error', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title('Prediction Errors')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rnn_lstm_gru_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'rnn_lstm_gru_comparison.png'")


def example_4_sentiment_analysis():
    """Example 4: Sentiment analysis for trading news."""
    print("\n" + "="*60)
    print("Example 4: Sentiment Analysis with LSTM")
    print("="*60)

    # Training data: financial news headlines with sentiment labels
    # 0 = negative, 1 = neutral, 2 = positive
    train_headlines = [
        # Positive
        "Company reports record profits, stock soars to new highs",
        "Excellent earnings beat expectations, analyst upgrades rating",
        "Strong growth drives stock price higher in morning trading",
        "Investor optimism surges on positive economic data",
        "Shares jump on successful product launch announcement",
        "Bullish sentiment continues as company expands operations",
        "Stock rallies on better than expected quarterly results",
        "Positive outlook fuels buying interest in shares",
        "Company beats revenue targets, stock climbs steadily",
        "Investor confidence grows on strong fundamental performance",
        # Negative
        "Stock plummets on disappointing earnings report",
        "Company misses targets, shares fall sharply",
        "Regulatory concerns weigh heavily on stock price",
        "Bearish sentiment drives selling pressure",
        "Disappointing guidance sends stock tumbling",
        "Shares drop on weak quarterly performance",
        "Stock falls as company warns of lower profits",
        "Investor fears grow on weak economic outlook",
        "Company struggles lead to share price decline",
        "Stock sinks on poor earnings and weak forecast",
        # Neutral
        "Company reports earnings in line with expectations",
        "Stock remains stable as market awaits news",
        "Trading flat ahead of quarterly earnings release",
        "Shares unchanged as investors digest recent news",
        "Stock holds steady amid mixed market conditions",
        "Company provides guidance matching analyst estimates",
        "Market reaction muted to quarterly results",
        "Stock trading sideways as investors remain cautious",
        "Shares little changed following earnings announcement",
        "Neutral sentiment prevails in trading session",
    ] * 5  # Repeat to have more training data

    train_labels = [2] * 50 + [0] * 50 + [1] * 50  # positive, negative, neutral

    # Shuffle training data
    indices = np.random.permutation(len(train_headlines))
    train_headlines = [train_headlines[i] for i in indices]
    train_labels = [train_labels[i] for i in indices]

    # Train sentiment classifier
    print("\nTraining Sentiment LSTM...")
    classifier = SentimentClassifier()
    history = classifier.fit(
        train_headlines,
        train_labels,
        epochs=15,
        batch_size=16,
        max_len=20,
        embed_dim=100,
        hidden_dim=128
    )

    print(f"\nFinal Training Accuracy: {history['train_acc'][-1]:.2%}")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2%}")

    # Test on new headlines
    test_headlines = [
        "Stock rises on strong earnings report",
        "Shares decline amid regulatory concerns",
        "Company reports results meeting expectations",
        "Dramatic losses lead to sharp selloff",
        "Outstanding performance drives shares higher",
        "Stock unchanged following quarterly update"
    ]

    print("\nPredicting sentiment for new headlines:")
    print("-" * 60)

    predictions = classifier.predict(test_headlines)
    probabilities = classifier.predict_proba(test_headlines)

    sentiment_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    for i, headline in enumerate(test_headlines):
        pred_label = sentiment_map[predictions[i]]
        probs = probabilities[i]

        print(f"\nHeadline: {headline}")
        print(f"Predicted: {pred_label}")
        print(f"Probabilities - Neg: {probs[0]:.2%}, Neu: {probs[1]:.2%}, Pos: {probs[2]:.2%}")

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Sentiment LSTM: Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Sentiment LSTM: Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rnn_sentiment_example.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'rnn_sentiment_example.png'")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("RNN Trading Examples")
    print("="*60)

    # Run examples
    example_1_univariate_lstm()
    example_2_multivariate_lstm()
    example_3_gru_comparison()
    example_4_sentiment_analysis()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
