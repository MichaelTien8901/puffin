"""
Demo script for CNN-based trading models.

This script demonstrates the three CNN approaches:
1. 1D CNN for time series prediction
2. CNN-TA for image-based technical analysis
3. Transfer learning for financial forecasting

Run with: python examples/cnn_trading_demo.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Import Puffin CNN models
from puffin.deep.cnn import TradingCNN
from puffin.deep.cnn_ta import TradingCNNTA, series_to_image


def demo_1d_cnn():
    """Demonstrate 1D CNN for price prediction."""
    print("\n" + "="*60)
    print("Demo 1: 1D CNN for Time Series Prediction")
    print("="*60)

    # Generate synthetic price data
    np.random.seed(42)
    n_samples = 500
    prices = np.cumsum(np.random.randn(n_samples)) + 100

    print(f"\nGenerated {n_samples} synthetic price points")

    # Prepare sequences
    lookback = 20
    X_seq, y = TradingCNN.prepare_sequences(
        prices.reshape(-1, 1),
        lookback=lookback
    )

    print(f"Prepared {len(X_seq)} sequences with lookback={lookback}")
    print(f"Input shape: {X_seq.shape}, Target shape: {y.shape}")

    # Initialize model
    model = TradingCNN(
        input_channels=1,
        seq_length=lookback,
        n_filters=[32, 64],
        kernel_sizes=[3, 3],
        output_dim=1,
        device='cpu'
    )

    print("\nTraining 1D CNN model...")
    history = model.fit(
        X_seq,
        y,
        epochs=20,
        lr=0.001,
        batch_size=32,
        validation_split=0.2
    )

    # Make predictions
    predictions = model.predict(X_seq[-10:])
    actuals = y[-10:]

    print("\nPredictions vs Actuals (last 10):")
    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        print(f"  {i+1}. Predicted: {pred:.2f}, Actual: {actual:.2f}, "
              f"Error: {abs(pred - actual):.2f}")

    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    print(f"\nTest MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")


def demo_cnn_ta():
    """Demonstrate CNN-TA with real market data."""
    print("\n" + "="*60)
    print("Demo 2: CNN-TA (2D Image Approach)")
    print("="*60)

    # Download real market data
    print("\nDownloading SPY data...")
    ticker = yf.Ticker("SPY")
    df = ticker.history(period="6mo")

    print(f"Downloaded {len(df)} days of data")

    # Convert single window to image (for visualization)
    window = 30
    sample_window = df.iloc[-window:]
    image = series_to_image(
        sample_window,
        window=window,
        indicators=['sma', 'rsi', 'macd']
    )

    print(f"\nImage representation: {image.shape[0]} features x {window} time steps")
    print(f"Features include: OHLC, Volume, SMA, RSI, MACD")

    # Prepare training samples
    print("\nPreparing training samples...")
    ohlcv_samples = []
    labels = []

    for i in range(len(df) - window - 10):
        window_df = df.iloc[i:i+window]
        ohlcv_samples.append(window_df)

        # Label: 0=down, 1=flat, 2=up based on next 5 days
        future_return = (
            df['Close'].iloc[i+window+5] - df['Close'].iloc[i+window]
        ) / df['Close'].iloc[i+window]

        if future_return < -0.02:
            labels.append(0)  # Down
        elif future_return > 0.02:
            labels.append(2)  # Up
        else:
            labels.append(1)  # Flat

    labels = np.array(labels)
    print(f"Created {len(ohlcv_samples)} samples")
    print(f"Label distribution - Down: {sum(labels==0)}, "
          f"Flat: {sum(labels==1)}, Up: {sum(labels==2)}")

    # Initialize and train model
    model = TradingCNNTA(
        window=window,
        indicators=['sma', 'rsi', 'macd'],
        n_classes=3,
        device='cpu'
    )

    print("\nTraining CNN-TA model...")
    history = model.fit(
        ohlcv_samples,
        labels,
        epochs=15,
        lr=0.001,
        batch_size=16,
        validation_split=0.2
    )

    # Make predictions
    test_samples = ohlcv_samples[-5:]
    test_labels = labels[-5:]

    predictions = model.predict(test_samples)
    probabilities = model.predict_proba(test_samples)

    print("\nTest Predictions:")
    signal_names = ['DOWN', 'FLAT', 'UP']
    for i, (pred, prob, actual) in enumerate(zip(predictions, probabilities, test_labels)):
        print(f"  Sample {i+1}:")
        print(f"    Predicted: {signal_names[pred]} ({prob[pred]:.2%} confidence)")
        print(f"    Actual: {signal_names[actual]}")
        print(f"    Probabilities - Down: {prob[0]:.2%}, "
              f"Flat: {prob[1]:.2%}, Up: {prob[2]:.2%}")

    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels)
    print(f"\nTest Accuracy: {accuracy:.2%}")


def demo_comparison():
    """Compare CNN approaches on same dataset."""
    print("\n" + "="*60)
    print("Demo 3: Approach Comparison")
    print("="*60)

    print("\nSummary of CNN Approaches:")
    print("\n1. 1D CNN:")
    print("   - Fast training and inference")
    print("   - Good for single-feature time series")
    print("   - Simple architecture, easy to interpret")
    print("   - Use for: Price forecasting, volatility prediction")

    print("\n2. CNN-TA (2D Image):")
    print("   - Rich multi-indicator representation")
    print("   - Captures relationships between indicators")
    print("   - More parameters, needs more data")
    print("   - Use for: Pattern recognition, multi-signal strategies")

    print("\n3. Transfer Learning:")
    print("   - Leverages pretrained models")
    print("   - Works well with limited data")
    print("   - Requires GPU for practical use")
    print("   - Use for: Small datasets, ensemble methods")

    print("\nBest Practices:")
    print("  - Always use validation data")
    print("  - Monitor for overfitting")
    print("  - Normalize inputs")
    print("  - Use appropriate sequence lengths (10-50)")
    print("  - Combine with risk management")


def main():
    """Run all demos."""
    print("CNN Trading Models Demo")
    print("This demo shows three approaches to using CNNs in trading")

    try:
        # Demo 1: 1D CNN
        demo_1d_cnn()

        # Demo 2: CNN-TA
        demo_cnn_ta()

        # Demo 3: Comparison
        demo_comparison()

        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
