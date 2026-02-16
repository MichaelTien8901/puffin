"""
Machine learning trading pipeline.

This example demonstrates:
1. Feature engineering
2. Model training with cross-validation
3. ML-based signal generation
4. Backtesting
5. Performance evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from puffin.data import YFinanceProvider
from puffin.ml import compute_features
from puffin.backtest import Backtester
from puffin.risk import PortfolioRiskManager


def main():
    """Run ML trading pipeline."""
    print("=" * 60)
    print("Machine Learning Trading Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    provider = YFinanceProvider()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)

    ticker = 'AAPL'
    data = provider.fetch_historical(
        symbol=ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
    )

    # 2. Feature engineering
    print("\n2. Engineering features...")
    features = compute_features(data)

    feature_cols = [c for c in features.columns if c in [
        'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr',
        'return_1', 'return_5', 'return_21', 'volume_ratio'
    ]]

    # Merge features with data
    df = data.join(features[feature_cols]).dropna()

    # Create labels: 1 if next day return > 0
    df['future_return'] = df['close'].pct_change().shift(-1)
    df['label'] = (df['future_return'] > 0).astype(int)
    df = df.dropna()

    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {feature_cols}")
    print(f"   Positive labels: {df['label'].sum()} ({df['label'].mean():.1%})")

    # 3. Train model
    print("\n3. Training Random Forest...")

    split_idx = int(len(df) * 0.7)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    y_train = train['label']
    X_test = scaler.transform(test[feature_cols])
    y_test = test['label']

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print(f"   Train accuracy: {model.score(X_train, y_train):.3f}")
    print(f"   Test accuracy: {model.score(X_test, y_test):.3f}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n   Top features:")
    for feat, imp in importance.head().items():
        print(f"   - {feat}: {imp:.3f}")

    # 4. Generate signals
    print("\n4. Generating signals on test set...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    test = test.copy()
    test['signal'] = predictions
    test['prob'] = probabilities

    long_days = (test['signal'] == 1).sum()
    print(f"   Long signals: {long_days} / {len(test)} days")

    # 5. Simple P&L calculation
    print("\n5. Performance...")
    test['strategy_return'] = test['future_return'] * test['signal']
    cumulative = (1 + test['strategy_return']).cumprod()

    total_return = cumulative.iloc[-1] - 1
    sharpe = test['strategy_return'].mean() / test['strategy_return'].std() * np.sqrt(252)

    buy_hold = (1 + test['future_return']).cumprod().iloc[-1] - 1

    print(f"   Strategy return: {total_return:.2%}")
    print(f"   Buy & hold return: {buy_hold:.2%}")
    print(f"   Sharpe ratio: {sharpe:.2f}")

    # 6. Risk metrics
    print("\n6. Risk metrics...")
    portfolio_rm = PortfolioRiskManager()
    returns = test['strategy_return'].dropna()
    var = portfolio_rm.compute_var(returns, confidence=0.95)
    es = portfolio_rm.compute_expected_shortfall(returns, confidence=0.95)
    print(f"   95% VaR: {var:.4f}")
    print(f"   Expected Shortfall: {es:.4f}")

    print("\n" + "=" * 60)
    print("ML pipeline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
