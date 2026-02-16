"""
Gradient boosting long-short strategy.

This example demonstrates:
1. Multi-asset feature engineering
2. Gradient boosting model training
3. Long-short portfolio construction
4. Risk assessment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from puffin.data import YFinanceProvider
from puffin.ml import compute_features
from puffin.risk import PortfolioRiskManager


def main():
    """Run long-short strategy."""
    print("=" * 60)
    print("Gradient Boosting Long-Short Strategy")
    print("=" * 60)

    # 1. Load multi-asset data
    print("\n1. Loading multi-asset data...")
    provider = YFinanceProvider()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
    data_dict = {}

    for ticker in tickers:
        try:
            df = provider.fetch_historical(
                symbol=ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
            )
            data_dict[ticker] = df
            print(f"   Loaded {ticker}: {len(df)} bars")
        except Exception as e:
            print(f"   Error loading {ticker}: {e}")

    # 2. Feature engineering
    print("\n2. Engineering features...")
    feature_cols = []
    all_data = []

    for ticker, df in data_dict.items():
        features = compute_features(df)
        cols = [c for c in features.columns if not features[c].isna().all()]
        if not feature_cols:
            feature_cols = cols[:8]  # Use first 8 available features

        combined = df.join(features[feature_cols]).dropna()
        combined['future_return'] = combined['close'].pct_change().shift(-1)
        combined['label'] = (combined['future_return'] > 0).astype(int)
        combined['ticker'] = ticker
        combined = combined.dropna()
        all_data.append(combined)

    full_data = pd.concat(all_data, ignore_index=True)
    print(f"   Combined dataset: {full_data.shape}")
    print(f"   Features: {feature_cols}")

    # 3. Train model
    print("\n3. Training gradient boosting model...")

    split_idx = int(len(full_data) * 0.7)
    train = full_data.iloc[:split_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    y_train = train['label']

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"   Training score: {model.score(X_train, y_train):.3f}")

    # 4. Rank assets for long-short portfolio
    print("\n4. Generating long-short portfolio...")

    rankings = []
    for ticker, df in data_dict.items():
        features = compute_features(df)
        latest = features[feature_cols].iloc[-1:]
        if latest.isna().any(axis=1).iloc[0]:
            continue
        X = scaler.transform(latest)
        prob = model.predict_proba(X)[0, 1]
        rankings.append({'ticker': ticker, 'score': prob, 'price': df['close'].iloc[-1]})

    rankings_df = pd.DataFrame(rankings).sort_values('score', ascending=False)

    print("\n   Asset Rankings:")
    for _, row in rankings_df.iterrows():
        print(f"   {row['ticker']:6s}  score={row['score']:.3f}  price=${row['price']:.2f}")

    longs = rankings_df.head(3)
    shorts = rankings_df.tail(3)

    print(f"\n   LONG:  {', '.join(longs['ticker'].tolist())}")
    print(f"   SHORT: {', '.join(shorts['ticker'].tolist())}")

    # 5. Risk assessment
    print("\n5. Risk assessment...")
    portfolio_rm = PortfolioRiskManager()
    equity_curve = pd.Series(np.cumsum(np.random.randn(100) * 500) + 100000)
    ok, dd = portfolio_rm.check_drawdown(equity_curve, max_dd=0.10)
    print(f"   Drawdown: {dd:.2%}, Within limits: {ok}")

    print("\n" + "=" * 60)
    print("Long-short strategy complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
