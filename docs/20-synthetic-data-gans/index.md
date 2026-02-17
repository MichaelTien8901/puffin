---
layout: default
title: "Part 20: Synthetic Data with GANs"
nav_order: 21
---

# Synthetic Data with GANs

Generative Adversarial Networks (GANs) generate realistic synthetic data by training two neural networks in competition. In trading, GANs create synthetic market data for backtesting, data augmentation, and stress testing.

## What Are GANs?

A GAN consists of two networks:
- **Generator**: Creates fake data from random noise
- **Discriminator**: Distinguishes real from fake data

They train together in an adversarial game:
1. Generator tries to fool the discriminator
2. Discriminator tries to detect fake data
3. Both improve until fake data becomes indistinguishable from real

### Architecture

```
Random Noise → Generator → Fake Data
                              ↓
Real Data ──────────────→ Discriminator → Real/Fake Classification
```

The generator learns to mimic the distribution of real data.

## Why Synthetic Data for Trading?

### 1. Data Augmentation
- Limited historical data for rare events (crashes, extreme volatility)
- Generate additional training examples
- Balance datasets for machine learning

### 2. Stress Testing
- Test strategies under hypothetical scenarios
- Generate market conditions not seen historically
- Risk assessment for extreme events

### 3. Privacy
- Share synthetic data without exposing proprietary information
- Preserve statistical properties while protecting confidentiality

### 4. Research
- Experiment with different market dynamics
- Test robustness of strategies across diverse conditions

## Standard GAN for Market Data

### Basic Implementation

```python
import numpy as np
from puffin.deep.gan import GAN

# Create market data (e.g., daily returns and features)
np.random.seed(42)
n_samples = 1000
n_features = 20

# Simulate market data (returns, volume, volatility, etc.)
real_data = np.random.randn(n_samples, n_features)

# Create GAN
gan = GAN(
    latent_dim=10,  # Dimension of random noise
    data_dim=20     # Dimension of market data
)

# Train GAN
history = gan.train(
    real_data,
    epochs=100,
    batch_size=64,
    lr=0.0002,
    verbose=True
)

# Generate synthetic market data
synthetic_data = gan.generate(n_samples=1000)
print(f"Generated synthetic data shape: {synthetic_data.shape}")
```

### Monitoring Training

```python
import matplotlib.pyplot as plt

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(history['g_loss'], label='Generator Loss')
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Training Progress')
plt.show()
```

**What to Look For**:
- Generator and discriminator losses should stabilize
- If discriminator loss goes to 0, generator is failing (mode collapse)
- If generator loss explodes, training is unstable

### Use Case: Data Augmentation

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Original dataset (small)
X = np.random.randn(500, 20)
y = np.random.choice([0, 1], size=500)

# Train GAN on features
gan = GAN(latent_dim=10, data_dim=20)
gan.train(X, epochs=100, batch_size=32)

# Generate additional synthetic samples
X_synthetic = gan.generate(n_samples=500)
y_synthetic = np.random.choice([0, 1], size=500)  # Or use a model to predict

# Augmented dataset
X_augmented = np.vstack([X, X_synthetic])
y_augmented = np.concatenate([y, y_synthetic])

print(f"Original size: {len(X)}")
print(f"Augmented size: {len(X_augmented)}")

# Train model on augmented data
clf = RandomForestClassifier()
clf.fit(X_augmented, y_augmented)
```

## TimeGAN for Financial Time Series

Standard GANs don't capture temporal dependencies. TimeGAN is designed for time series with sequential structure.

### Architecture

TimeGAN has four components:
1. **Embedder**: Maps real sequences to latent space
2. **Recovery**: Maps latent sequences back to data space
3. **Generator**: Generates latent sequences from noise
4. **Supervisor**: Models temporal dynamics
5. **Discriminator**: Classifies real vs fake latent sequences

### Implementation

```python
from puffin.deep.gan import TimeGAN

# Create time series data (e.g., 100 sequences of 30 days, 5 features)
np.random.seed(42)
n_sequences = 100
seq_length = 30
n_features = 5

# Simulate market time series (returns, volume, volatility, etc.)
real_sequences = np.random.randn(n_sequences, seq_length, n_features)

# Add some temporal structure (autocorrelation)
for i in range(1, seq_length):
    real_sequences[:, i, :] = 0.7 * real_sequences[:, i-1, :] + 0.3 * real_sequences[:, i, :]

# Create TimeGAN
tgan = TimeGAN(
    seq_length=30,
    n_features=5,
    hidden_dim=24,  # RNN hidden dimension
    latent_dim=24   # Latent space dimension
)

# Train TimeGAN
history = tgan.train(
    real_sequences,
    epochs=100,
    batch_size=32,
    lr=0.001,
    verbose=True
)

# Generate synthetic time series
synthetic_sequences = tgan.generate(n_sequences=100)
print(f"Generated sequences shape: {synthetic_sequences.shape}")
```

### Monitoring TimeGAN Training

```python
# Plot all losses
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history['recon_loss'])
plt.title('Reconstruction Loss')
plt.xlabel('Epoch')

plt.subplot(2, 2, 2)
plt.plot(history['sup_loss'])
plt.title('Supervisor Loss')
plt.xlabel('Epoch')

plt.subplot(2, 2, 3)
plt.plot(history['g_loss'])
plt.title('Generator Loss')
plt.xlabel('Epoch')

plt.subplot(2, 2, 4)
plt.plot(history['d_loss'])
plt.title('Discriminator Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()
```

### Use Case: Scenario Generation

```python
# Generate multiple market scenarios
n_scenarios = 50
scenarios = tgan.generate(n_sequences=n_scenarios)

# Use for stress testing
for i, scenario in enumerate(scenarios):
    # Run backtest on this scenario
    # returns = backtest_strategy(scenario)
    # print(f"Scenario {i}: Return = {returns[-1]:.2%}")
    pass
```

## Evaluating Synthetic Data Quality

Critical question: How realistic is the synthetic data?

### 1. Distribution Comparison

```python
from puffin.deep.gan import SyntheticDataEvaluator

evaluator = SyntheticDataEvaluator()

# Compare distributions
dist_results = evaluator.compare_distributions(real_data, synthetic_data)

print(f"Average KS statistic: {dist_results['avg_ks_statistic']:.4f}")
print(f"Average KS p-value: {dist_results['avg_ks_pvalue']:.4f}")
print(f"Average mean difference: {dist_results['avg_mean_diff']:.4f}")
print(f"Average std difference: {dist_results['avg_std_diff']:.4f}")
```

**Interpretation**:
- KS statistic close to 0: Similar distributions
- KS p-value > 0.05: Cannot reject null hypothesis (distributions are similar)
- Small mean/std differences: Good match

### 2. Autocorrelation Structure

For time series, check temporal dependencies:

```python
# Compare autocorrelation
acf_results = evaluator.compare_autocorrelation(
    real_sequences[:, :, 0].flatten(),  # First feature
    synthetic_sequences[:, :, 0].flatten(),
    lags=20
)

print(f"Average ACF difference: {acf_results['avg_acf_diff']:.4f}")
print(f"Max ACF difference: {acf_results['max_acf_diff']:.4f}")
```

### 3. PCA Comparison

Check if synthetic data captures the same variance structure:

```python
# Compare PCA representations
pca_results = evaluator.compare_pca(real_data, synthetic_data, n_components=5)

print("Explained variance (real):", pca_results['explained_variance_real'])
print(f"Average PC distance: {pca_results['avg_pc_distance']:.4f}")
```

### 4. Full Evaluation

```python
# Comprehensive evaluation
full_results = evaluator.full_evaluation(real_data, synthetic_data)

# Access all metrics
print("\n=== Distribution Metrics ===")
print(f"KS statistic: {full_results['distribution']['avg_ks_statistic']:.4f}")

print("\n=== PCA Metrics ===")
print(f"Avg variance diff: {full_results['pca']['avg_variance_diff']:.4f}")

if 'autocorrelation' in full_results:
    print("\n=== Autocorrelation Metrics ===")
    print(f"ACF difference: {full_results['autocorrelation']['avg_acf_diff']:.4f}")
```

## Advanced Techniques

### 1. Conditional GAN

Generate data conditional on specific features (e.g., market regime):

```python
# Pseudo-code (not implemented in basic GAN)
# condition = np.array([1, 0, 0])  # Bull market
# synthetic_bull = conditional_gan.generate(n_samples=100, condition=condition)
#
# condition = np.array([0, 0, 1])  # Bear market
# synthetic_bear = conditional_gan.generate(n_samples=100, condition=condition)
```

### 2. Wasserstein GAN (WGAN)

Improved stability using Wasserstein distance:
- Better convergence
- No mode collapse
- More stable training

### 3. Progressive GAN

Generate high-resolution data by progressively adding layers.

## Complete Trading Example

End-to-end workflow for synthetic data generation and backtesting:

```python
import numpy as np
import pandas as pd
from puffin.deep.gan import TimeGAN, SyntheticDataEvaluator
from sklearn.preprocessing import StandardScaler

# Step 1: Load historical market data
# Simulate for example
np.random.seed(42)
n_days = 1000
dates = pd.date_range('2020-01-01', periods=n_days)

# Create features: returns, volume, volatility, etc.
real_data = pd.DataFrame({
    'returns': np.random.randn(n_days) * 0.02,
    'volume': np.abs(np.random.randn(n_days)) * 1e6,
    'volatility': np.abs(np.random.randn(n_days)) * 0.1,
    'rsi': np.random.uniform(20, 80, n_days),
    'macd': np.random.randn(n_days) * 0.01
}, index=dates)

# Step 2: Create sequences
seq_length = 20
n_features = len(real_data.columns)

sequences = []
for i in range(len(real_data) - seq_length):
    seq = real_data.iloc[i:i+seq_length].values
    sequences.append(seq)

sequences = np.array(sequences)
print(f"Created {len(sequences)} sequences of length {seq_length}")

# Step 3: Normalize data
scaler = StandardScaler()
sequences_flat = sequences.reshape(-1, n_features)
sequences_normalized = scaler.fit_transform(sequences_flat).reshape(sequences.shape)

# Step 4: Train TimeGAN
tgan = TimeGAN(
    seq_length=seq_length,
    n_features=n_features,
    hidden_dim=32,
    latent_dim=32
)

history = tgan.train(
    sequences_normalized,
    epochs=50,
    batch_size=64,
    lr=0.001,
    verbose=True
)

# Step 5: Generate synthetic scenarios
n_scenarios = 100
synthetic_sequences = tgan.generate(n_sequences=n_scenarios)

# Denormalize
synthetic_sequences_flat = synthetic_sequences.reshape(-1, n_features)
synthetic_sequences = scaler.inverse_transform(synthetic_sequences_flat).reshape(
    synthetic_sequences.shape
)

print(f"Generated {n_scenarios} synthetic scenarios")

# Step 6: Evaluate quality
evaluator = SyntheticDataEvaluator()

# Compare first timestep of each sequence
real_first = sequences[:, 0, :]
synthetic_first = synthetic_sequences[:, 0, :]

results = evaluator.full_evaluation(real_first, synthetic_first)
print(f"\nQuality Metrics:")
print(f"KS statistic: {results['distribution']['avg_ks_statistic']:.4f}")
print(f"PCA distance: {results['pca']['avg_pc_distance']:.4f}")

# Step 7: Use for backtesting
for i, scenario in enumerate(synthetic_sequences[:5]):
    scenario_df = pd.DataFrame(
        scenario,
        columns=real_data.columns
    )
    print(f"\nScenario {i} summary:")
    print(scenario_df.describe())

    # Run your trading strategy on this scenario
    # returns = backtest_strategy(scenario_df)
```

## Best Practices

### 1. Training Stability

GANs are notoriously difficult to train. Tips:
- Use proper learning rates (0.0001 - 0.0003)
- Balance discriminator and generator updates
- Monitor both losses
- Use gradient clipping

```python
# Example with better training stability
gan = GAN(latent_dim=10, data_dim=20)
history = gan.train(
    real_data,
    epochs=200,
    batch_size=64,
    lr=0.0002,  # Conservative learning rate
    verbose=True
)

# Check for issues
if history['d_loss'][-1] < 0.01:
    print("Warning: Discriminator too strong, generator may have collapsed")
if history['g_loss'][-1] > 10:
    print("Warning: Generator struggling, try lowering learning rate")
```

### 2. Data Preprocessing

Always normalize data before training:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
real_data_normalized = scaler.fit_transform(real_data)

# Train on normalized data
gan.train(real_data_normalized, epochs=100)

# Denormalize synthetic data
synthetic_data_normalized = gan.generate(n_samples=100)
synthetic_data = scaler.inverse_transform(synthetic_data_normalized)
```

### 3. Validation

Never use synthetic data without validation:
1. Visual inspection (plot distributions)
2. Statistical tests (KS test, autocorrelation)
3. Out-of-sample testing
4. Domain expert review

### 4. Sample Diversity

Check for mode collapse (generator producing similar samples):
```python
synthetic_data = gan.generate(n_samples=1000)

# Check diversity
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
synthetic_pca = pca.fit_transform(synthetic_data)

plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], alpha=0.5)
plt.title('Synthetic Data Diversity')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Should see spread, not clusters
```

## Common Pitfalls

### 1. Mode Collapse
Generator produces limited variety of samples.

**Solution**: Use diverse training data, adjust learning rates, try WGAN.

### 2. Training Instability
Losses oscillate wildly.

**Solution**: Lower learning rate, add gradient clipping, use spectral normalization.

### 3. Poor Quality Samples
Synthetic data doesn't match real data statistics.

**Solution**: More training epochs, better architecture, more training data.

### 4. Overfitting
GAN memorizes training data instead of learning distribution.

**Solution**: More diverse training data, regularization, early stopping.

## Summary

GANs provide powerful tools for synthetic data generation in trading:
- **Standard GAN**: Cross-sectional market data
- **TimeGAN**: Time series with temporal dependencies
- **Evaluation**: Statistical tests and quality metrics

Key applications:
- Data augmentation for rare events
- Stress testing and scenario analysis
- Privacy-preserving data sharing
- Research and experimentation

Best practices:
- Always normalize input data
- Monitor training stability
- Validate synthetic data quality
- Check for mode collapse and diversity

Next steps:
- Experiment with conditional GANs
- Use Wasserstein distance for stability
- Apply to real market data
- Integrate with backtesting frameworks
