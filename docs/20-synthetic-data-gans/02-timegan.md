---
layout: default
title: "TimeGAN for Financial Data"
parent: "Part 20: Synthetic Data with GANs"
nav_order: 2
---

# TimeGAN for Financial Data

Standard GANs treat each sample independently and cannot capture temporal dependencies -- the autocorrelation, momentum, and mean-reversion patterns that define financial time series. TimeGAN addresses this by combining autoencoding, supervised learning, and adversarial training in a unified framework that preserves both the distributional and temporal properties of the original data.

{: .note }
> TimeGAN was introduced by Yoon, Jarrett, and van der Schaar (2019). It is the most
> widely cited GAN architecture for time-series generation and is particularly suited
> to financial data where temporal structure is critical.

## Architecture Components

TimeGAN has five cooperating networks, each handling a different aspect of temporal data generation:

1. **Embedder**: An RNN (GRU) that maps real sequences from data space to a lower-dimensional latent space
2. **Recovery**: A feedforward network that maps latent representations back to data space
3. **Generator**: An RNN (GRU) that generates latent sequences from random noise
4. **Supervisor**: An RNN (GRU) that models temporal dynamics in latent space by predicting the next latent state
5. **Discriminator**: An RNN (GRU) that classifies latent sequences as real or generated

The key insight is that the adversarial game happens in latent space (not data space), and the supervisor provides an additional temporal consistency signal.

## Basic Usage

```python
import numpy as np
from puffin.deep.gan import TimeGAN

# Create time series: 100 sequences of 30 days, 5 features each
# (returns, volume, volatility, RSI, MACD)
np.random.seed(42)
n_sequences = 100
seq_length = 30
n_features = 5

real_sequences = np.random.randn(n_sequences, seq_length, n_features)

# Add temporal structure (autocorrelation, as in real markets)
for i in range(1, seq_length):
    real_sequences[:, i, :] = (
        0.7 * real_sequences[:, i-1, :] + 0.3 * real_sequences[:, i, :]
    )

# Initialize TimeGAN
tgan = TimeGAN(
    seq_length=30,      # Trading days per sequence
    n_features=5,       # Market features per timestep
    hidden_dim=24,      # GRU hidden state dimension
    latent_dim=24       # Latent representation dimension
)

# Train (three-phase: autoencoding, supervised, adversarial)
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
# Output: (100, 30, 5)
```

{: .tip }
> The `hidden_dim` and `latent_dim` parameters control the capacity of the temporal
> model. For financial data with 5-10 features, values of 24-48 work well. For
> higher-dimensional data (50+ features), increase to 64-128.

## Three-Phase Training

TimeGAN training proceeds in three interleaved phases per batch, each targeting a different aspect of data fidelity:

### Phase 1: Autoencoding (Embedder + Recovery)

The embedder-recovery pair learns to compress and reconstruct real sequences. This creates a meaningful latent space where temporal patterns are preserved.

```
Real Sequence -> Embedder (GRU) -> Latent Representation -> Recovery (MLP) -> Reconstructed Sequence
```

Loss: MSE between original and reconstructed sequences.

### Phase 2: Supervised Learning (Supervisor)

The supervisor learns to predict the next latent state from the current one, capturing temporal dynamics like autocorrelation and trend.

```
Latent State [t] -> Supervisor (GRU) -> Predicted Latent State [t+1]
```

Loss: MSE between predicted and actual next latent states.

### Phase 3: Adversarial Training (Generator + Discriminator)

The generator produces fake latent sequences from noise, while the discriminator tries to distinguish them from real latent sequences produced by the embedder.

```
Random Noise -> Generator (GRU) -> Fake Latent Sequence -> Discriminator -> Real/Fake
Real Sequence -> Embedder (GRU) -> Real Latent Sequence -> Discriminator -> Real/Fake
```

Loss: BCE for both generator and discriminator.

{: .warning }
> All three phases run on every batch, not sequentially. This joint training is
> what makes TimeGAN effective -- the autoencoder and supervisor regularize the
> adversarial game, preventing mode collapse.

## Monitoring Training Progress

Track all four loss components to diagnose training health:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history['recon_loss'])
plt.title('Reconstruction Loss (Embedder + Recovery)')
plt.xlabel('Epoch')
plt.ylabel('MSE')

plt.subplot(2, 2, 2)
plt.plot(history['sup_loss'])
plt.title('Supervisor Loss (Temporal Dynamics)')
plt.xlabel('Epoch')
plt.ylabel('MSE')

plt.subplot(2, 2, 3)
plt.plot(history['g_loss'])
plt.title('Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('BCE')

plt.subplot(2, 2, 4)
plt.plot(history['d_loss'])
plt.title('Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('BCE')

plt.tight_layout()
plt.show()
```

**Diagnostic guide**:

| Loss | Healthy Behavior | Problem Indicator |
|---|---|---|
| `recon_loss` | Steadily decreasing | Plateaus high: increase hidden_dim |
| `sup_loss` | Steadily decreasing | Unstable: reduce learning rate |
| `g_loss` | Moderate, stabilizes | Explodes: mode collapse |
| `d_loss` | Moderate, stabilizes | Near zero: discriminator dominates |

## Scenario Generation for Stress Testing

Generate multiple market scenarios for Monte Carlo-style risk analysis:

```python
from puffin.deep.gan import TimeGAN
import numpy as np

# Assume tgan is already trained on historical data
n_scenarios = 50
scenarios = tgan.generate(n_sequences=n_scenarios)

# Analyze scenario statistics
for i in range(min(5, n_scenarios)):
    scenario = scenarios[i]
    returns = scenario[:, 0]  # First feature = returns
    print(f"Scenario {i}: "
          f"mean={returns.mean():.4f}, "
          f"std={returns.std():.4f}, "
          f"min={returns.min():.4f}, "
          f"max={returns.max():.4f}")
```

{: .tip }
> For stress testing, generate hundreds of scenarios and focus on the tail
> outcomes. The 1st and 99th percentile scenarios reveal how your strategy
> behaves under extreme conditions that may not appear in the historical record.

## Complete Trading Pipeline

End-to-end workflow from raw market data to synthetic scenario generation:

```python
import numpy as np
import pandas as pd
from puffin.deep.gan import TimeGAN
from sklearn.preprocessing import StandardScaler

# Step 1: Load historical market data
np.random.seed(42)
n_days = 1000
dates = pd.date_range('2020-01-01', periods=n_days)

real_data = pd.DataFrame({
    'returns': np.random.randn(n_days) * 0.02,
    'volume': np.abs(np.random.randn(n_days)) * 1e6,
    'volatility': np.abs(np.random.randn(n_days)) * 0.1,
    'rsi': np.random.uniform(20, 80, n_days),
    'macd': np.random.randn(n_days) * 0.01
}, index=dates)

# Step 2: Create sequences (sliding window)
seq_length = 20
n_features = len(real_data.columns)

sequences = []
for i in range(len(real_data) - seq_length):
    seq = real_data.iloc[i:i + seq_length].values
    sequences.append(seq)

sequences = np.array(sequences)
print(f"Created {len(sequences)} sequences of length {seq_length}")

# Step 3: Normalize (critical for GAN training)
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

# Step 5: Generate and denormalize synthetic scenarios
n_scenarios = 100
synthetic_normalized = tgan.generate(n_sequences=n_scenarios)

synthetic_flat = synthetic_normalized.reshape(-1, n_features)
synthetic_sequences = scaler.inverse_transform(synthetic_flat).reshape(
    synthetic_normalized.shape
)

print(f"Generated {n_scenarios} synthetic scenarios")

# Step 6: Convert to DataFrames for backtesting
for i, scenario in enumerate(synthetic_sequences[:5]):
    scenario_df = pd.DataFrame(scenario, columns=real_data.columns)
    print(f"\nScenario {i} summary:")
    print(scenario_df.describe())
```

{: .warning }
> Always normalize data before training TimeGAN. The GRU networks are sensitive
> to input scale, and unnormalized data (e.g., volume in millions vs. returns
> in decimals) will cause the reconstruction loss to be dominated by high-scale features.

## Hyperparameter Tuning

| Parameter | Range | Effect |
|---|---|---|
| `seq_length` | 10 - 60 | Longer captures more temporal structure but needs more data |
| `hidden_dim` | 16 - 128 | Larger captures more complex dynamics but risks overfitting |
| `latent_dim` | 16 - 128 | Controls information bottleneck; usually matches hidden_dim |
| `lr` | 0.0005 - 0.005 | Higher LR for TimeGAN than standard GAN (autoencoder regularizes) |
| `epochs` | 50 - 500 | Monitor recon_loss plateau as stopping criterion |
| `batch_size` | 16 - 128 | Smaller batches for small datasets (< 500 sequences) |

## Source Code

The `TimeGAN` class is implemented in [`puffin/deep/gan.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/gan.py).

Key architecture details:
- `embedder`: 2-layer GRU mapping data to latent space
- `recovery`: 2-layer MLP mapping latent back to data space
- `generator`: 2-layer GRU producing latent sequences from noise
- `supervisor`: 1-layer GRU modeling temporal transitions
- `discriminator`: 2-layer GRU classifying sequences in latent space
- All GRU modules use `batch_first=True` for shape (batch, seq, features)
