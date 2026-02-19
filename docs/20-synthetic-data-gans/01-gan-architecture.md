---
layout: default
title: "GAN Architecture Fundamentals"
parent: "Part 20: Synthetic Data with GANs"
nav_order: 1
---

# GAN Architecture Fundamentals

Generative Adversarial Networks consist of two neural networks -- a generator and a discriminator -- trained simultaneously in an adversarial game. The generator learns to produce realistic synthetic data while the discriminator learns to distinguish real from fake samples. When training succeeds, the generator produces data indistinguishable from the real distribution.

{: .note }
> GANs were introduced by Ian Goodfellow in 2014 and have since become one of the most
> powerful generative modeling frameworks. In finance, they address the fundamental problem
> of limited historical data for backtesting and model training.

## Generator Network

The generator maps random noise vectors z to synthetic data samples. In `puffin.deep.gan`, it uses a feedforward network with batch normalization and ReLU activations, outputting through a Tanh layer to bound values in [-1, 1].

```python
from puffin.deep.gan import Generator

# Create a generator that maps 10-dim noise to 20-dim market features
gen = Generator(
    latent_dim=10,       # Random noise dimension
    output_dim=20,       # Output data dimension (number of market features)
    hidden_dims=[128, 256]  # Two hidden layers
)

# Forward pass: noise -> synthetic data
import torch
z = torch.randn(64, 10)   # Batch of 64 noise vectors
fake_data = gen(z)         # Shape: (64, 20)
print(f"Generated shape: {fake_data.shape}")
```

The architecture stacks Linear -> BatchNorm1d -> ReLU blocks, with a final Linear -> Tanh output. BatchNorm stabilizes training by normalizing intermediate activations.

{: .tip }
> The latent dimension controls the diversity of generated samples. Too small and the generator
> cannot capture the full data distribution. Too large and training becomes harder. Start with
> latent_dim roughly half of data_dim.

## Discriminator Network

The discriminator classifies inputs as real or fake, outputting a probability via sigmoid activation. It uses LeakyReLU (slope 0.2) instead of ReLU to avoid dead neurons, and dropout (0.3) for regularization.

```python
from puffin.deep.gan import Discriminator

# Create a discriminator for 20-dim market features
disc = Discriminator(
    input_dim=20,            # Must match data dimension
    hidden_dims=[256, 128]   # Mirror of generator (but reversed)
)

# Forward pass: data -> probability of being real
import torch
x = torch.randn(64, 20)    # Batch of 64 data points
prob = disc(x)              # Shape: (64, 1), values in [0, 1]
print(f"Real probability: {prob.mean():.4f}")
```

{: .warning }
> If the discriminator becomes too powerful too quickly, it provides no useful gradient
> signal to the generator (vanishing gradients). This is why the architectures are kept
> roughly symmetric in capacity.

## The GAN Class: End-to-End Training

The `GAN` class orchestrates training of both networks. It alternates between discriminator updates (learning to detect fakes) and generator updates (learning to fool the discriminator).

```python
import numpy as np
from puffin.deep.gan import GAN

# Create market data: 1000 samples with 20 features
# (returns, volume, volatility, RSI, MACD, etc.)
np.random.seed(42)
real_data = np.random.randn(1000, 20)

# Initialize GAN
gan = GAN(
    latent_dim=10,   # Noise vector dimension
    data_dim=20      # Market feature dimension
)

# Train the adversarial game
history = gan.train(
    real_data,
    epochs=100,
    batch_size=64,
    lr=0.0002,       # Adam optimizer with conservative LR
    verbose=True
)
```

### Training Loop Internals

Each epoch proceeds as follows:

1. **Discriminator step**: Sample a real batch and a fake batch (from generator). Compute BCE loss for both and backpropagate through the discriminator only.
2. **Generator step**: Generate a new fake batch. Pass through discriminator. Compute BCE loss with "real" labels (generator wants the discriminator to believe fakes are real). Backpropagate through the generator only.

The Adam optimizer uses betas=(0.5, 0.999), which is the standard configuration for GAN training -- the lower beta1 reduces momentum, helping with the non-stationary training dynamics.

## Monitoring Training Stability

```python
import matplotlib.pyplot as plt

# Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(history['g_loss'], label='Generator Loss')
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Training Progress')
plt.show()
```

**What to look for**:
- Both losses should stabilize (not necessarily converge to the same value)
- If discriminator loss goes to 0: generator has collapsed (mode collapse)
- If generator loss explodes: training is unstable, reduce learning rate

{: .warning }
> GANs are notoriously difficult to train. A common failure mode is the discriminator
> "winning" too decisively, causing the generator to receive near-zero gradients and
> stop improving entirely.

## Data Augmentation for Trading

A key application is augmenting small datasets with realistic synthetic samples. This improves downstream ML model performance, especially when historical data for specific market conditions is scarce.

```python
import numpy as np
from puffin.deep.gan import GAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Original dataset (small -- only 500 labeled samples)
X = np.random.randn(500, 20)
y = np.random.choice([0, 1], size=500)

# Train GAN on features only
gan = GAN(latent_dim=10, data_dim=20)
gan.train(X, epochs=100, batch_size=32)

# Generate 500 additional synthetic samples
X_synthetic = gan.generate(n_samples=500)
y_synthetic = np.random.choice([0, 1], size=500)

# Augmented dataset
X_augmented = np.vstack([X, X_synthetic])
y_augmented = np.concatenate([y, y_synthetic])

print(f"Original size: {len(X)}")
print(f"Augmented size: {len(X_augmented)}")

# Train model on augmented data
clf = RandomForestClassifier()
clf.fit(X_augmented, y_augmented)
```

{: .tip }
> For classification tasks, train separate GANs for each class label to preserve
> the conditional distribution P(X|y). This produces higher quality augmented data
> than training a single GAN on all classes.

## Training Stability Best Practices

```python
from puffin.deep.gan import GAN

# Example with stability monitoring
gan = GAN(latent_dim=10, data_dim=20)
history = gan.train(
    real_data,
    epochs=200,
    batch_size=64,
    lr=0.0002,       # Conservative learning rate
    verbose=True
)

# Check for common issues
if history['d_loss'][-1] < 0.01:
    print("Warning: Discriminator too strong, generator may have collapsed")
if history['g_loss'][-1] > 10:
    print("Warning: Generator struggling, try lowering learning rate")
```

### Key Hyperparameters

| Parameter | Recommended Range | Notes |
|---|---|---|
| `lr` | 0.0001 - 0.0003 | Lower is more stable but slower |
| `batch_size` | 32 - 128 | Larger batches stabilize discriminator |
| `latent_dim` | 0.5x - 1.0x data_dim | Controls sample diversity |
| `epochs` | 100 - 500 | Monitor losses for convergence |

### Data Preprocessing

Always normalize data before training. The Tanh output layer produces values in [-1, 1], so input data should be standardized.

```python
from sklearn.preprocessing import StandardScaler
from puffin.deep.gan import GAN

scaler = StandardScaler()
real_data_normalized = scaler.fit_transform(real_data)

# Train on normalized data
gan = GAN(latent_dim=10, data_dim=20)
gan.train(real_data_normalized, epochs=100)

# Denormalize synthetic data back to original scale
synthetic_normalized = gan.generate(n_samples=100)
synthetic_data = scaler.inverse_transform(synthetic_normalized)
```

## Advanced GAN Variants

### Wasserstein GAN (WGAN)

Replaces BCE loss with the Wasserstein distance, providing smoother gradients and better convergence:
- No mode collapse
- More stable training dynamics
- The discriminator (called "critic") is not bounded to [0, 1]
- Requires weight clipping or gradient penalty

### Conditional GAN (cGAN)

Generates data conditioned on auxiliary information such as market regime:

```python
# Conceptual example (conditional generation)
# condition = np.array([1, 0, 0])  # Bull market one-hot encoding
# synthetic_bull = conditional_gan.generate(n_samples=100, condition=condition)
#
# condition = np.array([0, 0, 1])  # Bear market
# synthetic_bear = conditional_gan.generate(n_samples=100, condition=condition)
```

### Progressive GAN

Generates high-resolution data by progressively adding layers during training, starting from low-dimensional representations and gradually increasing complexity.

## Source Code

The `Generator`, `Discriminator`, and `GAN` classes are implemented in [`puffin/deep/gan.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/gan.py).

Key classes:
- `Generator(latent_dim, output_dim, hidden_dims)` -- Feedforward generator with BatchNorm
- `Discriminator(input_dim, hidden_dims)` -- Feedforward discriminator with LeakyReLU and Dropout
- `GAN(latent_dim, data_dim, device)` -- End-to-end training and generation
