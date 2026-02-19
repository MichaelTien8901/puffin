---
layout: default
title: "Standard & Denoising Autoencoders"
parent: "Part 19: Autoencoders"
nav_order: 1
---

# Standard & Denoising Autoencoders

Standard autoencoders learn compressed representations by minimizing reconstruction error. Denoising autoencoders add a twist: they corrupt the input with noise during training, forcing the network to learn robust features that look past microstructure noise, bid-ask bounce, and data quality issues.

{: .note }
> Both the `Autoencoder` and `DenoisingAutoencoder` classes live in `puffin.deep.autoencoder`.
> They share the same `AETrainer` for fitting and feature extraction.

## Standard Autoencoder

### Architecture

A feedforward autoencoder compresses `n_features` inputs through progressively smaller hidden layers to an `encoding_dim` bottleneck, then mirrors the architecture back out to reconstruct the original input.

```python
import numpy as np
from puffin.deep.autoencoder import Autoencoder, AETrainer

# Create market data (e.g., 100 features from technical indicators)
np.random.seed(42)
n_samples = 1000
n_features = 100

# Simulate market features
market_data = np.random.randn(n_samples, n_features)

# Create autoencoder with progressive compression
model = Autoencoder(
    input_dim=100,
    encoding_dim=20,  # Compress to 20 features
    hidden_dims=[80, 50, 30]  # Encoder layers
)

# Train the model
trainer = AETrainer()
history = trainer.fit(
    model,
    market_data,
    epochs=100,
    lr=0.001,
    batch_size=64
)

# Extract compressed features
compressed_features = trainer.extract_features(model, market_data)
print(f"Original shape: {market_data.shape}")       # (1000, 100)
print(f"Compressed shape: {compressed_features.shape}")  # (1000, 20)
```

The `hidden_dims` parameter controls the encoder's layer progression. The decoder automatically mirrors this in reverse order: 30 -> 50 -> 80 -> 100.

### Use Cases for Standard Autoencoders

- **Dimensionality reduction**: Compress hundreds of technical indicators to a compact set of latent factors before feeding them to a downstream classifier or regression model
- **Pre-processing**: Reduce collinearity in feature sets where many indicators are correlated
- **Feature extraction from tick data**: Compress high-frequency order book snapshots into a manageable representation

{: .tip }
> A standard autoencoder with linear activations and MSE loss is mathematically equivalent to PCA.
> Adding nonlinear activations (ReLU) enables the network to capture nonlinear relationships
> that PCA misses entirely.

### Choosing the Encoding Dimension

The encoding dimension determines the level of information compression. Too small and you lose signal; too large and the model memorizes noise.

```python
# Rule of thumb: start with 10-30% of input dimension
input_dim = 100
encoding_dim = int(0.2 * input_dim)  # 20 features
```

**Progressive compression** works best -- avoid aggressive jumps:

```python
# Good: Gradual compression
Autoencoder(
    input_dim=100,
    encoding_dim=10,
    hidden_dims=[80, 60, 40, 20]
)

# Avoid: Too aggressive compression
Autoencoder(
    input_dim=100,
    encoding_dim=10,
    hidden_dims=[10]  # Direct jump loses too much
)
```

{: .warning }
> If the encoding dimension is too close to the input dimension, the autoencoder will learn
> an identity mapping and provide no useful compression. Monitor reconstruction loss on a
> held-out validation set to find the sweet spot.

## Denoising Autoencoder

### How Noise Injection Works

A denoising autoencoder deliberately corrupts each training sample with Gaussian noise, then trains to reconstruct the clean original. This forces the network to learn features that are robust to perturbations -- exactly what you need when working with noisy market data.

```python
from puffin.deep.autoencoder import DenoisingAutoencoder, AETrainer

# Create denoising autoencoder
model = DenoisingAutoencoder(
    input_dim=100,
    encoding_dim=20,
    noise_factor=0.3  # Standard deviation of Gaussian noise
)

# Train with automatic noise injection
trainer = AETrainer()
history = trainer.fit(model, market_data, epochs=100)

# Extract robust features (no noise added during inference)
robust_features = trainer.extract_features(model, market_data)
```

During training, the `DenoisingAutoencoder.forward()` method adds noise when `self.training` is `True`. At inference time (after calling `model.eval()`), noise injection is disabled automatically.

### Noise Factor Selection

The `noise_factor` parameter controls how much corruption is applied:

| noise_factor | Effect | When to Use |
|-------------|--------|------------|
| 0.1 | Light corruption | Clean institutional data |
| 0.3 | Moderate corruption | General market data with bid-ask noise |
| 0.5 | Heavy corruption | Extremely noisy alternative data |

{: .tip }
> Start with `noise_factor=0.3` and tune based on validation reconstruction error.
> If the model cannot reconstruct well even on training data, reduce the noise.

### Use Cases for Denoising Autoencoders

- **Microstructure noise**: Handle bid-ask spreads and tick-level noise in high-frequency data
- **Data quality robustness**: Learn features resistant to missing values and reporting errors
- **Generalization**: Denoising training acts as a regularizer, improving performance on new market conditions

## Feature Extraction for Strategy Development

Both standard and denoising autoencoders excel at producing compact features for downstream models:

```python
import numpy as np
from puffin.deep.autoencoder import Autoencoder, AETrainer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Simulate market features and returns
np.random.seed(42)
X = np.random.randn(1000, 50)
y = np.random.randn(1000)

# Always normalize inputs before training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train autoencoder to extract key features
ae = Autoencoder(input_dim=50, encoding_dim=10)
trainer = AETrainer()
trainer.fit(ae, X_scaled, epochs=50)

# Use compressed features for a trading model
compressed_X = trainer.extract_features(ae, X_scaled)

model = Ridge()
model.fit(compressed_X, y)
predictions = model.predict(compressed_X)
```

{: .warning }
> Always normalize inputs with `StandardScaler` before training autoencoders. Unnormalized data
> with features on different scales will cause the reconstruction loss to be dominated by
> high-variance features, ignoring low-variance but potentially informative ones.

## Anomaly Detection

High reconstruction error signals that a sample does not fit the patterns the autoencoder learned from normal data. This is a natural anomaly detector for market surveillance.

```python
import torch
import numpy as np
from puffin.deep.autoencoder import Autoencoder, AETrainer
from sklearn.preprocessing import StandardScaler

# Simulate and normalize market data
np.random.seed(42)
X = np.random.randn(1000, 50)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train autoencoder on normal market data
ae = Autoencoder(input_dim=50, encoding_dim=10)
trainer = AETrainer()
trainer.fit(ae, X_scaled, epochs=50)

# Calculate reconstruction errors
ae.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    reconstructed = ae(X_tensor).numpy()

reconstruction_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)

# Define anomaly threshold (e.g., 95th percentile)
threshold = np.percentile(reconstruction_errors, 95)
anomalies = reconstruction_errors > threshold

print(f"Detected {anomalies.sum()} anomalies out of {len(X_scaled)} samples")
```

{: .note }
> Anomaly detection works best when the autoencoder is trained on a representative sample
> of "normal" market behavior. Anomalies are then defined as samples that fall outside
> the learned distribution -- flash crashes, liquidity events, or regime shifts.

## Regime Detection via Latent Clustering

The latent space learned by an autoencoder can reveal hidden market regimes when combined with clustering:

```python
from puffin.deep.autoencoder import Autoencoder, AETrainer
from sklearn.cluster import KMeans
import numpy as np

# Assume ae and trainer are already fitted from above
latent_features = trainer.extract_features(ae, X_scaled)

# Cluster into market regimes
n_regimes = 4
kmeans = KMeans(n_clusters=n_regimes, random_state=42)
regimes = kmeans.fit_predict(latent_features)

# Analyze regime characteristics
for regime in range(n_regimes):
    regime_mask = regimes == regime
    print(f"Regime {regime}: {regime_mask.sum()} samples")
```

Combining autoencoder features with K-Means or Gaussian Mixture Models gives you a data-driven regime indicator that can condition strategy behavior -- for example, reducing position sizes in high-volatility regimes.

## Training Best Practices

### Learning Rate and Convergence

Start with a learning rate of 0.001 and monitor the training and validation loss curves:

```python
import matplotlib.pyplot as plt
from puffin.deep.autoencoder import Autoencoder, AETrainer

ae = Autoencoder(input_dim=50, encoding_dim=10)
trainer = AETrainer()
history = trainer.fit(ae, X_scaled, epochs=100, lr=0.001)

# Plot loss to check convergence
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Autoencoder Training Convergence')
plt.show()
```

### Early Stopping

The `AETrainer` tracks validation loss at each epoch. Implement early stopping to prevent overfitting:

```python
from puffin.deep.training import EarlyStopping

best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(200):
    # Training step (simplified)
    # ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model weights
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### Evaluation Metrics

Track both reconstruction error and latent space quality:

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reconstruction metrics
mse = np.mean((X_scaled - reconstructed) ** 2)
mae = np.mean(np.abs(X_scaled - reconstructed))
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")

# Visualize latent space
latent = trainer.extract_features(ae, X_scaled)
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent)

plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Latent Space Visualization')
plt.show()
```

{: .tip }
> A well-trained autoencoder's latent space should show meaningful structure when projected
> to 2D -- clusters, gradients, or separations that correspond to known market conditions.

## Source Code

- Autoencoder and DenoisingAutoencoder classes: [`puffin/deep/autoencoder.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/autoencoder.py)
- AETrainer and training utilities: [`puffin/deep/training.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/training.py)
