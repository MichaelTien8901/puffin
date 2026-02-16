---
layout: default
title: "Chapter 1: Autoencoders for Trading"
parent: "Part 19: Autoencoders"
nav_order: 1
---

# Chapter 1: Autoencoders for Trading

Autoencoders are neural networks designed to learn efficient representations of data through unsupervised learning. In trading, they compress high-dimensional market data into meaningful features and detect anomalies in market behavior.

## What Are Autoencoders?

An autoencoder consists of two parts:
- **Encoder**: Compresses input data into a lower-dimensional latent representation
- **Decoder**: Reconstructs the original data from the latent representation

The network learns by minimizing reconstruction error, forcing it to capture the most important features in the compressed representation.

### Architecture

```
Input (n_features) → Encoder → Latent Space (encoding_dim) → Decoder → Reconstruction (n_features)
```

The bottleneck in the middle forces the network to learn a compressed representation.

## Types of Autoencoders for Trading

### 1. Standard Autoencoder

Basic feedforward autoencoder for dimensionality reduction:

```python
import numpy as np
from puffin.deep.autoencoder import Autoencoder, AETrainer

# Create market data (e.g., 100 features from technical indicators)
np.random.seed(42)
n_samples = 1000
n_features = 100

# Simulate market features
market_data = np.random.randn(n_samples, n_features)

# Create autoencoder
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
print(f"Original shape: {market_data.shape}")
print(f"Compressed shape: {compressed_features.shape}")
```

**Use Cases**:
- Reduce hundreds of technical indicators to a few key factors
- Pre-processing for other machine learning models
- Feature extraction from high-frequency tick data

### 2. Denoising Autoencoder

Adds noise during training to learn robust features:

```python
from puffin.deep.autoencoder import DenoisingAutoencoder

# Create denoising autoencoder
model = DenoisingAutoencoder(
    input_dim=100,
    encoding_dim=20,
    noise_factor=0.3  # Standard deviation of noise
)

# Train with noisy data
trainer = AETrainer()
history = trainer.fit(model, market_data, epochs=100)

# Extract robust features
robust_features = trainer.extract_features(model, market_data)
```

**Use Cases**:
- Handle noisy market data (bid-ask spreads, microstructure noise)
- Learn features resistant to data quality issues
- Improve generalization to new market conditions

### 3. Variational Autoencoder (VAE)

A probabilistic autoencoder that learns a distribution over latent space:

```python
from puffin.deep.autoencoder import VAE

# Create VAE
model = VAE(
    input_dim=100,
    latent_dim=20,
    hidden_dims=[80, 50, 30]
)

# Train VAE
trainer = AETrainer()
history = trainer.fit(model, market_data, epochs=100)

# Extract features (mean of latent distribution)
latent_features = trainer.extract_features(model, market_data)

# Generate synthetic market data
synthetic_data = model.sample(n=100)
print(f"Generated synthetic samples: {synthetic_data.shape}")
```

**VAE Loss Function**:
```
Loss = Reconstruction Loss + KL Divergence
     = MSE(x, reconstructed_x) + KL(q(z|x) || p(z))
```

**Use Cases**:
- Generate synthetic market scenarios for stress testing
- Learn smooth latent representations of market states
- Anomaly detection (high reconstruction error indicates unusual patterns)
- Data augmentation for rare events

### 4. Conditional Autoencoder

Incorporates external conditions (market regime, macroeconomic factors):

```python
from puffin.deep.autoencoder import ConditionalAutoencoder
import torch

# Create conditional autoencoder
model = ConditionalAutoencoder(
    input_dim=100,      # Asset features
    condition_dim=10,   # Market conditions
    encoding_dim=20
)

# Example: condition on market regime
asset_features = torch.randn(64, 100)
market_conditions = torch.randn(64, 10)  # VIX, interest rates, etc.

# Forward pass
reconstruction = model(asset_features, market_conditions)
```

**Use Cases**:
- Asset pricing conditional on market regime
- Factor models with macroeconomic conditions
- Multi-asset modeling with cross-sectional information

## Trading Applications

### 1. Feature Extraction for Strategy Development

```python
import pandas as pd
from puffin.deep.autoencoder import Autoencoder, AETrainer

# Load market data with many features
# data = pd.read_csv('market_features.csv')
# X = data.drop('returns', axis=1).values
# y = data['returns'].values

# Simulate data
X = np.random.randn(1000, 50)
y = np.random.randn(1000)

# Train autoencoder to extract key features
ae = Autoencoder(input_dim=50, encoding_dim=10)
trainer = AETrainer()
trainer.fit(ae, X, epochs=50)

# Use compressed features for strategy
compressed_X = trainer.extract_features(ae, X)

# Now use compressed_X for your trading model
from sklearn.linear_model import Ridge

model = Ridge()
model.fit(compressed_X, y)
predictions = model.predict(compressed_X)
```

### 2. Anomaly Detection

High reconstruction error indicates unusual market behavior:

```python
import torch

# Train autoencoder on normal market data
ae = Autoencoder(input_dim=50, encoding_dim=10)
trainer = AETrainer()
trainer.fit(ae, X, epochs=50)

# Calculate reconstruction errors
ae.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X)
    reconstructed = ae(X_tensor).numpy()

reconstruction_errors = np.mean((X - reconstructed) ** 2, axis=1)

# Define anomaly threshold (e.g., 95th percentile)
threshold = np.percentile(reconstruction_errors, 95)
anomalies = reconstruction_errors > threshold

print(f"Detected {anomalies.sum()} anomalies out of {len(X)} samples")
```

### 3. Regime Detection

Cluster latent representations to identify market regimes:

```python
from sklearn.cluster import KMeans

# Extract latent features
latent_features = trainer.extract_features(ae, X)

# Cluster into market regimes
n_regimes = 4
kmeans = KMeans(n_clusters=n_regimes, random_state=42)
regimes = kmeans.fit_predict(latent_features)

# Analyze regime characteristics
for regime in range(n_regimes):
    regime_mask = regimes == regime
    print(f"Regime {regime}: {regime_mask.sum()} samples")
```

### 4. Synthetic Data Generation with VAE

Generate realistic market scenarios:

```python
# Train VAE on historical data
vae = VAE(input_dim=50, latent_dim=10)
trainer = AETrainer()
trainer.fit(vae, X, epochs=100)

# Generate synthetic scenarios
n_scenarios = 1000
synthetic_scenarios = vae.sample(n=n_scenarios)

print(f"Generated {n_scenarios} synthetic market scenarios")

# Use for backtesting or stress testing
# backtest_strategy(synthetic_scenarios)
```

## Best Practices

### 1. Choosing Encoding Dimension

The encoding dimension determines information compression:
- Too small: Loss of important information
- Too large: Overfitting, inefficient compression

**Rule of thumb**: Start with 10-30% of input dimension:
```python
input_dim = 100
encoding_dim = int(0.2 * input_dim)  # 20 features
```

### 2. Architecture Design

Progressive compression works best:
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
    hidden_dims=[10]  # Direct compression
)
```

### 3. Training Strategies

**Data Normalization**: Always normalize inputs:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
```

**Learning Rate**: Start with 0.001 and adjust based on convergence:
```python
history = trainer.fit(model, X, epochs=100, lr=0.001)

# Plot loss to check convergence
import matplotlib.pyplot as plt

plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**Early Stopping**: Monitor validation loss:
```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    # Training code...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save model
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### 4. Evaluation Metrics

**Reconstruction Error**:
```python
mse = np.mean((X - reconstructed) ** 2)
mae = np.mean(np.abs(X - reconstructed))
```

**Latent Space Quality**: Check if latent space is meaningful:
```python
from sklearn.decomposition import PCA

# Visualize latent space
latent = trainer.extract_features(ae, X)
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent)

plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Latent Space Visualization')
plt.show()
```

## Complete Trading Example

Combining autoencoders with a trading strategy:

```python
import numpy as np
import pandas as pd
from puffin.deep.autoencoder import Autoencoder, AETrainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Generate/load market data
np.random.seed(42)
n_samples = 2000
n_features = 80

X = np.random.randn(n_samples, n_features)
y = np.random.choice([0, 1], size=n_samples)  # Buy/sell signals

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train autoencoder for feature extraction
ae = Autoencoder(
    input_dim=n_features,
    encoding_dim=15,
    hidden_dims=[60, 40, 25]
)

trainer = AETrainer()
history = trainer.fit(
    ae,
    X_train_scaled,
    epochs=50,
    lr=0.001,
    verbose=True
)

# Step 5: Extract compressed features
X_train_compressed = trainer.extract_features(ae, X_train_scaled)
X_test_compressed = trainer.extract_features(ae, X_test_scaled)

print(f"Compressed from {n_features} to {X_train_compressed.shape[1]} features")

# Step 6: Train trading model on compressed features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_compressed, y_train)

# Step 7: Evaluate
train_score = clf.score(X_train_compressed, y_train)
test_score = clf.score(X_test_compressed, y_test)

print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")
```

## Summary

Autoencoders provide powerful tools for trading:
- **Standard AE**: Dimensionality reduction, feature extraction
- **Denoising AE**: Robust features from noisy data
- **VAE**: Generative modeling, synthetic data
- **Conditional AE**: Regime-dependent modeling

Key advantages:
- Unsupervised learning (no labels needed)
- Automatic feature extraction
- Anomaly detection
- Data generation

Next steps:
- Experiment with different architectures
- Combine with other ML models
- Use for risk management and portfolio optimization
- Explore variational methods for uncertainty quantification
