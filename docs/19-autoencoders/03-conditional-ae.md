---
layout: default
title: "Conditional Autoencoders for Asset Pricing"
parent: "Part 19: Autoencoders"
nav_order: 3
---

# Conditional Autoencoders for Asset Pricing

A conditional autoencoder extends the standard architecture by incorporating external conditions -- market regime indicators, macroeconomic variables, or cross-sectional characteristics -- into both the encoder and decoder. This enables the model to learn latent representations that are sensitive to the prevailing market environment, making it a natural tool for regime-dependent asset pricing and factor modeling.

{: .note }
> The `ConditionalAutoencoder` in `puffin.deep.autoencoder` concatenates condition vectors
> with both the encoder input and the decoder input, allowing the latent space to be
> condition-aware at every stage of the forward pass.

## Why Conditioning Matters

Standard autoencoders treat all samples identically. But financial markets are non-stationary: the relationship between features and returns changes across regimes, sectors, and macro environments. A conditional autoencoder learns different compression strategies depending on the external state.

| Condition Type | Example Variables | Effect on Latent Space |
|---------------|-------------------|----------------------|
| Market regime | VIX level, trend direction | Different compression for calm vs. crisis |
| Macro factors | Interest rates, inflation, GDP | Economy-aware feature extraction |
| Sector membership | One-hot sector encoding | Sector-specific factor structure |
| Time features | Day-of-week, month, quarter | Seasonal pattern awareness |

## Architecture

The `ConditionalAutoencoder` concatenates the condition vector with both the encoder input and the decoder input:

```
Encoder: [x, condition] -> latent z
Decoder: [z, condition] -> reconstruction x_hat
```

This means the condition influences both how the data is compressed and how it is reconstructed.

```python
import torch
from puffin.deep.autoencoder import ConditionalAutoencoder

# Create conditional autoencoder
model = ConditionalAutoencoder(
    input_dim=100,      # Asset features (returns, indicators)
    condition_dim=10,   # Market conditions (VIX, rates, etc.)
    encoding_dim=20     # Latent dimension
)

# Example: condition on market regime indicators
asset_features = torch.randn(64, 100)
market_conditions = torch.randn(64, 10)  # VIX, interest rates, spreads, etc.

# Forward pass
reconstruction = model(asset_features, market_conditions)
print(f"Input shape: {asset_features.shape}")         # (64, 100)
print(f"Condition shape: {market_conditions.shape}")   # (64, 10)
print(f"Reconstruction shape: {reconstruction.shape}") # (64, 100)
```

{: .tip }
> The condition vector is not compressed through the bottleneck -- it passes directly to
> both encoder and decoder. This means conditions act as side information that modulates
> the learned representation without being subject to information loss.

## Asset Pricing with Market Regime Conditioning

The most powerful application of conditional autoencoders in trading is learning asset pricing factors that adapt to the market regime.

```python
import numpy as np
import torch
from puffin.deep.autoencoder import ConditionalAutoencoder
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Simulate asset cross-section data
n_assets = 500
n_features = 80   # Firm characteristics (size, value, momentum, etc.)
n_conditions = 5  # Macro conditions (VIX, term spread, credit spread, etc.)

# Asset characteristics
asset_chars = np.random.randn(n_assets, n_features)

# Market conditions (same for all assets in a given period)
macro_conditions = np.random.randn(n_assets, n_conditions)

# Normalize
char_scaler = StandardScaler()
cond_scaler = StandardScaler()

asset_chars_scaled = char_scaler.fit_transform(asset_chars)
conditions_scaled = cond_scaler.fit_transform(macro_conditions)

# Build conditional autoencoder
cae = ConditionalAutoencoder(
    input_dim=n_features,
    condition_dim=n_conditions,
    encoding_dim=15
)

# Training loop (simplified)
optimizer = torch.optim.Adam(cae.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

X_tensor = torch.FloatTensor(asset_chars_scaled)
C_tensor = torch.FloatTensor(conditions_scaled)

cae.train()
for epoch in range(100):
    optimizer.zero_grad()
    recon = cae(X_tensor, C_tensor)
    loss = loss_fn(recon, X_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.6f}")
```

{: .warning }
> When using macro conditions that are the same across all assets in a time period,
> ensure you are not introducing look-ahead bias. The condition vector should only
> contain information available at the time the asset features were observed.

## Extracting Regime-Dependent Latent Factors

Once trained, the conditional autoencoder produces different latent representations for the same asset depending on the market conditions:

```python
import torch
import numpy as np
from puffin.deep.autoencoder import ConditionalAutoencoder

# Assume cae is trained
cae.eval()

# Same asset features, different market conditions
asset = torch.FloatTensor(asset_chars_scaled[0:1])  # Single asset

# Calm market conditions
calm_conditions = torch.zeros(1, 5)
calm_conditions[0, 0] = -1.0  # Low VIX

# Stressed market conditions
stress_conditions = torch.zeros(1, 5)
stress_conditions[0, 0] = 2.5  # High VIX

with torch.no_grad():
    # Encode under calm regime
    calm_input = torch.cat([asset, calm_conditions], dim=1)
    calm_latent = cae.encoder(calm_input)

    # Encode under stress regime
    stress_input = torch.cat([asset, stress_conditions], dim=1)
    stress_latent = cae.encoder(stress_input)

    diff = torch.norm(calm_latent - stress_latent).item()
    print(f"Latent distance between regimes: {diff:.4f}")
```

{: .note }
> A large latent distance between regime conditions for the same asset indicates that the
> autoencoder has learned meaningfully different representations. This is the signal that
> regime conditioning is adding value over an unconditional model.

## Multi-Asset Factor Modeling

Conditional autoencoders are particularly well-suited for cross-sectional factor models where the factor structure itself changes with the macro environment:

```python
import numpy as np
import torch
from puffin.deep.autoencoder import ConditionalAutoencoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

np.random.seed(42)

# Simulate cross-section across multiple time periods
n_periods = 50
n_assets_per_period = 100
n_features = 60
n_conditions = 5

all_chars = []
all_conditions = []

for t in range(n_periods):
    # Asset characteristics for this period
    chars_t = np.random.randn(n_assets_per_period, n_features)
    # Macro conditions (same for all assets in period t)
    cond_t = np.tile(
        np.random.randn(1, n_conditions),
        (n_assets_per_period, 1)
    )
    all_chars.append(chars_t)
    all_conditions.append(cond_t)

X_all = np.vstack(all_chars)
C_all = np.vstack(all_conditions)

# Normalize
char_scaler = StandardScaler()
cond_scaler = StandardScaler()
X_scaled = char_scaler.fit_transform(X_all)
C_scaled = cond_scaler.fit_transform(C_all)

# Train conditional autoencoder
cae = ConditionalAutoencoder(
    input_dim=n_features,
    condition_dim=n_conditions,
    encoding_dim=10
)

optimizer = torch.optim.Adam(cae.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

X_t = torch.FloatTensor(X_scaled)
C_t = torch.FloatTensor(C_scaled)

cae.train()
for epoch in range(50):
    optimizer.zero_grad()
    recon = cae(X_t, C_t)
    loss = loss_fn(recon, X_t)
    loss.backward()
    optimizer.step()

# Extract latent factors and cluster
cae.eval()
with torch.no_grad():
    combined = torch.cat([X_t, C_t], dim=1)
    latent_factors = cae.encoder(combined).numpy()

# Cluster latent factors to find asset groups
kmeans = KMeans(n_clusters=5, random_state=42)
groups = kmeans.fit_predict(latent_factors)

print(f"Latent factors shape: {latent_factors.shape}")
for g in range(5):
    print(f"  Group {g}: {(groups == g).sum()} assets")
```

## Complete Trading Pipeline

Combining conditional autoencoders with a full prediction and evaluation workflow:

```python
import numpy as np
import pandas as pd
import torch
from puffin.deep.autoencoder import ConditionalAutoencoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Generate / load market data
np.random.seed(42)
n_samples = 2000
n_features = 80
n_conditions = 5

X = np.random.randn(n_samples, n_features)
conditions = np.random.randn(n_samples, n_conditions)
y = np.random.choice([0, 1], size=n_samples)  # Buy/sell signals

# Step 2: Split data (temporal split in production)
X_train, X_test, c_train, c_test, y_train, y_test = train_test_split(
    X, conditions, y, test_size=0.2, random_state=42
)

# Step 3: Normalize
x_scaler = StandardScaler()
c_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
c_train_scaled = c_scaler.fit_transform(c_train)
c_test_scaled = c_scaler.transform(c_test)

# Step 4: Train conditional autoencoder
cae = ConditionalAutoencoder(
    input_dim=n_features,
    condition_dim=n_conditions,
    encoding_dim=15
)

optimizer = torch.optim.Adam(cae.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

X_t = torch.FloatTensor(X_train_scaled)
C_t = torch.FloatTensor(c_train_scaled)

cae.train()
for epoch in range(50):
    optimizer.zero_grad()
    recon = cae(X_t, C_t)
    loss = loss_fn(recon, X_t)
    loss.backward()
    optimizer.step()

# Step 5: Extract condition-aware features
cae.eval()
with torch.no_grad():
    train_input = torch.cat([
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(c_train_scaled)
    ], dim=1)
    test_input = torch.cat([
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(c_test_scaled)
    ], dim=1)

    X_train_latent = cae.encoder(train_input).numpy()
    X_test_latent = cae.encoder(test_input).numpy()

print(f"Compressed from {n_features} to {X_train_latent.shape[1]} features")

# Step 6: Train trading model on compressed features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_latent, y_train)

# Step 7: Evaluate
train_score = clf.score(X_train_latent, y_train)
test_score = clf.score(X_test_latent, y_test)

print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")
```

{: .warning }
> In production, always use a temporal (walk-forward) split rather than random splitting.
> Random splits create look-ahead bias because future data points can leak information
> about past market conditions through the autoencoder's learned representation.

## Design Decisions and Trade-offs

### Encoding Dimension

For conditional autoencoders, the encoding dimension can often be smaller than for unconditional models because the condition vector carries supplementary information:

```python
# Unconditional: needs larger latent space
Autoencoder(input_dim=100, encoding_dim=20)

# Conditional: condition carries extra info, smaller latent suffices
ConditionalAutoencoder(input_dim=100, condition_dim=10, encoding_dim=12)
```

### Condition Vector Design

{: .tip }
> Keep the condition vector compact and meaningful. Avoid dumping raw time series into it.
> Instead, use pre-computed summary statistics: rolling volatility, regime labels,
> yield curve slope, credit spread level.

Good condition variables:
- VIX level (current volatility regime)
- Term spread (yield curve slope)
- Credit spread (risk appetite)
- Market return over past 20 days (momentum regime)
- Binary regime label from a hidden Markov model

Poor condition variables:
- Raw daily returns for 252 days (too high-dimensional)
- Individual stock prices (not informative about regime)
- Categorical IDs without economic meaning

## Summary

Conditional autoencoders bring regime-awareness to unsupervised feature learning:

- **Standard AE**: Fixed compression regardless of market state
- **Conditional AE**: Compression adapts to external conditions
- **Factor models**: Latent factors change meaning across regimes
- **Asset pricing**: Condition on macro to capture time-varying risk premia

The key insight is that financial relationships are non-stationary, and conditioning allows the model to learn separate compression strategies for different environments rather than averaging across all regimes.

## Source Code

- ConditionalAutoencoder class: [`puffin/deep/autoencoder.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/autoencoder.py)
- AETrainer (shared trainer): [`puffin/deep/autoencoder.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/autoencoder.py)
- Training utilities and EarlyStopping: [`puffin/deep/training.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/training.py)
