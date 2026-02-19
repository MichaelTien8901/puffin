---
layout: default
title: "Variational Autoencoders"
parent: "Part 19: Autoencoders"
nav_order: 2
---

# Variational Autoencoders (VAE)

A Variational Autoencoder replaces the deterministic bottleneck of a standard autoencoder with a probabilistic latent space. Instead of mapping each input to a single point, the encoder outputs the parameters of a distribution (mean and variance). This enables two capabilities that standard autoencoders lack: principled anomaly detection via likelihood estimation, and generation of new synthetic samples by sampling from the learned distribution.

{: .note }
> The VAE implementation in `puffin.deep.autoencoder` uses the reparameterization trick
> for backpropagation through stochastic nodes and supports a configurable beta parameter
> for disentangled representations (beta-VAE).

## How VAEs Differ from Standard Autoencoders

In a standard autoencoder, the encoder maps input **x** to a fixed latent vector **z**. In a VAE, the encoder maps **x** to a distribution over **z**, parameterized by mean (mu) and log-variance (log_var):

| Component | Standard AE | VAE |
|-----------|------------|-----|
| Encoder output | Single vector z | Distribution parameters (mu, log_var) |
| Latent space | Deterministic | Stochastic (sample from N(mu, sigma)) |
| Loss function | Reconstruction only | Reconstruction + KL divergence |
| Generation | Not supported | Sample z ~ N(0, I), decode |

## The VAE Loss Function

The VAE loss has two terms that balance reconstruction quality against latent space regularity:

```
Loss = Reconstruction Loss + KL Divergence
     = MSE(x, reconstructed_x) + KL(q(z|x) || p(z))
```

- **Reconstruction loss**: How well the decoder reproduces the input (MSE)
- **KL divergence**: How close the learned distribution q(z|x) is to the prior p(z) = N(0, I)

The KL term prevents the encoder from collapsing to a single point per input, ensuring the latent space is smooth and continuous -- which is what makes generation possible.

{: .warning }
> If the KL term dominates, the model ignores input information and produces blurry reconstructions
> (posterior collapse). If the reconstruction term dominates, the latent space becomes
> disconnected and generation quality degrades. Tune the beta parameter to balance these.

## The Reparameterization Trick

Sampling from N(mu, sigma) is not differentiable, so gradients cannot flow through it. The reparameterization trick rewrites the sampling as a deterministic function of the parameters plus external noise:

```
z = mu + sigma * epsilon,   where epsilon ~ N(0, I)
```

This makes the sampling differentiable with respect to mu and sigma, enabling standard backpropagation.

## Basic VAE Usage

```python
import numpy as np
from puffin.deep.autoencoder import VAE, AETrainer

# Simulate market data
np.random.seed(42)
n_samples = 1000
n_features = 100
market_data = np.random.randn(n_samples, n_features)

# Create VAE
model = VAE(
    input_dim=100,
    latent_dim=20,
    hidden_dims=[80, 50, 30]
)

# Train VAE (AETrainer auto-detects VAE and uses combined loss)
trainer = AETrainer()
history = trainer.fit(model, market_data, epochs=100)

# Extract features (mean of latent distribution)
latent_features = trainer.extract_features(model, market_data)
print(f"Latent shape: {latent_features.shape}")  # (1000, 20)

# Generate synthetic market data
synthetic_data = model.sample(n=100)
print(f"Generated synthetic samples: {synthetic_data.shape}")  # (100, 100)
```

{: .tip }
> The `AETrainer` automatically detects when a VAE is passed and switches to the combined
> reconstruction + KL divergence loss. No special configuration is needed.

## VAE Loss Implementation

The `AETrainer.vae_loss` static method computes the combined loss with an optional beta weight:

```python
from puffin.deep.autoencoder import AETrainer
import torch

# The loss function used internally by AETrainer
# recon: reconstructed output, x: original input
# mu, log_var: encoder distribution parameters
# beta: weight on KL term (default 1.0)

# Reconstruction loss (MSE, summed over features)
recon_loss = torch.nn.MSELoss(reduction='sum')(recon, x)

# KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# Total loss, averaged over batch
total_loss = (recon_loss + beta * kl_loss) / x.size(0)
```

### Beta-VAE for Disentangled Representations

Setting beta > 1 encourages more disentangled latent dimensions, where each dimension captures a single independent factor of variation:

| beta | Behavior | Trading Use |
|------|----------|------------|
| 0.5 | Favor reconstruction | Better feature extraction accuracy |
| 1.0 | Standard VAE | Balanced reconstruction and generation |
| 2.0+ | Favor disentanglement | Interpretable latent factors (e.g., momentum, volatility) |

## Synthetic Data Generation

One of the most powerful applications of VAEs in trading is generating synthetic market scenarios for stress testing and data augmentation.

```python
import numpy as np
from puffin.deep.autoencoder import VAE, AETrainer
from sklearn.preprocessing import StandardScaler

# Prepare and normalize data
np.random.seed(42)
X = np.random.randn(1000, 50)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train VAE on historical data
vae = VAE(input_dim=50, latent_dim=10)
trainer = AETrainer()
trainer.fit(vae, X_scaled, epochs=100)

# Generate synthetic scenarios
n_scenarios = 1000
synthetic_scenarios = vae.sample(n=n_scenarios)
print(f"Generated {n_scenarios} synthetic market scenarios")
print(f"Shape: {synthetic_scenarios.shape}")  # (1000, 50)

# Inverse transform to original scale
synthetic_original = scaler.inverse_transform(
    synthetic_scenarios.detach().numpy()
)
```

{: .note }
> Synthetic data from a VAE preserves the statistical relationships (correlations, distributions)
> present in the training data while introducing realistic variation. This is valuable for
> backtesting strategies on scenarios that have not occurred historically.

### Targeted Scenario Generation

You can interpolate in latent space to generate scenarios between known market states:

```python
import torch
from puffin.deep.autoencoder import VAE

# Assume vae is already trained
vae.eval()

# Encode two known market states
with torch.no_grad():
    state_a = torch.FloatTensor(X_scaled[0:1])  # Calm market
    state_b = torch.FloatTensor(X_scaled[100:101])  # Volatile market

    mu_a, _ = vae.encode(state_a)
    mu_b, _ = vae.encode(state_b)

# Interpolate between states
n_steps = 10
interpolated = []
for alpha in np.linspace(0, 1, n_steps):
    z = (1 - alpha) * mu_a + alpha * mu_b
    scenario = vae.decode(z)
    interpolated.append(scenario.numpy())

interpolated = np.vstack(interpolated)
print(f"Generated {n_steps} interpolated scenarios: {interpolated.shape}")
```

{: .tip }
> Latent space interpolation is a powerful tool for stress testing. You can generate
> a smooth transition from "normal market" to "crisis" conditions by interpolating between
> the encoded representations of historical calm and stress periods.

## Anomaly Detection with VAE

VAEs provide a principled anomaly score: the combination of reconstruction error and the distance of the encoded distribution from the prior.

```python
import torch
import numpy as np
from puffin.deep.autoencoder import VAE, AETrainer

# Assume vae and trainer are already fitted
vae.eval()

with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    recon, mu, log_var = vae(X_tensor)

    # Reconstruction error per sample
    recon_error = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()

    # KL divergence per sample (measures how unusual the encoding is)
    kl_per_sample = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp(), dim=1
    ).numpy()

    # Combined anomaly score
    anomaly_score = recon_error + 0.1 * kl_per_sample

# Flag anomalies
threshold = np.percentile(anomaly_score, 95)
anomalies = anomaly_score > threshold
print(f"Detected {anomalies.sum()} anomalous market states")
```

{: .warning }
> The relative weighting between reconstruction error and KL divergence in the anomaly score
> is a hyperparameter. Reconstruction error alone often works well for detecting gross anomalies,
> while the KL term catches subtler distributional shifts.

## Data Augmentation for Rare Events

Markets exhibit rare but important events (crashes, squeezes, flash crashes) that appear only a few times in historical data. A VAE can augment these samples:

```python
import torch
import numpy as np
from puffin.deep.autoencoder import VAE

# Assume vae is trained and we have identified rare event samples
# rare_indices = identify_rare_events(X_scaled)
rare_indices = np.array([10, 50, 200])  # Example indices

vae.eval()
with torch.no_grad():
    rare_samples = torch.FloatTensor(X_scaled[rare_indices])
    mu_rare, log_var_rare = vae.encode(rare_samples)

    # Generate variations around rare events
    n_augmented = 50
    augmented = []
    for _ in range(n_augmented):
        # Sample near the rare event encoding
        std = torch.exp(0.5 * log_var_rare)
        eps = torch.randn_like(std) * 0.5  # Smaller noise for proximity
        z = mu_rare + eps * std
        generated = vae.decode(z)
        augmented.append(generated.numpy())

augmented = np.vstack(augmented)
print(f"Augmented rare events: {augmented.shape}")
```

This gives downstream models more examples of tail events to learn from, reducing the bias toward common market conditions.

## Monitoring VAE Training

Track both components of the loss separately to diagnose training issues:

```python
import matplotlib.pyplot as plt
from puffin.deep.autoencoder import VAE, AETrainer

vae = VAE(input_dim=50, latent_dim=10)
trainer = AETrainer()
history = trainer.fit(vae, X_scaled, epochs=100, verbose=True)

# Plot combined loss
plt.figure(figsize=(10, 4))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Recon + KL)')
plt.legend()
plt.title('VAE Training Progress')
plt.show()
```

{: .note }
> If the validation loss plateaus while training loss continues to drop, the model is
> overfitting. Increase the beta parameter or add dropout to the encoder/decoder layers.

## Source Code

- VAE class and reparameterization: [`puffin/deep/autoencoder.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/autoencoder.py)
- VAE loss function (`AETrainer.vae_loss`): [`puffin/deep/autoencoder.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/autoencoder.py)
- Early stopping utility: [`puffin/deep/training.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/training.py)
