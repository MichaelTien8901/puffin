---
layout: default
title: "Part 19: Autoencoders"
nav_order: 20
has_children: true
permalink: /19-autoencoders/
---

# Autoencoders for Trading

## Overview

Autoencoders are neural networks designed to learn efficient representations of data through unsupervised learning. In trading, they compress high-dimensional market data into meaningful latent features, detect anomalies in market behavior, and generate synthetic scenarios for stress testing.

Unlike supervised models that map inputs to labels, autoencoders learn by reconstructing their own input through a bottleneck layer. This forces the network to discover the most salient structure in the data -- structure that can then drive feature engineering, risk modeling, and portfolio construction.

{: .note }
> Chapter 20 of *Machine Learning for Algorithmic Trading* covers autoencoders for conditional risk factors,
> variational autoencoders, and asset pricing applications. This part implements those techniques
> in the `puffin.deep.autoencoder` module.

## Autoencoder Architecture

The core idea is simple: compress, then reconstruct. The quality of the reconstruction tells us how well the bottleneck captures the essential information.

```mermaid
graph LR
    subgraph Input
        I1[Feature 1]
        I2[Feature 2]
        I3[Feature ...]
        I4[Feature n]
    end

    subgraph Encoder
        E1[Hidden 80]
        E2[Hidden 50]
        E3[Hidden 30]
    end

    subgraph Latent["Latent Space"]
        L1[z1]
        L2[z2]
        L3[z...]
        L4[zk]
    end

    subgraph Decoder
        D1[Hidden 30]
        D2[Hidden 50]
        D3[Hidden 80]
    end

    subgraph Output["Reconstruction"]
        O1[Feature 1]
        O2[Feature 2]
        O3[Feature ...]
        O4[Feature n]
    end

    I1 --> E1
    I4 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> L1
    E3 --> L4
    L1 --> D1
    L4 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> O1
    D3 --> O4

    classDef input fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4
    classDef encoder fill:#1a3a5c,stroke:#0d2137,color:#e8e0d4
    classDef latent fill:#6b2d5b,stroke:#4a1a4a,color:#e8e0d4
    classDef decoder fill:#8b4513,stroke:#5a2d0a,color:#e8e0d4
    classDef output fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4

    class I1,I2,I3,I4 input
    class E1,E2,E3 encoder
    class L1,L2,L3,L4 latent
    class D1,D2,D3 decoder
    class O1,O2,O3,O4 output

    linkStyle default stroke:#4a5568,stroke-width:2px
```

The bottleneck (latent space) has fewer dimensions than the input, forcing the network to learn a compressed representation that retains only the most important information.

## Autoencoder Variants for Trading

| Variant | Key Idea | Trading Application |
|---------|----------|-------------------|
| **Standard AE** | Minimize reconstruction error | Dimensionality reduction, feature extraction |
| **Denoising AE** | Corrupt input, reconstruct clean | Robust features from noisy market data |
| **Variational AE** | Learn latent distribution, sample | Synthetic data generation, anomaly detection |
| **Conditional AE** | Condition on external factors | Regime-dependent asset pricing |

{: .tip }
> Start with a standard autoencoder to verify your data pipeline works correctly.
> Move to denoising or variational variants only after validating the basic approach.

## Puffin Module Structure

All autoencoder implementations live in a single module with a shared trainer:

```python
from puffin.deep.autoencoder import (
    Autoencoder,              # Standard feedforward AE
    DenoisingAutoencoder,     # Noise-injection AE
    VAE,                      # Variational AE
    ConditionalAutoencoder,   # Condition-aware AE
    AETrainer                 # Unified training loop
)
```

The `AETrainer` class handles training for all four variants, including automatic detection of VAE loss (reconstruction + KL divergence) versus standard MSE reconstruction loss.

## Key Advantages

- **Unsupervised**: No labels required -- learns structure from raw market data
- **Flexible compression**: Reduce hundreds of technical indicators to a handful of latent factors
- **Anomaly detection**: High reconstruction error signals unusual market conditions
- **Generative capability**: VAE can synthesize realistic market scenarios for stress testing

## What You Will Learn

1. [**Standard & Denoising Autoencoders**](01-standard-denoising.md) -- Dimensionality reduction, robust feature extraction, anomaly detection, and regime detection from latent representations
2. [**Variational Autoencoders**](02-variational-ae.md) -- Probabilistic latent spaces, the reparameterization trick, synthetic data generation, and the VAE loss function
3. [**Conditional Autoencoders for Asset Pricing**](03-conditional-ae.md) -- Regime-dependent modeling, macro-conditioned factor extraction, and a complete trading pipeline

{: .tip }
> **Notebook**: Run the examples interactively in [`deep_learning.ipynb`](https://github.com/MichaelTien8901/puffin/blob/main/notebooks/deep_learning.ipynb)

## Related Chapters

- [Part 16: Deep Learning Fundamentals]({{ site.baseurl }}/16-deep-learning/) -- Foundational neural network training techniques that autoencoders build upon
- [Part 12: Unsupervised Learning]({{ site.baseurl }}/12-unsupervised-learning/) -- PCA and clustering provide classical unsupervised counterparts to autoencoder-based dimensionality reduction
- [Part 20: Synthetic Data with GANs]({{ site.baseurl }}/20-synthetic-data-gans/) -- GANs extend the generative modeling ideas introduced by variational autoencoders
- [Part 5: Portfolio Optimization]({{ site.baseurl }}/05-portfolio-optimization/) -- Latent factors extracted by autoencoders can drive portfolio construction and risk modeling

## Source Code

Browse the implementation: [`puffin/deep/autoencoder.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/autoencoder.py)
