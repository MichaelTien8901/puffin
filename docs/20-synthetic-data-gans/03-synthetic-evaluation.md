---
layout: default
title: "Synthetic Data Evaluation"
parent: "Part 20: Synthetic Data with GANs"
nav_order: 3
---

# Synthetic Data Evaluation

Generating synthetic data is only half the challenge. The critical question is: *how realistic is it?* If the synthetic data fails to reproduce the statistical properties of real market data -- marginal distributions, temporal dependencies, variance structure -- then models trained on it will learn spurious patterns. This page covers rigorous evaluation methods using the `SyntheticDataEvaluator` class.

{: .note }
> Never use synthetic data for backtesting or model training without first validating
> its quality against real data. Evaluation should be a mandatory step in every
> synthetic data pipeline.

## The SyntheticDataEvaluator

The `SyntheticDataEvaluator` class provides three complementary evaluation methods, plus a convenience method that runs all of them:

```python
from puffin.deep.gan import SyntheticDataEvaluator

evaluator = SyntheticDataEvaluator()
```

Each method targets a different aspect of data fidelity:
- **Distribution comparison**: Are the marginal distributions of each feature preserved?
- **Autocorrelation comparison**: Are the temporal dependencies preserved?
- **PCA comparison**: Is the variance and covariance structure preserved?

## Distribution Comparison (KS Test)

The Kolmogorov-Smirnov test compares the empirical CDFs of each feature between real and synthetic data. It is non-parametric -- no assumptions about the underlying distribution.

```python
import numpy as np
from puffin.deep.gan import GAN, SyntheticDataEvaluator

# Generate real and synthetic data
np.random.seed(42)
real_data = np.random.randn(1000, 20)

gan = GAN(latent_dim=10, data_dim=20)
gan.train(real_data, epochs=100, batch_size=64, lr=0.0002, verbose=False)
synthetic_data = gan.generate(n_samples=1000)

# Compare distributions
evaluator = SyntheticDataEvaluator()
dist_results = evaluator.compare_distributions(real_data, synthetic_data)

print(f"Average KS statistic: {dist_results['avg_ks_statistic']:.4f}")
print(f"Average KS p-value:   {dist_results['avg_ks_pvalue']:.4f}")
print(f"Average mean diff:    {dist_results['avg_mean_diff']:.4f}")
print(f"Average std diff:     {dist_results['avg_std_diff']:.4f}")
```

### Interpreting KS Results

| Metric | Good | Acceptable | Poor |
|---|---|---|---|
| KS statistic | < 0.05 | 0.05 - 0.15 | > 0.15 |
| KS p-value | > 0.10 | 0.01 - 0.10 | < 0.01 |
| Mean difference | < 0.01 | 0.01 - 0.05 | > 0.05 |
| Std difference | < 0.02 | 0.02 - 0.10 | > 0.10 |

{: .tip }
> A KS p-value above 0.05 means you *cannot reject* the null hypothesis that the
> distributions are identical. This is the desired outcome -- it does not prove they
> are identical, but it means the test found no evidence of difference.

### Per-Feature Analysis

The evaluator also returns per-feature results for detailed diagnostics:

```python
# Check each feature individually
for i, ks in enumerate(dist_results['ks_test']):
    status = "PASS" if ks['p_value'] > 0.05 else "FAIL"
    print(f"Feature {i:2d}: KS={ks['statistic']:.4f}, "
          f"p={ks['p_value']:.4f} [{status}]")
```

{: .warning }
> Features that consistently fail the KS test indicate the GAN has not learned
> their distribution. This may require more training epochs, a larger network,
> or better preprocessing for that feature.

## Autocorrelation Structure

For time-series data, matching marginal distributions is necessary but not sufficient. The temporal dependencies (autocorrelation) must also be preserved. A synthetic series with correct marginals but wrong autocorrelation will produce misleading backtest results.

```python
from puffin.deep.gan import TimeGAN, SyntheticDataEvaluator

# Assume tgan is trained and we have real and synthetic sequences
# Compare autocorrelation of the first feature
evaluator = SyntheticDataEvaluator()

acf_results = evaluator.compare_autocorrelation(
    real_sequences[:, :, 0].flatten(),      # First feature, all sequences
    synthetic_sequences[:, :, 0].flatten(),  # First feature, synthetic
    lags=20
)

print(f"Average ACF difference: {acf_results['avg_acf_diff']:.4f}")
print(f"Max ACF difference:     {acf_results['max_acf_diff']:.4f}")
```

### What the ACF Comparison Reveals

The autocorrelation function (ACF) at lag k measures the correlation between a time series and its k-step lagged version. The evaluator computes ACF for both real and synthetic data and reports the absolute difference at each lag.

| Metric | Good | Acceptable | Poor |
|---|---|---|---|
| Avg ACF difference | < 0.05 | 0.05 - 0.15 | > 0.15 |
| Max ACF difference | < 0.10 | 0.10 - 0.25 | > 0.25 |

{: .note }
> Financial returns typically have near-zero autocorrelation (weak form efficiency),
> but squared returns and absolute returns exhibit significant autocorrelation
> (volatility clustering). A good synthetic generator should reproduce both properties.

### Multi-Feature Autocorrelation

Check autocorrelation preservation across all features:

```python
n_features = real_sequences.shape[2]

for feat_idx in range(n_features):
    acf = evaluator.compare_autocorrelation(
        real_sequences[:, :, feat_idx].flatten(),
        synthetic_sequences[:, :, feat_idx].flatten(),
        lags=20
    )
    print(f"Feature {feat_idx}: avg_acf_diff={acf['avg_acf_diff']:.4f}, "
          f"max_acf_diff={acf['max_acf_diff']:.4f}")
```

## PCA Comparison

PCA comparison checks whether synthetic data preserves the covariance structure of the real data. If the first few principal components explain the same proportion of variance in both datasets, the synthetic data captures the same dominant factors.

```python
from puffin.deep.gan import SyntheticDataEvaluator

evaluator = SyntheticDataEvaluator()

pca_results = evaluator.compare_pca(real_data, synthetic_data, n_components=5)

print("Explained variance (real):",
      [f"{v:.4f}" for v in pca_results['explained_variance_real']])
print(f"Average variance diff:     {pca_results['avg_variance_diff']:.4f}")
print(f"Average PC distance:       {pca_results['avg_pc_distance']:.4f}")
```

### How PCA Comparison Works

1. Fit PCA on real data to find the principal axes
2. Project both real and synthetic data onto those axes
3. Compare the variance along each axis (do they explain similar amounts?)
4. Compare the distributions in PC space using KS tests

{: .tip }
> If the first PC explains 40% of variance in real data but 70% in synthetic data,
> the GAN is likely suffering from mode collapse -- generating samples concentrated
> along a single direction rather than spread across the full data manifold.

### Visualizing in PC Space

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
real_pca = pca.fit_transform(real_data)
synthetic_pca = pca.transform(synthetic_data)

plt.figure(figsize=(8, 6))
plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, label='Real', c='blue')
plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], alpha=0.3, label='Synthetic', c='red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('Real vs Synthetic Data in PC Space')
plt.show()
```

A well-trained GAN should produce overlapping point clouds. Gaps or clustering in the synthetic data indicate distribution mismatch or mode collapse.

## Full Evaluation Pipeline

The `full_evaluation` method runs all available tests and returns a comprehensive report:

```python
from puffin.deep.gan import SyntheticDataEvaluator

evaluator = SyntheticDataEvaluator()

# Comprehensive evaluation (runs distribution + PCA + autocorrelation)
full_results = evaluator.full_evaluation(real_data, synthetic_data)

# Distribution metrics
print("=== Distribution Metrics ===")
print(f"KS statistic: {full_results['distribution']['avg_ks_statistic']:.4f}")
print(f"KS p-value:   {full_results['distribution']['avg_ks_pvalue']:.4f}")
print(f"Mean diff:     {full_results['distribution']['avg_mean_diff']:.4f}")
print(f"Std diff:      {full_results['distribution']['avg_std_diff']:.4f}")

# PCA metrics
print("\n=== PCA Metrics ===")
print(f"Avg variance diff: {full_results['pca']['avg_variance_diff']:.4f}")
print(f"Avg PC distance:   {full_results['pca']['avg_pc_distance']:.4f}")

# Autocorrelation metrics (included when len(data) > 50)
if 'autocorrelation' in full_results:
    print("\n=== Autocorrelation Metrics ===")
    print(f"Avg ACF diff: {full_results['autocorrelation']['avg_acf_diff']:.4f}")
    print(f"Max ACF diff: {full_results['autocorrelation']['max_acf_diff']:.4f}")
```

{: .warning }
> The autocorrelation comparison is only included when the dataset has more than
> 50 samples. For smaller datasets, the ACF estimate is unreliable.

## Sample Diversity Check

Beyond statistical tests, visually inspect the diversity of generated samples to detect mode collapse:

```python
from sklearn.decomposition import PCA
from puffin.deep.gan import GAN
import matplotlib.pyplot as plt

# Generate a large batch
synthetic_data = gan.generate(n_samples=1000)

# Project to 2D via PCA
pca = PCA(n_components=2)
synthetic_pca = pca.fit_transform(synthetic_data)

plt.figure(figsize=(8, 6))
plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], alpha=0.5)
plt.title('Synthetic Data Diversity (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Good: Points spread broadly across both dimensions
# Bad: Tight cluster or elongated line (mode collapse)
```

{: .tip }
> If you see a single tight cluster in PC space, the generator has collapsed to
> producing nearly identical samples. Try: (1) reducing the learning rate,
> (2) using a Wasserstein loss, or (3) training the discriminator less frequently.

## Evaluation Checklist

Before using synthetic data in production, verify all of the following:

1. **Marginal distributions**: KS p-value > 0.05 for all features
2. **Moments**: Mean and standard deviation within 5% of real data
3. **Temporal structure**: ACF difference < 0.10 at all lags
4. **Covariance structure**: PCA variance ratios within 10% of real data
5. **Visual inspection**: Overlapping point clouds in PC space
6. **Domain review**: Synthetic scenarios make economic sense (no negative volumes, RSI outside [0, 100], etc.)

## Source Code

The `SyntheticDataEvaluator` class is implemented in [`puffin/deep/gan.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/deep/gan.py).

Key methods:
- `compare_distributions(real, synthetic)` -- KS test, mean/std comparison per feature
- `compare_autocorrelation(real, synthetic, lags)` -- ACF difference at specified lags
- `compare_pca(real, synthetic, n_components)` -- Variance and distribution comparison in PC space
- `full_evaluation(real, synthetic)` -- Runs all available tests in a single call
