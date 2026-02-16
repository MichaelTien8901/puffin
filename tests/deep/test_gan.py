"""Tests for GAN architectures."""

import numpy as np
import pytest
import torch

from puffin.deep.gan import (
    Generator,
    Discriminator,
    GAN,
    TimeGAN,
    SyntheticDataEvaluator
)


class TestGenerator:
    """Test generator network."""

    def test_initialization(self):
        """Test generator initialization."""
        gen = Generator(latent_dim=10, output_dim=20, hidden_dims=[64, 128])
        assert gen.latent_dim == 10
        assert gen.output_dim == 20
        assert gen.hidden_dims == [64, 128]

    def test_forward_shape(self):
        """Test forward pass output shape."""
        gen = Generator(latent_dim=10, output_dim=20)
        z = torch.randn(8, 10)
        output = gen(z)
        assert output.shape == (8, 20)

    def test_output_range(self):
        """Test that output is in [-1, 1] due to tanh."""
        gen = Generator(latent_dim=10, output_dim=20)
        z = torch.randn(8, 10)
        output = gen(z)
        assert torch.all(output >= -1) and torch.all(output <= 1)


class TestDiscriminator:
    """Test discriminator network."""

    def test_initialization(self):
        """Test discriminator initialization."""
        disc = Discriminator(input_dim=20, hidden_dims=[128, 64])
        assert disc.input_dim == 20
        assert disc.hidden_dims == [128, 64]

    def test_forward_shape(self):
        """Test forward pass output shape."""
        disc = Discriminator(input_dim=20)
        x = torch.randn(8, 20)
        output = disc(x)
        assert output.shape == (8, 1)

    def test_output_range(self):
        """Test that output is in [0, 1] due to sigmoid."""
        disc = Discriminator(input_dim=20)
        x = torch.randn(8, 20)
        output = disc(x)
        assert torch.all(output >= 0) and torch.all(output <= 1)


class TestGAN:
    """Test GAN class."""

    def test_initialization(self):
        """Test GAN initialization."""
        gan = GAN(latent_dim=10, data_dim=20)
        assert gan.latent_dim == 10
        assert gan.data_dim == 20
        assert isinstance(gan.generator, Generator)
        assert isinstance(gan.discriminator, Discriminator)

    def test_generate_shape(self):
        """Test generation output shape."""
        gan = GAN(latent_dim=10, data_dim=20)
        samples = gan.generate(n_samples=50)
        assert samples.shape == (50, 20)

    def test_training_basic(self):
        """Test basic GAN training."""
        gan = GAN(latent_dim=10, data_dim=20)

        # Create simple training data
        np.random.seed(42)
        real_data = np.random.randn(100, 20)

        # Train for a few epochs
        history = gan.train(real_data, epochs=5, batch_size=32, verbose=False)

        # Check history structure
        assert 'g_loss' in history
        assert 'd_loss' in history
        assert len(history['g_loss']) == 5
        assert len(history['d_loss']) == 5

    def test_training_improves(self):
        """Test that training produces reasonable losses."""
        gan = GAN(latent_dim=10, data_dim=20)

        np.random.seed(42)
        real_data = np.random.randn(200, 20)

        history = gan.train(real_data, epochs=20, batch_size=32, verbose=False)

        # Losses should be positive
        assert all(loss > 0 for loss in history['g_loss'])
        assert all(loss > 0 for loss in history['d_loss'])


class TestTimeGAN:
    """Test TimeGAN class."""

    def test_initialization(self):
        """Test TimeGAN initialization."""
        tgan = TimeGAN(seq_length=10, n_features=5, hidden_dim=16, latent_dim=16)
        assert tgan.seq_length == 10
        assert tgan.n_features == 5
        assert tgan.hidden_dim == 16
        assert tgan.latent_dim == 16

    def test_generate_shape(self):
        """Test generation output shape."""
        tgan = TimeGAN(seq_length=10, n_features=5)
        sequences = tgan.generate(n_sequences=20)
        assert sequences.shape == (20, 10, 5)

    def test_training_basic(self):
        """Test basic TimeGAN training."""
        tgan = TimeGAN(seq_length=10, n_features=5, hidden_dim=16, latent_dim=16)

        # Create simple time series data
        np.random.seed(42)
        real_sequences = np.random.randn(50, 10, 5)

        # Train for a few epochs
        history = tgan.train(real_sequences, epochs=3, batch_size=16, verbose=False)

        # Check history structure
        assert 'recon_loss' in history
        assert 'g_loss' in history
        assert 'd_loss' in history
        assert 'sup_loss' in history
        assert len(history['recon_loss']) == 3

    def test_training_components(self):
        """Test that all training components work."""
        tgan = TimeGAN(seq_length=10, n_features=5, hidden_dim=16, latent_dim=16)

        np.random.seed(42)
        real_sequences = np.random.randn(100, 10, 5)

        history = tgan.train(real_sequences, epochs=5, batch_size=16, verbose=False)

        # All losses should be positive
        assert all(loss > 0 for loss in history['recon_loss'])
        assert all(loss > 0 for loss in history['g_loss'])
        assert all(loss > 0 for loss in history['d_loss'])
        assert all(loss > 0 for loss in history['sup_loss'])


class TestSyntheticDataEvaluator:
    """Test synthetic data evaluator."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = SyntheticDataEvaluator()
        assert evaluator is not None

    def test_compare_distributions(self):
        """Test distribution comparison."""
        evaluator = SyntheticDataEvaluator()

        np.random.seed(42)
        real = np.random.randn(100, 10)
        synthetic = np.random.randn(100, 10)

        results = evaluator.compare_distributions(real, synthetic)

        # Check structure
        assert 'ks_test' in results
        assert 'mean_diff' in results
        assert 'std_diff' in results
        assert 'avg_ks_statistic' in results
        assert 'avg_ks_pvalue' in results
        assert len(results['ks_test']) == 10
        assert len(results['mean_diff']) == 10

    def test_compare_distributions_similar(self):
        """Test that similar distributions get good scores."""
        evaluator = SyntheticDataEvaluator()

        np.random.seed(42)
        real = np.random.randn(1000, 5)
        # Synthetic from same distribution
        synthetic = np.random.randn(1000, 5)

        results = evaluator.compare_distributions(real, synthetic)

        # Average mean and std differences should be small
        assert results['avg_mean_diff'] < 0.3
        assert results['avg_std_diff'] < 0.3

    def test_compare_autocorrelation(self):
        """Test autocorrelation comparison."""
        evaluator = SyntheticDataEvaluator()

        np.random.seed(42)
        # Create data with some autocorrelation
        real = np.cumsum(np.random.randn(200, 5), axis=0)
        synthetic = np.cumsum(np.random.randn(200, 5), axis=0)

        results = evaluator.compare_autocorrelation(real, synthetic, lags=10)

        # Check structure
        assert 'avg_acf_diff' in results
        assert 'max_acf_diff' in results
        assert 'acf_diffs_per_feature' in results
        assert len(results['acf_diffs_per_feature']) == 5

    def test_compare_pca(self):
        """Test PCA comparison."""
        evaluator = SyntheticDataEvaluator()

        np.random.seed(42)
        real = np.random.randn(100, 10)
        synthetic = np.random.randn(100, 10)

        results = evaluator.compare_pca(real, synthetic, n_components=5)

        # Check structure
        assert 'explained_variance_real' in results
        assert 'variance_diff' in results
        assert 'avg_variance_diff' in results
        assert 'pc_distribution_distances' in results
        assert len(results['explained_variance_real']) == 5
        assert len(results['variance_diff']) == 5

    def test_full_evaluation(self):
        """Test full evaluation pipeline."""
        evaluator = SyntheticDataEvaluator()

        np.random.seed(42)
        real = np.random.randn(100, 10)
        synthetic = np.random.randn(100, 10)

        results = evaluator.full_evaluation(real, synthetic)

        # Check structure
        assert 'distribution' in results
        assert 'pca' in results
        assert 'autocorrelation' in results
        assert 'ks_test' in results['distribution']
        assert 'explained_variance_real' in results['pca']


class TestIntegration:
    """Integration tests for GANs."""

    def test_gan_pipeline(self):
        """Test complete GAN pipeline."""
        # Create data
        np.random.seed(42)
        real_data = np.random.randn(200, 15)

        # Create and train GAN
        gan = GAN(latent_dim=10, data_dim=15)
        history = gan.train(real_data, epochs=10, batch_size=32, verbose=False)

        # Generate synthetic data
        synthetic_data = gan.generate(n_samples=200)

        # Evaluate
        evaluator = SyntheticDataEvaluator()
        results = evaluator.full_evaluation(real_data, synthetic_data)

        # Check shapes
        assert synthetic_data.shape == real_data.shape
        assert 'distribution' in results

    def test_timegan_pipeline(self):
        """Test complete TimeGAN pipeline."""
        # Create time series data
        np.random.seed(42)
        real_sequences = np.random.randn(100, 20, 5)

        # Create and train TimeGAN
        tgan = TimeGAN(seq_length=20, n_features=5, hidden_dim=16, latent_dim=16)
        history = tgan.train(real_sequences, epochs=5, batch_size=16, verbose=False)

        # Generate synthetic sequences
        synthetic_sequences = tgan.generate(n_sequences=100)

        # Check shapes
        assert synthetic_sequences.shape == real_sequences.shape
        assert len(history['recon_loss']) == 5

    def test_evaluation_workflow(self):
        """Test evaluation of GAN-generated data."""
        # Train GAN
        np.random.seed(42)
        real_data = np.random.randn(300, 20)

        gan = GAN(latent_dim=15, data_dim=20)
        gan.train(real_data, epochs=15, batch_size=32, verbose=False)

        # Generate synthetic data
        synthetic_data = gan.generate(n_samples=300)

        # Evaluate quality
        evaluator = SyntheticDataEvaluator()

        # Test individual components
        dist_results = evaluator.compare_distributions(real_data, synthetic_data)
        pca_results = evaluator.compare_pca(real_data, synthetic_data)

        # Should have results
        assert 'avg_ks_statistic' in dist_results
        assert 'avg_pc_distance' in pca_results

        # Full evaluation
        full_results = evaluator.full_evaluation(real_data, synthetic_data)
        assert 'distribution' in full_results
        assert 'pca' in full_results
