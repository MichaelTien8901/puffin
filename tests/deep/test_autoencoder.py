"""Tests for autoencoder architectures."""

import numpy as np
import pytest
import torch

from puffin.deep.autoencoder import (
    Autoencoder,
    DenoisingAutoencoder,
    VAE,
    ConditionalAutoencoder,
    AETrainer
)


class TestAutoencoder:
    """Test standard autoencoder."""

    def test_initialization(self):
        """Test autoencoder initialization."""
        model = Autoencoder(input_dim=50, encoding_dim=10, hidden_dims=[32, 16])
        assert model.input_dim == 50
        assert model.encoding_dim == 10
        assert model.hidden_dims == [32, 16]

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = Autoencoder(input_dim=50, encoding_dim=10)
        x = torch.randn(8, 50)
        output = model(x)
        assert output.shape == (8, 50)

    def test_encode_shape(self):
        """Test encoder output shape."""
        model = Autoencoder(input_dim=50, encoding_dim=10)
        x = torch.randn(8, 50)
        latent = model.encode(x)
        assert latent.shape == (8, 10)

    def test_decode_shape(self):
        """Test decoder output shape."""
        model = Autoencoder(input_dim=50, encoding_dim=10)
        latent = torch.randn(8, 10)
        output = model.decode(latent)
        assert output.shape == (8, 50)

    def test_reconstruction_quality(self):
        """Test that model can reconstruct simple patterns."""
        torch.manual_seed(42)
        np.random.seed(42)
        model = Autoencoder(input_dim=20, encoding_dim=10, hidden_dims=[15])
        trainer = AETrainer()

        # Create simple data with pattern
        X = np.random.randn(100, 20)

        # Train
        history = trainer.fit(model, X, epochs=50, verbose=False)

        # Check that loss decreases
        assert history['train_loss'][-1] < history['train_loss'][0]


class TestDenoisingAutoencoder:
    """Test denoising autoencoder."""

    def test_initialization(self):
        """Test denoising autoencoder initialization."""
        model = DenoisingAutoencoder(input_dim=50, encoding_dim=10, noise_factor=0.3)
        assert model.input_dim == 50
        assert model.encoding_dim == 10
        assert model.noise_factor == 0.3

    def test_forward_with_noise(self):
        """Test forward pass with noise."""
        model = DenoisingAutoencoder(input_dim=50, encoding_dim=10, noise_factor=0.3)
        model.train()
        x = torch.randn(8, 50)
        output = model(x, add_noise=True)
        assert output.shape == (8, 50)

    def test_forward_without_noise(self):
        """Test forward pass without noise."""
        model = DenoisingAutoencoder(input_dim=50, encoding_dim=10, noise_factor=0.3)
        model.eval()
        x = torch.randn(8, 50)
        output = model(x, add_noise=False)
        assert output.shape == (8, 50)

    def test_training_improves(self):
        """Test that training reduces loss."""
        model = DenoisingAutoencoder(input_dim=20, encoding_dim=10)
        trainer = AETrainer()

        np.random.seed(42)
        X = np.random.randn(100, 20)

        history = trainer.fit(model, X, epochs=20, verbose=False)
        assert history['train_loss'][-1] < history['train_loss'][0]


class TestVAE:
    """Test variational autoencoder."""

    def test_initialization(self):
        """Test VAE initialization."""
        model = VAE(input_dim=50, latent_dim=10, hidden_dims=[32, 16])
        assert model.input_dim == 50
        assert model.latent_dim == 10
        assert model.hidden_dims == [32, 16]

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        model = VAE(input_dim=50, latent_dim=10)
        x = torch.randn(8, 50)
        recon, mu, log_var = model(x)
        assert recon.shape == (8, 50)
        assert mu.shape == (8, 10)
        assert log_var.shape == (8, 10)

    def test_encode_shape(self):
        """Test encoder output shapes."""
        model = VAE(input_dim=50, latent_dim=10)
        x = torch.randn(8, 50)
        mu, log_var = model.encode(x)
        assert mu.shape == (8, 10)
        assert log_var.shape == (8, 10)

    def test_reparameterize(self):
        """Test reparameterization trick."""
        model = VAE(input_dim=50, latent_dim=10)
        mu = torch.randn(8, 10)
        log_var = torch.randn(8, 10)
        z = model.reparameterize(mu, log_var)
        assert z.shape == (8, 10)

    def test_decode_shape(self):
        """Test decoder output shape."""
        model = VAE(input_dim=50, latent_dim=10)
        z = torch.randn(8, 10)
        output = model.decode(z)
        assert output.shape == (8, 50)

    def test_sample(self):
        """Test sampling from learned distribution."""
        model = VAE(input_dim=50, latent_dim=10)
        samples = model.sample(n=5)
        assert samples.shape == (5, 50)

    def test_vae_training(self):
        """Test VAE training with VAE loss."""
        model = VAE(input_dim=20, latent_dim=10, hidden_dims=[15])
        trainer = AETrainer()

        np.random.seed(42)
        X = np.random.randn(100, 20)

        history = trainer.fit(model, X, epochs=20, verbose=False)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 20


class TestConditionalAutoencoder:
    """Test conditional autoencoder."""

    def test_initialization(self):
        """Test conditional autoencoder initialization."""
        model = ConditionalAutoencoder(input_dim=50, condition_dim=5, encoding_dim=10)
        assert model.input_dim == 50
        assert model.condition_dim == 5
        assert model.encoding_dim == 10

    def test_forward_shape(self):
        """Test forward pass with conditioning."""
        model = ConditionalAutoencoder(input_dim=50, condition_dim=5, encoding_dim=10)
        x = torch.randn(8, 50)
        condition = torch.randn(8, 5)
        output = model(x, condition)
        assert output.shape == (8, 50)

    def test_different_conditions(self):
        """Test that different conditions produce different outputs."""
        model = ConditionalAutoencoder(input_dim=50, condition_dim=5, encoding_dim=10)
        x = torch.randn(8, 50)
        condition1 = torch.randn(8, 5)
        condition2 = torch.randn(8, 5)

        output1 = model(x, condition1)
        output2 = model(x, condition2)

        # Outputs should be different for different conditions
        assert not torch.allclose(output1, output2, atol=1e-5)


class TestAETrainer:
    """Test autoencoder trainer."""

    def test_initialization(self):
        """Test trainer initialization."""
        trainer = AETrainer()
        assert trainer.device is not None

    def test_fit_autoencoder(self):
        """Test training standard autoencoder."""
        model = Autoencoder(input_dim=20, encoding_dim=10)
        trainer = AETrainer()

        np.random.seed(42)
        X = np.random.randn(100, 20)

        history = trainer.fit(model, X, epochs=10, verbose=False)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 10

    def test_fit_vae(self):
        """Test training VAE."""
        model = VAE(input_dim=20, latent_dim=10)
        trainer = AETrainer()

        np.random.seed(42)
        X = np.random.randn(100, 20)

        history = trainer.fit(model, X, epochs=10, verbose=False)
        assert 'train_loss' in history
        assert 'val_loss' in history

    def test_extract_features_autoencoder(self):
        """Test feature extraction from autoencoder."""
        model = Autoencoder(input_dim=20, encoding_dim=10)
        trainer = AETrainer()

        np.random.seed(42)
        X = np.random.randn(100, 20)

        # Train first
        trainer.fit(model, X, epochs=5, verbose=False)

        # Extract features
        features = trainer.extract_features(model, X)
        assert features.shape == (100, 10)

    def test_extract_features_vae(self):
        """Test feature extraction from VAE."""
        model = VAE(input_dim=20, latent_dim=10)
        trainer = AETrainer()

        np.random.seed(42)
        X = np.random.randn(100, 20)

        # Train first
        trainer.fit(model, X, epochs=5, verbose=False)

        # Extract features (mean of latent distribution)
        features = trainer.extract_features(model, X)
        assert features.shape == (100, 10)

    def test_vae_loss(self):
        """Test VAE loss computation."""
        recon = torch.randn(8, 50)
        x = torch.randn(8, 50)
        mu = torch.randn(8, 10)
        log_var = torch.randn(8, 10)

        loss = AETrainer.vae_loss(recon, x, mu, log_var)
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0


class TestIntegration:
    """Integration tests for autoencoders."""

    def test_full_pipeline_standard_ae(self):
        """Test complete pipeline with standard autoencoder."""
        # Create data
        np.random.seed(42)
        X = np.random.randn(200, 30)

        # Create and train model
        model = Autoencoder(input_dim=30, encoding_dim=15, hidden_dims=[25, 20])
        trainer = AETrainer()

        history = trainer.fit(model, X, epochs=15, lr=0.001, verbose=False)

        # Check training worked
        assert history['train_loss'][-1] < history['train_loss'][0]

        # Extract features
        features = trainer.extract_features(model, X)
        assert features.shape == (200, 15)

    def test_full_pipeline_vae(self):
        """Test complete pipeline with VAE."""
        # Create data
        np.random.seed(42)
        X = np.random.randn(200, 30)

        # Create and train model
        model = VAE(input_dim=30, latent_dim=15, hidden_dims=[25, 20])
        trainer = AETrainer()

        history = trainer.fit(model, X, epochs=15, lr=0.001, verbose=False)

        # Check training worked
        assert 'train_loss' in history

        # Extract features
        features = trainer.extract_features(model, X)
        assert features.shape == (200, 15)

        # Generate samples
        samples = model.sample(n=10)
        assert samples.shape == (10, 30)
