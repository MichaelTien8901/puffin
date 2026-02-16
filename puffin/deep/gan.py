"""Generative Adversarial Networks for synthetic financial data generation."""

from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats


class Generator(nn.Module):
    """Generator network for GAN."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 256]
    ):
        """
        Initialize generator.

        Args:
            latent_dim: Dimension of latent noise vector.
            output_dim: Dimension of generated data.
            hidden_dims: List of hidden layer dimensions.
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Build network
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate data from noise.

        Args:
            z: Latent noise tensor of shape (batch_size, latent_dim).

        Returns:
            Generated data of shape (batch_size, output_dim).
        """
        return self.network(z)


class Discriminator(nn.Module):
    """Discriminator network for GAN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128]
    ):
        """
        Initialize discriminator.

        Args:
            input_dim: Dimension of input data.
            hidden_dims: List of hidden layer dimensions.
        """
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Probability of being real

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify data as real or fake.

        Args:
            x: Input data of shape (batch_size, input_dim).

        Returns:
            Probability of being real, shape (batch_size, 1).
        """
        return self.network(x)


class GAN:
    """Generative Adversarial Network for synthetic data generation."""

    def __init__(
        self,
        latent_dim: int,
        data_dim: int,
        device: Optional[str] = None
    ):
        """
        Initialize GAN.

        Args:
            latent_dim: Dimension of latent noise vector.
            data_dim: Dimension of real/generated data.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        # Create generator and discriminator
        self.generator = Generator(
            latent_dim=latent_dim,
            output_dim=data_dim,
            hidden_dims=[128, 256]
        ).to(self.device)

        self.discriminator = Discriminator(
            input_dim=data_dim,
            hidden_dims=[256, 128]
        ).to(self.device)

    def train(
        self,
        real_data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.0002,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the GAN.

        Args:
            real_data: Real training data of shape (n_samples, data_dim).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            lr: Learning rate.
            verbose: Whether to print training progress.

        Returns:
            Dictionary containing training history with keys:
            - 'g_loss': Generator losses per epoch.
            - 'd_loss': Discriminator losses per epoch.
        """
        # Convert to tensor
        real_data_tensor = torch.FloatTensor(real_data).to(self.device)

        # Create data loader
        dataset = TensorDataset(real_data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizers
        g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        # Loss function
        criterion = nn.BCELoss()

        # Training loop
        history = {'g_loss': [], 'd_loss': []}

        for epoch in range(epochs):
            g_losses = []
            d_losses = []

            for (real_batch,) in dataloader:
                batch_size_actual = real_batch.size(0)

                # Labels
                real_labels = torch.ones(batch_size_actual, 1).to(self.device)
                fake_labels = torch.zeros(batch_size_actual, 1).to(self.device)

                # Train Discriminator
                d_optimizer.zero_grad()

                # Real data
                real_output = self.discriminator(real_batch)
                d_loss_real = criterion(real_output, real_labels)

                # Fake data
                z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = criterion(fake_output, fake_labels)

                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()

                # Train Generator
                g_optimizer.zero_grad()

                z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data)

                # Generator wants discriminator to classify fake as real
                g_loss = criterion(fake_output, real_labels)
                g_loss.backward()
                g_optimizer.step()

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

            # Record epoch losses
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            history['g_loss'].append(avg_g_loss)
            history['d_loss'].append(avg_d_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"G Loss: {avg_g_loss:.6f}, "
                      f"D Loss: {avg_d_loss:.6f}")

        return history

    def generate(self, n_samples: int = 100) -> np.ndarray:
        """
        Generate synthetic data.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Generated data of shape (n_samples, data_dim).
        """
        self.generator.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            fake_data = self.generator(z).cpu().numpy()

        return fake_data


class TimeGAN:
    """Time-series GAN for financial data with temporal dependencies."""

    def __init__(
        self,
        seq_length: int,
        n_features: int,
        hidden_dim: int = 24,
        latent_dim: int = 24,
        device: Optional[str] = None
    ):
        """
        Initialize TimeGAN.

        Args:
            seq_length: Length of time series sequences.
            n_features: Number of features per timestep.
            hidden_dim: Dimension of hidden states in RNNs.
            latent_dim: Dimension of latent representation.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.seq_length = seq_length
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Embedder: Maps real data to latent space
        self.embedder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        ).to(self.device)

        self.embed_fc = nn.Linear(hidden_dim, latent_dim).to(self.device)

        # Recovery: Maps latent space back to data space
        self.recovery = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features)
        ).to(self.device)

        # Generator: Generates latent sequences from noise
        self.generator = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        ).to(self.device)

        self.gen_fc = nn.Linear(hidden_dim, latent_dim).to(self.device)

        # Discriminator: Classifies real vs fake latent sequences
        self.discriminator = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        ).to(self.device)

        self.disc_fc = nn.Linear(hidden_dim, 1).to(self.device)

        # Supervisor: Models temporal dynamics in latent space
        self.supervisor = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        ).to(self.device)

        self.sup_fc = nn.Linear(hidden_dim, latent_dim).to(self.device)

    def train(
        self,
        real_sequences: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train TimeGAN.

        Args:
            real_sequences: Real time series data of shape (n_samples, seq_length, n_features).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            lr: Learning rate.
            verbose: Whether to print training progress.

        Returns:
            Dictionary containing training history.
        """
        # Convert to tensor
        real_data = torch.FloatTensor(real_sequences).to(self.device)

        # Create data loader
        dataset = TensorDataset(real_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizers
        e_optimizer = optim.Adam(
            list(self.embedder.parameters()) + list(self.embed_fc.parameters()),
            lr=lr
        )
        r_optimizer = optim.Adam(self.recovery.parameters(), lr=lr)
        g_optimizer = optim.Adam(
            list(self.generator.parameters()) + list(self.gen_fc.parameters()) +
            list(self.supervisor.parameters()) + list(self.sup_fc.parameters()),
            lr=lr
        )
        d_optimizer = optim.Adam(
            list(self.discriminator.parameters()) + list(self.disc_fc.parameters()),
            lr=lr
        )

        # Loss
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        history = {'recon_loss': [], 'g_loss': [], 'd_loss': [], 'sup_loss': []}

        for epoch in range(epochs):
            recon_losses = []
            g_losses = []
            d_losses = []
            sup_losses = []

            for (real_batch,) in dataloader:
                batch_size_actual = real_batch.size(0)

                # Phase 1: Autoencoding (Embedder + Recovery)
                e_optimizer.zero_grad()
                r_optimizer.zero_grad()

                # Embed
                h_real, _ = self.embedder(real_batch)
                e_real = self.embed_fc(h_real)

                # Recover
                x_recon = self.recovery(e_real)

                # Reconstruction loss
                recon_loss = mse_loss(x_recon, real_batch)
                recon_loss.backward()

                e_optimizer.step()
                r_optimizer.step()

                recon_losses.append(recon_loss.item())

                # Phase 2: Supervised learning (Supervisor)
                g_optimizer.zero_grad()

                h_real, _ = self.embedder(real_batch.detach())
                e_real = self.embed_fc(h_real)

                h_sup, _ = self.supervisor(e_real[:, :-1, :])
                e_sup = self.sup_fc(h_sup)

                # Supervisor loss (predict next latent state)
                sup_loss = mse_loss(e_sup, e_real[:, 1:, :])
                sup_loss.backward()

                g_optimizer.step()
                sup_losses.append(sup_loss.item())

                # Phase 3: GAN training
                # Train Discriminator
                d_optimizer.zero_grad()

                # Real latent sequences
                h_real, _ = self.embedder(real_batch.detach())
                e_real = self.embed_fc(h_real)

                # Fake latent sequences
                z = torch.randn(batch_size_actual, self.seq_length, self.latent_dim).to(self.device)
                h_fake, _ = self.generator(z)
                e_fake = self.gen_fc(h_fake)

                # Discriminate
                h_d_real, _ = self.discriminator(e_real.detach())
                d_real = self.disc_fc(h_d_real[:, -1, :])

                h_d_fake, _ = self.discriminator(e_fake.detach())
                d_fake = self.disc_fc(h_d_fake[:, -1, :])

                # Discriminator loss
                d_loss_real = bce_loss(d_real, torch.ones_like(d_real))
                d_loss_fake = bce_loss(d_fake, torch.zeros_like(d_fake))
                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                d_optimizer.step()

                d_losses.append(d_loss.item())

                # Train Generator
                g_optimizer.zero_grad()

                z = torch.randn(batch_size_actual, self.seq_length, self.latent_dim).to(self.device)
                h_fake, _ = self.generator(z)
                e_fake = self.gen_fc(h_fake)

                h_d_fake, _ = self.discriminator(e_fake)
                d_fake = self.disc_fc(h_d_fake[:, -1, :])

                # Generator loss (fool discriminator)
                g_loss = bce_loss(d_fake, torch.ones_like(d_fake))
                g_loss.backward()

                g_optimizer.step()
                g_losses.append(g_loss.item())

            # Record epoch losses
            history['recon_loss'].append(np.mean(recon_losses))
            history['g_loss'].append(np.mean(g_losses))
            history['d_loss'].append(np.mean(d_losses))
            history['sup_loss'].append(np.mean(sup_losses))

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Recon: {history['recon_loss'][-1]:.6f}, "
                      f"G: {history['g_loss'][-1]:.6f}, "
                      f"D: {history['d_loss'][-1]:.6f}, "
                      f"Sup: {history['sup_loss'][-1]:.6f}")

        return history

    def generate(self, n_sequences: int = 100) -> np.ndarray:
        """
        Generate synthetic time series.

        Args:
            n_sequences: Number of sequences to generate.

        Returns:
            Generated sequences of shape (n_sequences, seq_length, n_features).
        """
        self.generator.eval()
        self.gen_fc.eval()
        self.recovery.eval()

        with torch.no_grad():
            # Generate latent sequences
            z = torch.randn(n_sequences, self.seq_length, self.latent_dim).to(self.device)
            h_fake, _ = self.generator(z)
            e_fake = self.gen_fc(h_fake)

            # Recover to data space
            x_fake = self.recovery(e_fake)

        return x_fake.cpu().numpy()


class SyntheticDataEvaluator:
    """Evaluator for comparing real and synthetic data quality."""

    def __init__(self):
        """Initialize synthetic data evaluator."""
        pass

    def compare_distributions(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare marginal distributions using statistical tests.

        Args:
            real: Real data of shape (n_samples, n_features).
            synthetic: Synthetic data of shape (n_samples, n_features).

        Returns:
            Dictionary with test results for each feature.
        """
        n_features = real.shape[1]
        results = {
            'ks_test': [],
            'mean_diff': [],
            'std_diff': []
        }

        for i in range(n_features):
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(real[:, i], synthetic[:, i])
            results['ks_test'].append({'statistic': ks_stat, 'p_value': ks_pval})

            # Mean difference
            mean_diff = np.abs(np.mean(real[:, i]) - np.mean(synthetic[:, i]))
            results['mean_diff'].append(mean_diff)

            # Std difference
            std_diff = np.abs(np.std(real[:, i]) - np.std(synthetic[:, i]))
            results['std_diff'].append(std_diff)

        # Average across features
        results['avg_ks_statistic'] = np.mean([r['statistic'] for r in results['ks_test']])
        results['avg_ks_pvalue'] = np.mean([r['p_value'] for r in results['ks_test']])
        results['avg_mean_diff'] = np.mean(results['mean_diff'])
        results['avg_std_diff'] = np.mean(results['std_diff'])

        return results

    def compare_autocorrelation(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
        lags: int = 20
    ) -> Dict[str, Any]:
        """
        Compare autocorrelation structure.

        Args:
            real: Real time series data.
            synthetic: Synthetic time series data.
            lags: Number of lags to compute.

        Returns:
            Dictionary with autocorrelation comparison.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        def compute_acf(x, lags):
            """Compute autocorrelation function."""
            acf = []
            mean = np.mean(x)
            var = np.var(x)

            for lag in range(lags + 1):
                if lag == 0:
                    acf.append(1.0)
                else:
                    cov = np.mean((x[:-lag] - mean) * (x[lag:] - mean))
                    acf.append(cov / var if var > 0 else 0)

            return np.array(acf)

        # Compute ACF for each feature
        n_features = real.shape[1] if real.ndim > 1 else 1

        if real.ndim == 1:
            real = real.reshape(-1, 1)
            synthetic = synthetic.reshape(-1, 1)

        acf_diffs = []
        for i in range(n_features):
            real_acf = compute_acf(real[:, i], lags)
            synthetic_acf = compute_acf(synthetic[:, i], lags)
            acf_diff = np.mean(np.abs(real_acf - synthetic_acf))
            acf_diffs.append(acf_diff)

        return {
            'avg_acf_diff': np.mean(acf_diffs),
            'max_acf_diff': np.max(acf_diffs),
            'acf_diffs_per_feature': acf_diffs
        }

    def compare_pca(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
        n_components: int = 5
    ) -> Dict[str, Any]:
        """
        Compare PCA representations.

        Args:
            real: Real data.
            synthetic: Synthetic data.
            n_components: Number of principal components.

        Returns:
            Dictionary with PCA comparison.
        """
        from sklearn.decomposition import PCA

        # Fit PCA on real data
        pca = PCA(n_components=n_components)
        pca.fit(real)

        # Transform both datasets
        real_transformed = pca.transform(real)
        synthetic_transformed = pca.transform(synthetic)

        # Compare explained variance
        real_var = np.var(real_transformed, axis=0)
        synthetic_var = np.var(synthetic_transformed, axis=0)
        var_diff = np.abs(real_var - synthetic_var)

        # Compare distributions in PC space
        pc_dists = []
        for i in range(n_components):
            ks_stat, _ = stats.ks_2samp(real_transformed[:, i], synthetic_transformed[:, i])
            pc_dists.append(ks_stat)

        return {
            'explained_variance_real': pca.explained_variance_ratio_.tolist(),
            'variance_diff': var_diff.tolist(),
            'avg_variance_diff': np.mean(var_diff),
            'pc_distribution_distances': pc_dists,
            'avg_pc_distance': np.mean(pc_dists)
        }

    def full_evaluation(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of synthetic data quality.

        Args:
            real: Real data.
            synthetic: Synthetic data.

        Returns:
            Dictionary with all evaluation metrics.
        """
        results = {
            'distribution': self.compare_distributions(real, synthetic),
            'pca': self.compare_pca(real, synthetic)
        }

        # Add autocorrelation if data has temporal structure
        if len(real) > 50:
            results['autocorrelation'] = self.compare_autocorrelation(real, synthetic)

        return results
