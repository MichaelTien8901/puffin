"""Autoencoder architectures for trading feature extraction and generation."""

from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class Autoencoder(nn.Module):
    """Standard feedforward autoencoder for dimensionality reduction."""

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 32,
        hidden_dims: List[int] = [128, 64]
    ):
        """
        Initialize autoencoder.

        Args:
            input_dim: Number of input features.
            encoding_dim: Dimension of the encoded representation.
            hidden_dims: List of hidden layer dimensions in encoder.
        """
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (reverse architecture)
        decoder_layers = []
        prev_dim = encoding_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Encoded tensor of shape (batch_size, encoding_dim).
        """
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            latent: Latent tensor of shape (batch_size, encoding_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction


class DenoisingAutoencoder(nn.Module):
    """Denoising autoencoder that adds noise during training for robustness."""

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 32,
        noise_factor: float = 0.3
    ):
        """
        Initialize denoising autoencoder.

        Args:
            input_dim: Number of input features.
            encoding_dim: Dimension of the encoded representation.
            noise_factor: Standard deviation of Gaussian noise to add during training.
        """
        super(DenoisingAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.noise_factor = noise_factor

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """
        Forward pass with optional noise addition.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            add_noise: Whether to add Gaussian noise to input (for training).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        if add_noise and self.training:
            noise = torch.randn_like(x) * self.noise_factor
            x_noisy = x + noise
        else:
            x_noisy = x

        latent = self.encoder(x_noisy)
        reconstruction = self.decoder(latent)
        return reconstruction


class VAE(nn.Module):
    """Variational autoencoder for generative modeling."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: List[int] = [128, 64]
    ):
        """
        Initialize variational autoencoder.

        Args:
            input_dim: Number of input features.
            latent_dim: Dimension of the latent space.
            hidden_dims: List of hidden layer dimensions.
        """
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.encoder_layers = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of (mu, log_var) tensors, each of shape (batch_size, latent_dim).
        """
        h = self.encoder_layers(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from N(mu, var).

        Args:
            mu: Mean tensor of shape (batch_size, latent_dim).
            log_var: Log variance tensor of shape (batch_size, latent_dim).

        Returns:
            Sampled latent tensor of shape (batch_size, latent_dim).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        return self.decoder_layers(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of (reconstruction, mu, log_var).
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    def sample(self, n: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate samples from the learned distribution.

        Args:
            n: Number of samples to generate.
            device: Device to generate samples on.

        Returns:
            Generated samples of shape (n, input_dim).
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(n, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


class ConditionalAutoencoder(nn.Module):
    """Conditional autoencoder for asset pricing with external conditions."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        encoding_dim: int = 32
    ):
        """
        Initialize conditional autoencoder.

        Args:
            input_dim: Number of input features.
            condition_dim: Number of conditioning features.
            encoding_dim: Dimension of the encoded representation.
        """
        super(ConditionalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.encoding_dim = encoding_dim

        # Encoder (takes input + condition)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )

        # Decoder (takes latent + condition)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim + condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditioning.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            condition: Condition tensor of shape (batch_size, condition_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        # Encode with condition
        x_cond = torch.cat([x, condition], dim=1)
        latent = self.encoder(x_cond)

        # Decode with condition
        latent_cond = torch.cat([latent, condition], dim=1)
        reconstruction = self.decoder(latent_cond)
        return reconstruction


class AETrainer:
    """Trainer class for autoencoder models."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize autoencoder trainer.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def fit(
        self,
        model: nn.Module,
        X: np.ndarray,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 64,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train an autoencoder model.

        Args:
            model: Autoencoder model to train.
            X: Training data of shape (n_samples, n_features).
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size for training.
            validation_split: Fraction of data to use for validation.
            verbose: Whether to print training progress.

        Returns:
            Dictionary containing training history with keys:
            - 'train_loss': List of training losses per epoch.
            - 'val_loss': List of validation losses per epoch.
        """
        model = model.to(self.device)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)

        # Split into train and validation
        n_samples = len(X_tensor)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        X_train = X_tensor[train_indices]
        X_val = X_tensor[val_indices]

        # Create data loaders
        train_dataset = TensorDataset(X_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Determine if this is a VAE
        is_vae = isinstance(model, VAE)

        # Training loop
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []

            for (batch_X,) in train_loader:
                batch_X = batch_X.to(self.device)

                optimizer.zero_grad()

                if is_vae:
                    recon, mu, log_var = model(batch_X)
                    loss = self.vae_loss(recon, batch_X, mu, log_var)
                else:
                    recon = model(batch_X)
                    loss = nn.MSELoss()(recon, batch_X)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            model.eval()
            val_losses = []

            with torch.no_grad():
                for (batch_X,) in val_loader:
                    batch_X = batch_X.to(self.device)

                    if is_vae:
                        recon, mu, log_var = model(batch_X)
                        loss = self.vae_loss(recon, batch_X, mu, log_var)
                    else:
                        recon = model(batch_X)
                        loss = nn.MSELoss()(recon, batch_X)

                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}")

        return history

    def extract_features(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """
        Extract encoded features from trained autoencoder.

        Args:
            model: Trained autoencoder model.
            X: Input data of shape (n_samples, n_features).

        Returns:
            Encoded features of shape (n_samples, encoding_dim).
        """
        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            if isinstance(model, VAE):
                mu, _ = model.encode(X_tensor)
                features = mu.cpu().numpy()
            elif isinstance(model, (Autoencoder, DenoisingAutoencoder)):
                features = model.encoder(X_tensor).cpu().numpy()
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

        return features

    @staticmethod
    def vae_loss(
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Compute VAE loss (reconstruction + KL divergence).

        Args:
            recon: Reconstructed output.
            x: Original input.
            mu: Mean of latent distribution.
            log_var: Log variance of latent distribution.
            beta: Weight for KL divergence term (beta-VAE).

        Returns:
            Total VAE loss.
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss(reduction='sum')(recon, x)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        total_loss = recon_loss + beta * kl_loss
        return total_loss / x.size(0)  # Average over batch
