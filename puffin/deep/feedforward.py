"""Feedforward neural networks for trading with PyTorch."""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class FeedforwardNet(nn.Module):
    """PyTorch feedforward neural network for trading predictions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize feedforward neural network.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Number of output units (1 for regression, n for classification).
            dropout: Dropout probability (0 to 1).
            activation: Activation function ('relu', 'tanh', 'leaky_relu', 'elu').
        """
        super(FeedforwardNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_p = dropout
        self.activation_name = activation

        # Build activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation for regression, add softmax/sigmoid externally if needed)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)


class TradingFFN:
    """Wrapper class for training and using feedforward networks for trading."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        dropout: float = 0.3,
        activation: str = 'relu',
        device: Optional[str] = None
    ):
        """
        Initialize trading feedforward network.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Number of output units.
            dropout: Dropout probability.
            activation: Activation function name.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = FeedforwardNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation
        ).to(self.device)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.metadata: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the neural network.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,) or (n_samples, n_outputs).
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
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)

        # Handle y shape
        if y.ndim == 1:
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
        else:
            y_tensor = torch.FloatTensor(y)

        # Split into train and validation
        n_samples = len(X_tensor)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
        X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Training loop
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}")

        self.is_fitted = True

        # Store metadata
        self.metadata = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.model.hidden_dims,
            'dropout': self.model.dropout_p,
            'activation': self.model.activation_name,
            'n_samples': n_samples,
            'n_features': X.shape[1],
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
        }

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        # Squeeze if single output
        if self.output_dim == 1:
            predictions = predictions.squeeze()

        return predictions

    def save(self, path: str):
        """
        Save model and metadata to disk.

        Args:
            path: Directory path to save the model.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), path / 'model.pt')

        # Save metadata
        with open(path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TradingFFN':
        """
        Load a saved model.

        Args:
            path: Directory path containing the saved model.

        Returns:
            Loaded TradingFFN instance.
        """
        path = Path(path)

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(
            input_dim=metadata['input_dim'],
            hidden_dims=metadata['hidden_dims'],
            output_dim=metadata['output_dim'],
            dropout=metadata['dropout'],
            activation=metadata['activation']
        )

        # Load model state
        instance.model.load_state_dict(torch.load(path / 'model.pt'))
        instance.is_fitted = True
        instance.metadata = metadata

        return instance
