"""1D Convolutional Neural Networks for time series prediction."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class Conv1DNet(nn.Module):
    """1D CNN for autoregressive time-series prediction.

    Architecture:
    - Multiple 1D convolutional layers with ReLU activation
    - Max pooling for dimension reduction
    - Fully connected layers for final prediction
    """

    def __init__(
        self,
        input_channels: int,
        seq_length: int,
        n_filters: list = None,
        kernel_sizes: list = None,
        output_dim: int = 1,
    ):
        """Initialize Conv1DNet.

        Args:
            input_channels: Number of input features/channels.
            seq_length: Length of input sequences.
            n_filters: List of filter counts for each conv layer.
            kernel_sizes: List of kernel sizes for each conv layer.
            output_dim: Dimension of output (1 for regression, n_classes for classification).
        """
        super(Conv1DNet, self).__init__()

        if n_filters is None:
            n_filters = [32, 64]
        if kernel_sizes is None:
            kernel_sizes = [3, 3]

        if len(n_filters) != len(kernel_sizes):
            raise ValueError("n_filters and kernel_sizes must have same length")

        self.input_channels = input_channels
        self.seq_length = seq_length
        self.output_dim = output_dim

        # Build convolutional layers
        layers = []
        in_channels = input_channels
        current_length = seq_length

        for n_filter, kernel_size in zip(n_filters, kernel_sizes):
            # Conv layer
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_filter,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(n_filter))

            # Max pooling
            layers.append(nn.MaxPool1d(kernel_size=2))
            current_length = current_length // 2

            in_channels = n_filter

        self.conv_layers = nn.Sequential(*layers)

        # Calculate flattened size
        self.flat_size = n_filters[-1] * current_length

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_channels).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)

        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x


class TradingCNN:
    """Wrapper class for trading with 1D CNN models.

    Provides high-level interface for training and prediction on financial time series.
    """

    def __init__(
        self,
        input_channels: int = 1,
        seq_length: int = 20,
        n_filters: list = None,
        kernel_sizes: list = None,
        output_dim: int = 1,
        device: str = None,
    ):
        """Initialize TradingCNN.

        Args:
            input_channels: Number of input features.
            seq_length: Length of input sequences.
            n_filters: List of filter counts for each conv layer.
            kernel_sizes: List of kernel sizes for each conv layer.
            output_dim: Output dimension (1 for regression, n for classification).
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = Conv1DNet(
            input_channels=input_channels,
            seq_length=seq_length,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            output_dim=output_dim,
        ).to(self.device)

        self.input_channels = input_channels
        self.seq_length = seq_length
        self.output_dim = output_dim

    def fit(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> dict:
        """Train the CNN model.

        Args:
            X_seq: Sequential input data of shape (n_samples, seq_length, n_features).
            y: Target values of shape (n_samples,) or (n_samples, output_dim).
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size for training.
            validation_split: Fraction of data to use for validation.

        Returns:
            Dict containing training history (losses per epoch).
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Reshape y if needed
        if len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(1)

        # Train/validation split
        n_samples = len(X_tensor)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
        y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        if self.output_dim == 1:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
        }

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_train_loss)

            # Validation
            if n_val > 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    history['val_loss'].append(val_loss)
                self.model.train()

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                if n_val > 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        return history

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            X_seq: Sequential input data of shape (n_samples, seq_length, n_features).

        Returns:
            Predictions as numpy array.
        """
        self.model.eval()

        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = outputs.cpu().numpy()

        return predictions.squeeze()

    @staticmethod
    def prepare_sequences(
        X: np.ndarray,
        lookback: int = 20,
    ) -> tuple:
        """Prepare sequences for autoregressive prediction.

        Creates sliding windows of lookback periods, where each sequence
        predicts the next value.

        Args:
            X: Input data of shape (n_samples, n_features).
            lookback: Number of time steps to look back.

        Returns:
            Tuple of (X_seq, y) where:
            - X_seq: shape (n_samples - lookback, lookback, n_features)
            - y: shape (n_samples - lookback,) - next value to predict
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        X_seq = []
        y = []

        for i in range(lookback, n_samples):
            X_seq.append(X[i - lookback:i])
            # Predict the next value (first feature)
            y.append(X[i, 0])

        return np.array(X_seq), np.array(y)

    def save(self, path: str):
        """Save model weights.

        Args:
            path: Path to save the model.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_channels': self.input_channels,
            'seq_length': self.seq_length,
            'output_dim': self.output_dim,
        }, path)

    def load(self, path: str):
        """Load model weights.

        Args:
            path: Path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
