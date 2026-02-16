"""
RNN-based models for time series prediction in algorithmic trading.

This module provides LSTM and GRU implementations for both univariate
and multivariate time series forecasting.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, Optional, Union
import pandas as pd


class LSTMNet(nn.Module):
    """
    LSTM network for time series prediction.

    Parameters
    ----------
    input_dim : int, default=1
        Number of input features
    hidden_dim : int, default=64
        Number of hidden units in LSTM layers
    num_layers : int, default=2
        Number of stacked LSTM layers
    output_dim : int, default=1
        Number of output features
    dropout : float, default=0.2
        Dropout probability between LSTM layers
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get output from last time step
        out = self.fc(out[:, -1, :])

        return out


class TradingLSTM:
    """
    LSTM-based predictor for trading time series.

    Wraps LSTMNet with training and prediction functionality.
    """

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lookback = None
        self.mean = None
        self.std = None

    def prepare_sequences(
        self,
        series: np.ndarray,
        lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input-output sequences for training.

        Parameters
        ----------
        series : np.ndarray
            Time series data
        lookback : int
            Number of past time steps to use as input

        Returns
        -------
        X : np.ndarray
            Input sequences of shape (n_samples, lookback, 1)
        y : np.ndarray
            Target values of shape (n_samples,)
        """
        X, y = [], []
        for i in range(lookback, len(series)):
            X.append(series[i-lookback:i])
            y.append(series[i])

        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)

        return X, y

    def fit(
        self,
        series: np.ndarray,
        lookback: int = 20,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model on a time series.

        Parameters
        ----------
        series : np.ndarray
            Time series data
        lookback : int, default=20
            Number of past time steps to use as input
        epochs : int, default=50
            Number of training epochs
        lr : float, default=0.001
            Learning rate
        batch_size : int, default=32
            Batch size for training
        validation_split : float, default=0.2
            Fraction of data to use for validation

        Returns
        -------
        history : dict
            Training history with 'train_loss' and 'val_loss' keys
        """
        self.lookback = lookback

        # Normalize data
        self.mean = np.mean(series)
        self.std = np.std(series)
        series_norm = (series - self.mean) / (self.std + 1e-8)

        # Prepare sequences
        X, y = self.prepare_sequences(series_norm, lookback)

        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        # Initialize model
        self.model = LSTMNet(input_dim=1).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()

            # Mini-batch training
            train_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= (len(X_train) / batch_size)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val).item()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        return history

    def predict(
        self,
        series: np.ndarray,
        steps: int = 1
    ) -> np.ndarray:
        """
        Make predictions on future time steps.

        Parameters
        ----------
        series : np.ndarray
            Time series data (at least lookback points)
        steps : int, default=1
            Number of steps to predict ahead

        Returns
        -------
        predictions : np.ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()

        # Normalize input
        series_norm = (series - self.mean) / (self.std + 1e-8)

        predictions = []
        current_seq = series_norm[-self.lookback:].copy()

        with torch.no_grad():
            for _ in range(steps):
                # Prepare input
                X = torch.FloatTensor(current_seq.reshape(1, self.lookback, 1)).to(self.device)

                # Predict
                pred_norm = self.model(X).cpu().numpy()[0, 0]

                # Denormalize
                pred = pred_norm * self.std + self.mean
                predictions.append(pred)

                # Update sequence for next prediction
                current_seq = np.append(current_seq[1:], pred_norm)

        return np.array(predictions)


class StackedLSTM(nn.Module):
    """
    Stacked LSTM with configurable hidden dimensions per layer.

    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_dims : list of int, default=[64, 32]
        Hidden dimensions for each LSTM layer
    output_dim : int, default=1
        Number of output features
    dropout : float, default=0.2
        Dropout probability between layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super(StackedLSTM, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()

        for i in range(self.num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dims[i-1]
            self.lstm_layers.append(
                nn.LSTM(
                    layer_input_dim,
                    hidden_dims[i],
                    1,
                    batch_first=True
                )
            )

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacked LSTM layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        out = x

        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(out)

            # Apply dropout except for last layer
            if i < self.num_layers - 1:
                out = self.dropout(out)

        # Get output from last time step
        out = self.fc(out[:, -1, :])

        return out


class MultivariateLSTM:
    """
    LSTM for multivariate time series prediction.

    Predicts a target column based on multiple input features.
    """

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lookback = None
        self.feature_cols = None
        self.target_col = None
        self.means = None
        self.stds = None

    def prepare_sequences(
        self,
        features: np.ndarray,
        target: np.ndarray,
        lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare multivariate input sequences.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        target : np.ndarray
            Target values of shape (n_samples,)
        lookback : int
            Number of past time steps to use

        Returns
        -------
        X : np.ndarray
            Input sequences of shape (n_samples, lookback, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
        """
        X, y = [], []

        for i in range(lookback, len(features)):
            X.append(features[i-lookback:i])
            y.append(target[i])

        X = np.array(X)
        y = np.array(y)

        return X, y

    def fit(
        self,
        features_df: pd.DataFrame,
        target_col: str,
        lookback: int = 20,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2,
        hidden_dims: List[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the multivariate LSTM model.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame containing features and target
        target_col : str
            Name of the target column
        lookback : int, default=20
            Number of past time steps to use
        epochs : int, default=50
            Number of training epochs
        lr : float, default=0.001
            Learning rate
        batch_size : int, default=32
            Batch size for training
        validation_split : float, default=0.2
            Fraction of data for validation
        hidden_dims : list of int, optional
            Hidden dimensions for each layer

        Returns
        -------
        history : dict
            Training history
        """
        self.lookback = lookback
        self.target_col = target_col
        self.feature_cols = [col for col in features_df.columns if col != target_col]

        # Normalize data
        features = features_df[self.feature_cols].values
        target = features_df[target_col].values

        self.means = features.mean(axis=0)
        self.stds = features.std(axis=0) + 1e-8
        features_norm = (features - self.means) / self.stds

        target_mean = target.mean()
        target_std = target.std() + 1e-8
        target_norm = (target - target_mean) / target_std

        # Store target normalization parameters
        self.target_mean = target_mean
        self.target_std = target_std

        # Prepare sequences
        X, y = self.prepare_sequences(features_norm, target_norm, lookback)

        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        # Initialize model
        input_dim = len(self.feature_cols)
        self.model = StackedLSTM(
            input_dim=input_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()

            # Mini-batch training
            train_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= (len(X_train) / batch_size)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val).item()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        return history

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with feature columns (at least lookback rows)

        Returns
        -------
        predictions : np.ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()

        # Normalize features
        features = features_df[self.feature_cols].values
        features_norm = (features - self.means) / self.stds

        # Take last lookback points
        X = features_norm[-self.lookback:].reshape(1, self.lookback, -1)
        X = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(X).cpu().numpy()[0, 0]

        # Denormalize
        pred = pred_norm * self.target_std + self.target_mean

        return np.array([pred])


class GRUNet(nn.Module):
    """
    GRU network for time series prediction.

    Similar to LSTMNet but uses GRU cells instead of LSTM.
    GRU has fewer parameters and can be faster to train.

    Parameters
    ----------
    input_dim : int, default=1
        Number of input features
    hidden_dim : int, default=64
        Number of hidden units in GRU layers
    num_layers : int, default=2
        Number of stacked GRU layers
    output_dim : int, default=1
        Number of output features
    dropout : float, default=0.2
        Dropout probability between GRU layers
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)

        # Get output from last time step
        out = self.fc(out[:, -1, :])

        return out


class TradingGRU:
    """
    GRU-based predictor for trading time series.

    Provides the same interface as TradingLSTM but uses GRU cells.
    GRU can be more efficient and sometimes performs better than LSTM.
    """

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lookback = None
        self.mean = None
        self.std = None

    def prepare_sequences(
        self,
        series: np.ndarray,
        lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input-output sequences for training.

        Parameters
        ----------
        series : np.ndarray
            Time series data
        lookback : int
            Number of past time steps to use as input

        Returns
        -------
        X : np.ndarray
            Input sequences of shape (n_samples, lookback, 1)
        y : np.ndarray
            Target values of shape (n_samples,)
        """
        X, y = [], []
        for i in range(lookback, len(series)):
            X.append(series[i-lookback:i])
            y.append(series[i])

        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)

        return X, y

    def fit(
        self,
        series: np.ndarray,
        lookback: int = 20,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        Train the GRU model on a time series.

        Parameters
        ----------
        series : np.ndarray
            Time series data
        lookback : int, default=20
            Number of past time steps to use as input
        epochs : int, default=50
            Number of training epochs
        lr : float, default=0.001
            Learning rate
        batch_size : int, default=32
            Batch size for training
        validation_split : float, default=0.2
            Fraction of data to use for validation

        Returns
        -------
        history : dict
            Training history with 'train_loss' and 'val_loss' keys
        """
        self.lookback = lookback

        # Normalize data
        self.mean = np.mean(series)
        self.std = np.std(series)
        series_norm = (series - self.mean) / (self.std + 1e-8)

        # Prepare sequences
        X, y = self.prepare_sequences(series_norm, lookback)

        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        # Initialize model
        self.model = GRUNet(input_dim=1).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()

            # Mini-batch training
            train_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= (len(X_train) / batch_size)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val).item()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        return history

    def predict(
        self,
        series: np.ndarray,
        steps: int = 1
    ) -> np.ndarray:
        """
        Make predictions on future time steps.

        Parameters
        ----------
        series : np.ndarray
            Time series data (at least lookback points)
        steps : int, default=1
            Number of steps to predict ahead

        Returns
        -------
        predictions : np.ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()

        # Normalize input
        series_norm = (series - self.mean) / (self.std + 1e-8)

        predictions = []
        current_seq = series_norm[-self.lookback:].copy()

        with torch.no_grad():
            for _ in range(steps):
                # Prepare input
                X = torch.FloatTensor(current_seq.reshape(1, self.lookback, 1)).to(self.device)

                # Predict
                pred_norm = self.model(X).cpu().numpy()[0, 0]

                # Denormalize
                pred = pred_norm * self.std + self.mean
                predictions.append(pred)

                # Update sequence for next prediction
                current_seq = np.append(current_seq[1:], pred_norm)

        return np.array(predictions)
