"""Feedforward neural networks for trading with TensorFlow/Keras."""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

# Try to import TensorFlow, gracefully fail if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    models = None


def build_feedforward_tf(
    input_dim: int,
    hidden_dims: List[int] = [64, 32],
    output_dim: int = 1,
    dropout: float = 0.3,
    activation: str = 'relu'
) -> 'keras.Model':
    """
    Build a TensorFlow/Keras feedforward neural network.

    Args:
        input_dim: Number of input features.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Number of output units.
        dropout: Dropout probability.
        activation: Activation function ('relu', 'tanh', 'elu', 'selu').

    Returns:
        Compiled Keras model.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed. "
            "Install it with: pip install tensorflow"
        )

    # Build model
    model_layers = []

    # Input layer
    model_layers.append(layers.Input(shape=(input_dim,)))

    # Hidden layers
    for hidden_dim in hidden_dims:
        model_layers.append(layers.Dense(hidden_dim))
        model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.Activation(activation))
        model_layers.append(layers.Dropout(dropout))

    # Output layer
    model_layers.append(layers.Dense(output_dim))

    # Create sequential model
    model = models.Sequential(model_layers)

    return model


class TradingFFN_TF:
    """TensorFlow/Keras wrapper for trading feedforward networks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        dropout: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize TensorFlow trading feedforward network.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Number of output units.
            dropout: Dropout probability.
            activation: Activation function name.

        Raises:
            ImportError: If TensorFlow is not installed.
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is not installed. "
                "Install it with: pip install tensorflow"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation

        self.model = build_feedforward_tf(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation
        )

        self.metadata: Dict[str, Any] = {}
        self.is_fitted = False
        self.history = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
        validation_split: float = 0.2,
        verbose: int = 1
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
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch).

        Returns:
            Dictionary containing training history with keys like 'loss', 'val_loss'.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed.")

        # Handle y shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

        self.is_fitted = True
        self.history = history

        # Store metadata
        self.metadata = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'activation': self.activation,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
        }

        # Return history as dictionary of lists
        return {key: [float(v) for v in values] for key, values in history.history.items()}

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

        predictions = self.model.predict(X, verbose=0)

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

        # Save model (TensorFlow format)
        self.model.save(path / 'model.keras')

        # Save metadata
        with open(path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TradingFFN_TF':
        """
        Load a saved model.

        Args:
            path: Directory path containing the saved model.

        Returns:
            Loaded TradingFFN_TF instance.

        Raises:
            ImportError: If TensorFlow is not installed.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed.")

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

        # Load model
        instance.model = keras.models.load_model(path / 'model.keras')
        instance.is_fitted = True
        instance.metadata = metadata

        return instance


def check_tensorflow_available() -> bool:
    """
    Check if TensorFlow is available.

    Returns:
        True if TensorFlow is installed and importable, False otherwise.
    """
    return TF_AVAILABLE


def get_tensorflow_info() -> Dict[str, Any]:
    """
    Get TensorFlow installation information.

    Returns:
        Dictionary with TensorFlow version and device info.
    """
    if not TF_AVAILABLE:
        return {
            'available': False,
            'version': None,
            'devices': [],
            'gpu_available': False
        }

    return {
        'available': True,
        'version': tf.__version__,
        'devices': [device.name for device in tf.config.list_physical_devices()],
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
    }
