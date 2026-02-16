"""CNN for Technical Analysis using 2D image representation of time series."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def series_to_image(
    ohlcv: pd.DataFrame,
    window: int = 20,
    indicators: list = None,
) -> np.ndarray:
    """Convert OHLCV time series to 2D image representation.

    Each row represents a different feature/indicator, and columns represent time steps.
    This creates a "heatmap" style representation that CNNs can process.

    Args:
        ohlcv: DataFrame with OHLC and volume columns.
        window: Number of time steps to include.
        indicators: List of indicators to compute. Options: 'sma', 'rsi', 'macd', 'bbands'.

    Returns:
        2D numpy array of shape (n_features, window).
    """
    if indicators is None:
        indicators = ['sma', 'rsi', 'macd']

    if len(ohlcv) < window:
        raise ValueError(f"Need at least {window} samples, got {len(ohlcv)}")

    # Take the last 'window' samples
    df = ohlcv.iloc[-window:].copy()

    features = []

    # Normalize OHLC by closing price
    if 'close' in df.columns:
        close_norm = df['close'] / df['close'].iloc[0]
        features.append(close_norm.values)

        if 'open' in df.columns:
            open_norm = df['open'] / df['close'].iloc[0]
            features.append(open_norm.values)

        if 'high' in df.columns:
            high_norm = df['high'] / df['close'].iloc[0]
            features.append(high_norm.values)

        if 'low' in df.columns:
            low_norm = df['low'] / df['close'].iloc[0]
            features.append(low_norm.values)

    # Volume (log-normalized)
    if 'volume' in df.columns:
        vol = df['volume'].values
        vol_norm = np.log1p(vol) / np.log1p(vol.max())
        features.append(vol_norm)

    # Technical indicators
    if 'sma' in indicators and 'close' in df.columns:
        # Simple Moving Average
        sma = df['close'].rolling(window=min(10, window)).mean()
        sma_norm = sma / df['close']
        features.append(sma_norm.fillna(1.0).values)

    if 'rsi' in indicators and 'close' in df.columns:
        # Relative Strength Index
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=min(14, window)).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=min(14, window)).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_norm = rsi / 100.0
        features.append(rsi_norm.fillna(0.5).values)

    if 'macd' in indicators and 'close' in df.columns:
        # MACD (simplified)
        ema12 = df['close'].ewm(span=min(12, window), adjust=False).mean()
        ema26 = df['close'].ewm(span=min(26, window), adjust=False).mean()
        macd = ema12 - ema26
        macd_norm = macd / df['close']
        features.append(macd_norm.fillna(0.0).values)

    if 'bbands' in indicators and 'close' in df.columns:
        # Bollinger Bands
        sma20 = df['close'].rolling(window=min(20, window)).mean()
        std20 = df['close'].rolling(window=min(20, window)).std()
        upper = (sma20 + 2 * std20) / df['close']
        lower = (sma20 - 2 * std20) / df['close']
        features.append(upper.fillna(1.0).values)
        features.append(lower.fillna(1.0).values)

    # Stack features into 2D array
    image = np.vstack(features)

    return image


class CNNTA(nn.Module):
    """2D CNN for treating time series as images.

    Uses 2D convolutions to detect patterns in the "image" representation
    of technical indicators over time.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        n_classes: int = 3,
    ):
        """Initialize CNNTA.

        Args:
            image_height: Number of features/indicators (rows).
            image_width: Time window size (columns).
            n_classes: Number of output classes (e.g., 3 for buy/hold/sell).
        """
        super(CNNTA, self).__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.n_classes = n_classes

        # 2D Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # Calculate flattened size after conv layers
        # After 2 pooling layers, dimensions are reduced by 4
        h_out = image_height // 4
        w_out = image_width // 4
        self.flat_size = 128 * h_out * w_out

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, image_height, image_width).

        Returns:
            Output tensor of shape (batch_size, n_classes).
        """
        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x


class TradingCNNTA:
    """Wrapper for trading with 2D CNN-TA models.

    Converts OHLCV data to 2D images and uses CNN for classification.
    """

    def __init__(
        self,
        window: int = 20,
        indicators: list = None,
        n_classes: int = 3,
        device: str = None,
    ):
        """Initialize TradingCNNTA.

        Args:
            window: Time window size for image representation.
            indicators: List of indicators to include.
            n_classes: Number of output classes.
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if indicators is None:
            indicators = ['sma', 'rsi', 'macd']

        self.window = window
        self.indicators = indicators
        self.n_classes = n_classes

        # Model will be initialized after first image is created
        self.model = None

    def _prepare_images(self, ohlcv_data: list) -> np.ndarray:
        """Convert list of OHLCV DataFrames to images.

        Args:
            ohlcv_data: List of DataFrames, each with at least 'window' rows.

        Returns:
            4D numpy array of shape (n_samples, 1, height, width).
        """
        images = []

        for df in ohlcv_data:
            img = series_to_image(df, window=self.window, indicators=self.indicators)
            images.append(img)

        # Stack into 4D array and add channel dimension
        images_array = np.array(images)
        images_array = np.expand_dims(images_array, axis=1)  # Add channel dimension

        return images_array

    def fit(
        self,
        ohlcv_data: list,
        labels: np.ndarray,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> dict:
        """Train the CNN-TA model.

        Args:
            ohlcv_data: List of OHLCV DataFrames, one per sample.
            labels: Target labels of shape (n_samples,).
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size for training.
            validation_split: Fraction of data to use for validation.

        Returns:
            Dict containing training history.
        """
        # Convert to images
        X = self._prepare_images(ohlcv_data)

        # Initialize model if not already done
        if self.model is None:
            _, _, image_height, image_width = X.shape
            self.model = CNNTA(
                image_height=image_height,
                image_width=image_width,
                n_classes=self.n_classes
            ).to(self.device)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(labels).to(self.device)

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
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += batch_y.size(0)
                epoch_correct += (predicted == batch_y).sum().item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_acc = epoch_correct / epoch_total

            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)

            # Validation
            if n_val > 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_acc = (val_predicted == y_val).sum().item() / len(y_val)

                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                if n_val > 0:
                    print(f"Epoch {epoch+1}/{epochs}, "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        return history

    def predict(self, ohlcv_data: list) -> np.ndarray:
        """Generate predictions.

        Args:
            ohlcv_data: List of OHLCV DataFrames.

        Returns:
            Predicted class labels as numpy array.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")

        self.model.eval()

        X = self._prepare_images(ohlcv_data)
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.cpu().numpy()

        return predictions

    def predict_proba(self, ohlcv_data: list) -> np.ndarray:
        """Generate probability predictions.

        Args:
            ohlcv_data: List of OHLCV DataFrames.

        Returns:
            Predicted probabilities as numpy array of shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")

        self.model.eval()

        X = self._prepare_images(ohlcv_data)
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            probabilities = probs.cpu().numpy()

        return probabilities

    def save(self, path: str):
        """Save model weights.

        Args:
            path: Path to save the model.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized yet.")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'window': self.window,
            'indicators': self.indicators,
            'n_classes': self.n_classes,
            'image_height': self.model.image_height,
            'image_width': self.model.image_width,
        }, path)

    def load(self, path: str):
        """Load model weights.

        Args:
            path: Path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.window = checkpoint['window']
        self.indicators = checkpoint['indicators']
        self.n_classes = checkpoint['n_classes']

        self.model = CNNTA(
            image_height=checkpoint['image_height'],
            image_width=checkpoint['image_width'],
            n_classes=checkpoint['n_classes']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
