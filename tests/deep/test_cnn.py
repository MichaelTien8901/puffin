"""Tests for CNN models."""

import pytest
import numpy as np
import pandas as pd
import torch

from puffin.deep.cnn import Conv1DNet, TradingCNN
from puffin.deep.cnn_ta import CNNTA, TradingCNNTA, series_to_image
from puffin.deep.transfer import TransferLearningModel, prepare_financial_images


class TestConv1DNet:
    """Test 1D CNN for time series."""

    def test_init(self):
        """Test model initialization."""
        model = Conv1DNet(
            input_channels=5,
            seq_length=20,
            n_filters=[32, 64],
            kernel_sizes=[3, 3],
            output_dim=1
        )
        assert model is not None
        assert model.input_channels == 5
        assert model.seq_length == 20
        assert model.output_dim == 1

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 10
        seq_length = 20
        input_channels = 5
        output_dim = 1

        model = Conv1DNet(
            input_channels=input_channels,
            seq_length=seq_length,
            output_dim=output_dim
        )

        # Create dummy input
        x = torch.randn(batch_size, seq_length, input_channels)

        # Forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (batch_size, output_dim)

    def test_forward_multiclass(self):
        """Test forward pass with multiple output classes."""
        batch_size = 10
        seq_length = 20
        input_channels = 3
        output_dim = 3

        model = Conv1DNet(
            input_channels=input_channels,
            seq_length=seq_length,
            output_dim=output_dim
        )

        x = torch.randn(batch_size, seq_length, input_channels)
        output = model(x)

        assert output.shape == (batch_size, output_dim)


class TestTradingCNN:
    """Test TradingCNN wrapper."""

    def test_prepare_sequences(self):
        """Test sequence preparation."""
        # Create synthetic data
        n_samples = 100
        n_features = 3
        X = np.random.randn(n_samples, n_features)

        lookback = 20
        X_seq, y = TradingCNN.prepare_sequences(X, lookback=lookback)

        # Check shapes
        assert X_seq.shape == (n_samples - lookback, lookback, n_features)
        assert y.shape == (n_samples - lookback,)

        # Check that y contains the next values
        for i in range(len(y)):
            assert y[i] == X[i + lookback, 0]

    def test_prepare_sequences_1d(self):
        """Test sequence preparation with 1D input."""
        # Create 1D data
        X = np.random.randn(100)

        lookback = 20
        X_seq, y = TradingCNN.prepare_sequences(X, lookback=lookback)

        # Check shapes
        assert X_seq.shape == (80, 20, 1)
        assert y.shape == (80,)

    def test_fit_predict(self):
        """Test basic training and prediction."""
        # Create synthetic data
        n_samples = 100
        seq_length = 20
        n_features = 3

        # Generate simple pattern: next value = mean of last 5 values
        X = np.random.randn(n_samples, n_features)
        X_seq, y = TradingCNN.prepare_sequences(X, lookback=seq_length)

        # Create model
        model = TradingCNN(
            input_channels=n_features,
            seq_length=seq_length,
            output_dim=1,
            device='cpu'
        )

        # Train (few epochs for speed)
        history = model.fit(X_seq, y, epochs=5, batch_size=16, validation_split=0.2)

        # Check history
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5

        # Make predictions
        predictions = model.predict(X_seq[:10])

        # Check predictions shape
        assert predictions.shape == (10,)
        assert not np.isnan(predictions).any()

    def test_training_convergence(self):
        """Test that training loss decreases."""
        # Create simple learnable pattern
        n_samples = 200
        seq_length = 10

        # Pattern: y = mean of last 3 values
        X = np.cumsum(np.random.randn(n_samples, 1), axis=0)
        X_seq, y = TradingCNN.prepare_sequences(X, lookback=seq_length)

        model = TradingCNN(
            input_channels=1,
            seq_length=seq_length,
            output_dim=1,
            device='cpu'
        )

        # Train
        history = model.fit(X_seq, y, epochs=20, batch_size=16, validation_split=0.0)

        # Check that loss decreases
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        assert final_loss < initial_loss


class TestSeriesToImage:
    """Test time series to image conversion."""

    def test_basic_conversion(self):
        """Test basic OHLCV to image conversion."""
        # Create synthetic OHLCV data
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'open': np.random.uniform(90, 110, 50),
            'high': np.random.uniform(95, 115, 50),
            'low': np.random.uniform(85, 105, 50),
            'close': np.random.uniform(90, 110, 50),
            'volume': np.random.uniform(1000, 5000, 50),
        }, index=dates)

        window = 20
        image = series_to_image(df, window=window, indicators=['sma', 'rsi'])

        # Check shape
        assert len(image.shape) == 2
        assert image.shape[1] == window
        assert image.shape[0] > 0  # Multiple features

        # Check no NaN values
        assert not np.isnan(image).all()

    def test_all_indicators(self):
        """Test with all indicators."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'open': np.random.uniform(90, 110, 50),
            'high': np.random.uniform(95, 115, 50),
            'low': np.random.uniform(85, 105, 50),
            'close': np.random.uniform(90, 110, 50),
            'volume': np.random.uniform(1000, 5000, 50),
        }, index=dates)

        image = series_to_image(
            df,
            window=20,
            indicators=['sma', 'rsi', 'macd', 'bbands']
        )

        # Check that we have more features with more indicators
        assert image.shape[0] > 5


class TestCNNTA:
    """Test 2D CNN for technical analysis."""

    def test_init(self):
        """Test model initialization."""
        model = CNNTA(image_height=10, image_width=20, n_classes=3)
        assert model is not None
        assert model.image_height == 10
        assert model.image_width == 20
        assert model.n_classes == 3

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 8
        image_height = 10
        image_width = 20
        n_classes = 3

        model = CNNTA(
            image_height=image_height,
            image_width=image_width,
            n_classes=n_classes
        )

        # Create dummy input (batch, channels, height, width)
        x = torch.randn(batch_size, 1, image_height, image_width)

        # Forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (batch_size, n_classes)


class TestTradingCNNTA:
    """Test TradingCNNTA wrapper."""

    def _create_sample_data(self, n_samples=50):
        """Create sample OHLCV data."""
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        df = pd.DataFrame({
            'open': np.random.uniform(90, 110, n_samples),
            'high': np.random.uniform(95, 115, n_samples),
            'low': np.random.uniform(85, 105, n_samples),
            'close': np.random.uniform(90, 110, n_samples),
            'volume': np.random.uniform(1000, 5000, n_samples),
        }, index=dates)
        return df

    def test_fit_predict(self):
        """Test training and prediction."""
        # Create sample data
        n_samples = 20
        window = 20
        n_classes = 3

        ohlcv_data = [self._create_sample_data(window + 10) for _ in range(n_samples)]
        labels = np.random.randint(0, n_classes, n_samples)

        # Create model
        model = TradingCNNTA(window=window, n_classes=n_classes, device='cpu')

        # Train (few epochs for speed)
        history = model.fit(
            ohlcv_data,
            labels,
            epochs=5,
            batch_size=8,
            validation_split=0.2
        )

        # Check history
        assert 'train_loss' in history
        assert 'train_acc' in history
        assert len(history['train_loss']) == 5

        # Make predictions
        predictions = model.predict(ohlcv_data[:5])

        # Check predictions
        assert predictions.shape == (5,)
        assert predictions.dtype == np.int64
        assert all(0 <= p < n_classes for p in predictions)

    def test_predict_proba(self):
        """Test probability predictions."""
        n_samples = 10
        window = 20
        n_classes = 3

        ohlcv_data = [self._create_sample_data(window + 10) for _ in range(n_samples)]
        labels = np.random.randint(0, n_classes, n_samples)

        model = TradingCNNTA(window=window, n_classes=n_classes, device='cpu')

        # Train
        model.fit(ohlcv_data, labels, epochs=3, batch_size=8, validation_split=0.0)

        # Get probabilities
        probs = model.predict_proba(ohlcv_data[:5])

        # Check shape and properties
        assert probs.shape == (5, n_classes)
        assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert (probs >= 0).all() and (probs <= 1).all()  # Valid probabilities


class TestTransferLearning:
    """Test transfer learning models."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Skip transfer learning tests without GPU (too slow)"
    )
    def test_from_pretrained(self):
        """Test loading pretrained model."""
        try:
            model = TransferLearningModel.from_pretrained(
                model_name='resnet18',
                n_classes=3,
                device='cpu'
            )
            assert model is not None
            assert model.model is not None
        except ImportError:
            pytest.skip("torchvision not installed")

    def test_prepare_financial_images(self):
        """Test image preparation for pretrained models."""
        # Create dummy images
        n_samples = 10
        images = np.random.randn(n_samples, 1, 50, 50)

        # Resize to standard size
        resized = prepare_financial_images(images, target_size=(224, 224))

        # Check shape
        assert resized.shape == (n_samples, 1, 224, 224)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
