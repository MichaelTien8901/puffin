"""
Tests for RNN-based models.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from puffin.deep.rnn import (
    LSTMNet,
    TradingLSTM,
    StackedLSTM,
    MultivariateLSTM,
    GRUNet,
    TradingGRU
)


class TestLSTMNet:
    """Tests for LSTMNet module."""

    def test_forward_pass_univariate(self):
        """Test forward pass with univariate input."""
        model = LSTMNet(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
        x = torch.randn(16, 20, 1)  # batch_size=16, seq_len=20, input_dim=1
        output = model(x)

        assert output.shape == (16, 1)

    def test_forward_pass_multivariate(self):
        """Test forward pass with multivariate input."""
        model = LSTMNet(input_dim=5, hidden_dim=32, num_layers=2, output_dim=1)
        x = torch.randn(16, 20, 5)  # 5 features
        output = model(x)

        assert output.shape == (16, 1)

    def test_forward_pass_multi_output(self):
        """Test forward pass with multiple outputs."""
        model = LSTMNet(input_dim=3, hidden_dim=32, num_layers=2, output_dim=3)
        x = torch.randn(16, 20, 3)
        output = model(x)

        assert output.shape == (16, 3)

    def test_single_layer_no_dropout(self):
        """Test that single layer LSTM doesn't use dropout."""
        model = LSTMNet(num_layers=1, dropout=0.5)
        # Should not raise error even though dropout is specified
        x = torch.randn(8, 10, 1)
        output = model(x)
        assert output.shape == (8, 1)


class TestTradingLSTM:
    """Tests for TradingLSTM class."""

    def test_prepare_sequences(self):
        """Test sequence preparation."""
        lstm = TradingLSTM()
        series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        X, y = lstm.prepare_sequences(series, lookback=3)

        # Should have 7 samples (10 - 3)
        assert X.shape == (7, 3, 1)
        assert y.shape == (7,)

        # Check first sequence
        np.testing.assert_array_equal(X[0, :, 0], [1, 2, 3])
        assert y[0] == 4

        # Check last sequence
        np.testing.assert_array_equal(X[-1, :, 0], [7, 8, 9])
        assert y[-1] == 10

    def test_fit_basic(self):
        """Test basic training."""
        # Generate synthetic data
        np.random.seed(42)
        t = np.linspace(0, 10, 200)
        series = np.sin(t) + 0.1 * np.random.randn(200)

        lstm = TradingLSTM()
        history = lstm.fit(series, lookback=10, epochs=5, lr=0.01)

        # Check history structure
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5

        # Training loss should decrease (or at least not explode)
        assert history['train_loss'][-1] < 10

    def test_predict_single_step(self):
        """Test single-step prediction."""
        np.random.seed(42)
        t = np.linspace(0, 10, 200)
        series = np.sin(t) + 0.1 * np.random.randn(200)

        lstm = TradingLSTM()
        lstm.fit(series, lookback=10, epochs=5, lr=0.01)

        # Predict next value
        predictions = lstm.predict(series, steps=1)

        assert predictions.shape == (1,)
        assert not np.isnan(predictions[0])

    def test_predict_multi_step(self):
        """Test multi-step prediction."""
        np.random.seed(42)
        t = np.linspace(0, 10, 200)
        series = np.sin(t) + 0.1 * np.random.randn(200)

        lstm = TradingLSTM()
        lstm.fit(series, lookback=10, epochs=5, lr=0.01)

        # Predict 5 steps ahead
        predictions = lstm.predict(series, steps=5)

        assert predictions.shape == (5,)
        assert not np.any(np.isnan(predictions))

    def test_predict_before_fit_raises(self):
        """Test that predict raises error before fit."""
        lstm = TradingLSTM()
        series = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="Model must be trained"):
            lstm.predict(series)


class TestStackedLSTM:
    """Tests for StackedLSTM module."""

    def test_forward_pass_default_dims(self):
        """Test forward pass with default hidden dimensions."""
        model = StackedLSTM(input_dim=3)
        x = torch.randn(16, 20, 3)
        output = model(x)

        assert output.shape == (16, 1)

    def test_forward_pass_custom_dims(self):
        """Test forward pass with custom hidden dimensions."""
        model = StackedLSTM(input_dim=5, hidden_dims=[128, 64, 32], output_dim=1)
        x = torch.randn(8, 15, 5)
        output = model(x)

        assert output.shape == (8, 1)
        assert len(model.lstm_layers) == 3

    def test_multiple_outputs(self):
        """Test with multiple output dimensions."""
        model = StackedLSTM(input_dim=4, hidden_dims=[64, 32], output_dim=3)
        x = torch.randn(16, 20, 4)
        output = model(x)

        assert output.shape == (16, 3)


class TestMultivariateLSTM:
    """Tests for MultivariateLSTM class."""

    def test_prepare_sequences(self):
        """Test multivariate sequence preparation."""
        lstm = MultivariateLSTM()

        features = np.random.randn(100, 5)
        target = np.random.randn(100)

        X, y = lstm.prepare_sequences(features, target, lookback=10)

        assert X.shape == (90, 10, 5)  # 100 - 10 samples
        assert y.shape == (90,)

    def test_fit_basic(self):
        """Test basic multivariate training."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 200
        n_features = 3

        # Create correlated features
        t = np.linspace(0, 10, n_samples)
        features = {
            'feature1': np.sin(t) + 0.1 * np.random.randn(n_samples),
            'feature2': np.cos(t) + 0.1 * np.random.randn(n_samples),
            'feature3': np.sin(2*t) + 0.1 * np.random.randn(n_samples),
            'target': np.sin(t + 0.5) + 0.1 * np.random.randn(n_samples)
        }
        df = pd.DataFrame(features)

        lstm = MultivariateLSTM()
        history = lstm.fit(
            df,
            target_col='target',
            lookback=10,
            epochs=5,
            lr=0.01
        )

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5

    def test_predict(self):
        """Test multivariate prediction."""
        np.random.seed(42)
        n_samples = 200

        t = np.linspace(0, 10, n_samples)
        features = {
            'feature1': np.sin(t) + 0.1 * np.random.randn(n_samples),
            'feature2': np.cos(t) + 0.1 * np.random.randn(n_samples),
            'target': np.sin(t + 0.5) + 0.1 * np.random.randn(n_samples)
        }
        df = pd.DataFrame(features)

        lstm = MultivariateLSTM()
        lstm.fit(df, target_col='target', lookback=10, epochs=5, lr=0.01)

        # Predict using last samples
        predictions = lstm.predict(df.iloc[-20:])

        assert predictions.shape == (1,)
        assert not np.isnan(predictions[0])

    def test_predict_before_fit_raises(self):
        """Test that predict raises error before fit."""
        lstm = MultivariateLSTM()
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        with pytest.raises(ValueError, match="Model must be trained"):
            lstm.predict(df)


class TestGRUNet:
    """Tests for GRUNet module."""

    def test_forward_pass(self):
        """Test GRU forward pass."""
        model = GRUNet(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
        x = torch.randn(16, 20, 1)
        output = model(x)

        assert output.shape == (16, 1)

    def test_gru_vs_lstm_shapes(self):
        """Test that GRU produces same output shape as LSTM."""
        gru = GRUNet(input_dim=3, hidden_dim=64, output_dim=1)
        lstm = LSTMNet(input_dim=3, hidden_dim=64, output_dim=1)

        x = torch.randn(8, 15, 3)

        gru_out = gru(x)
        lstm_out = lstm(x)

        assert gru_out.shape == lstm_out.shape


class TestTradingGRU:
    """Tests for TradingGRU class."""

    def test_prepare_sequences(self):
        """Test GRU sequence preparation."""
        gru = TradingGRU()
        series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        X, y = gru.prepare_sequences(series, lookback=3)

        assert X.shape == (7, 3, 1)
        assert y.shape == (7,)

    def test_fit_basic(self):
        """Test basic GRU training."""
        np.random.seed(42)
        t = np.linspace(0, 10, 200)
        series = np.sin(t) + 0.1 * np.random.randn(200)

        gru = TradingGRU()
        history = gru.fit(series, lookback=10, epochs=5, lr=0.01)

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5

    def test_predict(self):
        """Test GRU prediction."""
        np.random.seed(42)
        t = np.linspace(0, 10, 200)
        series = np.sin(t) + 0.1 * np.random.randn(200)

        gru = TradingGRU()
        gru.fit(series, lookback=10, epochs=5, lr=0.01)

        predictions = gru.predict(series, steps=3)

        assert predictions.shape == (3,)
        assert not np.any(np.isnan(predictions))

    def test_convergence_on_simple_pattern(self):
        """Test that GRU can learn a simple pattern."""
        # Create a simple repeating pattern
        pattern = np.array([1, 2, 3, 4, 5] * 50)

        gru = TradingGRU()
        history = gru.fit(pattern, lookback=5, epochs=20, lr=0.01)

        # Loss should decrease
        assert history['train_loss'][-1] < history['train_loss'][0]


class TestDeviceHandling:
    """Tests for device handling (CPU/GPU)."""

    def test_trading_lstm_uses_correct_device(self):
        """Test that TradingLSTM uses available device."""
        lstm = TradingLSTM()
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert lstm.device.type == expected_device

    def test_trading_gru_uses_correct_device(self):
        """Test that TradingGRU uses available device."""
        gru = TradingGRU()
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert gru.device.type == expected_device

    def test_multivariate_lstm_uses_correct_device(self):
        """Test that MultivariateLSTM uses available device."""
        lstm = MultivariateLSTM()
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert lstm.device.type == expected_device


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
