"""Tests for feedforward neural networks."""

import pytest
import numpy as np
import torch

from puffin.deep.feedforward import FeedforwardNet, TradingFFN


class TestFeedforwardNet:
    """Tests for PyTorch FeedforwardNet."""

    def test_model_creation(self):
        """Test model can be created with different configurations."""
        model = FeedforwardNet(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=1,
            dropout=0.3,
            activation='relu'
        )
        assert model is not None
        assert model.input_dim == 10
        assert model.output_dim == 1

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = FeedforwardNet(input_dim=10, hidden_dims=[32, 16], output_dim=1)
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 1)

    def test_multiple_outputs(self):
        """Test model with multiple output units."""
        model = FeedforwardNet(input_dim=10, hidden_dims=[32], output_dim=3)
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 3)

    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ['relu', 'tanh', 'leaky_relu', 'elu']:
            model = FeedforwardNet(
                input_dim=10,
                hidden_dims=[32],
                output_dim=1,
                activation=activation
            )
            x = torch.randn(5, 10)
            output = model(x)
            assert output.shape == (5, 1)

    def test_invalid_activation(self):
        """Test invalid activation raises error."""
        with pytest.raises(ValueError):
            FeedforwardNet(
                input_dim=10,
                hidden_dims=[32],
                output_dim=1,
                activation='invalid'
            )


class TestTradingFFN:
    """Tests for TradingFFN wrapper."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic regression data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(100) * 0.1
        return X, y

    def test_model_creation(self):
        """Test TradingFFN can be created."""
        model = TradingFFN(input_dim=10, hidden_dims=[32, 16], output_dim=1)
        assert model is not None
        assert not model.is_fitted

    def test_fit_and_predict(self, synthetic_data):
        """Test model can be trained and make predictions."""
        X, y = synthetic_data
        model = TradingFFN(input_dim=10, hidden_dims=[32, 16], output_dim=1)

        # Train model
        history = model.fit(X, y, epochs=10, lr=0.01, batch_size=32, verbose=False)

        assert model.is_fitted
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 10
        assert len(history['val_loss']) == 10

        # Make predictions
        predictions = model.predict(X)
        assert predictions.shape == (100,)
        assert not np.isnan(predictions).any()

    def test_training_convergence(self, synthetic_data):
        """Test that training loss decreases over epochs."""
        X, y = synthetic_data
        model = TradingFFN(input_dim=10, hidden_dims=[64, 32], output_dim=1)

        history = model.fit(X, y, epochs=20, lr=0.01, batch_size=32, verbose=False)

        # Loss should generally decrease
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        assert final_loss < initial_loss

    def test_prediction_without_fit(self):
        """Test prediction fails if model not fitted."""
        model = TradingFFN(input_dim=10, hidden_dims=[32], output_dim=1)
        X = np.random.randn(10, 10)

        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_save_and_load(self, synthetic_data, tmp_path):
        """Test model can be saved and loaded."""
        X, y = synthetic_data
        model = TradingFFN(input_dim=10, hidden_dims=[32, 16], output_dim=1)

        # Train model
        model.fit(X, y, epochs=5, lr=0.01, batch_size=32, verbose=False)

        # Make predictions before saving
        pred_before = model.predict(X[:10])

        # Save model
        save_path = tmp_path / "model"
        model.save(str(save_path))

        # Load model
        loaded_model = TradingFFN.load(str(save_path))

        # Check loaded model
        assert loaded_model.is_fitted
        assert loaded_model.input_dim == 10
        assert loaded_model.output_dim == 1

        # Make predictions with loaded model
        pred_after = loaded_model.predict(X[:10])

        # Predictions should be the same
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)

    def test_different_batch_sizes(self, synthetic_data):
        """Test training with different batch sizes."""
        X, y = synthetic_data
        for batch_size in [16, 32, 64]:
            model = TradingFFN(input_dim=10, hidden_dims=[32], output_dim=1)
            history = model.fit(X, y, epochs=5, batch_size=batch_size, verbose=False)
            assert len(history['train_loss']) == 5

    def test_validation_split(self, synthetic_data):
        """Test different validation split ratios."""
        X, y = synthetic_data
        for val_split in [0.1, 0.2, 0.3]:
            model = TradingFFN(input_dim=10, hidden_dims=[32], output_dim=1)
            history = model.fit(
                X, y, epochs=5, validation_split=val_split, verbose=False
            )
            assert 'val_loss' in history

    def test_gpu_availability(self):
        """Test GPU detection."""
        model = TradingFFN(input_dim=10, hidden_dims=[32], output_dim=1)
        # Just check device is set
        assert model.device is not None

    def test_metadata_storage(self, synthetic_data):
        """Test metadata is stored after training."""
        X, y = synthetic_data
        model = TradingFFN(input_dim=10, hidden_dims=[32, 16], output_dim=1)
        model.fit(X, y, epochs=5, lr=0.01, batch_size=32, verbose=False)

        assert 'input_dim' in model.metadata
        assert 'output_dim' in model.metadata
        assert 'hidden_dims' in model.metadata
        assert 'n_samples' in model.metadata
        assert 'final_train_loss' in model.metadata
        assert 'final_val_loss' in model.metadata

    def test_multioutput_prediction(self):
        """Test model with multiple outputs."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.column_stack([
            X[:, 0] * 2,
            X[:, 1] * 1.5,
            X[:, 2] * 0.5
        ])

        model = TradingFFN(input_dim=10, hidden_dims=[32], output_dim=3)
        history = model.fit(X, y, epochs=10, verbose=False)

        predictions = model.predict(X)
        assert predictions.shape == (100, 3)
        assert not np.isnan(predictions).any()
