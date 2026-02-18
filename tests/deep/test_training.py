"""Tests for training utilities."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from puffin.deep.training import (
    EarlyStopping,
    LRScheduler,
    create_dataloaders,
    training_loop,
    compute_class_weights,
    set_seed
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_early_stopping_creation(self):
        """Test early stopping can be created."""
        es = EarlyStopping(patience=5, min_delta=0.001)
        assert es.patience == 5
        assert es.min_delta == 0.001
        assert not es.should_stop

    def test_early_stopping_improvement(self):
        """Test early stopping tracks improvement."""
        model = SimpleModel()
        es = EarlyStopping(patience=3, verbose=False)

        # Simulate improving loss
        assert not es(1.0, model)  # First call
        assert not es(0.9, model)  # Improvement
        assert not es(0.8, model)  # Improvement
        assert not es(0.7, model)  # Improvement
        assert not es.should_stop

    def test_early_stopping_triggers(self):
        """Test early stopping triggers after patience."""
        model = SimpleModel()
        es = EarlyStopping(patience=2, verbose=False)

        assert not es(1.0, model)  # First call
        assert not es(0.9, model)  # Improvement
        assert not es(0.95, model)  # No improvement (1)
        assert not es(0.96, model)  # No improvement (2)
        assert es(0.97, model)  # No improvement (3) - should stop
        assert es.should_stop

    def test_restore_best_weights(self):
        """Test best weights are restored."""
        model = SimpleModel()
        es = EarlyStopping(patience=2, restore_best_weights=True, verbose=False)

        # First call: saves initial random weights as best, best_loss=1.0
        es(1.0, model)

        # Improve: saves new best weights at loss=0.8
        es(0.8, model)
        best_weights = model.fc.weight.data.clone()

        # Change model weights after saving best
        model.fc.weight.data.fill_(99.0)

        # Trigger early stopping with worse losses
        es(0.9, model)   # Worse - counter=1
        es(1.0, model)   # Worse - counter=2
        should_stop = es(1.1, model)  # Worse - counter=3, triggers stop

        assert should_stop
        # Check weights were restored to best (not the modified 99.0 values)
        assert torch.allclose(model.fc.weight.data, best_weights)
        assert not torch.allclose(model.fc.weight.data, torch.full_like(model.fc.weight.data, 99.0))

    def test_min_delta(self):
        """Test minimum delta for improvement."""
        model = SimpleModel()
        es = EarlyStopping(patience=2, min_delta=0.1, verbose=False)

        es(1.0, model)
        es(0.95, model)  # Improvement < min_delta, counts as no improvement
        es(0.94, model)  # No improvement
        assert es(0.93, model)  # Should stop


class TestLRScheduler:
    """Tests for learning rate scheduler."""

    def test_step_scheduler(self):
        """Test step learning rate schedule."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = LRScheduler(optimizer, schedule_type='step', step_size=5, gamma=0.1)

        initial_lr = optimizer.param_groups[0]['lr']
        assert initial_lr == 0.1

        # Step through epochs
        for _ in range(5):
            scheduler.step()

        # LR should have decreased
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr == pytest.approx(0.01, rel=1e-5)

    def test_cosine_scheduler(self):
        """Test cosine annealing schedule."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = LRScheduler(
            optimizer,
            schedule_type='cosine',
            T_max=10,
            eta_min=0.001
        )

        initial_lr = optimizer.param_groups[0]['lr']
        assert initial_lr == 0.1

        # Step through epochs
        for _ in range(10):
            scheduler.step()

        # LR should be close to eta_min
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr <= initial_lr

    def test_warmup_cosine_scheduler(self):
        """Test warmup + cosine schedule."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = LRScheduler(
            optimizer,
            schedule_type='warmup_cosine',
            warmup_epochs=5,
            T_max=20,
            eta_min=0.001
        )

        # During warmup, LR should increase
        initial_lr = optimizer.param_groups[0]['lr']

        for _ in range(3):
            scheduler.step()

        warmup_lr = optimizer.param_groups[0]['lr']
        assert warmup_lr > initial_lr

    def test_invalid_schedule_type(self):
        """Test invalid schedule type raises error."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        with pytest.raises(ValueError):
            LRScheduler(optimizer, schedule_type='invalid')

    def test_get_last_lr(self):
        """Test getting current learning rate."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = LRScheduler(optimizer, schedule_type='step', step_size=5)

        lrs = scheduler.get_last_lr()
        assert len(lrs) > 0
        assert lrs[0] == 0.1


class TestCreateDataLoaders:
    """Tests for data loader creation."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        return X, y

    def test_create_dataloaders(self, synthetic_data):
        """Test data loaders are created correctly."""
        X, y = synthetic_data
        train_loader, val_loader = create_dataloaders(
            X, y, batch_size=32, val_split=0.2
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_data_splits(self, synthetic_data):
        """Test data is split correctly."""
        X, y = synthetic_data
        train_loader, val_loader = create_dataloaders(
            X, y, batch_size=32, val_split=0.2
        )

        # Count samples
        train_samples = sum(len(batch[0]) for batch in train_loader)
        val_samples = sum(len(batch[0]) for batch in val_loader)

        assert train_samples + val_samples == 100
        assert val_samples == 20  # 20% of 100

    def test_batch_size(self, synthetic_data):
        """Test batch size is respected."""
        X, y = synthetic_data
        train_loader, val_loader = create_dataloaders(
            X, y, batch_size=16, val_split=0.2
        )

        # Check first batch size (may be different for last batch)
        for batch_X, batch_y in train_loader:
            assert len(batch_X) <= 16
            break

    def test_tensor_shapes(self, synthetic_data):
        """Test tensors have correct shapes."""
        X, y = synthetic_data
        train_loader, val_loader = create_dataloaders(
            X, y, batch_size=32, val_split=0.2
        )

        for batch_X, batch_y in train_loader:
            assert batch_X.shape[1] == 10  # Feature dimension
            assert batch_y.shape[1] == 1  # Target dimension (reshaped)
            break

    def test_random_seed(self, synthetic_data):
        """Test random seed produces same splits."""
        X, y = synthetic_data

        train_loader1, _ = create_dataloaders(
            X, y, batch_size=32, val_split=0.2, random_seed=42
        )
        train_loader2, _ = create_dataloaders(
            X, y, batch_size=32, val_split=0.2, random_seed=42
        )

        # Get first batch from each
        batch1 = next(iter(train_loader1))
        batch2 = next(iter(train_loader2))

        # Should be identical
        assert torch.allclose(batch1[0], batch2[0])
        assert torch.allclose(batch1[1], batch2[1])


class TestTrainingLoop:
    """Tests for generic training loop."""

    @pytest.fixture
    def setup(self):
        """Setup model and data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        train_loader, val_loader = create_dataloaders(
            X, y, batch_size=32, val_split=0.2
        )

        model = SimpleModel(input_dim=10, output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        return model, train_loader, val_loader, optimizer, criterion

    def test_training_loop_basic(self, setup):
        """Test basic training loop execution."""
        model, train_loader, val_loader, optimizer, criterion = setup

        history = training_loop(
            model, train_loader, val_loader,
            epochs=5,
            optimizer=optimizer,
            criterion=criterion,
            verbose=False
        )

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5

    def test_training_reduces_loss(self, setup):
        """Test that training reduces loss."""
        model, train_loader, val_loader, optimizer, criterion = setup

        history = training_loop(
            model, train_loader, val_loader,
            epochs=20,
            optimizer=optimizer,
            criterion=criterion,
            verbose=False
        )

        # Loss should generally decrease
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        assert final_loss < initial_loss

    def test_training_with_callback(self, setup):
        """Test training loop with callbacks."""
        model, train_loader, val_loader, optimizer, criterion = setup

        callback_called = []

        def test_callback(epoch, train_loss, val_loss, model):
            callback_called.append(epoch)
            return False  # Don't stop

        history = training_loop(
            model, train_loader, val_loader,
            epochs=5,
            optimizer=optimizer,
            criterion=criterion,
            callbacks=[test_callback],
            verbose=False
        )

        assert len(callback_called) == 5

    def test_training_with_early_stopping(self, setup):
        """Test training loop with early stopping callback."""
        model, train_loader, val_loader, optimizer, criterion = setup

        es = EarlyStopping(patience=3, verbose=False)

        def es_callback(epoch, train_loss, val_loss, model):
            return es(val_loss, model)

        history = training_loop(
            model, train_loader, val_loader,
            epochs=100,  # Large number
            optimizer=optimizer,
            criterion=criterion,
            callbacks=[es_callback],
            verbose=False
        )

        # Should stop before 100 epochs
        assert len(history['train_loss']) < 100


class TestComputeClassWeights:
    """Tests for class weight computation."""

    def test_balanced_classes(self):
        """Test weights for balanced classes."""
        y = np.array([0, 0, 1, 1, 2, 2])
        weights = compute_class_weights(y)

        # All weights should be 1.0 for balanced classes
        assert torch.allclose(weights, torch.ones(3))

    def test_imbalanced_classes(self):
        """Test weights for imbalanced classes."""
        y = np.array([0, 0, 0, 0, 1, 2])  # Class 0 is overrepresented
        weights = compute_class_weights(y)

        # Weight for class 0 should be smaller
        assert weights[0] < weights[1]
        assert weights[0] < weights[2]


class TestSetSeed:
    """Tests for seed setting."""

    def test_set_seed_reproducibility(self):
        """Test setting seed produces reproducible results."""
        set_seed(42)
        random1 = torch.randn(10)

        set_seed(42)
        random2 = torch.randn(10)

        assert torch.allclose(random1, random2)

    def test_set_seed_numpy(self):
        """Test seed affects NumPy random state."""
        set_seed(42)
        np_random1 = np.random.randn(10)

        set_seed(42)
        np_random2 = np.random.randn(10)

        np.testing.assert_array_almost_equal(np_random1, np_random2)
