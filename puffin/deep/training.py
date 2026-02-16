"""Training utilities for deep learning models."""

from typing import Optional, Callable, Tuple, List, Dict, Any
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


class EarlyStopping:
    """Early stopping callback to stop training when validation loss stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum change in monitored value to qualify as improvement.
            restore_best_weights: Whether to restore model weights from epoch with best value.
            verbose: Whether to print messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.should_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss.
            model: PyTorch model.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            return False

        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best_weights:
                    if self.verbose:
                        print(f"Restoring best weights (val_loss={self.best_loss:.6f})")
                    model.load_state_dict(self.best_weights)
                return True

        return False


class LRScheduler:
    """Learning rate scheduler with multiple schedule types."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str = 'step',
        step_size: int = 30,
        gamma: float = 0.1,
        warmup_epochs: int = 0,
        T_max: Optional[int] = None,
        eta_min: float = 0.0
    ):
        """
        Initialize learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer.
            schedule_type: Type of schedule ('step', 'cosine', 'warmup_cosine').
            step_size: Period of learning rate decay for step schedule.
            gamma: Multiplicative factor of learning rate decay for step schedule.
            warmup_epochs: Number of warmup epochs for warmup schedules.
            T_max: Maximum number of iterations for cosine schedule.
            eta_min: Minimum learning rate for cosine schedule.
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min

        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

        # Create PyTorch scheduler if needed
        if schedule_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif schedule_type == 'cosine':
            if T_max is None:
                raise ValueError("T_max must be specified for cosine schedule")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        elif schedule_type == 'warmup_cosine':
            # Manual implementation
            self.scheduler = None
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def step(self):
        """Update learning rate for next epoch."""
        if self.schedule_type == 'warmup_cosine':
            self._warmup_cosine_step()
        else:
            self.scheduler.step()
        self.current_epoch += 1

    def _warmup_cosine_step(self):
        """Warmup + cosine annealing schedule."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            if self.T_max is None:
                raise ValueError("T_max must be specified for warmup_cosine schedule")

            progress = (self.current_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            lr = self.eta_min + (self.initial_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        if self.schedule_type == 'warmup_cosine':
            return [param_group['lr'] for param_group in self.optimizer.param_groups]
        else:
            return self.scheduler.get_last_lr()


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.2,
    shuffle: bool = True,
    random_seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders for training and validation.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Target array of shape (n_samples,) or (n_samples, n_outputs).
        batch_size: Batch size for data loaders.
        val_split: Fraction of data to use for validation.
        shuffle: Whether to shuffle training data.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)

    if y.ndim == 1:
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
    else:
        y_tensor = torch.FloatTensor(y)

    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split into train and val
    n_samples = len(dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    if random_seed is not None:
        generator = torch.Generator().manual_seed(random_seed)
    else:
        generator = None

    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=generator
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    callbacks: Optional[List[Callable]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Generic training loop for PyTorch models.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of epochs to train.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        callbacks: List of callback functions to call after each epoch.
                   Each callback receives (epoch, train_loss, val_loss, model).
        device: Device to use for training.
        verbose: Whether to print progress.

    Returns:
        Dictionary with training history containing 'train_loss' and 'val_loss'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")

        # Execute callbacks
        if callbacks:
            for callback in callbacks:
                result = callback(epoch, avg_train_loss, avg_val_loss, model)
                # Check if callback signals to stop (e.g., early stopping)
                if result is True:
                    if verbose:
                        print(f"Stopping early at epoch {epoch + 1}")
                    break

    return history


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced classification.

    Args:
        y: Target labels of shape (n_samples,).

    Returns:
        Tensor of class weights.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)

    # Compute weights inversely proportional to class frequencies
    weights = n_samples / (n_classes * counts)

    return torch.FloatTensor(weights)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
