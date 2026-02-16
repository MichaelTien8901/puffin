"""TensorBoard logging for deep learning model training."""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """TensorBoard logger for tracking neural network training."""

    def __init__(
        self,
        log_dir: str = 'runs',
        experiment_name: Optional[str] = None,
        flush_secs: int = 120
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Base directory for TensorBoard logs.
            experiment_name: Name of experiment. If None, uses timestamp.
            flush_secs: How often, in seconds, to flush pending events to disk.
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d-%H%M%S')

        self.log_path = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(log_dir=str(self.log_path), flush_secs=flush_secs)
        self.experiment_name = experiment_name

    def log_scalars(self, step: int, **kwargs):
        """
        Log scalar values (e.g., loss, accuracy).

        Args:
            step: Current training step/epoch.
            **kwargs: Key-value pairs of scalar metrics to log.

        Example:
            logger.log_scalars(epoch, loss=0.5, accuracy=0.95, lr=0.001)
        """
        for name, value in kwargs.items():
            self.writer.add_scalar(name, value, step)

    def log_histogram(self, step: int, name: str, values: torch.Tensor):
        """
        Log histogram of values (e.g., weights, gradients).

        Args:
            step: Current training step/epoch.
            name: Name/tag for the histogram.
            values: Tensor of values to histogram.
        """
        self.writer.add_histogram(name, values, step)

    def log_model_graph(self, model: nn.Module, sample_input: torch.Tensor):
        """
        Log model computational graph.

        Args:
            model: PyTorch model.
            sample_input: Sample input tensor matching model's expected input shape.
        """
        self.writer.add_graph(model, sample_input)

    def log_weights(self, step: int, model: nn.Module):
        """
        Log all model weights as histograms.

        Args:
            step: Current training step/epoch.
            model: PyTorch model.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'weights/{name}', param.data, step)

    def log_gradients(self, step: int, model: nn.Module):
        """
        Log all model gradients as histograms.

        Args:
            step: Current training step/epoch.
            model: PyTorch model.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)

    def log_learning_rate(self, step: int, optimizer: torch.optim.Optimizer):
        """
        Log current learning rate.

        Args:
            step: Current training step/epoch.
            optimizer: PyTorch optimizer.
        """
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'learning_rate/group_{i}', param_group['lr'], step)

    def log_text(self, tag: str, text: str, step: int = 0):
        """
        Log text data.

        Args:
            tag: Name/tag for the text.
            text: Text string to log.
            step: Current training step/epoch.
        """
        self.writer.add_text(tag, text, step)

    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log hyperparameters and optional metrics.

        Args:
            hparams: Dictionary of hyperparameters.
            metrics: Dictionary of metric values.
        """
        if metrics is None:
            metrics = {}
        self.writer.add_hparams(hparams, metrics)

    def log_embeddings(
        self,
        mat: torch.Tensor,
        metadata: Optional[list] = None,
        label_img: Optional[torch.Tensor] = None,
        tag: str = 'default'
    ):
        """
        Log embeddings for visualization.

        Args:
            mat: Matrix where each row is the feature vector of a data point.
            metadata: List of labels/tags for each data point.
            label_img: Images corresponding to each data point.
            tag: Name/tag for the embedding.
        """
        self.writer.add_embedding(mat, metadata=metadata, label_img=label_img, tag=tag)

    def log_pr_curve(
        self,
        tag: str,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        step: int = 0
    ):
        """
        Log precision-recall curve.

        Args:
            tag: Name/tag for the curve.
            labels: Ground truth labels (binary).
            predictions: Predicted probabilities.
            step: Current training step/epoch.
        """
        self.writer.add_pr_curve(tag, labels, predictions, step)

    def log_custom_scalar(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: int
    ):
        """
        Log multiple scalars that will be plotted together.

        Args:
            main_tag: Parent name for the group.
            tag_scalar_dict: Dictionary of {tag: scalar_value}.
            step: Current training step/epoch.
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def close(self):
        """Close the TensorBoard writer and flush all pending events."""
        self.writer.flush()
        self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MetricsTracker:
    """Simple metrics tracker for training monitoring."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}

    def update(self, **kwargs):
        """
        Update metrics.

        Args:
            **kwargs: Key-value pairs of metrics to track.
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

    def get_average(self, name: str, last_n: Optional[int] = None) -> float:
        """
        Get average of a metric.

        Args:
            name: Name of the metric.
            last_n: If specified, average over last N values only.

        Returns:
            Average value.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found")

        values = self.metrics[name]
        if last_n is not None:
            values = values[-last_n:]

        return sum(values) / len(values) if values else 0.0

    def get_best(self, name: str, mode: str = 'min') -> float:
        """
        Get best value of a metric.

        Args:
            name: Name of the metric.
            mode: 'min' or 'max'.

        Returns:
            Best value.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found")

        values = self.metrics[name]
        if not values:
            return float('inf') if mode == 'min' else float('-inf')

        return min(values) if mode == 'min' else max(values)

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}

    def get_all(self) -> Dict[str, list]:
        """Get all tracked metrics."""
        return self.metrics.copy()


def create_training_logger(
    model_name: str,
    hparams: Dict[str, Any],
    log_dir: str = 'runs'
) -> TrainingLogger:
    """
    Create a training logger with automatic experiment naming.

    Args:
        model_name: Name of the model.
        hparams: Dictionary of hyperparameters.
        log_dir: Base directory for logs.

    Returns:
        TrainingLogger instance.
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_name = f"{model_name}_{timestamp}"

    logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)

    # Log hyperparameters as text
    hparams_text = "\n".join([f"- {k}: {v}" for k, v in hparams.items()])
    logger.log_text('hyperparameters', hparams_text)

    return logger
