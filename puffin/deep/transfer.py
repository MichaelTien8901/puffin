"""Transfer learning for financial time series using pretrained models."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class TransferLearningModel:
    """Transfer learning wrapper for financial predictions.

    Uses pretrained computer vision models (ResNet, VGG, etc.) and adapts them
    for financial time series classification by replacing the final layer.
    """

    def __init__(
        self,
        model_name: str = 'resnet18',
        n_classes: int = 3,
        device: str = None,
    ):
        """Initialize TransferLearningModel.

        Args:
            model_name: Name of pretrained model ('resnet18', 'resnet34', 'resnet50',
                       'vgg16', 'vgg19', 'mobilenet_v2').
            n_classes: Number of output classes.
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.n_classes = n_classes
        self.model = None

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = 'resnet18',
        n_classes: int = 3,
        device: str = None,
    ):
        """Create model from pretrained weights.

        Args:
            model_name: Name of pretrained model.
            n_classes: Number of output classes.
            device: Device to use.

        Returns:
            TransferLearningModel instance with pretrained base.
        """
        instance = cls(model_name=model_name, n_classes=n_classes, device=device)
        instance._load_pretrained()
        return instance

    def _load_pretrained(self):
        """Load pretrained model and modify final layer."""
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError(
                "torchvision is required for transfer learning. "
                "Install with: pip install torchvision"
            )

        # Load pretrained model
        if self.model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, self.n_classes)

        elif self.model_name == 'resnet34':
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, self.n_classes)

        elif self.model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, self.n_classes)

        elif self.model_name == 'vgg16':
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            n_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(n_features, self.n_classes)

        elif self.model_name == 'vgg19':
            self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            n_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(n_features, self.n_classes)

        elif self.model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            n_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(n_features, self.n_classes)

        else:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Supported: resnet18, resnet34, resnet50, vgg16, vgg19, mobilenet_v2"
            )

        self.model = self.model.to(self.device)

    def _freeze_base_layers(self):
        """Freeze all layers except the final classifier."""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze final layer
        if 'resnet' in self.model_name:
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif 'vgg' in self.model_name:
            for param in self.model.classifier[6].parameters():
                param.requires_grad = True
        elif 'mobilenet' in self.model_name:
            for param in self.model.classifier[1].parameters():
                param.requires_grad = True

    def _unfreeze_all_layers(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def fine_tune(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        lr: float = 0.001,
        freeze_layers: bool = True,
        val_loader: DataLoader = None,
    ) -> dict:
        """Fine-tune the pretrained model on new data.

        Args:
            train_loader: DataLoader for training data.
            epochs: Number of training epochs.
            lr: Learning rate.
            freeze_layers: If True, freeze base layers and only train final layer.
                          If False, fine-tune all layers.
            val_loader: Optional DataLoader for validation data.

        Returns:
            Dict containing training history.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call from_pretrained() first.")

        # Freeze/unfreeze layers
        if freeze_layers:
            self._freeze_base_layers()
            print(f"Freezing base layers, training only final classifier")
        else:
            self._unfreeze_all_layers()
            print(f"Fine-tuning all layers")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

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
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Handle single channel images by repeating to 3 channels
                if batch_X.shape[1] == 1:
                    batch_X = batch_X.repeat(1, 3, 1, 1)

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
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        return history

    def _validate(self, val_loader: DataLoader, criterion) -> tuple:
        """Run validation.

        Args:
            val_loader: Validation data loader.
            criterion: Loss function.

        Returns:
            Tuple of (loss, accuracy).
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Handle single channel images
                if batch_X.shape[1] == 1:
                    batch_X = batch_X.repeat(1, 3, 1, 1)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        return avg_val_loss, val_acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Input images of shape (n_samples, channels, height, width).

        Returns:
            Predicted class labels as numpy array.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call from_pretrained() first.")

        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        # Handle single channel images
        if X_tensor.shape[1] == 1:
            X_tensor = X_tensor.repeat(1, 3, 1, 1)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.cpu().numpy()

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions.

        Args:
            X: Input images of shape (n_samples, channels, height, width).

        Returns:
            Predicted probabilities as numpy array of shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call from_pretrained() first.")

        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        # Handle single channel images
        if X_tensor.shape[1] == 1:
            X_tensor = X_tensor.repeat(1, 3, 1, 1)

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
            raise RuntimeError("Model not loaded.")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'n_classes': self.n_classes,
        }, path)

    def load(self, path: str):
        """Load model weights.

        Args:
            path: Path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model_name = checkpoint['model_name']
        self.n_classes = checkpoint['n_classes']

        # Load pretrained architecture
        self._load_pretrained()

        # Load fine-tuned weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


def prepare_financial_images(
    images: np.ndarray,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """Prepare financial time series images for pretrained models.

    Resizes images to the expected input size for pretrained models (typically 224x224)
    and ensures proper channel configuration.

    Args:
        images: Input images of shape (n_samples, channels, height, width).
        target_size: Target size (height, width) for pretrained models.

    Returns:
        Resized images ready for pretrained models.
    """
    try:
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch is required for image preparation")

    # Convert to tensor
    if not isinstance(images, torch.Tensor):
        images = torch.FloatTensor(images)

    # Resize to target size
    resized = F.interpolate(
        images,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )

    return resized.numpy()
