"""Supervised ML model training for trading signal prediction."""

import json
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
)


class TradingModel:
    """Wrapper for scikit-learn models with trading-specific evaluation."""

    def __init__(self, model, model_type: str = "classification"):
        """
        Args:
            model: A scikit-learn estimator (classifier or regressor).
            model_type: 'classification' or 'regression'.
        """
        self.model = model
        self.model_type = model_type
        self.is_fitted = False
        self.metadata: dict = {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict:
        """Train with time-series cross-validation.

        Args:
            X: Feature matrix.
            y: Target variable.
            n_splits: Number of CV splits.

        Returns:
            Dict of cross-validation metrics.
        """
        # Drop NaN rows
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            if self.model_type == "classification":
                metrics = {
                    "fold": fold + 1,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                }
            else:
                metrics = {
                    "fold": fold + 1,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "r2": float(r2_score(y_test, y_pred)),
                }
            fold_metrics.append(metrics)

        # Final fit on all data
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True

        # Store metadata
        self.metadata = {
            "model_class": type(self.model).__name__,
            "model_type": self.model_type,
            "n_features": X_clean.shape[1],
            "n_samples": X_clean.shape[0],
            "feature_names": list(X_clean.columns),
            "trained_at": datetime.now().isoformat(),
            "data_hash": hashlib.md5(
                pd.util.hash_pandas_object(X_clean).values.tobytes()
            ).hexdigest()[:12],
            "cv_metrics": fold_metrics,
        }

        # Average metrics
        avg_metrics = {}
        for key in fold_metrics[0]:
            if key == "fold":
                continue
            values = [m[key] for m in fold_metrics]
            avg_metrics[f"avg_{key}"] = float(np.mean(values))
            avg_metrics[f"std_{key}"] = float(np.std(values))

        return avg_metrics

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet. Call train() first.")
        mask = X.notna().all(axis=1)
        predictions = pd.Series(np.nan, index=X.index)
        if mask.any():
            predictions[mask] = self.model.predict(X[mask])
        return predictions

    def save(self, path: str):
        """Save model and metadata."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TradingModel":
        """Load a saved model."""
        import joblib

        path = Path(path)
        model = joblib.load(path / "model.joblib")
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        instance = cls(model, model_type=metadata.get("model_type", "classification"))
        instance.is_fitted = True
        instance.metadata = metadata
        return instance


def list_saved_models(models_dir: str = "models") -> pd.DataFrame:
    """List all saved models with their metrics."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return pd.DataFrame()

    records = []
    for model_dir in models_path.iterdir():
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            record = {
                "name": model_dir.name,
                "model_class": meta.get("model_class"),
                "model_type": meta.get("model_type"),
                "n_samples": meta.get("n_samples"),
                "trained_at": meta.get("trained_at"),
            }
            # Add average CV metrics
            if "cv_metrics" in meta and meta["cv_metrics"]:
                for key in meta["cv_metrics"][0]:
                    if key != "fold":
                        values = [m[key] for m in meta["cv_metrics"]]
                        record[f"avg_{key}"] = np.mean(values)
            records.append(record)

    return pd.DataFrame(records)
