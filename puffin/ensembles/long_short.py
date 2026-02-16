"""Ensemble-based long-short trading strategy."""

import numpy as np
import pandas as pd
from typing import List, Dict, Union


class EnsembleLongShort:
    """Long-short trading strategy using ensemble model predictions."""

    def __init__(self, models: Union[list, dict] = None):
        """
        Args:
            models: List of models or dict of {name: model} pairs.
                   Models should have fit() and predict() or predict_proba() methods.
        """
        if models is None:
            self.models = []
        elif isinstance(models, dict):
            self.models = list(models.values())
            self.model_names = list(models.keys())
        else:
            self.models = models
            self.model_names = [f"model_{i}" for i in range(len(models))]

        self.is_fitted = False
        self.weights_ = None

    def add_model(self, model, name: str = None):
        """Add a model to the ensemble.

        Args:
            model: Model with fit() and predict() methods.
            name: Optional name for the model.
        """
        self.models.append(model)
        if name is None:
            name = f"model_{len(self.models) - 1}"
        if hasattr(self, "model_names"):
            self.model_names.append(name)
        else:
            self.model_names = [name]

    def fit(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        equal_weights: bool = True
    ) -> "EnsembleLongShort":
        """Fit all models in the ensemble.

        Args:
            features: Feature matrix.
            returns: Target returns (for regression) or binary signals (for classification).
            equal_weights: If True, use equal weights for all models.
                          If False, weight by validation performance (future implementation).

        Returns:
            self for chaining.
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Add models with add_model().")

        # Clean data
        mask = features.notna().all(axis=1) & returns.notna()
        features_clean = features[mask]
        returns_clean = returns[mask]

        # Fit each model
        for i, model in enumerate(self.models):
            # Check if model needs to be fitted
            if hasattr(model, "is_fitted") and model.is_fitted:
                continue

            if hasattr(model, "fit"):
                model.fit(features_clean, returns_clean)
            else:
                raise ValueError(f"Model {i} does not have a fit() method.")

        # Set weights (equal weights for now)
        if equal_weights:
            self.weights_ = np.ones(len(self.models)) / len(self.models)
        else:
            # Future: Implement performance-based weighting
            self.weights_ = np.ones(len(self.models)) / len(self.models)

        self.is_fitted = True
        return self

    def predict_ensemble(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get predictions from all models in the ensemble.

        Args:
            features: Feature matrix.

        Returns:
            DataFrame with predictions from each model and ensemble prediction.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        predictions = pd.DataFrame(index=features.index)

        # Get predictions from each model
        for i, model in enumerate(self.models):
            if hasattr(model, "predict_proba"):
                # For classification models, use probability of positive class
                proba = model.predict_proba(features)
                # Handle both binary and multi-class
                if proba.shape[1] == 2:
                    predictions[self.model_names[i]] = proba[:, 1]
                else:
                    # For multi-class, use max probability
                    predictions[self.model_names[i]] = proba.max(axis=1)
            elif hasattr(model, "predict"):
                predictions[self.model_names[i]] = model.predict(features)
            else:
                raise ValueError(f"Model {i} does not have predict() or predict_proba().")

        # Calculate weighted ensemble prediction
        predictions["ensemble"] = (predictions[self.model_names] * self.weights_).sum(axis=1)

        return predictions

    def generate_signals(
        self,
        features: pd.DataFrame,
        top_pct: float = 0.2,
        bottom_pct: float = 0.2,
        method: str = "ensemble"
    ) -> pd.DataFrame:
        """Generate long-short trading signals based on predictions.

        Args:
            features: Feature matrix.
            top_pct: Percentage of assets to go long (e.g., 0.2 = top 20%).
            bottom_pct: Percentage of assets to go short (e.g., 0.2 = bottom 20%).
            method: Prediction method - 'ensemble' uses weighted average,
                   or specify a model name to use individual model.

        Returns:
            DataFrame with columns: prediction, long, short, signal.
            signal: 1 for long, -1 for short, 0 for neutral.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Get predictions
        predictions = self.predict_ensemble(features)

        # Select prediction method
        if method == "ensemble":
            pred_col = "ensemble"
        elif method in self.model_names:
            pred_col = method
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'ensemble' or one of {self.model_names}."
            )

        # Create signals DataFrame
        signals = pd.DataFrame(index=features.index)
        signals["prediction"] = predictions[pred_col]

        # Calculate percentile thresholds for each timestamp (for cross-sectional ranking)
        signals["long"] = 0
        signals["short"] = 0
        signals["signal"] = 0

        # Handle NaN predictions
        valid_mask = signals["prediction"].notna()

        if valid_mask.sum() > 0:
            # Calculate thresholds
            top_threshold = signals.loc[valid_mask, "prediction"].quantile(1 - top_pct)
            bottom_threshold = signals.loc[valid_mask, "prediction"].quantile(bottom_pct)

            # Assign signals
            signals.loc[valid_mask & (signals["prediction"] >= top_threshold), "long"] = 1
            signals.loc[valid_mask & (signals["prediction"] <= bottom_threshold), "short"] = 1

            # Combined signal: 1 for long, -1 for short, 0 for neutral
            signals["signal"] = signals["long"] - signals["short"]

        return signals

    def backtest_signals(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        top_pct: float = 0.2,
        bottom_pct: float = 0.2
    ) -> dict:
        """Backtest the long-short strategy.

        Args:
            features: Feature matrix.
            returns: Forward returns for each observation.
            top_pct: Percentage for long positions.
            bottom_pct: Percentage for short positions.

        Returns:
            Dict with performance metrics.
        """
        # Generate signals
        signals = self.generate_signals(features, top_pct, bottom_pct)

        # Calculate strategy returns
        strategy_returns = signals["signal"] * returns

        # Filter valid returns
        valid_returns = strategy_returns[strategy_returns.notna()]

        if len(valid_returns) == 0:
            return {
                "total_return": 0.0,
                "mean_return": 0.0,
                "sharpe_ratio": 0.0,
                "hit_rate": 0.0,
                "n_trades": 0,
            }

        # Calculate metrics
        total_return = valid_returns.sum()
        mean_return = valid_returns.mean()
        std_return = valid_returns.std()
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0

        # Hit rate (percentage of profitable trades)
        hit_rate = (valid_returns > 0).sum() / len(valid_returns)

        # Number of trades (non-zero signals)
        n_trades = (signals["signal"] != 0).sum()

        return {
            "total_return": float(total_return),
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "sharpe_ratio": float(sharpe_ratio),
            "hit_rate": float(hit_rate),
            "n_trades": int(n_trades),
        }

    def get_model_weights(self) -> pd.Series:
        """Get the weights assigned to each model.

        Returns:
            Series of model weights.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        return pd.Series(self.weights_, index=self.model_names)
