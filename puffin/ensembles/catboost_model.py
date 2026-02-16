"""CatBoost model for trading with native categorical feature handling."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.model_selection import TimeSeriesSplit


class CatBoostTrader:
    """CatBoost wrapper for trading signal generation with categorical support."""

    def __init__(self, task: str = "classification", random_state: int = 42):
        """
        Args:
            task: 'classification' or 'regression'.
            random_state: Random seed for reproducibility.
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost is not installed. Install it with: pip install catboost"
            )

        self.task = task
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names_ = None
        self.cat_features_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: list = None,
        params: dict = None
    ) -> "CatBoostTrader":
        """Fit the CatBoost model with financial-optimized defaults.

        Args:
            X: Feature matrix.
            y: Target variable.
            cat_features: List of categorical feature names or indices.
            params: CatBoost parameters. If None, uses financial-optimized defaults.

        Returns:
            self for chaining.
        """
        # Store feature names
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.cat_features_ = cat_features

        # Clean data
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        # Default parameters optimized for financial data
        default_params = {
            "learning_rate": 0.01,
            "depth": 5,
            "l2_leaf_reg": 3.0,  # L2 regularization
            "iterations": 100,
            "random_state": self.random_state,
            "verbose": False,
            "allow_writing_files": False,  # Don't write temp files
        }

        # Update with user-provided params
        if params is not None:
            default_params.update(params)

        # Initialize model
        if self.task == "classification":
            # Determine loss function based on number of classes
            n_classes = len(np.unique(y_clean))
            if n_classes == 2:
                default_params["loss_function"] = "Logloss"
            else:
                default_params["loss_function"] = "MultiClass"

            self.model = CatBoostClassifier(**default_params)
        else:
            default_params["loss_function"] = "RMSE"
            self.model = CatBoostRegressor(**default_params)

        # Fit model with categorical features
        self.model.fit(X_clean, y_clean, cat_features=cat_features)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        mask = X.notna().all(axis=1)
        predictions = np.full(len(X), np.nan)

        if mask.any():
            predictions[mask] = self.model.predict(X[mask])

        return predictions

    def feature_importance(self) -> pd.Series:
        """Get feature importances sorted in descending order.

        Returns:
            Series of feature importances indexed by feature names.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importances = self.model.get_feature_importance()

        if self.feature_names_ is not None:
            importance_series = pd.Series(importances, index=self.feature_names_)
        else:
            importance_series = pd.Series(
                importances, index=[f"feature_{i}" for i in range(len(importances))]
            )

        return importance_series.sort_values(ascending=False)

    def plot_importance(self, max_features: int = 20) -> plt.Figure:
        """Plot feature importance.

        Args:
            max_features: Maximum number of features to display.

        Returns:
            Matplotlib figure object.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get feature importance
        importance_series = self.feature_importance()

        # Limit to top features
        importance_df = importance_series.head(max_features).sort_values(ascending=True)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, max_features * 0.3)))
        ax.barh(importance_df.index, importance_df.values)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {max_features} Feature Importances (CatBoost)")
        plt.tight_layout()

        return fig

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        cat_features: list = None,
        params: dict = None
    ) -> dict:
        """Perform time-series cross-validation.

        Args:
            X: Feature matrix.
            y: Target variable.
            cv: Number of cross-validation folds.
            cat_features: List of categorical feature names or indices.
            params: CatBoost parameters.

        Returns:
            Dict with mean_score, std_score, and fold_scores.
        """
        # Clean data
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        # Default parameters
        default_params = {
            "learning_rate": 0.01,
            "depth": 5,
            "iterations": 100,
            "random_state": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
        }

        if params is not None:
            default_params.update(params)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)
        fold_scores = []

        for train_idx, test_idx in tscv.split(X_clean):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

            # Initialize and fit model
            if self.task == "classification":
                model = CatBoostClassifier(**default_params)
            else:
                model = CatBoostRegressor(**default_params)

            model.fit(X_train, y_train, cat_features=cat_features)
            score = model.score(X_test, y_test)
            fold_scores.append(score)

        return {
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "fold_scores": [float(s) for s in fold_scores],
        }

    def get_best_iteration(self) -> int:
        """Get the best iteration (tree count) from the model.

        Returns:
            Best iteration number.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.get_best_iteration()
