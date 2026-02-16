"""LightGBM model for trading with categorical feature support."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


class LightGBMTrader:
    """LightGBM wrapper for trading signal generation with categorical feature support."""

    def __init__(self, task: str = "classification", random_state: int = 42):
        """
        Args:
            task: 'classification' or 'regression'.
            random_state: Random seed for reproducibility.
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install it with: pip install lightgbm"
            )

        self.task = task
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names_ = None
        self.categorical_features_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict = None,
        categorical_features: list = None
    ) -> "LightGBMTrader":
        """Fit the LightGBM model with financial-optimized defaults.

        Args:
            X: Feature matrix.
            y: Target variable.
            params: LightGBM parameters. If None, uses financial-optimized defaults.
            categorical_features: List of categorical feature names or indices.

        Returns:
            self for chaining.
        """
        # Store feature names
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.categorical_features_ = categorical_features

        # Clean data
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        # Default parameters optimized for financial data
        default_params = {
            "learning_rate": 0.01,
            "max_depth": 5,
            "num_leaves": 31,  # Should be < 2^max_depth
            "min_child_samples": 20,  # Minimum samples per leaf
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,  # L1 regularization
            "reg_lambda": 1.0,  # L2 regularization
            "n_estimators": 100,
            "random_state": self.random_state,
            "verbose": -1,  # Suppress warnings
        }

        # Update with user-provided params
        if params is not None:
            default_params.update(params)

        # Initialize model
        if self.task == "classification":
            # Determine objective based on number of classes
            n_classes = len(np.unique(y_clean))
            if n_classes == 2:
                default_params["objective"] = "binary"
                default_params["metric"] = "binary_logloss"
            else:
                default_params["objective"] = "multiclass"
                default_params["num_class"] = n_classes
                default_params["metric"] = "multi_logloss"

            self.model = lgb.LGBMClassifier(**default_params)
        else:
            default_params["objective"] = "regression"
            default_params["metric"] = "rmse"
            self.model = lgb.LGBMRegressor(**default_params)

        # Fit model with categorical features
        if categorical_features is not None:
            self.model.fit(X_clean, y_clean, categorical_feature=categorical_features)
        else:
            self.model.fit(X_clean, y_clean)

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

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict = None,
        cv: int = 5,
        categorical_features: list = None
    ) -> dict:
        """Tune hyperparameters using time-series cross-validation.

        Args:
            X: Feature matrix.
            y: Target variable.
            param_grid: Dictionary of parameters to search. If None, uses default grid.
            cv: Number of cross-validation folds.
            categorical_features: List of categorical feature names or indices.

        Returns:
            Dictionary of best parameters.
        """
        # Clean data
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "num_leaves": [15, 31, 63],
                "min_child_samples": [10, 20, 30],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "n_estimators": [50, 100, 200],
            }

        # Initialize base model
        if self.task == "classification":
            base_model = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        else:
            base_model = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)

        # Grid search
        fit_params = {}
        if categorical_features is not None:
            fit_params["categorical_feature"] = categorical_features

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring="accuracy" if self.task == "classification" else "r2",
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_clean, y_clean, **fit_params)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        self.categorical_features_ = categorical_features

        return grid_search.best_params_

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
        importance = self.model.feature_importances_

        if self.feature_names_ is not None:
            importance_df = pd.DataFrame({
                "feature": self.feature_names_,
                "importance": importance
            })
        else:
            importance_df = pd.DataFrame({
                "feature": [f"feature_{i}" for i in range(len(importance))],
                "importance": importance
            })

        # Sort and limit
        importance_df = importance_df.sort_values("importance", ascending=True).tail(max_features)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, max_features * 0.3)))
        ax.barh(importance_df["feature"], importance_df["importance"])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {max_features} Feature Importances (LightGBM)")
        plt.tight_layout()

        return fig

    def get_booster(self):
        """Get the underlying LightGBM Booster object.

        Returns:
            LightGBM Booster object.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.booster_
