"""Random Forest model for trading signal prediction."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
)


class RandomForestTrader:
    """Random Forest wrapper for trading signal generation with feature importance."""

    def __init__(self, task: str = "classification", random_state: int = 42):
        """
        Args:
            task: 'classification' or 'regression'.
            random_state: Random seed for reproducibility.
        """
        self.task = task
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 10,
        **kwargs
    ) -> "RandomForestTrader":
        """Fit the random forest model.

        Args:
            X: Feature matrix.
            y: Target variable.
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree.
            **kwargs: Additional parameters for RandomForest.

        Returns:
            self for chaining.
        """
        # Store feature names
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None

        # Clean data
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        # Initialize model
        if self.task == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                **kwargs
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                **kwargs
            )

        # Fit model
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

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions (classification only).

        Args:
            X: Feature matrix.

        Returns:
            Array of class probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.task != "classification":
            raise ValueError("predict_proba only available for classification tasks.")

        mask = X.notna().all(axis=1)
        n_classes = len(self.model.classes_)
        predictions = np.full((len(X), n_classes), np.nan)

        if mask.any():
            predictions[mask] = self.model.predict_proba(X[mask])

        return predictions

    def feature_importance(self) -> pd.Series:
        """Get feature importances sorted in descending order.

        Returns:
            Series of feature importances indexed by feature names.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importances = self.model.feature_importances_

        if self.feature_names_ is not None:
            importance_series = pd.Series(importances, index=self.feature_names_)
        else:
            importance_series = pd.Series(
                importances, index=[f"feature_{i}" for i in range(len(importances))]
            )

        return importance_series.sort_values(ascending=False)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: TimeSeriesSplit = None,
        n_estimators: int = 100,
        max_depth: int = 10,
        **kwargs
    ) -> dict:
        """Perform time-series cross-validation.

        Args:
            X: Feature matrix.
            y: Target variable.
            cv: TimeSeriesSplit object. If None, uses TimeSeriesSplit(5).
            n_estimators: Number of trees.
            max_depth: Maximum depth.
            **kwargs: Additional parameters for RandomForest.

        Returns:
            Dict with mean_score, std_score, and fold_scores.
        """
        if cv is None:
            cv = TimeSeriesSplit(n_splits=5)

        # Clean data
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        fold_scores = []

        for train_idx, test_idx in cv.split(X_clean):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

            # Initialize and fit model
            if self.task == "classification":
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                    **kwargs
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                # Use accuracy as the score for classification
                score = accuracy_score(y_test, y_pred)
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                    **kwargs
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                # Use R2 score for regression
                score = r2_score(y_test, y_pred)

            fold_scores.append(score)

        return {
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "fold_scores": [float(s) for s in fold_scores],
        }
