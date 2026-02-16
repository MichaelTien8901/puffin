"""
Linear models for algorithmic trading.

This module provides linear regression models including OLS, Ridge, Lasso,
and logistic regression for direction prediction.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


class OLSModel:
    """
    Ordinary Least Squares regression model wrapper around statsmodels.

    This class provides a clean interface for OLS regression with automatic
    constant term addition and comprehensive diagnostics.

    Attributes:
        model: Fitted statsmodels OLS model
        coefficients: Model coefficients including intercept
        r_squared: R-squared value
        p_values: P-values for coefficients
        residuals: Model residuals
    """

    def __init__(self, add_constant: bool = True):
        """
        Initialize OLS model.

        Args:
            add_constant: If True, automatically add constant term to X
        """
        self.add_constant = add_constant
        self.model = None
        self._feature_names = None
        self._scaler = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'OLSModel':
        """
        Fit OLS model to data.

        Args:
            X: Features (n_samples, n_features)
            y: Target variable (n_samples,)

        Returns:
            self: Fitted model
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        else:
            self._feature_names = [f'x{i}' for i in range(X.shape[1])]

        # Convert y to array if Series
        if isinstance(y, pd.Series):
            y = y.values

        # Add constant if requested
        if self.add_constant:
            X = sm.add_constant(X)

        # Fit model
        self.model = sm.OLS(y, X).fit()

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using fitted model.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            predictions: Predicted values (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Convert DataFrame to array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Add constant if needed
        if self.add_constant:
            X = sm.add_constant(X)

        return self.model.predict(X)

    def summary(self) -> Dict[str, Any]:
        """
        Get model summary statistics.

        Returns:
            dict: Dictionary containing model diagnostics
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting summary")

        return {
            'coefficients': self.coefficients,
            'r_squared': self.r_squared,
            'adj_r_squared': self.model.rsquared_adj,
            'p_values': self.p_values,
            'aic': self.model.aic,
            'bic': self.model.bic,
            'f_statistic': self.model.fvalue,
            'f_pvalue': self.model.f_pvalue,
            'mse': self.model.mse_resid,
            'rmse': np.sqrt(self.model.mse_resid),
            'condition_number': self.model.condition_number
        }

    @property
    def coefficients(self) -> pd.Series:
        """Get model coefficients."""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing coefficients")

        names = ['const'] + self._feature_names if self.add_constant else self._feature_names
        return pd.Series(self.model.params, index=names)

    @property
    def r_squared(self) -> float:
        """Get R-squared value."""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing r_squared")
        return self.model.rsquared

    @property
    def p_values(self) -> pd.Series:
        """Get p-values for coefficients."""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing p_values")

        names = ['const'] + self._feature_names if self.add_constant else self._feature_names
        return pd.Series(self.model.pvalues, index=names)

    @property
    def residuals(self) -> np.ndarray:
        """Get model residuals."""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing residuals")
        return self.model.resid


class RidgeModel:
    """
    Ridge regression with cross-validated alpha selection.

    Ridge regression adds L2 regularization to prevent overfitting by
    penalizing large coefficient values.

    Attributes:
        model: Fitted sklearn Ridge or RidgeCV model
        alpha: Regularization strength
    """

    def __init__(self, alphas: Optional[np.ndarray] = None, cv: int = 5,
                 normalize: bool = True):
        """
        Initialize Ridge model.

        Args:
            alphas: Array of alpha values to try in cross-validation.
                   If None, uses default range.
            cv: Number of cross-validation folds
            normalize: If True, normalize features before fitting
        """
        if alphas is None:
            alphas = np.logspace(-3, 3, 50)

        self.alphas = alphas
        self.cv = cv
        self.normalize = normalize
        self.model = None
        self._feature_names = None
        self._scaler = StandardScaler() if normalize else None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'RidgeModel':
        """
        Fit Ridge model with cross-validated alpha selection.

        Args:
            X: Features (n_samples, n_features)
            y: Target variable (n_samples,)

        Returns:
            self: Fitted model
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        else:
            self._feature_names = [f'x{i}' for i in range(X.shape[1])]

        # Convert y to array if Series
        if isinstance(y, pd.Series):
            y = y.values

        # Normalize features if requested
        if self.normalize:
            X = self._scaler.fit_transform(X)

        # Fit with cross-validation
        self.model = RidgeCV(alphas=self.alphas, cv=self.cv)
        self.model.fit(X, y)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using fitted model.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            predictions: Predicted values (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Convert DataFrame to array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize if needed
        if self.normalize:
            X = self._scaler.transform(X)

        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        """
        Get feature importance based on absolute coefficient values.

        Returns:
            Series: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        importance = np.abs(self.model.coef_)
        return pd.Series(importance, index=self._feature_names).sort_values(ascending=False)

    @property
    def alpha(self) -> float:
        """Get selected alpha value."""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing alpha")
        return self.model.alpha_

    @property
    def coefficients(self) -> pd.Series:
        """Get model coefficients."""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing coefficients")
        return pd.Series(self.model.coef_, index=self._feature_names)


class LassoModel:
    """
    Lasso regression with cross-validated alpha selection.

    Lasso regression adds L1 regularization which can drive coefficients
    to exactly zero, performing automatic feature selection.

    Attributes:
        model: Fitted sklearn Lasso or LassoCV model
        alpha: Regularization strength
    """

    def __init__(self, alphas: Optional[np.ndarray] = None, cv: int = 5,
                 normalize: bool = True, max_iter: int = 10000):
        """
        Initialize Lasso model.

        Args:
            alphas: Array of alpha values to try in cross-validation.
                   If None, uses default range.
            cv: Number of cross-validation folds
            normalize: If True, normalize features before fitting
            max_iter: Maximum iterations for optimization
        """
        if alphas is None:
            alphas = np.logspace(-4, 0, 50)

        self.alphas = alphas
        self.cv = cv
        self.normalize = normalize
        self.max_iter = max_iter
        self.model = None
        self._feature_names = None
        self._scaler = StandardScaler() if normalize else None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'LassoModel':
        """
        Fit Lasso model with cross-validated alpha selection.

        Args:
            X: Features (n_samples, n_features)
            y: Target variable (n_samples,)

        Returns:
            self: Fitted model
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        else:
            self._feature_names = [f'x{i}' for i in range(X.shape[1])]

        # Convert y to array if Series
        if isinstance(y, pd.Series):
            y = y.values

        # Normalize features if requested
        if self.normalize:
            X = self._scaler.fit_transform(X)

        # Fit with cross-validation
        self.model = LassoCV(alphas=self.alphas, cv=self.cv, max_iter=self.max_iter)
        self.model.fit(X, y)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using fitted model.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            predictions: Predicted values (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Convert DataFrame to array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize if needed
        if self.normalize:
            X = self._scaler.transform(X)

        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        """
        Get feature importance based on absolute coefficient values.

        Non-zero coefficients indicate selected features.

        Returns:
            Series: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        importance = np.abs(self.model.coef_)
        return pd.Series(importance, index=self._feature_names).sort_values(ascending=False)

    @property
    def alpha(self) -> float:
        """Get selected alpha value."""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing alpha")
        return self.model.alpha_

    @property
    def coefficients(self) -> pd.Series:
        """Get model coefficients."""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing coefficients")
        return pd.Series(self.model.coef_, index=self._feature_names)

    @property
    def selected_features(self) -> list:
        """Get features with non-zero coefficients."""
        if self.model is None:
            raise ValueError("Model must be fitted before getting selected features")
        return [name for name, coef in zip(self._feature_names, self.model.coef_) if coef != 0]


class DirectionClassifier:
    """
    Logistic regression classifier for predicting price direction.

    This classifier predicts whether prices will move up or down,
    which is useful for directional trading strategies.

    Attributes:
        model: Fitted sklearn LogisticRegression model
    """

    def __init__(self, class_weight: Optional[str] = 'balanced',
                 normalize: bool = True, max_iter: int = 1000):
        """
        Initialize direction classifier.

        Args:
            class_weight: How to weight classes. 'balanced' adjusts weights
                         inversely proportional to class frequencies.
            normalize: If True, normalize features before fitting
            max_iter: Maximum iterations for optimization
        """
        self.class_weight = class_weight
        self.normalize = normalize
        self.max_iter = max_iter
        self.model = None
        self._feature_names = None
        self._scaler = StandardScaler() if normalize else None

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y_direction: Union[pd.Series, np.ndarray]) -> 'DirectionClassifier':
        """
        Fit classifier to data.

        Args:
            X: Features (n_samples, n_features)
            y_direction: Direction labels (n_samples,). Should be binary:
                        1 for up, 0 for down (or -1 for down)

        Returns:
            self: Fitted model
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        else:
            self._feature_names = [f'x{i}' for i in range(X.shape[1])]

        # Convert y to array if Series
        if isinstance(y_direction, pd.Series):
            y_direction = y_direction.values

        # Normalize features if requested
        if self.normalize:
            X = self._scaler.fit_transform(X)

        # Fit logistic regression
        self.model = LogisticRegression(
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=42
        )
        self.model.fit(X, y_direction)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict direction for new data.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            directions: Predicted directions (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Convert DataFrame to array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize if needed
        if self.normalize:
            X = self._scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probability of each direction.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            probabilities: Class probabilities (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Convert DataFrame to array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize if needed
        if self.normalize:
            X = self._scaler.transform(X)

        return self.model.predict_proba(X)

    def feature_importance(self) -> pd.Series:
        """
        Get feature importance based on absolute coefficient values.

        Returns:
            Series: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        importance = np.abs(self.model.coef_[0])
        return pd.Series(importance, index=self._feature_names).sort_values(ascending=False)

    @property
    def coefficients(self) -> pd.Series:
        """Get model coefficients."""
        if self.model is None:
            raise ValueError("Model must be fitted before accessing coefficients")
        return pd.Series(self.model.coef_[0], index=self._feature_names)
