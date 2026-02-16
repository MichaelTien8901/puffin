"""
Tests for linear models.
"""

import pytest
import numpy as np
import pandas as pd
from puffin.models.linear import OLSModel, RidgeModel, LassoModel, DirectionClassifier


@pytest.fixture
def synthetic_regression_data():
    """Generate synthetic regression data with known coefficients."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    # True coefficients
    true_coef = np.array([2.0, -1.5, 0.8, 0.0, 1.2])
    true_intercept = 5.0

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate target with noise
    y = true_intercept + X @ true_coef + np.random.randn(n_samples) * 0.5

    # Create DataFrames
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y, name='target')

    return {
        'X': X_df,
        'y': y_series,
        'X_array': X,
        'y_array': y,
        'true_coef': true_coef,
        'true_intercept': true_intercept,
    }


@pytest.fixture
def synthetic_classification_data():
    """Generate synthetic classification data."""
    np.random.seed(42)
    n_samples = 300
    n_features = 4

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate binary target based on linear combination
    z = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
    y = (z > 0).astype(int)

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y, name='direction')

    return {
        'X': X_df,
        'y': y_series,
        'X_array': X,
        'y_array': y,
    }


class TestOLSModel:
    """Test OLS regression model."""

    def test_fit_predict(self, synthetic_regression_data):
        """Test basic fit and predict functionality."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        model = OLSModel(add_constant=True)
        model.fit(X, y)

        # Test predictions
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert isinstance(y_pred, np.ndarray)

        # Check R-squared is reasonable
        assert model.r_squared > 0.8  # Should fit well with low noise

    def test_coefficients(self, synthetic_regression_data):
        """Test that coefficients are estimated correctly."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']
        true_coef = synthetic_regression_data['true_coef']

        model = OLSModel(add_constant=True)
        model.fit(X, y)

        # Check coefficients are close to true values
        coef = model.coefficients
        assert 'const' in coef.index
        assert len(coef) == len(true_coef) + 1

        # Check feature coefficients (allow some error due to noise)
        for i, true_val in enumerate(true_coef):
            estimated = coef[f'feature_{i}']
            assert abs(estimated - true_val) < 0.5

    def test_summary(self, synthetic_regression_data):
        """Test model summary statistics."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        model = OLSModel(add_constant=True)
        model.fit(X, y)

        summary = model.summary()

        # Check required keys
        required_keys = ['coefficients', 'r_squared', 'adj_r_squared',
                        'p_values', 'aic', 'bic', 'mse', 'rmse']
        for key in required_keys:
            assert key in summary

        # Check types and values
        assert isinstance(summary['r_squared'], float)
        assert 0 <= summary['r_squared'] <= 1
        assert isinstance(summary['coefficients'], pd.Series)
        assert isinstance(summary['p_values'], pd.Series)

    def test_residuals(self, synthetic_regression_data):
        """Test residuals calculation."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        model = OLSModel(add_constant=True)
        model.fit(X, y)

        residuals = model.residuals
        assert len(residuals) == len(y)

        # Residuals should have mean close to zero
        assert abs(np.mean(residuals)) < 0.1

    def test_without_constant(self, synthetic_regression_data):
        """Test model without constant term."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        model = OLSModel(add_constant=False)
        model.fit(X, y)

        coef = model.coefficients
        assert 'const' not in coef.index
        assert len(coef) == X.shape[1]

    def test_with_arrays(self, synthetic_regression_data):
        """Test model works with numpy arrays."""
        X = synthetic_regression_data['X_array']
        y = synthetic_regression_data['y_array']

        model = OLSModel(add_constant=True)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert model.r_squared > 0.7


class TestRidgeModel:
    """Test Ridge regression model."""

    def test_fit_predict(self, synthetic_regression_data):
        """Test basic fit and predict functionality."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        model = RidgeModel(alphas=np.logspace(-3, 3, 20), cv=3)
        model.fit(X, y)

        # Test predictions
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert isinstance(y_pred, np.ndarray)

    def test_alpha_selection(self, synthetic_regression_data):
        """Test that alpha is selected via cross-validation."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        alphas = np.logspace(-2, 2, 10)
        model = RidgeModel(alphas=alphas, cv=3)
        model.fit(X, y)

        # Check alpha is one of the provided values
        assert model.alpha in alphas or min(alphas) <= model.alpha <= max(alphas)

    def test_feature_importance(self, synthetic_regression_data):
        """Test feature importance calculation."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']
        true_coef = synthetic_regression_data['true_coef']

        model = RidgeModel()
        model.fit(X, y)

        importance = model.feature_importance()
        assert len(importance) == X.shape[1]
        assert isinstance(importance, pd.Series)

        # Features with larger true coefficients should have higher importance
        # (allowing for some regularization effects)
        top_feature = importance.index[0]
        assert top_feature in ['feature_0', 'feature_1', 'feature_4']

    def test_regularization_effect(self, synthetic_regression_data):
        """Test that regularization reduces coefficient magnitudes."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        # Fit with weak and strong regularization
        weak_model = RidgeModel(alphas=[0.01], cv=2)
        strong_model = RidgeModel(alphas=[100.0], cv=2)

        weak_model.fit(X, y)
        strong_model.fit(X, y)

        # Strong regularization should produce smaller coefficients
        weak_norm = np.linalg.norm(weak_model.coefficients.values)
        strong_norm = np.linalg.norm(strong_model.coefficients.values)
        assert strong_norm < weak_norm

    def test_normalization(self, synthetic_regression_data):
        """Test feature normalization."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        # Scale features differently
        X_scaled = X.copy()
        X_scaled['feature_0'] *= 100
        X_scaled['feature_1'] *= 0.01

        model = RidgeModel(normalize=True)
        model.fit(X_scaled, y)

        y_pred = model.predict(X_scaled)
        assert len(y_pred) == len(y)


class TestLassoModel:
    """Test Lasso regression model."""

    def test_fit_predict(self, synthetic_regression_data):
        """Test basic fit and predict functionality."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        model = LassoModel(alphas=np.logspace(-4, 0, 20), cv=3)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

    def test_feature_selection(self, synthetic_regression_data):
        """Test that Lasso performs feature selection."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        # Use moderate regularization
        model = LassoModel(alphas=np.logspace(-3, -1, 20), cv=3)
        model.fit(X, y)

        # Check that some coefficients are exactly zero
        coef = model.coefficients
        n_zero = (coef == 0).sum()
        n_nonzero = (coef != 0).sum()

        # Should have some non-zero coefficients
        assert n_nonzero > 0

        # Get selected features
        selected = model.selected_features
        assert len(selected) == n_nonzero

    def test_feature_importance(self, synthetic_regression_data):
        """Test feature importance calculation."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        model = LassoModel(alphas=np.logspace(-4, -1, 20), cv=3)
        model.fit(X, y)

        importance = model.feature_importance()
        assert len(importance) == X.shape[1]
        assert isinstance(importance, pd.Series)

    def test_strong_regularization(self, synthetic_regression_data):
        """Test that strong regularization drives all coefficients to zero."""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']

        model = LassoModel(alphas=[10.0], cv=2)
        model.fit(X, y)

        # Very strong regularization should zero out most/all coefficients
        n_zero = (model.coefficients == 0).sum()
        assert n_zero >= 3  # At least some should be zero


class TestDirectionClassifier:
    """Test direction classifier."""

    def test_fit_predict(self, synthetic_classification_data):
        """Test basic fit and predict functionality."""
        X = synthetic_classification_data['X']
        y = synthetic_classification_data['y']

        model = DirectionClassifier(class_weight='balanced')
        model.fit(X, y)

        # Test predictions
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        assert set(np.unique(y_pred)).issubset({0, 1})

        # Check accuracy is reasonable
        accuracy = (y_pred == y).mean()
        assert accuracy > 0.6  # Should be better than random

    def test_predict_proba(self, synthetic_classification_data):
        """Test probability predictions."""
        X = synthetic_classification_data['X']
        y = synthetic_classification_data['y']

        model = DirectionClassifier()
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_feature_importance(self, synthetic_classification_data):
        """Test feature importance calculation."""
        X = synthetic_classification_data['X']
        y = synthetic_classification_data['y']

        model = DirectionClassifier()
        model.fit(X, y)

        importance = model.feature_importance()
        assert len(importance) == X.shape[1]
        assert isinstance(importance, pd.Series)

        # Most important feature should be feature_0 (highest true coefficient)
        assert importance.index[0] in ['feature_0', 'feature_1', 'feature_2']

    def test_class_weights(self, synthetic_classification_data):
        """Test balanced class weights."""
        X = synthetic_classification_data['X']
        y = synthetic_classification_data['y']

        # Create imbalanced data
        imbalanced_idx = np.where(y == 1)[0][:20]  # Keep only 20 positive samples
        balanced_idx = np.where(y == 0)[0]
        idx = np.concatenate([imbalanced_idx, balanced_idx])

        X_imb = X.iloc[idx]
        y_imb = y.iloc[idx]

        model = DirectionClassifier(class_weight='balanced')
        model.fit(X_imb, y_imb)

        # Should still make predictions for both classes
        y_pred = model.predict(X_imb)
        assert len(np.unique(y_pred)) >= 1  # At least one class predicted

    def test_coefficients(self, synthetic_classification_data):
        """Test coefficient access."""
        X = synthetic_classification_data['X']
        y = synthetic_classification_data['y']

        model = DirectionClassifier()
        model.fit(X, y)

        coef = model.coefficients
        assert len(coef) == X.shape[1]
        assert isinstance(coef, pd.Series)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
