"""Tests for Random Forest trading model."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from puffin.ensembles.random_forest import RandomForestTrader


@pytest.fixture
def synthetic_data():
    """Create synthetic data with known patterns."""
    np.random.seed(42)
    n = 300

    # Create features with predictive power
    feature_1 = np.random.randn(n)
    feature_2 = np.random.randn(n)
    feature_3 = np.random.randn(n)

    X = pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "feature_4": np.random.randn(n),  # Noise feature
    })

    # Create target with pattern: positive when feature_1 + feature_2 > 0
    signal = (feature_1 + feature_2) > 0
    y_cls = pd.Series(signal.astype(int), name="signal")

    # Create regression target
    y_reg = pd.Series(feature_1 * 0.5 + feature_2 * 0.3 + np.random.randn(n) * 0.1, name="returns")

    return X, y_cls, y_reg


class TestRandomForestTraderClassification:
    def test_fit(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = RandomForestTrader(task="classification", random_state=42)
        result = model.fit(X, y_cls, n_estimators=50, max_depth=5)

        assert result is model  # Check method chaining
        assert model.is_fitted
        assert model.feature_names_ == list(X.columns)

    def test_predict(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = RandomForestTrader(task="classification", random_state=42)
        model.fit(X, y_cls, n_estimators=50, max_depth=5)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert np.isfinite(predictions).sum() > 0
        assert set(np.unique(predictions[np.isfinite(predictions)])).issubset({0, 1})

    def test_predict_proba(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = RandomForestTrader(task="classification", random_state=42)
        model.fit(X, y_cls, n_estimators=50, max_depth=5)

        probas = model.predict_proba(X)
        assert probas.shape[0] == len(X)
        assert probas.shape[1] == 2  # Binary classification
        # Check probabilities sum to 1 for valid predictions
        valid_mask = ~np.isnan(probas[:, 0])
        assert np.allclose(probas[valid_mask].sum(axis=1), 1.0)

    def test_feature_importance(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = RandomForestTrader(task="classification", random_state=42)
        model.fit(X, y_cls, n_estimators=50, max_depth=5)

        importance = model.feature_importance()
        assert len(importance) == X.shape[1]
        assert all(importance >= 0)
        # Check that predictive features have higher importance
        assert importance["feature_1"] > importance["feature_4"]
        assert importance["feature_2"] > importance["feature_4"]

    def test_cross_validate(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = RandomForestTrader(task="classification", random_state=42)

        cv_results = model.cross_validate(
            X, y_cls, cv=TimeSeriesSplit(3), n_estimators=30, max_depth=5
        )

        assert "mean_score" in cv_results
        assert "std_score" in cv_results
        assert "fold_scores" in cv_results
        assert len(cv_results["fold_scores"]) == 3
        assert 0 <= cv_results["mean_score"] <= 1  # Accuracy should be in [0, 1]

    def test_not_fitted_error(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = RandomForestTrader(task="classification")

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(X)

        with pytest.raises(RuntimeError, match="not fitted"):
            model.feature_importance()


class TestRandomForestTraderRegression:
    def test_fit(self, synthetic_data):
        X, _, y_reg = synthetic_data
        model = RandomForestTrader(task="regression", random_state=42)
        result = model.fit(X, y_reg, n_estimators=50, max_depth=5)

        assert result is model
        assert model.is_fitted

    def test_predict(self, synthetic_data):
        X, _, y_reg = synthetic_data
        model = RandomForestTrader(task="regression", random_state=42)
        model.fit(X, y_reg, n_estimators=50, max_depth=5)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert np.isfinite(predictions).sum() > 0

    def test_predict_proba_error(self, synthetic_data):
        X, _, y_reg = synthetic_data
        model = RandomForestTrader(task="regression", random_state=42)
        model.fit(X, y_reg, n_estimators=50, max_depth=5)

        with pytest.raises(ValueError, match="only available for classification"):
            model.predict_proba(X)

    def test_cross_validate(self, synthetic_data):
        X, _, y_reg = synthetic_data
        model = RandomForestTrader(task="regression", random_state=42)

        cv_results = model.cross_validate(
            X, y_reg, cv=TimeSeriesSplit(3), n_estimators=30, max_depth=5
        )

        assert "mean_score" in cv_results
        assert "std_score" in cv_results
        assert "fold_scores" in cv_results
        assert len(cv_results["fold_scores"]) == 3


class TestRandomForestTraderWithNaN:
    def test_fit_with_nan(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        # Introduce NaN values
        X_nan = X.copy()
        X_nan.iloc[0:10, 0] = np.nan

        model = RandomForestTrader(task="classification", random_state=42)
        model.fit(X_nan, y_cls, n_estimators=30)

        assert model.is_fitted
        # Model should be trained on clean data only
        predictions = model.predict(X_nan)
        # Predictions for rows with NaN should be NaN
        assert np.isnan(predictions[0:10]).all()
        # Predictions for clean rows should be valid
        assert np.isfinite(predictions[10:]).sum() > 0
