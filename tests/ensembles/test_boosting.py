"""Tests for boosting models (XGBoost, LightGBM, CatBoost)."""

import numpy as np
import pandas as pd
import pytest

# Check for optional dependencies
try:
    from puffin.ensembles.xgboost_model import XGBoostTrader
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from puffin.ensembles.lightgbm_model import LightGBMTrader
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from puffin.ensembles.catboost_model import CatBoostTrader
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


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

    # Create target with pattern
    signal = (feature_1 + feature_2) > 0
    y_cls = pd.Series(signal.astype(int), name="signal")

    # Create regression target
    y_reg = pd.Series(feature_1 * 0.5 + feature_2 * 0.3 + np.random.randn(n) * 0.1, name="returns")

    return X, y_cls, y_reg


@pytest.fixture
def categorical_data():
    """Create data with categorical features."""
    np.random.seed(42)
    n = 300

    X = pd.DataFrame({
        "numeric_1": np.random.randn(n),
        "numeric_2": np.random.randn(n),
        "category_1": np.random.choice(["A", "B", "C"], size=n),
        "category_2": np.random.choice(["X", "Y"], size=n),
    })

    # Target depends on both numeric and categorical features
    signal = (X["numeric_1"] > 0) & (X["category_1"] == "A")
    y = pd.Series(signal.astype(int), name="signal")

    return X, y, ["category_1", "category_2"]


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
class TestXGBoostTrader:
    def test_fit_classification(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = XGBoostTrader(task="classification", random_state=42)
        result = model.fit(X, y_cls)

        assert result is model
        assert model.is_fitted

    def test_predict(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = XGBoostTrader(task="classification", random_state=42)
        model.fit(X, y_cls)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert np.isfinite(predictions).sum() > 0

    def test_custom_params(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        custom_params = {
            "learning_rate": 0.05,
            "max_depth": 3,
            "n_estimators": 50,
        }
        model = XGBoostTrader(task="classification", random_state=42)
        model.fit(X, y_cls, params=custom_params)

        assert model.is_fitted
        assert model.model.get_params()["learning_rate"] == 0.05

    def test_tune_hyperparameters(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = XGBoostTrader(task="classification", random_state=42)

        param_grid = {
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "n_estimators": [30, 50],
        }

        best_params = model.tune_hyperparameters(X, y_cls, param_grid=param_grid, cv=3)

        assert model.is_fitted
        assert "learning_rate" in best_params
        assert "max_depth" in best_params

    def test_plot_importance(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = XGBoostTrader(task="classification", random_state=42)
        model.fit(X, y_cls)

        fig = model.plot_importance(max_features=10)
        assert fig is not None

    def test_regression(self, synthetic_data):
        X, _, y_reg = synthetic_data
        model = XGBoostTrader(task="regression", random_state=42)
        model.fit(X, y_reg)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert np.isfinite(predictions).sum() > 0


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
class TestLightGBMTrader:
    def test_fit_classification(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = LightGBMTrader(task="classification", random_state=42)
        result = model.fit(X, y_cls)

        assert result is model
        assert model.is_fitted

    def test_predict(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = LightGBMTrader(task="classification", random_state=42)
        model.fit(X, y_cls)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert np.isfinite(predictions).sum() > 0

    def test_categorical_features(self, categorical_data):
        X, y, cat_features = categorical_data
        # Encode categorical features as category dtype
        for col in cat_features:
            X[col] = X[col].astype("category")

        model = LightGBMTrader(task="classification", random_state=42)
        model.fit(X, y, categorical_features=cat_features)

        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_tune_hyperparameters(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = LightGBMTrader(task="classification", random_state=42)

        param_grid = {
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
            "n_estimators": [30, 50],
        }

        best_params = model.tune_hyperparameters(X, y_cls, param_grid=param_grid, cv=3)

        assert model.is_fitted
        assert "learning_rate" in best_params

    def test_plot_importance(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = LightGBMTrader(task="classification", random_state=42)
        model.fit(X, y_cls)

        fig = model.plot_importance(max_features=10)
        assert fig is not None

    def test_regression(self, synthetic_data):
        X, _, y_reg = synthetic_data
        model = LightGBMTrader(task="regression", random_state=42)
        model.fit(X, y_reg)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert np.isfinite(predictions).sum() > 0


@pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not installed")
class TestCatBoostTrader:
    def test_fit_classification(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = CatBoostTrader(task="classification", random_state=42)
        result = model.fit(X, y_cls)

        assert result is model
        assert model.is_fitted

    def test_predict(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = CatBoostTrader(task="classification", random_state=42)
        model.fit(X, y_cls)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert np.isfinite(predictions).sum() > 0

    def test_categorical_features(self, categorical_data):
        X, y, cat_features = categorical_data

        model = CatBoostTrader(task="classification", random_state=42)
        model.fit(X, y, cat_features=cat_features)

        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_feature_importance(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = CatBoostTrader(task="classification", random_state=42)
        model.fit(X, y_cls)

        importance = model.feature_importance()
        assert len(importance) == X.shape[1]
        assert all(importance >= 0)

    def test_plot_importance(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = CatBoostTrader(task="classification", random_state=42)
        model.fit(X, y_cls)

        fig = model.plot_importance(max_features=10)
        assert fig is not None

    def test_cross_validate(self, synthetic_data):
        X, y_cls, _ = synthetic_data
        model = CatBoostTrader(task="classification", random_state=42)

        cv_results = model.cross_validate(X, y_cls, cv=3)

        assert "mean_score" in cv_results
        assert "std_score" in cv_results
        assert "fold_scores" in cv_results
        assert len(cv_results["fold_scores"]) == 3

    def test_regression(self, synthetic_data):
        X, _, y_reg = synthetic_data
        model = CatBoostTrader(task="regression", random_state=42)
        model.fit(X, y_reg)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert np.isfinite(predictions).sum() > 0


# Integration test with all models (if available)
def test_all_models_produce_consistent_results(synthetic_data):
    """Test that all available models can be trained and produce predictions."""
    X, y_cls, _ = synthetic_data
    models = []

    if XGBOOST_AVAILABLE:
        models.append(("XGBoost", XGBoostTrader(task="classification", random_state=42)))

    if LIGHTGBM_AVAILABLE:
        models.append(("LightGBM", LightGBMTrader(task="classification", random_state=42)))

    if CATBOOST_AVAILABLE:
        models.append(("CatBoost", CatBoostTrader(task="classification", random_state=42)))

    if len(models) == 0:
        pytest.skip("No boosting libraries installed")

    for name, model in models:
        model.fit(X, y_cls)
        predictions = model.predict(X)
        assert len(predictions) == len(X), f"{name} produced wrong number of predictions"
        assert np.isfinite(predictions).sum() > 0, f"{name} produced no valid predictions"
