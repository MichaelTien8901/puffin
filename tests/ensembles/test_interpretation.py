"""Tests for SHAP-based model interpretation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from puffin.ensembles.random_forest import RandomForestTrader

# Check for optional dependencies
try:
    from puffin.ensembles.interpretation import ModelInterpreter
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from puffin.ensembles.xgboost_model import XGBoostTrader
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@pytest.fixture
def synthetic_data():
    """Create synthetic data with known patterns."""
    np.random.seed(42)
    n = 200

    # Create features with clear patterns
    feature_1 = np.random.randn(n)
    feature_2 = np.random.randn(n)
    feature_3 = np.random.randn(n)

    X = pd.DataFrame({
        "feature_1": feature_1,
        "feature_2": feature_2,
        "feature_3": feature_3,
        "noise": np.random.randn(n),
    })

    # Create target: positive when feature_1 + feature_2 > 0
    signal = (feature_1 + feature_2) > 0
    y = pd.Series(signal.astype(int), name="signal")

    return X, y


@pytest.fixture
def trained_model(synthetic_data):
    """Create a trained model for testing."""
    X, y = synthetic_data
    model = RandomForestTrader(task="classification", random_state=42)
    model.fit(X, y, n_estimators=30, max_depth=5)
    return model.model  # Return the underlying sklearn model


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestModelInterpreter:
    def test_initialization(self):
        interpreter = ModelInterpreter()
        assert interpreter is not None

    def test_shap_values(self, trained_model, synthetic_data):
        X, _ = synthetic_data
        interpreter = ModelInterpreter()

        shap_values = interpreter.shap_values(trained_model, X)

        assert shap_values is not None
        assert hasattr(shap_values, "values")
        assert shap_values.values.shape[0] == len(X)
        assert shap_values.values.shape[1] == X.shape[1]

    def test_plot_summary(self, trained_model, synthetic_data):
        X, _ = synthetic_data
        interpreter = ModelInterpreter()

        # Test dot plot
        fig = interpreter.plot_summary(trained_model, X, plot_type="dot", max_display=4)
        assert fig is not None

        # Test bar plot
        fig = interpreter.plot_summary(trained_model, X, plot_type="bar", max_display=4)
        assert fig is not None

    def test_plot_summary_invalid_type(self, trained_model, synthetic_data):
        X, _ = synthetic_data
        interpreter = ModelInterpreter()

        with pytest.raises(ValueError, match="Unknown plot_type"):
            interpreter.plot_summary(trained_model, X, plot_type="invalid")

    def test_plot_dependence(self, trained_model, synthetic_data):
        X, _ = synthetic_data
        interpreter = ModelInterpreter()

        fig = interpreter.plot_dependence(trained_model, X, feature="feature_1")
        assert fig is not None

    def test_plot_waterfall(self, trained_model, synthetic_data):
        X, _ = synthetic_data
        interpreter = ModelInterpreter()

        fig = interpreter.plot_waterfall(trained_model, X, index=0)
        assert fig is not None

    def test_plot_force(self, trained_model, synthetic_data):
        X, _ = synthetic_data
        interpreter = ModelInterpreter()

        force_plot = interpreter.plot_force(trained_model, X, index=0)
        assert force_plot is not None

    def test_feature_importance_comparison_shap(self, synthetic_data):
        X, y = synthetic_data
        interpreter = ModelInterpreter()

        # Train multiple models
        model1 = RandomForestTrader(task="classification", random_state=42)
        model1.fit(X, y, n_estimators=20, max_depth=3)

        model2 = RandomForestTrader(task="classification", random_state=43)
        model2.fit(X, y, n_estimators=20, max_depth=5)

        models_dict = {
            "model_1": model1.model,
            "model_2": model2.model,
        }

        importance_df = interpreter.feature_importance_comparison(
            models_dict, X, method="shap"
        )

        assert importance_df is not None
        assert "model_1" in importance_df.columns
        assert "model_2" in importance_df.columns
        assert "mean_importance" in importance_df.columns
        assert len(importance_df) == X.shape[1]

    def test_feature_importance_comparison_native(self, synthetic_data):
        X, y = synthetic_data
        interpreter = ModelInterpreter()

        # Train multiple models
        model1 = RandomForestTrader(task="classification", random_state=42)
        model1.fit(X, y, n_estimators=20, max_depth=3)

        model2 = RandomForestTrader(task="classification", random_state=43)
        model2.fit(X, y, n_estimators=20, max_depth=5)

        models_dict = {
            "model_1": model1.model,
            "model_2": model2.model,
        }

        importance_df = interpreter.feature_importance_comparison(
            models_dict, X, method="native"
        )

        assert importance_df is not None
        assert "model_1" in importance_df.columns
        assert "model_2" in importance_df.columns
        assert len(importance_df) == X.shape[1]

    def test_plot_importance_comparison(self, synthetic_data):
        X, y = synthetic_data
        interpreter = ModelInterpreter()

        # Train multiple models
        model1 = RandomForestTrader(task="classification", random_state=42)
        model1.fit(X, y, n_estimators=20, max_depth=3)

        model2 = RandomForestTrader(task="classification", random_state=43)
        model2.fit(X, y, n_estimators=20, max_depth=5)

        models_dict = {
            "model_1": model1.model,
            "model_2": model2.model,
        }

        fig = interpreter.plot_importance_comparison(
            models_dict, X, max_features=4, method="native"
        )

        assert fig is not None

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_shap_values_xgboost(self, synthetic_data):
        X, y = synthetic_data
        interpreter = ModelInterpreter()

        # Train XGBoost model
        model = XGBoostTrader(task="classification", random_state=42)
        model.fit(X, y)

        shap_values = interpreter.shap_values(model.model, X)

        assert shap_values is not None
        assert hasattr(shap_values, "values")

    def test_feature_importance_invalid_method(self, trained_model, synthetic_data):
        X, _ = synthetic_data
        interpreter = ModelInterpreter()

        models_dict = {"model": trained_model}

        with pytest.raises(ValueError, match="Unknown method"):
            interpreter.feature_importance_comparison(models_dict, X, method="invalid")


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestModelInterpreterWithNaN:
    def test_shap_values_with_nan(self, trained_model, synthetic_data):
        X, _ = synthetic_data
        interpreter = ModelInterpreter()

        # Introduce NaN values
        X_nan = X.copy()
        X_nan.iloc[0:10, 0] = np.nan

        # Should handle NaN by removing them
        shap_values = interpreter.shap_values(trained_model, X_nan)

        assert shap_values is not None
        # SHAP values should only be for clean data
        assert shap_values.values.shape[0] == len(X_nan) - 10


# Test with sklearn RandomForestClassifier directly
@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
def test_sklearn_random_forest(synthetic_data):
    X, y = synthetic_data
    interpreter = ModelInterpreter()

    # Train sklearn model directly
    model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)

    shap_values = interpreter.shap_values(model, X)

    assert shap_values is not None
    assert hasattr(shap_values, "values")
