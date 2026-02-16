"""Tests for supervised ML trading models."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from puffin.ml.supervised import TradingModel


@pytest.fixture
def training_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.randn(n),
    })
    # Target: classification (up/down)
    y_cls = pd.Series((np.random.randn(n) > 0).astype(int), name="target")
    # Target: regression (returns)
    y_reg = pd.Series(np.random.randn(n) * 0.01, name="target")
    return X, y_cls, y_reg


class TestTradingModelClassification:
    def test_train(self, training_data):
        X, y_cls, _ = training_data
        model = TradingModel(RandomForestClassifier(n_estimators=10, random_state=42))
        metrics = model.train(X, y_cls, n_splits=3)
        assert "avg_accuracy" in metrics
        assert "avg_f1" in metrics
        assert model.is_fitted

    def test_predict(self, training_data):
        X, y_cls, _ = training_data
        model = TradingModel(RandomForestClassifier(n_estimators=10, random_state=42))
        model.train(X, y_cls)
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert predictions.notna().sum() > 0

    def test_save_load(self, training_data, tmp_path):
        X, y_cls, _ = training_data
        model = TradingModel(RandomForestClassifier(n_estimators=10, random_state=42))
        model.train(X, y_cls)

        save_path = str(tmp_path / "test_model")
        model.save(save_path)

        loaded = TradingModel.load(save_path)
        assert loaded.is_fitted
        assert loaded.metadata["model_class"] == "RandomForestClassifier"


class TestTradingModelRegression:
    def test_train(self, training_data):
        X, _, y_reg = training_data
        model = TradingModel(
            RandomForestRegressor(n_estimators=10, random_state=42),
            model_type="regression",
        )
        metrics = model.train(X, y_reg, n_splits=3)
        assert "avg_rmse" in metrics
        assert "avg_r2" in metrics
