"""Tests for ensemble-based long-short strategy."""

import numpy as np
import pandas as pd
import pytest

from puffin.ensembles.random_forest import RandomForestTrader
from puffin.ensembles.long_short import EnsembleLongShort


@pytest.fixture
def synthetic_panel_data():
    """Create synthetic panel data for multiple assets."""
    np.random.seed(42)
    n_assets = 20
    n_periods = 100

    data = []
    for asset in range(n_assets):
        for period in range(n_periods):
            # Create features with some predictive power
            feature_1 = np.random.randn()
            feature_2 = np.random.randn()
            feature_3 = np.random.randn()

            # Forward return depends on features
            forward_return = (feature_1 * 0.02 + feature_2 * 0.01 + np.random.randn() * 0.05)

            data.append({
                "asset": asset,
                "period": period,
                "feature_1": feature_1,
                "feature_2": feature_2,
                "feature_3": feature_3,
                "forward_return": forward_return,
            })

    df = pd.DataFrame(data)
    features = df[["feature_1", "feature_2", "feature_3"]]
    returns = df["forward_return"]

    return features, returns


@pytest.fixture
def trained_models(synthetic_panel_data):
    """Create trained models for testing."""
    features, returns = synthetic_panel_data

    # Convert returns to binary signals for classification
    signals = (returns > 0).astype(int)

    model1 = RandomForestTrader(task="classification", random_state=42)
    model1.fit(features, signals, n_estimators=20, max_depth=3)

    model2 = RandomForestTrader(task="classification", random_state=43)
    model2.fit(features, signals, n_estimators=30, max_depth=5)

    return [model1, model2]


class TestEnsembleLongShort:
    def test_initialization_empty(self):
        ensemble = EnsembleLongShort()
        assert len(ensemble.models) == 0

    def test_initialization_with_list(self, trained_models):
        ensemble = EnsembleLongShort(models=trained_models)
        assert len(ensemble.models) == 2

    def test_initialization_with_dict(self, trained_models):
        models_dict = {
            "model_1": trained_models[0],
            "model_2": trained_models[1],
        }
        ensemble = EnsembleLongShort(models=models_dict)
        assert len(ensemble.models) == 2
        assert ensemble.model_names == ["model_1", "model_2"]

    def test_add_model(self):
        ensemble = EnsembleLongShort()
        model = RandomForestTrader(task="classification", random_state=42)
        ensemble.add_model(model, name="test_model")

        assert len(ensemble.models) == 1
        assert ensemble.model_names[0] == "test_model"

    def test_fit(self, synthetic_panel_data):
        features, returns = synthetic_panel_data
        signals = (returns > 0).astype(int)

        model1 = RandomForestTrader(task="classification", random_state=42)
        model2 = RandomForestTrader(task="classification", random_state=43)

        ensemble = EnsembleLongShort(models=[model1, model2])
        result = ensemble.fit(features, signals)

        assert result is ensemble  # Check method chaining
        assert ensemble.is_fitted
        assert ensemble.weights_ is not None
        assert len(ensemble.weights_) == 2

    def test_fit_empty_ensemble(self, synthetic_panel_data):
        features, returns = synthetic_panel_data
        ensemble = EnsembleLongShort()

        with pytest.raises(ValueError, match="No models in ensemble"):
            ensemble.fit(features, returns)

    def test_predict_ensemble(self, trained_models, synthetic_panel_data):
        features, returns = synthetic_panel_data
        signals = (returns > 0).astype(int)

        ensemble = EnsembleLongShort(models=trained_models)
        ensemble.fit(features, signals)

        predictions = ensemble.predict_ensemble(features)

        assert "ensemble" in predictions.columns
        assert len(predictions) == len(features)
        assert predictions["ensemble"].notna().sum() > 0

    def test_generate_signals(self, trained_models, synthetic_panel_data):
        features, returns = synthetic_panel_data
        signals = (returns > 0).astype(int)

        ensemble = EnsembleLongShort(models=trained_models)
        ensemble.fit(features, signals)

        trading_signals = ensemble.generate_signals(
            features, top_pct=0.2, bottom_pct=0.2
        )

        assert "prediction" in trading_signals.columns
        assert "long" in trading_signals.columns
        assert "short" in trading_signals.columns
        assert "signal" in trading_signals.columns
        assert len(trading_signals) == len(features)

        # Check that signals are correct
        assert set(trading_signals["signal"].unique()).issubset({-1, 0, 1})

        # Check that approximately 20% are long and 20% are short
        long_pct = (trading_signals["long"] == 1).sum() / len(trading_signals)
        short_pct = (trading_signals["short"] == 1).sum() / len(trading_signals)

        assert 0.15 < long_pct < 0.25
        assert 0.15 < short_pct < 0.25

    def test_generate_signals_individual_model(self, trained_models, synthetic_panel_data):
        features, returns = synthetic_panel_data
        signals = (returns > 0).astype(int)

        ensemble = EnsembleLongShort(
            models={"model_1": trained_models[0], "model_2": trained_models[1]}
        )
        ensemble.fit(features, signals)

        # Generate signals using individual model
        trading_signals = ensemble.generate_signals(
            features, top_pct=0.2, bottom_pct=0.2, method="model_1"
        )

        assert "signal" in trading_signals.columns
        assert len(trading_signals) == len(features)

    def test_generate_signals_invalid_method(self, trained_models, synthetic_panel_data):
        features, returns = synthetic_panel_data
        signals = (returns > 0).astype(int)

        ensemble = EnsembleLongShort(models=trained_models)
        ensemble.fit(features, signals)

        with pytest.raises(ValueError, match="Unknown method"):
            ensemble.generate_signals(features, method="invalid_model")

    def test_backtest_signals(self, trained_models, synthetic_panel_data):
        features, returns = synthetic_panel_data
        signals = (returns > 0).astype(int)

        ensemble = EnsembleLongShort(models=trained_models)
        ensemble.fit(features, signals)

        results = ensemble.backtest_signals(
            features, returns, top_pct=0.2, bottom_pct=0.2
        )

        assert "total_return" in results
        assert "mean_return" in results
        assert "std_return" in results
        assert "sharpe_ratio" in results
        assert "hit_rate" in results
        assert "n_trades" in results

        assert isinstance(results["total_return"], float)
        assert isinstance(results["n_trades"], int)
        assert 0 <= results["hit_rate"] <= 1

    def test_backtest_empty_returns(self, trained_models):
        # Create data with all NaN returns
        features = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
        })
        returns = pd.Series([np.nan] * 100)
        signals = pd.Series([1] * 100)

        ensemble = EnsembleLongShort(models=trained_models)
        ensemble.fit(features, signals)

        results = ensemble.backtest_signals(features, returns)

        assert results["total_return"] == 0.0
        assert results["n_trades"] == 0

    def test_get_model_weights(self, trained_models, synthetic_panel_data):
        features, returns = synthetic_panel_data
        signals = (returns > 0).astype(int)

        ensemble = EnsembleLongShort(
            models={"model_1": trained_models[0], "model_2": trained_models[1]}
        )
        ensemble.fit(features, signals)

        weights = ensemble.get_model_weights()

        assert len(weights) == 2
        assert "model_1" in weights.index
        assert "model_2" in weights.index
        assert np.isclose(weights.sum(), 1.0)

    def test_not_fitted_error(self, trained_models, synthetic_panel_data):
        features, returns = synthetic_panel_data

        ensemble = EnsembleLongShort(models=trained_models)

        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.predict_ensemble(features)

        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.generate_signals(features)

        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.get_model_weights()


class TestEnsembleLongShortWithNaN:
    def test_fit_with_nan_features(self, trained_models):
        features = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
        })
        # Introduce NaN
        features.iloc[0:10, 0] = np.nan

        signals = pd.Series((np.random.randn(100) > 0).astype(int))

        # Create fresh models for this test
        model1 = RandomForestTrader(task="classification", random_state=42)
        model2 = RandomForestTrader(task="classification", random_state=43)

        ensemble = EnsembleLongShort(models=[model1, model2])
        ensemble.fit(features, signals)

        assert ensemble.is_fitted

    def test_generate_signals_with_nan(self, trained_models, synthetic_panel_data):
        features, returns = synthetic_panel_data
        signals = (returns > 0).astype(int)

        ensemble = EnsembleLongShort(models=trained_models)
        ensemble.fit(features, signals)

        # Introduce NaN in prediction features
        features_nan = features.copy()
        features_nan.iloc[0:10, 0] = np.nan

        trading_signals = ensemble.generate_signals(features_nan)

        # Signals for NaN rows should be 0 (neutral)
        assert (trading_signals.iloc[0:10]["signal"] == 0).all()
