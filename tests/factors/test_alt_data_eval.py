"""Tests for alternative data evaluation framework."""

import numpy as np
import pandas as pd
import pytest

from puffin.factors.alt_data_eval import AltDataEvaluator


@pytest.fixture
def evaluator():
    return AltDataEvaluator(risk_free_rate=0.02)


@pytest.fixture
def sample_signal_returns():
    """Generate sample signal and returns data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Create signal with some predictive power
    signal = pd.Series(np.random.randn(100), index=dates)

    # Returns correlated with signal
    noise = np.random.randn(100) * 0.5
    returns = signal * 0.3 + noise
    returns = pd.Series(returns / 100, index=dates)  # Scale to realistic returns

    return signal, returns


@pytest.fixture
def sample_data():
    """Generate sample alternative data."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    data = pd.DataFrame({
        "metric_a": np.random.randn(50),
        "metric_b": np.random.randn(50),
        "metric_c": np.random.randn(50),
    }, index=dates)

    # Add some missing values
    data.iloc[5:8, 0] = np.nan
    data.iloc[10, 1] = np.nan

    return data


def test_evaluator_initialization():
    evaluator = AltDataEvaluator(risk_free_rate=0.03)
    assert evaluator.risk_free_rate == 0.03


def test_evaluate_signal_quality(evaluator, sample_signal_returns):
    signal, returns = sample_signal_returns

    result = evaluator.evaluate_signal_quality(signal, returns)

    assert "ic" in result
    assert "ir" in result
    assert "t_stat" in result
    assert "p_value" in result
    assert "decay_half_life" in result

    # IC should be reasonable
    assert -1 <= result["ic"] <= 1

    # Should have valid statistics
    assert not np.isnan(result["t_stat"])
    assert not np.isnan(result["p_value"])


def test_evaluate_signal_quality_perfect_correlation(evaluator):
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    signal = pd.Series(np.arange(50), index=dates)
    returns = signal / 100  # Perfect correlation

    result = evaluator.evaluate_signal_quality(signal, returns)

    # Should have high IC close to 1
    assert result["ic"] > 0.9


def test_evaluate_signal_quality_no_correlation(evaluator):
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    signal = pd.Series(np.random.randn(50), index=dates)
    returns = pd.Series(np.random.randn(50), index=dates) / 100

    result = evaluator.evaluate_signal_quality(signal, returns)

    # IC should be low (close to 0)
    assert abs(result["ic"]) < 0.5


def test_evaluate_signal_quality_insufficient_data(evaluator):
    signal = pd.Series([1, 2, 3])
    returns = pd.Series([0.01, 0.02, 0.03])

    result = evaluator.evaluate_signal_quality(signal, returns)

    # Should return NaN for insufficient data
    assert np.isnan(result["ic"])
    assert np.isnan(result["ir"])


def test_evaluate_data_quality(evaluator, sample_data):
    result = evaluator.evaluate_data_quality(sample_data)

    assert "completeness" in result
    assert "missing_pct" in result
    assert "coverage_days" in result
    assert "update_frequency" in result
    assert "data_points" in result

    # Should calculate completeness correctly
    assert 0 <= result["completeness"] <= 100
    assert result["completeness"] + result["missing_pct"] == pytest.approx(100.0)

    # Should have coverage
    assert result["coverage_days"] > 0
    assert result["data_points"] == 50


def test_evaluate_data_quality_complete_data(evaluator):
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    data = pd.DataFrame({
        "metric_a": np.random.randn(30),
        "metric_b": np.random.randn(30),
    }, index=dates)

    result = evaluator.evaluate_data_quality(data)

    # Should be 100% complete
    assert result["completeness"] == 100.0
    assert result["missing_pct"] == 0.0


def test_evaluate_data_quality_empty_dataframe(evaluator):
    data = pd.DataFrame()

    result = evaluator.evaluate_data_quality(data)

    assert result["completeness"] == 0.0
    assert result["missing_pct"] == 100.0
    assert result["data_points"] == 0


def test_evaluate_data_quality_non_datetime_index(evaluator):
    data = pd.DataFrame({
        "metric_a": [1, 2, 3, 4, 5],
        "metric_b": [5, 4, 3, 2, 1],
    })

    result = evaluator.evaluate_data_quality(data)

    # Should still calculate completeness
    assert result["completeness"] == 100.0
    # Coverage should be row count for non-datetime index
    assert result["coverage_days"] == 5


def test_evaluate_technical(evaluator, sample_data):
    result = evaluator.evaluate_technical(sample_data)

    assert "storage_mb" in result
    assert "row_count" in result
    assert "column_count" in result
    assert "memory_per_row_kb" in result
    assert "has_datetime_index" in result
    assert "numeric_columns" in result

    # Should have correct dimensions
    assert result["row_count"] == 50
    assert result["column_count"] == 3
    assert result["has_datetime_index"] is True
    assert result["numeric_columns"] == 3
    assert result["storage_mb"] > 0


def test_evaluate_technical_empty_dataframe(evaluator):
    data = pd.DataFrame()

    result = evaluator.evaluate_technical(data)

    assert result["storage_mb"] == 0.0
    assert result["row_count"] == 0
    assert result["column_count"] == 0


def test_backtest_signal(evaluator, sample_signal_returns):
    signal, returns = sample_signal_returns

    result = evaluator.backtest_signal(signal, returns, quantiles=5)

    assert "quantile_returns" in result
    assert "long_short_return" in result
    assert "long_short_sharpe" in result

    # Should have returns for each quantile
    quantile_returns = result["quantile_returns"]
    assert len(quantile_returns) > 0

    # Each quantile should have statistics
    for q_name, q_stats in quantile_returns.items():
        assert "mean_return" in q_stats
        assert "std_return" in q_stats
        assert "sharpe" in q_stats
        assert "count" in q_stats


def test_backtest_signal_positive_signal(evaluator):
    """Test with signal that should predict positive returns."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Strong positive signal
    signal = pd.Series(np.random.randn(100), index=dates)
    returns = signal * 0.01 + np.random.randn(100) * 0.001  # Highly correlated

    result = evaluator.backtest_signal(signal, returns, quantiles=5)

    # Long-short should be positive
    assert result["long_short_return"] > 0


def test_backtest_signal_insufficient_data(evaluator):
    signal = pd.Series([1, 2, 3, 4, 5])
    returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.02])

    result = evaluator.backtest_signal(signal, returns, quantiles=5)

    # Should return NaN for insufficient data
    assert np.isnan(result["long_short_return"])


def test_backtest_signal_different_quantiles(evaluator, sample_signal_returns):
    signal, returns = sample_signal_returns

    result_3 = evaluator.backtest_signal(signal, returns, quantiles=3)
    result_10 = evaluator.backtest_signal(signal, returns, quantiles=10)

    # Should have different numbers of quantiles
    assert len(result_3["quantile_returns"]) <= 3
    assert len(result_10["quantile_returns"]) <= 10


def test_calculate_sharpe(evaluator):
    # Generate returns with positive mean
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)  # Daily returns, positive drift

    sharpe = evaluator._calculate_sharpe(returns)

    # Should return a reasonable Sharpe ratio
    assert not np.isnan(sharpe)
    assert -5 < sharpe < 5  # Reasonable range


def test_calculate_sharpe_insufficient_data(evaluator):
    returns = pd.Series([0.01])

    sharpe = evaluator._calculate_sharpe(returns)

    assert np.isnan(sharpe)


def test_calculate_sharpe_zero_volatility(evaluator):
    returns = pd.Series([0.01] * 100)  # Constant returns

    sharpe = evaluator._calculate_sharpe(returns)

    assert np.isnan(sharpe)


def test_decay_half_life_calculation(evaluator):
    """Test signal decay calculation."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Signal with decaying predictive power
    signal = pd.Series(np.random.randn(100), index=dates)
    returns = pd.Series(np.random.randn(100) / 100, index=dates)

    decay = evaluator._calculate_decay_half_life(signal, returns)

    # Should return a valid decay measure (or NaN if no clear decay)
    assert isinstance(decay, (int, float))
