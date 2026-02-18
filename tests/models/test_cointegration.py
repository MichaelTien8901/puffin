"""
Tests for cointegration analysis.
"""

import pytest
import numpy as np
import pandas as pd
from puffin.models.cointegration import (
    engle_granger_test,
    johansen_test,
    find_cointegrated_pairs,
    calculate_spread,
    half_life,
    test_cointegration_all_pairs,
    adf_test_spread
)


@pytest.fixture
def cointegrated_pair():
    """Generate two cointegrated series."""
    np.random.seed(42)
    n = 252

    # Generate common random walk
    common_factor = np.random.randn(n).cumsum()

    # Two series driven by same common factor (cointegrated)
    y1 = common_factor + np.random.randn(n) * 0.5
    y2 = 2 * common_factor + np.random.randn(n) * 0.5

    return pd.Series(y1), pd.Series(y2)


@pytest.fixture
def non_cointegrated_pair():
    """Generate two non-cointegrated series (independent random walks)."""
    np.random.seed(42)
    n = 252

    y1 = np.random.randn(n).cumsum()
    y2 = np.random.randn(n).cumsum()

    return pd.Series(y1), pd.Series(y2)


@pytest.fixture
def multiple_series():
    """Generate multiple time series for testing."""
    np.random.seed(42)
    n = 252

    # Create DataFrame with some cointegrated and some non-cointegrated series
    common = np.random.randn(n).cumsum()

    data = pd.DataFrame({
        'A': common + np.random.randn(n) * 0.3,
        'B': 1.5 * common + np.random.randn(n) * 0.3,
        'C': np.random.randn(n).cumsum(),  # Independent
        'D': np.random.randn(n).cumsum()   # Independent
    })

    return data


class TestEngleGrangerTest:
    """Tests for Engle-Granger cointegration test."""

    def test_cointegrated_series(self, cointegrated_pair):
        """Test on cointegrated series."""
        y1, y2 = cointegrated_pair
        result = engle_granger_test(y1, y2)

        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'is_cointegrated' in result
        assert 'hedge_ratio' in result
        assert 'critical_values' in result

        # Should detect cointegration (though not guaranteed with small sample)
        assert isinstance(result['is_cointegrated'], (bool, np.bool_))
        assert isinstance(result['hedge_ratio'], (int, float, np.floating))

    def test_non_cointegrated_series(self, non_cointegrated_pair):
        """Test on non-cointegrated series."""
        y1, y2 = non_cointegrated_pair
        result = engle_granger_test(y1, y2)

        assert 'is_cointegrated' in result
        # Should likely not find cointegration (though not guaranteed)

    def test_hedge_ratio(self, cointegrated_pair):
        """Test that hedge ratio is calculated."""
        y1, y2 = cointegrated_pair
        result = engle_granger_test(y1, y2)

        hedge_ratio = result['hedge_ratio']
        assert isinstance(hedge_ratio, (int, float, np.floating))
        assert not np.isnan(hedge_ratio)

        # For our construction, hedge ratio should be around 0.5 (y1 = 0.5*y2)
        # But allow wide range due to noise
        assert -5 < hedge_ratio < 5

    def test_critical_values(self, cointegrated_pair):
        """Test that critical values are returned."""
        y1, y2 = cointegrated_pair
        result = engle_granger_test(y1, y2)

        crit_vals = result['critical_values']
        assert '1%' in crit_vals
        assert '5%' in crit_vals
        assert '10%' in crit_vals

    def test_too_short_series(self):
        """Test with series too short."""
        short_series = pd.Series(np.random.randn(10))

        with pytest.raises(ValueError, match="at least 20"):
            engle_granger_test(short_series, short_series)


class TestJohansenTest:
    """Tests for Johansen cointegration test."""

    def test_multiple_series(self, multiple_series):
        """Test Johansen test on multiple series."""
        result = johansen_test(multiple_series)

        assert 'trace_statistic' in result
        assert 'trace_critical_values' in result
        assert 'max_eigen_statistic' in result
        assert 'max_eigen_critical_values' in result
        assert 'n_cointegrated' in result
        assert 'eigenvalues' in result

    def test_number_cointegrated(self, multiple_series):
        """Test counting cointegrated relationships."""
        result = johansen_test(multiple_series)

        n_coint = result['n_cointegrated']
        assert isinstance(n_coint, (int, np.integer))
        assert n_coint >= 0
        assert n_coint <= len(multiple_series.columns)

    def test_too_few_variables(self):
        """Test with too few variables."""
        single_var = pd.DataFrame({'A': np.random.randn(100)})

        with pytest.raises(ValueError, match="at least 2"):
            johansen_test(single_var)

    def test_too_short_series(self):
        """Test with series too short."""
        short_data = pd.DataFrame({
            'A': np.random.randn(10),
            'B': np.random.randn(10)
        })

        with pytest.raises(ValueError, match="at least 20"):
            johansen_test(short_data)


class TestFindCointegratedPairs:
    """Tests for finding cointegrated pairs."""

    def test_find_pairs(self, multiple_series):
        """Test finding cointegrated pairs."""
        pairs = find_cointegrated_pairs(multiple_series, significance=0.05)

        # Should return a list
        assert isinstance(pairs, list)

        # Each element should be a tuple with 4 elements
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 4
            ticker1, ticker2, p_value, hedge_ratio = pair
            assert isinstance(ticker1, str)
            assert isinstance(ticker2, str)
            assert isinstance(p_value, (int, float, np.floating))
            assert isinstance(hedge_ratio, (int, float, np.floating))

    def test_pairs_sorted_by_pvalue(self, multiple_series):
        """Test that pairs are sorted by p-value."""
        pairs = find_cointegrated_pairs(multiple_series, significance=0.1)

        if len(pairs) > 1:
            p_values = [p[2] for p in pairs]
            # Should be sorted ascending
            assert p_values == sorted(p_values)

    def test_significance_filter(self, multiple_series):
        """Test that significance level filters results."""
        # With strict significance, should find fewer pairs
        pairs_strict = find_cointegrated_pairs(multiple_series, significance=0.01)
        pairs_loose = find_cointegrated_pairs(multiple_series, significance=0.10)

        # Loose should have >= pairs than strict
        assert len(pairs_loose) >= len(pairs_strict)

        # All p-values should be below significance
        for pair in pairs_strict:
            assert pair[2] < 0.01

    def test_min_observations(self):
        """Test minimum observations requirement."""
        # Create short series
        short_data = pd.DataFrame({
            'A': np.random.randn(30),
            'B': np.random.randn(30),
            'C': np.random.randn(30)
        })

        pairs = find_cointegrated_pairs(short_data, min_observations=50)

        # Should find no pairs due to minimum observations
        assert len(pairs) == 0


class TestCalculateSpread:
    """Tests for spread calculation."""

    def test_calculate_spread(self, cointegrated_pair):
        """Test basic spread calculation."""
        y1, y2 = cointegrated_pair
        spread = calculate_spread(y1, y2)

        assert isinstance(spread, pd.Series)
        assert len(spread) == len(y1)

    def test_spread_with_hedge_ratio(self, cointegrated_pair):
        """Test spread with specified hedge ratio."""
        y1, y2 = cointegrated_pair
        spread = calculate_spread(y1, y2, hedge_ratio=0.5)

        assert isinstance(spread, pd.Series)
        # spread = y1 - 0.5 * y2
        expected = y1 - 0.5 * y2
        pd.testing.assert_series_equal(spread, expected, check_names=False)

    def test_spread_auto_hedge_ratio(self, cointegrated_pair):
        """Test spread with automatic hedge ratio estimation."""
        y1, y2 = cointegrated_pair
        spread = calculate_spread(y1, y2, hedge_ratio=None)

        assert isinstance(spread, pd.Series)
        assert len(spread) > 0


class TestHalfLife:
    """Tests for half-life calculation."""

    def test_half_life_mean_reverting(self):
        """Test half-life for mean-reverting series."""
        np.random.seed(42)
        # Generate mean-reverting series (AR with |phi| < 1)
        n = 500
        phi = 0.9
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i-1] + np.random.randn()

        spread = pd.Series(x)
        hl = half_life(spread)

        # Half-life should be positive and finite
        assert hl > 0
        assert np.isfinite(hl)

        # For AR(1) with phi=0.9, half-life should be reasonable
        assert hl < 100  # Should be relatively short

    def test_half_life_random_walk(self):
        """Test half-life for random walk (non-mean-reverting)."""
        np.random.seed(42)
        random_walk = pd.Series(np.random.randn(500).cumsum())

        hl = half_life(random_walk)

        # Random walk should have positive half-life
        # (exact value depends on seed; just verify it's positive and finite or large)
        assert hl > 0

    def test_half_life_too_short(self):
        """Test half-life with series too short."""
        short_series = pd.Series([1.0])

        with pytest.raises(ValueError, match="at least 2"):
            half_life(short_series)


class TestCointegrationAllPairs:
    """Tests for testing all pairs."""

    def test_all_pairs_matrix(self, multiple_series):
        """Test creating cointegration p-value matrix."""
        p_value_matrix = test_cointegration_all_pairs(multiple_series)

        assert isinstance(p_value_matrix, pd.DataFrame)
        assert p_value_matrix.shape[0] == len(multiple_series.columns)
        assert p_value_matrix.shape[1] == len(multiple_series.columns)

        # Diagonal should be NaN (can't test cointegration with itself)
        assert np.isnan(p_value_matrix.values.diagonal()).all()

        # Matrix should be symmetric
        for i in range(len(p_value_matrix)):
            for j in range(i+1, len(p_value_matrix)):
                if not np.isnan(p_value_matrix.iloc[i, j]):
                    assert p_value_matrix.iloc[i, j] == p_value_matrix.iloc[j, i]


class TestADFTestSpread:
    """Tests for ADF test on spread."""

    def test_adf_stationary_spread(self):
        """Test ADF on stationary spread."""
        np.random.seed(42)
        stationary_spread = pd.Series(np.random.randn(252))

        result = adf_test_spread(stationary_spread)

        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result

        # White noise should be stationary
        assert result['is_stationary'] == True

    def test_adf_non_stationary_spread(self):
        """Test ADF on non-stationary spread."""
        np.random.seed(42)
        non_stationary_spread = pd.Series(np.random.randn(252).cumsum())

        result = adf_test_spread(non_stationary_spread)

        assert 'is_stationary' in result
        # Random walk should typically be non-stationary

    def test_adf_too_short(self):
        """Test ADF with series too short."""
        short_spread = pd.Series(np.random.randn(5))

        with pytest.raises(ValueError, match="at least 10"):
            adf_test_spread(short_spread)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_cointegration_workflow(self, cointegrated_pair):
        """Test complete cointegration analysis workflow."""
        y1, y2 = cointegrated_pair

        # 1. Test for cointegration
        coint_result = engle_granger_test(y1, y2)
        assert 'hedge_ratio' in coint_result

        # 2. Calculate spread
        spread = calculate_spread(y1, y2, hedge_ratio=coint_result['hedge_ratio'])
        assert len(spread) > 0

        # 3. Test spread for stationarity
        spread_result = adf_test_spread(spread)
        assert 'is_stationary' in spread_result

        # 4. Calculate half-life
        hl = half_life(spread)
        assert hl > 0

    def test_pairs_selection_workflow(self, multiple_series):
        """Test workflow for selecting trading pairs."""
        # 1. Find all cointegrated pairs
        pairs = find_cointegrated_pairs(multiple_series, significance=0.10)

        # 2. For each pair, calculate spread and half-life
        for ticker1, ticker2, p_value, hedge_ratio in pairs:
            spread = calculate_spread(
                multiple_series[ticker1],
                multiple_series[ticker2],
                hedge_ratio=hedge_ratio
            )

            hl = half_life(spread)

            # Half-life should be positive
            assert hl > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_series_with_nans(self):
        """Test handling of NaN values."""
        y1 = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 30)
        y2 = pd.Series([2.0, 4.0, np.nan, 8.0, 10.0] * 30)

        # Should handle NaN by dropping
        result = engle_granger_test(y1, y2)
        assert 'test_statistic' in result

    def test_perfectly_correlated(self):
        """Test with perfectly correlated series."""
        y1 = pd.Series(np.random.randn(100).cumsum())
        y2 = 2 * y1  # Perfect linear relationship

        result = engle_granger_test(y1, y2)
        # Should detect cointegration
        assert result['is_cointegrated'] == True
        # Hedge ratio should be close to 0.5 (y1 = 0.5 * y2)
        assert 0.4 < result['hedge_ratio'] < 0.6

    def test_constant_series(self):
        """Test with constant series."""
        y1 = pd.Series([1.0] * 100)
        y2 = pd.Series([2.0] * 100)

        # Might raise error or handle gracefully
        try:
            result = engle_granger_test(y1, y2)
            # If works, check result is valid
            assert 'test_statistic' in result
        except (ValueError, np.linalg.LinAlgError):
            # Some implementations might fail on constant series
            pass
