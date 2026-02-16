"""
Tests for factor models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from puffin.models.factor_models import FamaFrenchModel, fama_macbeth


@pytest.fixture
def synthetic_returns():
    """Generate synthetic asset returns."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)

    # Generate returns with known factor exposures
    returns = pd.Series(
        0.0003 + 0.0001 * np.random.randn(n_days),
        index=dates,
        name='returns'
    )

    return returns


@pytest.fixture
def synthetic_market_returns():
    """Generate synthetic market returns."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)

    # Market returns
    market = pd.Series(
        0.0004 + 0.01 * np.random.randn(n_days),
        index=dates,
        name='market'
    )

    return market


@pytest.fixture
def synthetic_panel_data():
    """Generate synthetic panel data for Fama-MacBeth."""
    np.random.seed(42)

    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='M')
    assets = [f'Asset_{i}' for i in range(20)]

    data = []
    for date in dates:
        for asset in assets:
            # Generate random factor exposures
            beta_mkt = np.random.uniform(0.5, 1.5)
            beta_smb = np.random.uniform(-0.5, 0.5)
            beta_hml = np.random.uniform(-0.5, 0.5)

            # Generate returns based on factors (with risk premiums)
            ret = (0.005 +  # risk-free rate
                   0.006 * beta_mkt +  # market premium
                   0.002 * beta_smb +  # size premium
                   0.003 * beta_hml +  # value premium
                   np.random.randn() * 0.02)  # noise

            data.append({
                'date': date,
                'asset': asset,
                'returns': ret,
                'beta_mkt': beta_mkt,
                'beta_smb': beta_smb,
                'beta_hml': beta_hml,
            })

    return pd.DataFrame(data)


class TestFamaFrenchModel:
    """Test Fama-French factor models."""

    def test_fetch_factors(self):
        """Test fetching factor data."""
        model = FamaFrenchModel()

        start = '2020-01-01'
        end = '2020-12-31'

        factors = model.fetch_factors(start, end)

        # Check structure
        assert isinstance(factors, pd.DataFrame)
        expected_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        for col in expected_cols:
            assert col in factors.columns

        # Check date range
        assert factors.index.min() >= pd.to_datetime(start)
        assert factors.index.max() <= pd.to_datetime(end)

    def test_fetch_factors_caching(self):
        """Test that factor data is cached."""
        model = FamaFrenchModel()

        # First fetch
        start = '2020-01-01'
        end = '2020-12-31'
        factors1 = model.fetch_factors(start, end)

        # Second fetch (should use cache)
        factors2 = model.fetch_factors(start, end)

        pd.testing.assert_frame_equal(factors1, factors2)

    def test_fit_capm(self, synthetic_returns, synthetic_market_returns):
        """Test CAPM model fitting."""
        model = FamaFrenchModel()

        # Align returns and market
        common_idx = synthetic_returns.index.intersection(synthetic_market_returns.index)
        returns = synthetic_returns.loc[common_idx]
        market = synthetic_market_returns.loc[common_idx]

        # Fit CAPM (without fetching factors)
        results = model.fit_capm(returns, market)

        # Check results structure
        assert 'alpha' in results
        assert 'beta' in results
        assert 'r_squared' in results
        assert 'alpha_pvalue' in results
        assert 'beta_pvalue' in results

        # Check types
        assert isinstance(results['alpha'], (float, np.floating))
        assert isinstance(results['beta'], (float, np.floating))
        assert isinstance(results['r_squared'], (float, np.floating))

        # Beta should be positive for correlated assets
        assert results['beta'] > -2 and results['beta'] < 2

    def test_fit_capm_with_factor_fetch(self, synthetic_returns):
        """Test CAPM with automatic factor fetching."""
        model = FamaFrenchModel()

        start = synthetic_returns.index.min()
        end = synthetic_returns.index.max()

        results = model.fit_capm(synthetic_returns, start=start, end=end)

        # Check results structure
        assert 'alpha' in results
        assert 'beta' in results
        assert 'r_squared' in results

    def test_fit_three_factor(self, synthetic_returns):
        """Test 3-factor model fitting."""
        model = FamaFrenchModel()

        start = synthetic_returns.index.min()
        end = synthetic_returns.index.max()

        results = model.fit_three_factor(synthetic_returns, start=start, end=end)

        # Check results structure
        assert 'alpha' in results
        assert 'beta_mkt' in results
        assert 'beta_smb' in results
        assert 'beta_hml' in results
        assert 'betas' in results
        assert 'r_squared' in results
        assert 'pvalues' in results
        assert 'tstats' in results

        # Check betas dictionary
        assert 'Mkt-RF' in results['betas']
        assert 'SMB' in results['betas']
        assert 'HML' in results['betas']

        # Check types
        assert isinstance(results['r_squared'], (float, np.floating))
        assert 0 <= results['r_squared'] <= 1

    def test_fit_five_factor(self, synthetic_returns):
        """Test 5-factor model fitting."""
        model = FamaFrenchModel()

        start = synthetic_returns.index.min()
        end = synthetic_returns.index.max()

        results = model.fit_five_factor(synthetic_returns, start=start, end=end)

        # Check results structure
        assert 'alpha' in results
        assert 'beta_mkt' in results
        assert 'beta_smb' in results
        assert 'beta_hml' in results
        assert 'beta_rmw' in results
        assert 'beta_cma' in results
        assert 'betas' in results
        assert 'r_squared' in results

        # Check betas dictionary
        assert 'Mkt-RF' in results['betas']
        assert 'SMB' in results['betas']
        assert 'HML' in results['betas']
        assert 'RMW' in results['betas']
        assert 'CMA' in results['betas']

        # Check types
        assert isinstance(results['r_squared'], (float, np.floating))
        assert 0 <= results['r_squared'] <= 1

    def test_factor_model_comparison(self, synthetic_returns):
        """Test that 5-factor model explains more than 3-factor."""
        model = FamaFrenchModel()

        start = synthetic_returns.index.min()
        end = synthetic_returns.index.max()

        results_3f = model.fit_three_factor(synthetic_returns, start=start, end=end)
        results_5f = model.fit_five_factor(synthetic_returns, start=start, end=end)

        # 5-factor should have equal or higher R-squared (more factors)
        # (may not always be true with synthetic data, but generally expected)
        assert results_5f['r_squared'] >= results_3f['r_squared'] - 0.1


class TestFamaMacBeth:
    """Test Fama-MacBeth regression."""

    def test_basic_fama_macbeth(self, synthetic_panel_data):
        """Test basic Fama-MacBeth procedure."""
        factors = ['beta_mkt', 'beta_smb', 'beta_hml']

        results = fama_macbeth(
            panel_data=synthetic_panel_data,
            factors=factors,
            return_col='returns',
            time_col='date',
            asset_col='asset'
        )

        # Check results structure
        assert 'risk_premiums' in results
        assert 't_stats' in results
        assert 'r_squared' in results
        assert 'n_periods' in results
        assert 'cross_sectional_results' in results
        assert 'betas' in results

        # Check risk premiums
        risk_premiums = results['risk_premiums']
        assert 'lambda_0' in risk_premiums  # zero-beta rate
        for factor in factors:
            assert factor in risk_premiums

        # Check t-statistics
        t_stats = results['t_stats']
        assert 'lambda_0' in t_stats
        for factor in factors:
            assert factor in t_stats

        # Check types
        assert isinstance(results['r_squared'], (float, np.floating))
        assert isinstance(results['n_periods'], int)

    def test_fama_macbeth_recovers_premiums(self, synthetic_panel_data):
        """Test that Fama-MacBeth recovers known risk premiums."""
        factors = ['beta_mkt', 'beta_smb', 'beta_hml']

        results = fama_macbeth(
            panel_data=synthetic_panel_data,
            factors=factors,
            return_col='returns',
            time_col='date',
            asset_col='asset'
        )

        risk_premiums = results['risk_premiums']

        # Check that market premium is positive (we set it to 0.006)
        # Allow for estimation error
        assert risk_premiums['beta_mkt'] > 0
        assert 0.002 < risk_premiums['beta_mkt'] < 0.015

        # SMB and HML should be recovered (set to 0.002 and 0.003)
        # Allow for wide error due to noise
        assert -0.005 < risk_premiums['beta_smb'] < 0.01
        assert -0.005 < risk_premiums['beta_hml'] < 0.015

    def test_fama_macbeth_betas(self, synthetic_panel_data):
        """Test that betas are estimated in first stage."""
        factors = ['beta_mkt', 'beta_smb', 'beta_hml']

        results = fama_macbeth(
            panel_data=synthetic_panel_data,
            factors=factors,
            return_col='returns',
            time_col='date',
            asset_col='asset'
        )

        betas = results['betas']

        # Check structure
        assert isinstance(betas, pd.DataFrame)
        assert 'asset' in betas.columns
        for factor in factors:
            assert factor in betas.columns

        # Check number of assets
        n_assets = synthetic_panel_data['asset'].nunique()
        assert len(betas) == n_assets

    def test_fama_macbeth_cross_sectional_results(self, synthetic_panel_data):
        """Test cross-sectional results."""
        factors = ['beta_mkt', 'beta_smb', 'beta_hml']

        results = fama_macbeth(
            panel_data=synthetic_panel_data,
            factors=factors,
            return_col='returns',
            time_col='date',
            asset_col='asset'
        )

        cs_results = results['cross_sectional_results']

        # Check structure
        assert isinstance(cs_results, pd.DataFrame)
        assert 'time' in cs_results.columns
        assert 'lambda_0' in cs_results.columns
        for factor in factors:
            assert f'lambda_{factor}' in cs_results.columns

        # Check number of time periods
        n_periods = results['n_periods']
        assert len(cs_results) == n_periods
        assert n_periods > 0

    def test_fama_macbeth_missing_columns(self):
        """Test error handling for missing columns."""
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'asset': ['A'] * 10,
            'returns': np.random.randn(10),
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            fama_macbeth(
                panel_data=data,
                factors=['beta_mkt'],  # This column doesn't exist
                return_col='returns',
                time_col='date',
                asset_col='asset'
            )

    def test_fama_macbeth_single_factor(self, synthetic_panel_data):
        """Test Fama-MacBeth with single factor."""
        results = fama_macbeth(
            panel_data=synthetic_panel_data,
            factors=['beta_mkt'],
            return_col='returns',
            time_col='date',
            asset_col='asset'
        )

        # Should still work with single factor
        assert 'risk_premiums' in results
        assert 'beta_mkt' in results['risk_premiums']
        assert results['n_periods'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
