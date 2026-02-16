"""
Statistical and machine learning models for trading.
"""

from .linear import OLSModel, RidgeModel, LassoModel, DirectionClassifier
from .factor_models import FamaFrenchModel, fama_macbeth

# Time series models
from .ts_diagnostics import (
    decompose_series,
    test_stationarity,
    test_kpss,
    plot_acf_pacf,
    autocorrelation,
    check_stationarity
)
from .arima import ARIMAModel, auto_arima
from .var import VARModel, test_granger_causality_matrix
from .garch import GARCHModel, fit_garch_models, rolling_volatility_forecast
from .cointegration import (
    engle_granger_test,
    johansen_test,
    find_cointegrated_pairs,
    calculate_spread,
    half_life,
    test_cointegration_all_pairs,
    adf_test_spread
)
from .pairs_trading import PairsTradingStrategy, rank_pairs_by_performance

# Bayesian models (optional dependency)
try:
    from .bayesian import (
        BayesianLinearRegression,
        bayesian_sharpe,
        compare_strategies_bayesian,
        BayesianPairsTrading
    )
    from .stochastic_vol import StochasticVolatilityModel, estimate_volatility_regime

    __all__ = [
        'OLSModel',
        'RidgeModel',
        'LassoModel',
        'DirectionClassifier',
        'FamaFrenchModel',
        'fama_macbeth',
        # Time series models
        'decompose_series',
        'test_stationarity',
        'test_kpss',
        'plot_acf_pacf',
        'autocorrelation',
        'check_stationarity',
        'ARIMAModel',
        'auto_arima',
        'VARModel',
        'test_granger_causality_matrix',
        'GARCHModel',
        'fit_garch_models',
        'rolling_volatility_forecast',
        'engle_granger_test',
        'johansen_test',
        'find_cointegrated_pairs',
        'calculate_spread',
        'half_life',
        'test_cointegration_all_pairs',
        'adf_test_spread',
        'PairsTradingStrategy',
        'rank_pairs_by_performance',
        # Bayesian models
        'BayesianLinearRegression',
        'bayesian_sharpe',
        'compare_strategies_bayesian',
        'BayesianPairsTrading',
        'StochasticVolatilityModel',
        'estimate_volatility_regime',
    ]
except ImportError:
    __all__ = [
        'OLSModel',
        'RidgeModel',
        'LassoModel',
        'DirectionClassifier',
        'FamaFrenchModel',
        'fama_macbeth',
        # Time series models
        'decompose_series',
        'test_stationarity',
        'test_kpss',
        'plot_acf_pacf',
        'autocorrelation',
        'check_stationarity',
        'ARIMAModel',
        'auto_arima',
        'VARModel',
        'test_granger_causality_matrix',
        'GARCHModel',
        'fit_garch_models',
        'rolling_volatility_forecast',
        'engle_granger_test',
        'johansen_test',
        'find_cointegrated_pairs',
        'calculate_spread',
        'half_life',
        'test_cointegration_all_pairs',
        'adf_test_spread',
        'PairsTradingStrategy',
        'rank_pairs_by_performance',
    ]
