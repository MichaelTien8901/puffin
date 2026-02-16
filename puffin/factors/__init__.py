"""
Alpha factors package for algorithmic trading.

This package provides tools for:
- Computing alpha factors (momentum, value, volatility, quality)
- Technical analysis with TA-Lib integration
- Signal denoising with Kalman filters and wavelets
- Factor evaluation with Alphalens integration
- WorldQuant-style formulaic alphas
"""

from .alpha import (
    compute_momentum_factors,
    compute_value_factors,
    compute_volatility_factors,
    compute_quality_factors,
    compute_all_factors
)

from .technical import TechnicalIndicators

from .kalman import (
    KalmanFilter,
    AdaptiveKalmanFilter,
    extract_trend,
    dynamic_hedge_ratio,
    kalman_ma_crossover
)

from .wavelets import (
    wavelet_denoise,
    wavelet_decompose,
    reconstruct_from_levels,
    wavelet_smooth,
    wavelet_variance_decomposition,
    multiscale_decomposition,
    adaptive_wavelet_threshold
)

from .evaluation import (
    FactorEvaluator,
    factor_autocorrelation,
    factor_rank_autocorrelation,
    quantile_returns_analysis
)

from .formulaic import (
    AlphaExpression,
    evaluate_alpha,
    evaluate_alpha_library,
    to_multiindex_series,
    combine_alphas,
    neutralize_factor,
    ALPHA_LIBRARY
)

__all__ = [
    # Alpha factors
    'compute_momentum_factors',
    'compute_value_factors',
    'compute_volatility_factors',
    'compute_quality_factors',
    'compute_all_factors',

    # Technical indicators
    'TechnicalIndicators',

    # Kalman filter
    'KalmanFilter',
    'AdaptiveKalmanFilter',
    'extract_trend',
    'dynamic_hedge_ratio',
    'kalman_ma_crossover',

    # Wavelets
    'wavelet_denoise',
    'wavelet_decompose',
    'reconstruct_from_levels',
    'wavelet_smooth',
    'wavelet_variance_decomposition',
    'multiscale_decomposition',
    'adaptive_wavelet_threshold',

    # Evaluation
    'FactorEvaluator',
    'factor_autocorrelation',
    'factor_rank_autocorrelation',
    'quantile_returns_analysis',

    # Formulaic alphas
    'AlphaExpression',
    'evaluate_alpha',
    'evaluate_alpha_library',
    'to_multiindex_series',
    'combine_alphas',
    'neutralize_factor',
    'ALPHA_LIBRARY',
]
