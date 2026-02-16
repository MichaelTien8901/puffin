"""
Portfolio Optimization Module

Provides portfolio construction and optimization tools including:
- Mean-variance optimization (Markowitz)
- Risk parity and equal risk contribution
- Hierarchical Risk Parity (HRP)
- Performance analysis and tearsheets
- Portfolio rebalancing with transaction costs
"""

from .mean_variance import MeanVarianceOptimizer
from .risk_parity import (
    risk_parity_weights,
    inverse_volatility_weights,
    risk_contribution,
    equal_risk_contribution_analytical,
    minimum_correlation_algorithm,
    maximum_diversification_weights,
    diversification_ratio
)
from .hrp import (
    hrp_weights,
    hrp_weights_with_names,
    plot_dendrogram,
    hrp_allocation_stats
)
from .tearsheet import (
    compute_stats,
    generate_tearsheet,
    plot_returns,
    plot_drawdown,
    plot_monthly_returns,
    plot_rolling_metrics,
    print_tearsheet_summary
)
from .rebalance import (
    RebalanceEngine,
    Trade,
    CostModel,
    rebalance_schedule,
    backtest_rebalancing,
    compare_rebalancing_strategies
)

__all__ = [
    # Mean-variance optimization
    'MeanVarianceOptimizer',

    # Risk parity
    'risk_parity_weights',
    'inverse_volatility_weights',
    'risk_contribution',
    'equal_risk_contribution_analytical',
    'minimum_correlation_algorithm',
    'maximum_diversification_weights',
    'diversification_ratio',

    # Hierarchical Risk Parity
    'hrp_weights',
    'hrp_weights_with_names',
    'plot_dendrogram',
    'hrp_allocation_stats',

    # Tearsheet and performance
    'compute_stats',
    'generate_tearsheet',
    'plot_returns',
    'plot_drawdown',
    'plot_monthly_returns',
    'plot_rolling_metrics',
    'print_tearsheet_summary',

    # Rebalancing
    'RebalanceEngine',
    'Trade',
    'CostModel',
    'rebalance_schedule',
    'backtest_rebalancing',
    'compare_rebalancing_strategies',
]
