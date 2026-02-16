# Bayesian ML Models

This module provides Bayesian inference tools for algorithmic trading using PyMC.

## Overview

Bayesian methods provide:
- **Uncertainty Quantification**: Full probability distributions instead of point estimates
- **Robust Inference**: Natural handling of outliers through Student's t-distributions
- **Dynamic Estimation**: Online updating for time-varying parameters
- **Principled Comparison**: Compare strategies accounting for estimation uncertainty

## Installation

The Bayesian models require PyMC and ArviZ:

```bash
pip install pymc arviz
```

## Modules

### bayesian.py

Core Bayesian models for trading:

#### BayesianLinearRegression
MCMC-based linear regression with uncertainty quantification.

```python
from puffin.models.bayesian import BayesianLinearRegression

model = BayesianLinearRegression()
model.fit(X, y, samples=2000, tune=1000)
mean, (lower, upper) = model.predict(X_test)
```

#### bayesian_sharpe
Bayesian Sharpe ratio estimation using Student's t-distribution.

```python
from puffin.models.bayesian import bayesian_sharpe

sharpe_stats = bayesian_sharpe(returns, samples=5000)
print(f"Sharpe: {sharpe_stats['mean']:.2f}")
print(f"95% HDI: [{sharpe_stats['hdi_low']:.2f}, {sharpe_stats['hdi_high']:.2f}]")
print(f"P(Sharpe > 0): {sharpe_stats['prob_positive']:.1%}")
```

#### compare_strategies_bayesian
Compare multiple strategies with uncertainty quantification.

```python
from puffin.models.bayesian import compare_strategies_bayesian

results = compare_strategies_bayesian({
    'strategy_a': returns_a,
    'strategy_b': returns_b,
    'strategy_c': returns_c
}, samples=5000)
```

#### BayesianPairsTrading
Dynamic hedge ratio estimation for pairs trading.

```python
from puffin.models.bayesian import BayesianPairsTrading

pairs = BayesianPairsTrading()
hedge_ratios = pairs.fit_dynamic_hedge(y, x, window=60)
signals = pairs.generate_signals(entry_threshold=2.0, exit_threshold=0.5)
```

### stochastic_vol.py

Stochastic volatility models for time-varying volatility estimation.

#### StochasticVolatilityModel
Full stochastic volatility model using MCMC.

```python
from puffin.models.stochastic_vol import StochasticVolatilityModel

sv_model = StochasticVolatilityModel()
sv_model.fit(returns, samples=2000, tune=1000)

vol_path = sv_model.volatility_path
vol_forecast = sv_model.volatility_forecast
```

#### estimate_volatility_regime
Quick volatility estimation without storing full model.

```python
from puffin.models.stochastic_vol import estimate_volatility_regime

vol_df = estimate_volatility_regime(returns, samples=1000)
```

## Testing

Tests are in `tests/models/test_bayesian.py`. Slow tests (MCMC sampling) are marked with `@pytest.mark.slow`.

Run all tests:
```bash
pytest tests/models/test_bayesian.py -v
```

Run only fast tests:
```bash
pytest tests/models/test_bayesian.py -v -m "not slow"
```

## Performance Considerations

### Sample Size
- **Development**: 500-1000 samples for quick iterations
- **Production**: 2000-5000 samples for stable estimates
- **Critical Decisions**: 5000-10000 samples

### Computational Speed
- Use `cores=4` for parallel sampling
- Use smaller windows for rolling estimation
- Cache fitted models when possible

### Convergence Diagnostics
Always check:
- `r_hat < 1.01` (Gelman-Rubin statistic)
- `ess_bulk > 1000` (effective sample size)
- Visual inspection of trace plots

```python
import arviz as az

# Check diagnostics
summary = az.summary(trace)
print(summary[['r_hat', 'ess_bulk', 'ess_tail']])

# Visual checks
az.plot_trace(trace)
az.plot_rank(trace)
```

## Common Patterns

### Pattern 1: Strategy Evaluation
```python
from puffin.models.bayesian import bayesian_sharpe, compare_strategies_bayesian

# Single strategy
sharpe_stats = bayesian_sharpe(returns)
if sharpe_stats['prob_positive'] > 0.95:
    print("High confidence in positive Sharpe")

# Multiple strategies
comparison = compare_strategies_bayesian(strategy_dict)
best_strategy = comparison.iloc[0]['strategy']
```

### Pattern 2: Volatility-Adjusted Positions
```python
from puffin.models.stochastic_vol import estimate_volatility_regime

vol_df = estimate_volatility_regime(returns)
target_vol = 0.01
position_sizes = (target_vol / vol_df['volatility']).clip(upper=2.0)
```

### Pattern 3: Dynamic Pairs Trading
```python
from puffin.models.bayesian import BayesianPairsTrading

pairs = BayesianPairsTrading()
hedge_df = pairs.fit_dynamic_hedge(stock_y, stock_x, window=60)
signals = pairs.generate_signals(entry_threshold=2.0)

# Calculate strategy returns
ret_y = stock_y.pct_change()
ret_x = stock_x.pct_change()
strategy_ret = signals * (ret_y - hedge_df['hedge_ratio_mean'] * ret_x)
```

## Fallback Behavior

All models implement graceful fallbacks when MCMC sampling fails:

- `BayesianLinearRegression`: Falls back to OLS with bootstrap confidence intervals
- `StochasticVolatilityModel`: Falls back to EWMA volatility estimation
- `BayesianPairsTrading`: Falls back to rolling OLS for hedge ratios

Warning messages are issued when fallbacks are used.

## Model Selection Guide

| Use Case | Model | Key Parameters |
|----------|-------|----------------|
| Factor exposure with uncertainty | `BayesianLinearRegression` | `samples=2000` |
| Strategy comparison | `compare_strategies_bayesian` | `samples=5000` |
| Pairs trading | `BayesianPairsTrading` | `window=60` |
| Volatility forecasting | `StochasticVolatilityModel` | `samples=2000` |
| Quick vol estimate | `estimate_volatility_regime` | `samples=1000` |

## References

- PyMC Documentation: https://www.pymc.io/
- ArviZ Documentation: https://arviz-devs.github.io/arviz/
- Bayesian Methods for Hackers: https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
- Statistical Rethinking by Richard McElreath
