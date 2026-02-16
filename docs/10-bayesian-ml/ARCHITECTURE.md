# Bayesian ML Architecture

## Module Structure

```
puffin/models/
├── bayesian.py                    # Core Bayesian models
│   ├── BayesianLinearRegression  # MCMC-based regression
│   ├── bayesian_sharpe()         # Sharpe ratio estimation
│   ├── compare_strategies()      # Multi-strategy comparison
│   └── BayesianPairsTrading      # Dynamic hedge ratios
│
└── stochastic_vol.py             # Volatility models
    ├── StochasticVolatilityModel # Full SV model
    └── estimate_volatility_regime() # Quick estimation
```

## Class Hierarchy

```
BayesianLinearRegression
├── __init__()
├── fit(X, y, samples=2000, tune=1000)
├── predict(X, hdi_prob=0.94) → (mean, (lower, upper))
├── summary() → Dict
└── plot_posterior(param_name) → Figure

BayesianPairsTrading
├── __init__()
├── fit_dynamic_hedge(y, x, window=60) → DataFrame
└── generate_signals(spread, entry_threshold=2.0, exit_threshold=0.5) → Series

StochasticVolatilityModel
├── __init__()
├── fit(returns, samples=2000, tune=1000)
├── summary() → DataFrame
├── plot_volatility() → Figure
├── plot_posterior() → Figure
└── Properties:
    ├── volatility_path → ndarray
    └── volatility_forecast → float
```

## Function API

```
bayesian_sharpe(returns, samples=5000, risk_free_rate=0.0)
→ {'mean', 'std', 'hdi_low', 'hdi_high', 'prob_positive'}

compare_strategies_bayesian(returns_dict, samples=5000)
→ DataFrame with rankings and credible intervals

estimate_volatility_regime(returns, samples=1000)
→ DataFrame with 'volatility', 'vol_lower', 'vol_upper'
```

## Data Flow

### Bayesian Sharpe Ratio
```
Returns
  ↓
bayesian_sharpe()
  ├─ Center returns
  ├─ Build PyMC model (Student's t)
  ├─ MCMC sampling
  └─ Compute HDI & probabilities
  ↓
{'mean', 'hdi_low', 'hdi_high', 'prob_positive'}
```

### Pairs Trading
```
Stock Y, Stock X
  ↓
BayesianPairsTrading.fit_dynamic_hedge()
  ├─ Rolling windows
  ├─ Bayesian regression per window
  └─ Extract hedge ratio posteriors
  ↓
DataFrame(hedge_ratio_mean, hedge_ratio_std, spread)
  ↓
generate_signals()
  ├─ Calculate z-score
  ├─ Apply thresholds
  └─ Forward fill positions
  ↓
Signals (-1, 0, 1)
```

### Stochastic Volatility
```
Returns
  ↓
StochasticVolatilityModel.fit()
  ├─ Build SV model
  │   ├─ log(σ_t²) = μ + φ(log(σ_{t-1}²) - μ) + η_t
  │   └─ r_t = σ_t × ε_t
  ├─ MCMC sampling with scan
  └─ Extract volatility path
  ↓
volatility_path, volatility_forecast
```

## Dependency Graph

```
BayesianLinearRegression
  ├── numpy
  ├── pandas
  ├── pymc
  └── arviz

bayesian_sharpe
  ├── numpy
  ├── pandas
  ├── pymc
  └── arviz

BayesianPairsTrading
  ├── numpy
  ├── pandas
  ├── pymc
  └── warnings

StochasticVolatilityModel
  ├── numpy
  ├── pandas
  ├── pymc
  ├── arviz
  └── matplotlib (optional)
```

## Integration Points

### With Other Puffin Modules

```
puffin.strategies
  ├─ Use bayesian_sharpe() for strategy evaluation
  └─ Use BayesianPairsTrading for pairs strategies

puffin.risk
  ├─ Use StochasticVolatilityModel for risk estimation
  └─ Use volatility forecasts for position sizing

puffin.portfolio
  ├─ Use compare_strategies_bayesian() for allocation
  └─ Use uncertainty estimates for portfolio optimization

puffin.backtest
  ├─ Evaluate strategies with Bayesian metrics
  └─ Compare alternative approaches with credible intervals
```

## Error Handling Flow

```
User calls Bayesian function
  ↓
_check_pymc()
  ├─ PyMC available? → Continue
  └─ PyMC missing? → Raise ImportError with install instructions
  ↓
Try MCMC sampling
  ├─ Success? → Return results
  └─ Failure? → Fallback
      ├─ BayesianLinearRegression → OLS
      ├─ StochasticVolatilityModel → EWMA
      └─ BayesianPairsTrading → Rolling OLS
  ↓
Return results with warnings
```

## Testing Architecture

```
tests/models/test_bayesian.py
├── TestBayesianLinearRegression
│   ├── test_fit_predict
│   ├── test_parameter_recovery
│   ├── test_pandas_input
│   └── test_error_before_fit
│
├── TestBayesianSharpe
│   ├── test_positive_returns
│   ├── test_negative_returns
│   ├── test_pandas_series
│   └── test_compare_strategies
│
├── TestBayesianPairsTrading
│   ├── test_fit_dynamic_hedge
│   ├── test_generate_signals
│   ├── test_signals_with_custom_spread
│   └── test_error_without_fit
│
├── TestStochasticVolatilityModel
│   ├── test_fit_basic
│   ├── test_volatility_positive
│   ├── test_pandas_input
│   ├── test_fallback_on_failure
│   └── test_estimate_volatility_regime
│
├── TestIntegration
│   └── test_bayesian_workflow
│
└── TestEdgeCases
    ├── test_empty_returns
    ├── test_constant_returns
    └── test_very_short_series
```

## Performance Characteristics

| Operation | Time (approx) | Memory | Notes |
|-----------|---------------|--------|-------|
| BayesianLinearRegression.fit() | 10-30s | 100-500 MB | Depends on features |
| bayesian_sharpe() | 15-45s | 50-200 MB | 5000 samples |
| BayesianPairsTrading.fit_dynamic_hedge() | 5-10 min | 200-800 MB | Full series |
| StochasticVolatilityModel.fit() | 1-3 min | 100-400 MB | 2000 samples |
| estimate_volatility_regime() | 30-90s | 50-200 MB | Quick estimation |

## Parallelization Strategy

```
MCMC Sampling (within model)
├── Chain 1 (CPU core 1)
├── Chain 2 (CPU core 2)
├── Chain 3 (CPU core 3)
└── Chain 4 (CPU core 4)

Multiple Assets (user level)
├── Asset 1 → Process 1
├── Asset 2 → Process 2
├── Asset 3 → Process 3
└── Asset 4 → Process 4
```

Enable with:
```python
# Within model
model.fit(..., cores=4)

# Multiple assets
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(process_asset, asset_list)
```

## Recommended Workflows

### Workflow 1: Strategy Development
```
1. Generate strategy returns
2. bayesian_sharpe() → Evaluate Sharpe with uncertainty
3. If promising → Full backtest
4. compare_strategies_bayesian() → Compare alternatives
5. Select best with high confidence
```

### Workflow 2: Pairs Trading
```
1. Identify candidate pairs (cointegration test)
2. BayesianPairsTrading.fit_dynamic_hedge() → Get hedge ratios
3. Check hedge ratio stability
4. generate_signals() → Get trading signals
5. Backtest with puffin.backtest
6. Evaluate with bayesian_sharpe()
```

### Workflow 3: Risk Management
```
1. Collect returns data
2. StochasticVolatilityModel.fit() → Estimate volatility
3. Use volatility_forecast for next-period risk
4. Adjust positions: target_vol / forecast_vol
5. Update daily with new data
```

## Configuration Recommendations

### Development
```python
samples = 500
tune = 200
cores = 1
```

### Testing
```python
samples = 1000
tune = 500
cores = 2
```

### Production
```python
samples = 2000-5000
tune = 1000
cores = 4
```

### Critical Decisions
```python
samples = 5000-10000
tune = 2000
cores = 4
```

## Future Extensions

Potential additions to architecture:

```
puffin.models.bayesian
├── [Existing models]
└── [Future models]
    ├── BayesianHierarchical (multi-asset modeling)
    ├── BayesianOptimizer (hyperparameter tuning)
    ├── BayesianRegimeSwitching (state-space models)
    └── VariationalInference (faster approximation)

puffin.models.stochastic_vol
├── [Existing models]
└── [Future models]
    ├── MultivariateSV (portfolio vol)
    ├── SVWithJumps (crisis modeling)
    └── RealizedVolatilityBayes (high-freq data)
```
