---
layout: default
title: "Walk-Forward Analysis"
parent: "Part 7: Backtesting"
nav_order: 3
---

# Walk-Forward Analysis

## Overview

A strategy that looks great on a single backtest period may simply be overfit to that specific slice of history. Walk-forward analysis is the standard technique for testing whether a strategy generalizes across different market regimes. Instead of one train/test split, it rolls multiple windows through time, training on each in-sample period and evaluating on the subsequent out-of-sample period.

## Why Walk-Forward?

A single backtest has a critical flaw: you can keep tweaking parameters until the results look good, but that says nothing about future performance. Walk-forward analysis addresses this by:

- **Testing multiple time periods** -- Performance must be consistent, not just good on one lucky window.
- **Preventing data snooping** -- Each out-of-sample period is never seen during parameter selection.
- **Revealing regime sensitivity** -- If the strategy works in 2019 but fails in 2020, you learn that before deploying capital.

## Running Walk-Forward Analysis

The `walk_forward()` function splits your data into `n_splits` rolling windows. Within each window, `train_ratio` controls how much is used for training vs testing.

```python
from puffin.backtest import walk_forward

results = walk_forward(
    strategy=strategy,
    data=data,
    train_ratio=0.7,
    n_splits=5,
    initial_capital=100_000,
)

for r in results:
    train_sharpe = r['train_metrics']['sharpe_ratio']
    test_sharpe = r['test_metrics']['sharpe_ratio']
    print(f"Split {r['split']}: Train Sharpe={train_sharpe:.2f}, Test Sharpe={test_sharpe:.2f}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | -- | Strategy instance to test |
| `data` | -- | OHLCV DataFrame |
| `train_ratio` | 0.7 | Fraction of each window used for training |
| `n_splits` | 5 | Number of rolling windows |
| `initial_capital` | 100,000 | Starting capital per window |

Each result dictionary contains:

- `split` -- Window number (1-indexed)
- `train_start`, `train_end` -- In-sample date range
- `test_start`, `test_end` -- Out-of-sample date range
- `train_metrics` -- Metrics dict from the training period
- `test_metrics` -- Metrics dict from the test period

You can also pass any `Backtester` keyword arguments (like `slippage` and `commission`) through to `walk_forward()`:

```python
from puffin.backtest import SlippageModel, CommissionModel

results = walk_forward(
    strategy=strategy,
    data=data,
    train_ratio=0.7,
    n_splits=5,
    slippage=SlippageModel(fixed=0.01),
    commission=CommissionModel(flat=1.0),
)
```

## Interpreting Results

{: .warning }
If your strategy performs well in-sample but poorly out-of-sample, it is likely overfit to historical data. Walk-forward analysis helps detect this.

When reviewing walk-forward results, look for these patterns:

- **Consistent Sharpe ratios** across splits indicate a robust strategy. Some variation is normal, but the out-of-sample Sharpe should be positive across most windows.
- **Train >> Test performance** is a red flag for overfitting. If the in-sample Sharpe is 3.0 but out-of-sample is 0.2, your strategy has memorized the training data.
- **One bad split** may just reflect a regime shift (e.g., COVID-19 crash). Multiple bad splits indicate a fundamental problem.

## Visualization

Plot the train vs test Sharpe ratios side by side to quickly spot overfitting:

```python
import matplotlib.pyplot as plt

splits = [r['split'] for r in results]
train_sharpes = [r['train_metrics']['sharpe_ratio'] for r in results]
test_sharpes = [r['test_metrics']['sharpe_ratio'] for r in results]

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(splits))
width = 0.35
ax.bar([i - width/2 for i in x], train_sharpes, width, label='Train', color='#2d5016')
ax.bar([i + width/2 for i in x], test_sharpes, width, label='Test', color='#1a3a5c')
ax.set_xlabel('Split')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Walk-Forward: Train vs Test Sharpe')
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.legend()
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

A healthy result shows test bars that are somewhat lower than train bars, but still comfortably above zero. If test bars are consistently negative, the strategy is not viable.

## Common Pitfalls

{: .warning }
**Overfitting through repeated walk-forward analysis.** If you run walk-forward analysis, tweak parameters based on the results, and re-run, you are effectively overfitting to the walk-forward splits themselves. Set your parameters once using the training splits and validate once.

Other pitfalls to watch for:

- **Too few splits** -- With only 2 or 3 splits, results are not statistically meaningful. Use at least 5.
- **Too short test periods** -- If each test window is only a few weeks, the results are noisy and unreliable.
- **Survivorship bias in data** -- Walk-forward analysis cannot fix bad data. If your dataset only contains stocks that survived to today, your results are biased upward.
- **Ignoring transaction costs** -- Always pass realistic `SlippageModel` and `CommissionModel` to `walk_forward()`. A strategy that only works with zero costs is not a real strategy.

## Summary

- Event-driven backtesting prevents lookahead bias by processing bars sequentially
- Always model slippage and commissions -- they significantly impact results
- Key metrics: Sharpe ratio, max drawdown, win rate, profit factor
- Walk-forward analysis tests strategy robustness across time periods
- Visualize equity curves and drawdowns to understand strategy behavior
- If train performance far exceeds test performance, the strategy is likely overfit

## Exercises

1. Run walk-forward analysis with 5 splits on SPY from 2018--2024. Is the Sharpe ratio consistent across splits?
2. Try increasing `n_splits` to 10. Do the results become more or less stable?
3. Compare walk-forward results for two strategies (e.g., momentum with different window lengths). Which one generalizes better out-of-sample?
4. Add slippage and commission to your walk-forward analysis. Does the ranking of strategies change when costs are included?

## Next Steps

In Part 8, we apply **machine learning** to generate trading signals, moving beyond rule-based strategies to data-driven models.
