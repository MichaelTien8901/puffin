"""Alternative data evaluation framework for signal and data quality assessment."""

import numpy as np
import pandas as pd
from scipy import stats


class AltDataEvaluator:
    """Evaluator for alternative data signals and quality.

    Examples:
        >>> evaluator = AltDataEvaluator()
        >>> signal_quality = evaluator.evaluate_signal_quality(signal_data, returns_data)
        >>> data_quality = evaluator.evaluate_data_quality(raw_data)
        >>> backtest_results = evaluator.backtest_signal(signal_data, returns_data)
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize the evaluator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate

    def evaluate_signal_quality(
        self,
        signal: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """Evaluate signal quality using IC, IR, and decay analysis.

        Args:
            signal: Signal values (aligned with returns)
            returns: Forward returns

        Returns:
            Dictionary with IC, IR, decay_half_life, and t_stat
        """
        # Align signal and returns
        aligned = pd.DataFrame({"signal": signal, "returns": returns}).dropna()

        if len(aligned) < 10:
            return {
                "ic": np.nan,
                "ir": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "decay_half_life": np.nan,
            }

        # Calculate Information Coefficient (IC)
        # IC is the correlation between signal and forward returns
        ic, p_value = stats.spearmanr(aligned["signal"], aligned["returns"])

        # Calculate Information Ratio (IR)
        # IR = mean(IC) / std(IC) over time
        # For single period, estimate using std of residuals
        ir = ic / (1 / np.sqrt(len(aligned)))

        # Calculate t-statistic for statistical significance
        t_stat = ic * np.sqrt(len(aligned) - 2) / np.sqrt(1 - ic**2)

        # Decay analysis: measure how quickly signal predictive power decays
        decay_half_life = self._calculate_decay_half_life(aligned["signal"], aligned["returns"])

        return {
            "ic": ic,
            "ir": ir,
            "t_stat": t_stat,
            "p_value": p_value,
            "decay_half_life": decay_half_life,
        }

    def _calculate_decay_half_life(
        self,
        signal: pd.Series,
        returns: pd.Series,
    ) -> float:
        """Calculate signal decay half-life.

        Args:
            signal: Signal values
            returns: Returns

        Returns:
            Half-life in periods (days if daily data)
        """
        # Calculate IC at different lags
        max_lag = min(20, len(signal) // 4)
        ics = []

        for lag in range(1, max_lag + 1):
            shifted_returns = returns.shift(-lag)
            aligned = pd.DataFrame({"signal": signal, "returns": shifted_returns}).dropna()

            if len(aligned) >= 10:
                ic, _ = stats.spearmanr(aligned["signal"], aligned["returns"])
                ics.append(abs(ic))
            else:
                break

        if not ics or ics[0] == 0:
            return np.nan

        # Find where IC drops to half of original
        half_ic = ics[0] / 2
        for i, ic in enumerate(ics):
            if ic <= half_ic:
                return i + 1

        return np.nan

    def evaluate_data_quality(self, data: pd.DataFrame) -> dict:
        """Evaluate data quality metrics.

        Args:
            data: DataFrame with alternative data

        Returns:
            Dictionary with completeness, timeliness, and coverage metrics
        """
        if data.empty:
            return {
                "completeness": 0.0,
                "missing_pct": 100.0,
                "coverage_days": 0,
                "update_frequency": np.nan,
                "data_points": 0,
            }

        # Completeness: percentage of non-null values
        total_cells = data.size
        non_null_cells = data.count().sum()
        completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0.0
        missing_pct = 100.0 - completeness

        # Coverage: time span of data
        if isinstance(data.index, pd.DatetimeIndex):
            coverage_days = (data.index.max() - data.index.min()).days
            # Update frequency: average days between updates
            time_diffs = data.index.to_series().diff().dt.days.dropna()
            update_frequency = time_diffs.median() if len(time_diffs) > 0 else np.nan
        else:
            coverage_days = len(data)
            update_frequency = np.nan

        return {
            "completeness": completeness,
            "missing_pct": missing_pct,
            "coverage_days": coverage_days,
            "update_frequency": update_frequency,
            "data_points": len(data),
        }

    def evaluate_technical(self, data: pd.DataFrame) -> dict:
        """Evaluate technical aspects of data.

        Args:
            data: DataFrame with alternative data

        Returns:
            Dictionary with storage, latency, and format assessment
        """
        if data.empty:
            return {
                "storage_mb": 0.0,
                "row_count": 0,
                "column_count": 0,
                "memory_per_row_kb": 0.0,
                "has_datetime_index": False,
                "numeric_columns": 0,
            }

        # Storage metrics
        storage_bytes = data.memory_usage(deep=True).sum()
        storage_mb = storage_bytes / (1024 * 1024)
        memory_per_row_kb = (storage_bytes / len(data) / 1024) if len(data) > 0 else 0.0

        # Format assessment
        has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
        numeric_columns = len(data.select_dtypes(include=[np.number]).columns)

        return {
            "storage_mb": storage_mb,
            "row_count": len(data),
            "column_count": len(data.columns),
            "memory_per_row_kb": memory_per_row_kb,
            "has_datetime_index": has_datetime_index,
            "numeric_columns": numeric_columns,
        }

    def backtest_signal(
        self,
        signal: pd.Series,
        returns: pd.Series,
        quantiles: int = 5,
    ) -> dict:
        """Backtest a signal by creating quantile portfolios.

        Args:
            signal: Signal values
            returns: Forward returns
            quantiles: Number of quantiles to split signal into

        Returns:
            Dictionary with returns statistics for each quantile
        """
        # Align signal and returns
        aligned = pd.DataFrame({"signal": signal, "returns": returns}).dropna()

        if len(aligned) < quantiles * 2:
            return {
                "quantile_returns": {},
                "long_short_return": np.nan,
                "long_short_sharpe": np.nan,
            }

        # Assign quantiles based on signal
        aligned["quantile"] = pd.qcut(
            aligned["signal"],
            q=quantiles,
            labels=False,
            duplicates="drop",
        )

        # Calculate returns for each quantile
        quantile_returns = {}
        for q in range(quantiles):
            q_returns = aligned[aligned["quantile"] == q]["returns"]
            if len(q_returns) > 0:
                quantile_returns[f"Q{q+1}"] = {
                    "mean_return": q_returns.mean(),
                    "std_return": q_returns.std(),
                    "sharpe": self._calculate_sharpe(q_returns),
                    "count": len(q_returns),
                }

        # Long-short portfolio (top quantile minus bottom quantile)
        top_returns = aligned[aligned["quantile"] == quantiles - 1]["returns"]
        bottom_returns = aligned[aligned["quantile"] == 0]["returns"]

        if len(top_returns) > 0 and len(bottom_returns) > 0:
            # Align by index
            long_short = top_returns - bottom_returns.reindex(top_returns.index, fill_value=0)
            long_short_return = long_short.mean()
            long_short_sharpe = self._calculate_sharpe(long_short)
        else:
            long_short_return = np.nan
            long_short_sharpe = np.nan

        return {
            "quantile_returns": quantile_returns,
            "long_short_return": long_short_return,
            "long_short_sharpe": long_short_sharpe,
        }

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio.

        Args:
            returns: Series of returns

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2 or returns.std() < 1e-12:
            return np.nan

        # Assume daily returns, annualize
        excess_return = returns.mean() - (self.risk_free_rate / 252)
        sharpe = excess_return / returns.std() * np.sqrt(252)

        return sharpe
