"""
WorldQuant-style formulaic alpha expressions.

This module provides a framework for defining and evaluating alpha factors
using formula strings similar to WorldQuant's alpha expressions.

Supported operators:
- rank(): Cross-sectional rank
- delay(x, d): Value of x d periods ago
- delta(x, d): Change in x over d periods (x - delay(x, d))
- ts_mean(x, d): Time-series mean over d periods
- ts_std(x, d): Time-series standard deviation over d periods
- ts_rank(x, d): Time-series rank over d periods
- ts_max(x, d): Time-series maximum over d periods
- ts_min(x, d): Time-series minimum over d periods
- correlation(x, y, d): Rolling correlation between x and y over d periods
- covariance(x, y, d): Rolling covariance between x and y over d periods
- abs(x): Absolute value
- log(x): Natural logarithm
- sign(x): Sign (-1, 0, or 1)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union
import re


class AlphaExpression:
    """
    Parser and evaluator for WorldQuant-style alpha expressions.

    This class parses formula strings and evaluates them on market data
    to produce alpha factors.

    Parameters
    ----------
    expression : str
        Alpha formula string (e.g., "rank(delay(close, 5) - close)")

    Examples
    --------
    >>> data = {'close': pd.DataFrame({'AAPL': [100, 101, 102, 103, 104]})}
    >>> alpha = AlphaExpression("rank(delay(close, 2) - close)")
    >>> factor = alpha.evaluate(data)

    >>> # More complex example
    >>> alpha = AlphaExpression("rank(ts_mean(close, 5) / close - 1)")
    >>> factor = alpha.evaluate(data)
    """

    def __init__(self, expression: str):
        self.expression = expression
        self.operators = {
            'rank': self._rank,
            'delay': self._delay,
            'delta': self._delta,
            'ts_mean': self._ts_mean,
            'ts_std': self._ts_std,
            'ts_rank': self._ts_rank,
            'ts_max': self._ts_max,
            'ts_min': self._ts_min,
            'ts_sum': self._ts_sum,
            'correlation': self._correlation,
            'covariance': self._covariance,
            'abs': self._abs,
            'log': self._log,
            'sign': self._sign,
            'power': self._power,
            'scale': self._scale,
        }

    def evaluate(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Evaluate the alpha expression on market data.

        Parameters
        ----------
        data : dict
            Dictionary of DataFrames with keys like 'open', 'high', 'low',
            'close', 'volume'. Each DataFrame has dates as index and
            symbols as columns.

        Returns
        -------
        pd.DataFrame
            Factor values with same shape as input DataFrames

        Examples
        --------
        >>> data = {
        ...     'close': pd.DataFrame({'AAPL': [100, 101, 102]}),
        ...     'volume': pd.DataFrame({'AAPL': [1000, 1100, 1050]})
        ... }
        >>> alpha = AlphaExpression("rank(volume * close)")
        >>> result = alpha.evaluate(data)
        """
        # Store data for reference
        self.data = data

        # Evaluate expression
        result = self._parse_and_evaluate(self.expression)

        return result

    def _parse_and_evaluate(self, expr: str) -> pd.DataFrame:
        """Parse and evaluate an expression recursively."""
        expr = expr.strip()

        # Check if it's a basic data field
        if expr in self.data:
            return self.data[expr]

        # Check if it's a number
        try:
            value = float(expr)
            # Return a constant DataFrame matching data shape
            reference = list(self.data.values())[0]
            return pd.DataFrame(value, index=reference.index, columns=reference.columns)
        except ValueError:
            pass

        # Check for operators
        for op_name, op_func in self.operators.items():
            pattern = rf'{op_name}\s*\('
            if re.match(pattern, expr):
                # Extract arguments
                args = self._extract_arguments(expr, op_name)
                return op_func(*args)

        # Check for binary operators
        for op in ['+', '-', '*', '/', '<', '>', '<=', '>=', '==']:
            if op in expr:
                # Split by operator (handling nested parentheses)
                left, right = self._split_binary(expr, op)
                if left and right:
                    left_val = self._parse_and_evaluate(left)
                    right_val = self._parse_and_evaluate(right)

                    if op == '+':
                        return left_val + right_val
                    elif op == '-':
                        return left_val - right_val
                    elif op == '*':
                        return left_val * right_val
                    elif op == '/':
                        return left_val / (right_val + 1e-8)
                    elif op == '<':
                        return (left_val < right_val).astype(float)
                    elif op == '>':
                        return (left_val > right_val).astype(float)
                    elif op == '<=':
                        return (left_val <= right_val).astype(float)
                    elif op == '>=':
                        return (left_val >= right_val).astype(float)
                    elif op == '==':
                        return (left_val == right_val).astype(float)

        raise ValueError(f"Cannot parse expression: {expr}")

    def _extract_arguments(self, expr: str, func_name: str) -> list:
        """Extract function arguments from expression."""
        # Find the opening parenthesis
        start = expr.index('(')
        # Find matching closing parenthesis
        level = 0
        for i in range(start, len(expr)):
            if expr[i] == '(':
                level += 1
            elif expr[i] == ')':
                level -= 1
                if level == 0:
                    args_str = expr[start + 1:i]
                    break

        # Split arguments by comma (respecting nested parentheses)
        args = []
        current_arg = ""
        level = 0

        for char in args_str:
            if char == ',' and level == 0:
                args.append(self._parse_and_evaluate(current_arg.strip()))
                current_arg = ""
            else:
                if char == '(':
                    level += 1
                elif char == ')':
                    level -= 1
                current_arg += char

        if current_arg:
            args.append(self._parse_and_evaluate(current_arg.strip()))

        return args

    def _split_binary(self, expr: str, op: str) -> tuple:
        """Split expression by binary operator (respecting parentheses)."""
        level = 0
        # Find the last occurrence of operator at level 0
        op_pos = -1

        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == ')':
                level += 1
            elif expr[i] == '(':
                level -= 1
            elif level == 0 and expr[i:i + len(op)] == op:
                op_pos = i
                break

        if op_pos > 0:
            left = expr[:op_pos].strip()
            right = expr[op_pos + len(op):].strip()
            return left, right

        return None, None

    # Operator implementations

    def _rank(self, x: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank (percentile)."""
        return x.rank(axis=1, pct=True)

    def _delay(self, x: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
        """Delay operator: value d periods ago."""
        periods = int(d.iloc[0, 0]) if isinstance(d, pd.DataFrame) else int(d)
        return x.shift(periods)

    def _delta(self, x: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
        """Delta operator: change over d periods."""
        periods = int(d.iloc[0, 0]) if isinstance(d, pd.DataFrame) else int(d)
        return x - x.shift(periods)

    def _ts_mean(self, x: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
        """Time-series mean over d periods."""
        periods = int(d.iloc[0, 0]) if isinstance(d, pd.DataFrame) else int(d)
        return x.rolling(window=periods, min_periods=1).mean()

    def _ts_std(self, x: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
        """Time-series standard deviation over d periods."""
        periods = int(d.iloc[0, 0]) if isinstance(d, pd.DataFrame) else int(d)
        return x.rolling(window=periods, min_periods=2).std()

    def _ts_rank(self, x: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
        """Time-series rank over d periods."""
        periods = int(d.iloc[0, 0]) if isinstance(d, pd.DataFrame) else int(d)
        return x.rolling(window=periods, min_periods=1).apply(
            lambda s: (s.rank().iloc[-1] - 1) / (len(s) - 1) if len(s) > 1 else 0.5
        )

    def _ts_max(self, x: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
        """Time-series maximum over d periods."""
        periods = int(d.iloc[0, 0]) if isinstance(d, pd.DataFrame) else int(d)
        return x.rolling(window=periods, min_periods=1).max()

    def _ts_min(self, x: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
        """Time-series minimum over d periods."""
        periods = int(d.iloc[0, 0]) if isinstance(d, pd.DataFrame) else int(d)
        return x.rolling(window=periods, min_periods=1).min()

    def _ts_sum(self, x: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
        """Time-series sum over d periods."""
        periods = int(d.iloc[0, 0]) if isinstance(d, pd.DataFrame) else int(d)
        return x.rolling(window=periods, min_periods=1).sum()

    def _correlation(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        d: pd.DataFrame
    ) -> pd.DataFrame:
        """Rolling correlation between x and y over d periods."""
        periods = int(d.iloc[0, 0]) if isinstance(d, pd.DataFrame) else int(d)
        result = pd.DataFrame(index=x.index, columns=x.columns)

        for col in x.columns:
            if col in y.columns:
                result[col] = x[col].rolling(window=periods, min_periods=2).corr(y[col])

        return result

    def _covariance(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        d: pd.DataFrame
    ) -> pd.DataFrame:
        """Rolling covariance between x and y over d periods."""
        periods = int(d.iloc[0, 0]) if isinstance(d, pd.DataFrame) else int(d)
        result = pd.DataFrame(index=x.index, columns=x.columns)

        for col in x.columns:
            if col in y.columns:
                result[col] = x[col].rolling(window=periods, min_periods=2).cov(y[col])

        return result

    def _abs(self, x: pd.DataFrame) -> pd.DataFrame:
        """Absolute value."""
        return x.abs()

    def _log(self, x: pd.DataFrame) -> pd.DataFrame:
        """Natural logarithm."""
        return np.log(x + 1e-8)

    def _sign(self, x: pd.DataFrame) -> pd.DataFrame:
        """Sign function."""
        return np.sign(x)

    def _power(self, x: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
        """Power function."""
        power = p.iloc[0, 0] if isinstance(p, pd.DataFrame) else p
        return x ** power

    def _scale(self, x: pd.DataFrame) -> pd.DataFrame:
        """Scale to sum to 1 cross-sectionally (creates portfolio weights)."""
        return x.div(x.abs().sum(axis=1), axis=0)


# Predefined WorldQuant-style alphas

ALPHA_LIBRARY = {
    'alpha001': "rank(ts_rank(log(volume), 10))",
    'alpha002': "delta(log(volume), 2) / delta(log(close), 2)",
    'alpha003': "rank(correlation(close, volume, 10))",
    'alpha004': "ts_rank(rank(low), 9)",
    'alpha005': "rank(close / ts_mean(volume, 10))",
    'alpha006': "correlation(open, volume, 10)",
    'alpha007': "rank(abs(delta(close, 7))) * sign(delta(close, 7))",
    'alpha008': "rank(delta(ts_sum(open, 5), 1))",
    'alpha009': "delta(ts_mean(close, 10), 1)",
    'alpha010': "rank(abs(delta(close, 1))) * sign(delta(close, 1))",
}


def evaluate_alpha(
    expression: str,
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Evaluate an alpha expression.

    This is a convenience function for quickly evaluating alpha expressions.

    Parameters
    ----------
    expression : str
        Alpha formula string
    data : dict
        Market data dictionary

    Returns
    -------
    pd.DataFrame
        Factor values

    Examples
    --------
    >>> data = {'close': pd.DataFrame({'AAPL': [100, 101, 102, 103, 104]})}
    >>> factor = evaluate_alpha("rank(delta(close, 1))", data)
    """
    alpha = AlphaExpression(expression)
    return alpha.evaluate(data)


def evaluate_alpha_library(
    data: Dict[str, pd.DataFrame],
    alphas: Optional[list] = None
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate multiple alphas from the library.

    Parameters
    ----------
    data : dict
        Market data dictionary
    alphas : list of str, optional
        List of alpha names to evaluate. If None, evaluates all alphas.

    Returns
    -------
    dict
        Dictionary mapping alpha names to factor DataFrames

    Examples
    --------
    >>> data = {
    ...     'open': pd.DataFrame({'AAPL': [100, 101, 102]}),
    ...     'close': pd.DataFrame({'AAPL': [101, 102, 103]}),
    ...     'volume': pd.DataFrame({'AAPL': [1000, 1100, 1050]})
    ... }
    >>> results = evaluate_alpha_library(data, alphas=['alpha001', 'alpha002'])
    """
    if alphas is None:
        alphas = list(ALPHA_LIBRARY.keys())

    results = {}

    for alpha_name in alphas:
        if alpha_name in ALPHA_LIBRARY:
            expression = ALPHA_LIBRARY[alpha_name]
            try:
                results[alpha_name] = evaluate_alpha(expression, data)
            except Exception as e:
                print(f"Error evaluating {alpha_name}: {e}")

    return results


def to_multiindex_series(factor_df: pd.DataFrame) -> pd.Series:
    """
    Convert factor DataFrame to MultiIndex Series.

    This is useful for compatibility with factor evaluation tools
    that expect (date, symbol) MultiIndex format.

    Parameters
    ----------
    factor_df : pd.DataFrame
        Factor values with dates as index and symbols as columns

    Returns
    -------
    pd.Series
        Factor values with (date, symbol) MultiIndex

    Examples
    --------
    >>> factor_df = pd.DataFrame({'AAPL': [0.5, 0.6], 'MSFT': [0.3, 0.4]})
    >>> factor_series = to_multiindex_series(factor_df)
    """
    return factor_df.stack()


def combine_alphas(
    factors: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Combine multiple alpha factors with optional weights.

    Parameters
    ----------
    factors : dict
        Dictionary mapping factor names to factor DataFrames
    weights : dict, optional
        Dictionary mapping factor names to weights. If None, uses equal weights.

    Returns
    -------
    pd.DataFrame
        Combined factor

    Examples
    --------
    >>> factors = {
    ...     'alpha1': pd.DataFrame({'AAPL': [0.5, 0.6]}),
    ...     'alpha2': pd.DataFrame({'AAPL': [0.3, 0.4]})
    ... }
    >>> combined = combine_alphas(factors, weights={'alpha1': 0.7, 'alpha2': 0.3})
    """
    if weights is None:
        weights = {name: 1.0 / len(factors) for name in factors}

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Combine factors
    combined = None

    for name, factor_df in factors.items():
        weight = weights.get(name, 0)
        if combined is None:
            combined = factor_df * weight
        else:
            combined = combined + (factor_df * weight)

    return combined


def neutralize_factor(
    factor: pd.DataFrame,
    neutralizer: pd.DataFrame
) -> pd.DataFrame:
    """
    Neutralize factor against another factor (e.g., market beta).

    This removes the exposure of the factor to the neutralizer by
    performing cross-sectional regression at each time point.

    Parameters
    ----------
    factor : pd.DataFrame
        Factor to neutralize
    neutralizer : pd.DataFrame
        Factor to neutralize against (e.g., market beta, sector, size)

    Returns
    -------
    pd.DataFrame
        Neutralized factor (residuals from regression)

    Examples
    --------
    >>> factor = pd.DataFrame({'AAPL': [0.5, 0.6], 'MSFT': [0.3, 0.4]})
    >>> market_beta = pd.DataFrame({'AAPL': [1.2, 1.1], 'MSFT': [0.8, 0.9]})
    >>> neutral_factor = neutralize_factor(factor, market_beta)
    """
    result = pd.DataFrame(index=factor.index, columns=factor.columns)

    for date in factor.index:
        factor_row = factor.loc[date]
        neutralizer_row = neutralizer.loc[date]

        # Drop NaNs
        valid_mask = factor_row.notna() & neutralizer_row.notna()
        if valid_mask.sum() > 1:
            y = factor_row[valid_mask].values
            X = neutralizer_row[valid_mask].values.reshape(-1, 1)

            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])

            # OLS regression
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

            # Residuals
            residuals = y - X_with_intercept @ beta

            result.loc[date, valid_mask] = residuals

    return result
