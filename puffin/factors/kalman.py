"""
Kalman filter module for signal denoising and trend extraction.

This module provides Kalman filter implementations for:
- Signal denoising and smoothing
- Trend extraction from price series
- Dynamic hedge ratio estimation for pairs trading
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


class KalmanFilter:
    """
    Kalman filter for signal denoising and state estimation.

    The Kalman filter is an optimal recursive estimator that processes
    measurements over time to produce estimates of unknown variables that
    tend to be more accurate than those based on a single measurement alone.

    Parameters
    ----------
    transition_matrix : np.ndarray, optional
        State transition matrix (F). Default: identity matrix
    observation_matrix : np.ndarray, optional
        Observation matrix (H). Default: identity matrix
    process_covariance : np.ndarray or float, optional
        Process noise covariance (Q). Default: 1e-5
    observation_covariance : np.ndarray or float, optional
        Measurement noise covariance (R). Default: 1e-2
    initial_state : np.ndarray, optional
        Initial state estimate. Default: zeros
    initial_covariance : np.ndarray, optional
        Initial state covariance (P). Default: identity matrix

    Examples
    --------
    >>> kf = KalmanFilter()
    >>> prices = pd.Series([100, 101, 99, 102, 103])
    >>> filtered = kf.filter(prices)
    >>> smoothed = kf.smooth(prices)
    """

    def __init__(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        observation_matrix: Optional[np.ndarray] = None,
        process_covariance: Union[np.ndarray, float] = 1e-5,
        observation_covariance: Union[np.ndarray, float] = 1e-2,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None,
        dim_state: int = 1
    ):
        """Initialize Kalman filter with system matrices."""
        self.dim_state = dim_state

        # Default to identity matrices
        if transition_matrix is None:
            self.F = np.eye(dim_state)
        else:
            self.F = transition_matrix

        if observation_matrix is None:
            self.H = np.eye(dim_state)
        else:
            self.H = observation_matrix

        # Process noise covariance
        if isinstance(process_covariance, (int, float)):
            self.Q = np.eye(dim_state) * process_covariance
        else:
            self.Q = process_covariance

        # Measurement noise covariance
        if isinstance(observation_covariance, (int, float)):
            self.R = np.eye(dim_state) * observation_covariance
        else:
            self.R = observation_covariance

        # Initial state
        if initial_state is None:
            self.x = np.zeros((dim_state, 1))
        else:
            self.x = initial_state.reshape(-1, 1)

        # Initial covariance
        if initial_covariance is None:
            self.P = np.eye(dim_state)
        else:
            self.P = initial_covariance

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state (time update).

        Returns
        -------
        x_pred : np.ndarray
            Predicted state estimate
        P_pred : np.ndarray
            Predicted error covariance
        """
        # Predicted state estimate
        x_pred = self.F @ self.x

        # Predicted error covariance
        P_pred = self.F @ self.P @ self.F.T + self.Q

        return x_pred, P_pred

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state estimate with measurement (measurement update).

        Parameters
        ----------
        z : np.ndarray
            Measurement vector

        Returns
        -------
        x : np.ndarray
            Updated state estimate
        P : np.ndarray
            Updated error covariance
        """
        z = z.reshape(-1, 1)

        # Predict
        x_pred, P_pred = self.predict()

        # Innovation (measurement residual)
        y = z - self.H @ x_pred

        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Updated state estimate
        self.x = x_pred + K @ y

        # Updated error covariance
        I = np.eye(self.dim_state)
        self.P = (I - K @ self.H) @ P_pred

        return self.x, self.P

    def filter(self, signal: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Apply Kalman filter to signal (forward pass only).

        Parameters
        ----------
        signal : pd.Series or np.ndarray
            Input signal to filter

        Returns
        -------
        pd.Series
            Filtered signal
        """
        if isinstance(signal, pd.Series):
            values = signal.values
            index = signal.index
        else:
            values = signal
            index = None

        # Reset state
        self.x = np.zeros((self.dim_state, 1))
        self.P = np.eye(self.dim_state)

        filtered = np.zeros(len(values))

        for i, measurement in enumerate(values):
            self.update(np.array([measurement]))
            filtered[i] = self.x[0, 0]

        if index is not None:
            return pd.Series(filtered, index=index)
        else:
            return pd.Series(filtered)

    def smooth(self, signal: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Apply Kalman smoother (forward-backward pass).

        The smoother uses all measurements (past and future) to estimate
        the state at each time point, resulting in better estimates than
        the filter alone.

        Parameters
        ----------
        signal : pd.Series or np.ndarray
            Input signal to smooth

        Returns
        -------
        pd.Series
            Smoothed signal
        """
        if isinstance(signal, pd.Series):
            values = signal.values
            index = signal.index
        else:
            values = signal
            index = None

        # Reset state
        self.x = np.zeros((self.dim_state, 1))
        self.P = np.eye(self.dim_state)

        n = len(values)
        filtered_states = np.zeros((n, self.dim_state))
        filtered_covariances = np.zeros((n, self.dim_state, self.dim_state))
        predicted_states = np.zeros((n, self.dim_state))
        predicted_covariances = np.zeros((n, self.dim_state, self.dim_state))

        # Forward pass (filtering)
        for i, measurement in enumerate(values):
            # Store predicted state and covariance
            x_pred, P_pred = self.predict()
            predicted_states[i] = x_pred.flatten()
            predicted_covariances[i] = P_pred

            # Update
            self.update(np.array([measurement]))
            filtered_states[i] = self.x.flatten()
            filtered_covariances[i] = self.P

        # Backward pass (smoothing)
        smoothed_states = np.zeros((n, self.dim_state))
        smoothed_states[-1] = filtered_states[-1]

        for i in range(n - 2, -1, -1):
            # Smoother gain
            C = filtered_covariances[i] @ self.F.T @ np.linalg.inv(predicted_covariances[i + 1])

            # Smoothed state
            smoothed_states[i] = (
                filtered_states[i] +
                C @ (smoothed_states[i + 1] - predicted_states[i + 1])
            )

        if index is not None:
            return pd.Series(smoothed_states[:, 0], index=index)
        else:
            return pd.Series(smoothed_states[:, 0])


def extract_trend(
    prices: Union[pd.Series, np.ndarray],
    process_variance: float = 1e-5,
    observation_variance: float = 1e-2
) -> pd.Series:
    """
    Extract trend from price series using Kalman filter.

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series
    process_variance : float, optional
        Process noise variance (Q). Lower values = smoother trend.
    observation_variance : float, optional
        Measurement noise variance (R). Higher values = more smoothing.

    Returns
    -------
    pd.Series
        Extracted trend

    Examples
    --------
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104])
    >>> trend = extract_trend(prices)
    """
    kf = KalmanFilter(
        process_covariance=process_variance,
        observation_covariance=observation_variance
    )
    return kf.smooth(prices)


def dynamic_hedge_ratio(
    y: Union[pd.Series, np.ndarray],
    x: Union[pd.Series, np.ndarray],
    delta: float = 1e-5
) -> pd.Series:
    """
    Calculate dynamic hedge ratio for pairs trading using Kalman filter.

    This estimates the time-varying hedge ratio (beta) in the relationship:
    y_t = alpha + beta_t * x_t + epsilon_t

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Dependent variable (e.g., stock 1 price)
    x : pd.Series or np.ndarray
        Independent variable (e.g., stock 2 price)
    delta : float, optional
        Transition variance (process noise). Lower = more stable hedge ratio.

    Returns
    -------
    pd.Series
        Time-varying hedge ratio (beta)

    Examples
    --------
    >>> stock1 = pd.Series([100, 101, 102, 103])
    >>> stock2 = pd.Series([50, 51, 50, 52])
    >>> hedge_ratio = dynamic_hedge_ratio(stock1, stock2)
    """
    if isinstance(y, pd.Series):
        y_values = y.values
        index = y.index
    else:
        y_values = y
        index = None

    if isinstance(x, pd.Series):
        x_values = x.values
    else:
        x_values = x

    # State space model for linear regression
    # State: [beta, alpha]
    n = len(y_values)
    beta_estimates = np.zeros(n)

    # Initialize Kalman filter for online regression
    # State dimension: 2 (beta and alpha)
    kf = KalmanFilter(
        transition_matrix=np.eye(2),  # Random walk for beta and alpha
        observation_matrix=np.array([[1, 1]]),  # Will be updated with [x_t, 1]
        process_covariance=np.eye(2) * delta,
        observation_covariance=np.array([[1.0]]),
        dim_state=2
    )

    for i in range(n):
        # Observation matrix: [x_t, 1]
        kf.H = np.array([[x_values[i], 1.0]])

        # Update with observation y_t
        kf.update(np.array([y_values[i]]))

        # Store beta estimate
        beta_estimates[i] = kf.x[0, 0]

    if index is not None:
        return pd.Series(beta_estimates, index=index)
    else:
        return pd.Series(beta_estimates)


class AdaptiveKalmanFilter(KalmanFilter):
    """
    Adaptive Kalman filter with automatic covariance estimation.

    This variant automatically adjusts the process and measurement noise
    covariances based on the observed innovation sequence.

    Parameters
    ----------
    adaptation_rate : float, optional
        Rate of adaptation (0 to 1). Higher = faster adaptation.
    window : int, optional
        Window size for estimating covariances.
    """

    def __init__(
        self,
        adaptation_rate: float = 0.01,
        window: int = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate
        self.window = window
        self.innovations = []

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update with adaptive covariance estimation."""
        z = z.reshape(-1, 1)

        # Predict
        x_pred, P_pred = self.predict()

        # Innovation
        y = z - self.H @ x_pred
        self.innovations.append(y[0, 0])

        # Keep only recent innovations
        if len(self.innovations) > self.window:
            self.innovations.pop(0)

        # Adapt measurement noise covariance
        if len(self.innovations) >= 2:
            innovation_var = np.var(self.innovations)
            self.R = (1 - self.adaptation_rate) * self.R + \
                     self.adaptation_rate * innovation_var * np.eye(self.dim_state)

        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Updated state estimate
        self.x = x_pred + K @ y

        # Updated error covariance
        I = np.eye(self.dim_state)
        self.P = (I - K @ self.H) @ P_pred

        return self.x, self.P


def kalman_ma_crossover(
    prices: pd.Series,
    fast_variance: float = 1e-4,
    slow_variance: float = 1e-6
) -> pd.DataFrame:
    """
    Generate Kalman-filtered moving average crossover signals.

    Creates fast and slow Kalman-filtered price series and generates
    trading signals based on their crossover.

    Parameters
    ----------
    prices : pd.Series
        Price series
    fast_variance : float, optional
        Process variance for fast filter (higher = more responsive)
    slow_variance : float, optional
        Process variance for slow filter (lower = smoother)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: fast_ma, slow_ma, signal
        signal: 1 (long), -1 (short), 0 (no position)
    """
    # Fast Kalman filter
    kf_fast = KalmanFilter(process_covariance=fast_variance)
    fast_ma = kf_fast.filter(prices)

    # Slow Kalman filter
    kf_slow = KalmanFilter(process_covariance=slow_variance)
    slow_ma = kf_slow.filter(prices)

    # Generate signals
    result = pd.DataFrame(index=prices.index)
    result['fast_ma'] = fast_ma
    result['slow_ma'] = slow_ma
    result['signal'] = 0

    # Long when fast > slow
    result.loc[fast_ma > slow_ma, 'signal'] = 1
    # Short when fast < slow
    result.loc[fast_ma < slow_ma, 'signal'] = -1

    return result
