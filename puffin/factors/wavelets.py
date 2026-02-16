"""
Wavelet transformation module for signal denoising and decomposition.

This module provides wavelet-based signal processing techniques for:
- Signal denoising using wavelet thresholding
- Multi-resolution signal decomposition
- Signal reconstruction from wavelet components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings


# Try to import PyWavelets
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn(
        "PyWavelets not available. Install it for wavelet functionality: pip install PyWavelets",
        ImportWarning
    )


def wavelet_denoise(
    signal: Union[pd.Series, np.ndarray],
    wavelet: str = 'db4',
    level: int = 3,
    threshold_method: str = 'soft',
    threshold_mode: str = 'universal'
) -> pd.Series:
    """
    Denoise signal using wavelet transform and thresholding.

    This function decomposes the signal into wavelet coefficients,
    applies thresholding to remove noise, and reconstructs the signal.

    Parameters
    ----------
    signal : pd.Series or np.ndarray
        Input signal to denoise
    wavelet : str, optional
        Wavelet family to use (default: 'db4' - Daubechies 4)
        Common options: 'db1'-'db20', 'sym2'-'sym20', 'coif1'-'coif5', 'haar'
    level : int, optional
        Decomposition level (default: 3)
    threshold_method : str, optional
        Thresholding method: 'soft' or 'hard' (default: 'soft')
    threshold_mode : str, optional
        Threshold selection mode: 'universal', 'sqtwolog', 'minimax' (default: 'universal')

    Returns
    -------
    pd.Series
        Denoised signal

    Examples
    --------
    >>> signal = pd.Series([1, 2, 1.5, 3, 2.5, 4, 3.5, 5])
    >>> denoised = wavelet_denoise(signal, wavelet='db4', level=2)

    Notes
    -----
    The universal threshold is: threshold = sigma * sqrt(2 * log(n))
    where sigma is the noise standard deviation (estimated from detail coefficients)
    and n is the signal length.
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required for wavelet denoising. Install with: pip install PyWavelets")

    # Convert to numpy array
    if isinstance(signal, pd.Series):
        values = signal.values
        index = signal.index
    else:
        values = signal
        index = None

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(values, wavelet, level=level)

    # Estimate noise standard deviation from finest detail coefficients
    # Using Median Absolute Deviation (MAD)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Calculate threshold based on mode
    n = len(values)
    if threshold_mode == 'universal':
        threshold = sigma * np.sqrt(2 * np.log(n))
    elif threshold_mode == 'sqtwolog':
        threshold = sigma * np.sqrt(2 * np.log(n))
    elif threshold_mode == 'minimax':
        # Minimax threshold approximation
        if n > 32:
            threshold = sigma * (0.3936 + 0.1829 * np.log2(n))
        else:
            threshold = 0
    else:
        threshold = sigma * np.sqrt(2 * np.log(n))

    # Apply thresholding to detail coefficients (not approximation)
    coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
    for i in range(1, len(coeffs)):
        if threshold_method == 'soft':
            # Soft thresholding
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        elif threshold_method == 'hard':
            # Hard thresholding
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
        else:
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

    # Reconstruct signal
    denoised = pywt.waverec(coeffs_thresh, wavelet)

    # Ensure same length as input (reconstruction may add padding)
    denoised = denoised[:len(values)]

    if index is not None:
        return pd.Series(denoised, index=index)
    else:
        return pd.Series(denoised)


def wavelet_decompose(
    signal: Union[pd.Series, np.ndarray],
    wavelet: str = 'db4',
    level: int = 3
) -> Dict[str, pd.Series]:
    """
    Decompose signal into wavelet components.

    This function performs multi-resolution decomposition of the signal
    into approximation and detail components at multiple scales.

    Parameters
    ----------
    signal : pd.Series or np.ndarray
        Input signal to decompose
    wavelet : str, optional
        Wavelet family to use (default: 'db4')
    level : int, optional
        Decomposition level (default: 3)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'approximation': Approximation coefficients (low-frequency trend)
        - 'detail_1', 'detail_2', ...: Detail coefficients at each level
        - 'reconstructed': Full reconstructed signal (for verification)

    Examples
    --------
    >>> signal = pd.Series(np.sin(np.linspace(0, 10, 100)))
    >>> components = wavelet_decompose(signal, level=3)
    >>> print(components.keys())

    Notes
    -----
    - Level 1 details capture highest frequency components
    - Higher level details capture progressively lower frequencies
    - Approximation at level N captures the trend
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")

    # Convert to numpy array
    if isinstance(signal, pd.Series):
        values = signal.values
        index = signal.index
    else:
        values = signal
        index = None

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(values, wavelet, level=level)

    # Organize components
    components = {}

    # Approximation coefficients (lowest frequency)
    approx = coeffs[0]
    # Upsample to original length
    approx_full = pywt.upcoef('a', approx, wavelet, level=level, take=len(values))
    components['approximation'] = pd.Series(approx_full[:len(values)], index=index)

    # Detail coefficients (from coarse to fine)
    for i in range(1, len(coeffs)):
        detail = coeffs[i]
        # Upsample to original length
        detail_full = pywt.upcoef('d', detail, wavelet, level=level - i + 1, take=len(values))
        components[f'detail_{i}'] = pd.Series(detail_full[:len(values)], index=index)

    # Reconstruct full signal (for verification)
    reconstructed = pywt.waverec(coeffs, wavelet)[:len(values)]
    components['reconstructed'] = pd.Series(reconstructed, index=index)

    return components


def reconstruct_from_levels(
    components: Dict[str, Union[pd.Series, np.ndarray]],
    levels: Optional[List[str]] = None
) -> pd.Series:
    """
    Reconstruct signal from selected wavelet components.

    This allows you to reconstruct the signal using only specific
    frequency components (e.g., only trend, or only high-frequency details).

    Parameters
    ----------
    components : dict
        Dictionary of wavelet components (output from wavelet_decompose)
    levels : list of str, optional
        List of component names to include in reconstruction.
        If None, uses all components.
        Examples: ['approximation'], ['detail_1', 'detail_2'], etc.

    Returns
    -------
    pd.Series
        Reconstructed signal from selected components

    Examples
    --------
    >>> signal = pd.Series(np.sin(np.linspace(0, 10, 100)))
    >>> components = wavelet_decompose(signal, level=3)
    >>> # Reconstruct only the trend (approximation)
    >>> trend = reconstruct_from_levels(components, levels=['approximation'])
    >>> # Reconstruct only high-frequency components
    >>> noise = reconstruct_from_levels(components, levels=['detail_1', 'detail_2'])
    """
    if levels is None:
        levels = [k for k in components.keys() if k != 'reconstructed']

    # Get index from first component
    first_component = components[levels[0]]
    if isinstance(first_component, pd.Series):
        index = first_component.index
    else:
        index = None

    # Sum selected components
    reconstructed = np.zeros(len(first_component))
    for level in levels:
        if level in components:
            if isinstance(components[level], pd.Series):
                reconstructed += components[level].values
            else:
                reconstructed += components[level]

    if index is not None:
        return pd.Series(reconstructed, index=index)
    else:
        return pd.Series(reconstructed)


def wavelet_smooth(
    signal: Union[pd.Series, np.ndarray],
    wavelet: str = 'db4',
    level: int = 3,
    keep_approximation: bool = True,
    keep_details: Optional[List[int]] = None
) -> pd.Series:
    """
    Smooth signal by removing high-frequency wavelet components.

    This is a simpler interface to wavelet-based smoothing that
    automatically handles decomposition and reconstruction.

    Parameters
    ----------
    signal : pd.Series or np.ndarray
        Input signal to smooth
    wavelet : str, optional
        Wavelet family to use (default: 'db4')
    level : int, optional
        Decomposition level (default: 3)
    keep_approximation : bool, optional
        Whether to keep approximation coefficients (default: True)
    keep_details : list of int, optional
        Which detail levels to keep (1-indexed).
        If None, keeps no details (maximum smoothing).
        Example: [2, 3] keeps medium-frequency components.

    Returns
    -------
    pd.Series
        Smoothed signal

    Examples
    --------
    >>> signal = pd.Series([1, 2, 1.5, 3, 2.5, 4, 3.5, 5])
    >>> # Maximum smoothing (keep only trend)
    >>> trend = wavelet_smooth(signal, keep_approximation=True, keep_details=None)
    >>> # Keep some medium-frequency components
    >>> partial = wavelet_smooth(signal, keep_approximation=True, keep_details=[2, 3])
    """
    # Decompose signal
    components = wavelet_decompose(signal, wavelet=wavelet, level=level)

    # Select components to keep
    levels_to_keep = []

    if keep_approximation:
        levels_to_keep.append('approximation')

    if keep_details is not None:
        for detail_level in keep_details:
            levels_to_keep.append(f'detail_{detail_level}')

    # Reconstruct
    return reconstruct_from_levels(components, levels=levels_to_keep)


def wavelet_variance_decomposition(
    signal: Union[pd.Series, np.ndarray],
    wavelet: str = 'db4',
    level: int = 3
) -> pd.DataFrame:
    """
    Compute variance decomposition across wavelet scales.

    This shows how much of the signal's variance is contained
    in each frequency band.

    Parameters
    ----------
    signal : pd.Series or np.ndarray
        Input signal
    wavelet : str, optional
        Wavelet family to use (default: 'db4')
    level : int, optional
        Decomposition level (default: 3)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: component, variance, variance_pct

    Examples
    --------
    >>> signal = pd.Series(np.random.randn(100))
    >>> var_decomp = wavelet_variance_decomposition(signal, level=3)
    >>> print(var_decomp)
    """
    # Decompose signal
    components = wavelet_decompose(signal, wavelet=wavelet, level=level)

    # Calculate variance for each component
    results = []
    total_variance = 0

    for name, component in components.items():
        if name != 'reconstructed':
            var = np.var(component)
            total_variance += var
            results.append({'component': name, 'variance': var})

    # Calculate percentages
    for result in results:
        result['variance_pct'] = (result['variance'] / total_variance) * 100

    return pd.DataFrame(results)


def multiscale_decomposition(
    prices: pd.Series,
    wavelet: str = 'db4',
    level: int = 4
) -> pd.DataFrame:
    """
    Decompose price series into multiple time scales.

    This is particularly useful for separating different market dynamics:
    - High-frequency noise (day trading)
    - Medium-frequency patterns (swing trading)
    - Low-frequency trends (position trading)

    Parameters
    ----------
    prices : pd.Series
        Price series
    wavelet : str, optional
        Wavelet family (default: 'db4')
    level : int, optional
        Decomposition level (default: 4)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each time scale:
        - trend: Long-term trend (approximation)
        - D1, D2, D3, D4: Detail components from fine to coarse

    Examples
    --------
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
    >>> scales = multiscale_decomposition(prices, level=3)
    """
    components = wavelet_decompose(prices, wavelet=wavelet, level=level)

    result = pd.DataFrame(index=prices.index)
    result['trend'] = components['approximation']

    for i in range(1, level + 1):
        if f'detail_{i}' in components:
            result[f'D{i}'] = components[f'detail_{i}']

    return result


def adaptive_wavelet_threshold(
    signal: Union[pd.Series, np.ndarray],
    wavelet: str = 'db4',
    level: int = 3,
    noise_est_method: str = 'mad'
) -> pd.Series:
    """
    Apply adaptive wavelet thresholding with level-dependent thresholds.

    Unlike standard wavelet denoising, this method uses different
    thresholds for each decomposition level, which can be more effective
    for signals with varying noise levels across frequencies.

    Parameters
    ----------
    signal : pd.Series or np.ndarray
        Input signal
    wavelet : str, optional
        Wavelet family (default: 'db4')
    level : int, optional
        Decomposition level (default: 3)
    noise_est_method : str, optional
        Method for noise estimation: 'mad' or 'std' (default: 'mad')

    Returns
    -------
    pd.Series
        Denoised signal

    Examples
    --------
    >>> noisy_signal = pd.Series([1, 2, 1.5, 3, 2.5, 4, 3.5, 5])
    >>> denoised = adaptive_wavelet_threshold(noisy_signal)
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")

    if isinstance(signal, pd.Series):
        values = signal.values
        index = signal.index
    else:
        values = signal
        index = None

    # Decompose
    coeffs = pywt.wavedec(values, wavelet, level=level)

    # Apply level-dependent thresholding
    coeffs_thresh = [coeffs[0]]  # Keep approximation

    for i in range(1, len(coeffs)):
        detail = coeffs[i]

        # Estimate noise at this level
        if noise_est_method == 'mad':
            sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
        else:
            sigma = np.std(detail)

        # Calculate threshold
        n = len(detail)
        threshold = sigma * np.sqrt(2 * np.log(n))

        # Apply soft thresholding
        coeffs_thresh.append(pywt.threshold(detail, threshold, mode='soft'))

    # Reconstruct
    denoised = pywt.waverec(coeffs_thresh, wavelet)[:len(values)]

    if index is not None:
        return pd.Series(denoised, index=index)
    else:
        return pd.Series(denoised)
