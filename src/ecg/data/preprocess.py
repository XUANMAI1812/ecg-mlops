"""Preprocessing pipeline for PTB-XL ECG signals."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def filter_ecg(
    signal: np.ndarray,
    sampling_rate: int,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Reduce baseline wander and high-frequency noise.

    Input and output shape: (samples, leads).
    """
    signal = _validate_signal(signal, sampling_rate)

    if not 0 < lowcut < highcut < sampling_rate / 2:
        raise ValueError("Cần 0 < lowcut < highcut < Nyquist frequency")
    if order < 1:
        raise ValueError("order phải >= 1")

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    return sosfiltfilt(sos, signal, axis=0)


def resample_ecg(
    signal: np.ndarray,
    original_rate: int,
    target_rate: int = 100,
) -> np.ndarray:
    """Resample ECG while preserving the number of leads."""
    signal = _validate_signal(signal, original_rate)

    if target_rate <= 0:
        raise ValueError("target_rate phải > 0")

    ratio = Fraction(target_rate, original_rate)
    result = resample_poly(
        signal,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=0,
    )
    return result.astype(np.float64, copy=False)


def normalize_ecg(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply per-lead z-score normalization across the time axis."""
    signal = _validate_signal(signal, sampling_rate=None)

    if eps <= 0:
        raise ValueError("eps phải > 0")

    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)

    return (signal - mean) / std


def preprocess_ecg(
    signal: np.ndarray,
    original_rate: int,
    target_rate: int = 100,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Filter, resample, then normalize an ECG record."""
    signal = _validate_signal(signal, original_rate)

    filtered = filter_ecg(
        signal,
        sampling_rate=original_rate,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
    )
    resampled = resample_ecg(filtered, original_rate, target_rate)
    return normalize_ecg(resampled)


def _validate_signal(
    signal: np.ndarray,
    sampling_rate: int | None,
) -> np.ndarray:
    """Validate ECG shape, numeric values, and sampling rate."""
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 2 or signal.shape[0] < 2 or signal.shape[1] < 1:
        raise ValueError("signal phải có shape (samples, leads)")
    if not np.isfinite(signal).all():
        raise ValueError("signal chứa NaN hoặc Inf")
    if sampling_rate is not None and sampling_rate <= 0:
        raise ValueError("sampling_rate phải > 0")

    return signal
