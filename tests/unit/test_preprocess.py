import numpy as np
import pytest

from ecg.data.preprocess import (
    filter_ecg,
    normalize_ecg,
    preprocess_ecg,
    resample_ecg,
)


def test_filter_preserves_shape():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(1000, 12))

    result = filter_ecg(signal, sampling_rate=100)

    assert result.shape == signal.shape
    assert np.isfinite(result).all()


def test_resample_500hz_to_100hz():
    signal = np.random.default_rng(42).normal(size=(5000, 12))

    result = resample_ecg(signal, original_rate=500, target_rate=100)

    assert result.shape == (1000, 12)
    assert np.isfinite(result).all()


def test_normalize_per_lead():
    signal = np.random.default_rng(42).normal(loc=5.0, scale=2.0, size=(1000, 12))

    result = normalize_ecg(signal)

    np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(result.std(axis=0), 1.0, atol=1e-10)


def test_preprocess_pipeline():
    signal = np.random.default_rng(42).normal(size=(5000, 12))

    result = preprocess_ecg(signal, original_rate=500, target_rate=100)

    assert result.shape == (1000, 12)
    assert np.isfinite(result).all()


def test_reject_non_finite_signal():
    signal = np.ones((1000, 12))
    signal[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN hoặc Inf"):
        normalize_ecg(signal)


def test_reject_invalid_sampling_rate():
    signal = np.ones((1000, 12))

    with pytest.raises(ValueError, match="sampling_rate"):
        filter_ecg(signal, sampling_rate=0)
