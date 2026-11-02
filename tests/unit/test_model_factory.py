import pytest
import torch

from ecg.models.cnn1d import ECG1DCNN
from ecg.models.model_factory import SUPPORTED_MODELS, build_model


def test_build_cnn1d_model():
    model = build_model("cnn1d")
    assert isinstance(model, ECG1DCNN)


def test_build_model_with_custom_num_classes():
    model = build_model("cnn1d", num_classes=3)
    inputs = torch.randn(2, 12, 1000)

    outputs = model(inputs)

    assert outputs.shape == (2, 3)


def test_build_model_normalizes_name():
    model = build_model(" CNN1D ")
    assert isinstance(model, ECG1DCNN)


def test_build_model_rejects_unsupported_name():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("unknown")


def test_supported_models_contains_cnn1d():
    assert "cnn1d" in SUPPORTED_MODELS
