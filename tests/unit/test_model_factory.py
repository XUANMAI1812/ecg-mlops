import pytest
import torch

from ecg.models.cnn1d import ECG1DCNN
from ecg.models.resnet1d import ECGResNet1D
from ecg.models.model_factory import SUPPORTED_MODELS, build_model


@pytest.mark.parametrize(
    ("model_name", "model_class"),
    [
        ("cnn1d", ECG1DCNN),
        ("resnet1d", ECGResNet1D),
    ],
)
def test_build_model(model_name, model_class):
    model = build_model(model_name)
    assert isinstance(model, model_class)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_model_output_shape(model_name):
    model = build_model(model_name)
    inputs = torch.randn(2, 12, 1000)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 5)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_build_model_with_custom_num_classes(model_name):
    model = build_model(model_name, num_classes=3)
    inputs = torch.randn(2, 12, 1000)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 3)


def test_build_model_normalizes_name():
    model = build_model(" RESNET1D ")
    assert isinstance(model, ECGResNet1D)


def test_build_model_rejects_unsupported_name():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("unknown")


def test_supported_models_contains_both_models():
    assert SUPPORTED_MODELS == ("cnn1d", "resnet1d")
