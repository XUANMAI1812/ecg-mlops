"""Factory for creating ECG classification models."""

from torch import nn

from ecg.models.cnn1d import ECG1DCNN
from ecg.models.resnet1d import ECGResNet1D


SUPPORTED_MODELS = ("cnn1d", "resnet1d")


def build_model(
    model_name: str,
    num_classes: int = 5,
) -> nn.Module:
    """Build an ECG model by its registered name."""
    name = model_name.lower().strip()

    if name == "cnn1d":
        return ECG1DCNN(num_classes=num_classes)

    if name == "resnet1d":
        return ECGResNet1D(num_classes=num_classes)

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Supported models: {supported}"
    )
