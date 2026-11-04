"""ResNet1D for multi-label ECG classification."""

import torch
from torch import nn

NUM_LEADS = 12
NUM_CLASSES = 5


class ResidualBlock1D(nn.Module):
    """A basic 1D residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        return self.relu(out)


class ECGResNet1D(nn.Module):
    """A medium-sized ResNet1D for 12-lead ECG classification."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                NUM_LEADS,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = self._make_stage(32, 32, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(32, 64, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(64, 128, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [
            ResidualBlock1D(
                in_channels,
                out_channels,
                stride=stride,
            )
        ]

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits with shape [batch_size, num_classes]."""
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape [batch_size, 12, time_steps], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != NUM_LEADS:
            raise ValueError(f"Expected {NUM_LEADS} ECG leads, got {x.shape[1]}")

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)

        return self.classifier(x)
