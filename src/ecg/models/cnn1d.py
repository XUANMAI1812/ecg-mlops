"""Baseline 1D-CNN for multi-label ECG classification."""

import torch
from torch import nn

NUM_LEADS = 12
NUM_CLASSES = 5


class ECG1DCNN(nn.Module):
    """A lightweight 1D-CNN for 12-lead, 10-second ECG signals."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(NUM_LEADS, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits with shape [batch_size, num_classes]."""
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape [batch_size, 12, time_steps], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != NUM_LEADS:
            raise ValueError(
                f"Expected {NUM_LEADS} ECG leads, got {x.shape[1]}"
            )

        return self.classifier(self.features(x))
