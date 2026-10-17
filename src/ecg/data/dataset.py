"""Lazy PyTorch Dataset for multi-label PTB-XL classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg.data.labels import SUPERCLASSES, add_superclass_column
from ecg.data.load import (
    load_metadata,
    load_record,
    load_scp_statements,
    validate_signal,
)
from ecg.data.preprocess import preprocess_ecg


class ECGDataset(Dataset):
    """Load and preprocess one ECG record when requested."""

    def __init__(
        self,
        data_dir: str | Path,
        folds: tuple[int, ...],
        sampling_rate: int = 100,
    ) -> None:
        if sampling_rate not in (100, 500):
            raise ValueError("sampling_rate phải là 100 hoặc 500")

        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.filename_column = (
            "filename_lr" if sampling_rate == 100 else "filename_hr"
        )

        metadata = load_metadata(self.data_dir)
        statements = load_scp_statements(self.data_dir)
        metadata = add_superclass_column(metadata, statements)

        # Exclude records without any target superclass.
        metadata = metadata[
            metadata["diagnostic_superclass"].map(len) > 0
        ]

        # Select only the requested folds.
        self.metadata = metadata[
            metadata["strat_fold"].isin(folds)
        ].copy()

        if self.metadata.empty:
            raise ValueError(f"Không có bản ghi cho folds={folds}")

        self.label_to_index = {
            name: index for index, name in enumerate(SUPERCLASSES)
        }

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]

        signal, fields = load_record(
            self.data_dir, row[self.filename_column]
        )
        validate_signal(signal, fields, self.sampling_rate)

        signal = preprocess_ecg(
            signal,
            original_rate=self.sampling_rate,
            target_rate=100,
        )

        # Convert (time, leads) to (leads, time).
        signal = np.ascontiguousarray(signal.T, dtype=np.float32)

        target = np.zeros(len(SUPERCLASSES), dtype=np.float32)
        for label in row["diagnostic_superclass"]:
            target[self.label_to_index[label]] = 1.0

        return torch.from_numpy(signal), torch.from_numpy(target)
