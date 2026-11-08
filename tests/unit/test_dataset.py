"""Tests for ECGDataset using synthetic PTB-XL metadata."""

import numpy as np
import pandas as pd
import pytest

from ecg.data.dataset import ECGDataset


@pytest.fixture
def fake_dataset_files(tmp_path, monkeypatch):
    """Create synthetic metadata with one labeled and one unlabeled record."""
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2],
            "scp_codes": [
                {"NORM": 100.0},
                {"SR": 0.0},
            ],
            "strat_fold": [1, 1],
            "filename_lr": ["fake/record1", "fake/record2"],
            "filename_hr": ["fake/record1_hr", "fake/record2_hr"],
        }
    ).set_index("ecg_id")

    statements = pd.DataFrame(
        {
            "diagnostic": [1.0],
            "diagnostic_class": ["NORM"],
        },
        index=["NORM"],
    )

    monkeypatch.setattr(
        "ecg.data.dataset.load_metadata",
        lambda data_dir: metadata.copy(),
    )
    monkeypatch.setattr(
        "ecg.data.dataset.load_scp_statements",
        lambda data_dir: statements.copy(),
    )

    def fake_load_record(data_dir, filename):
        signal = np.zeros((1000, 12), dtype=np.float32)
        fields = {
            "fs": 100,
            "sig_name": [
                "I",
                "II",
                "III",
                "AVR",
                "AVL",
                "AVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ],
        }
        return signal, fields

    monkeypatch.setattr(
        "ecg.data.dataset.load_record",
        fake_load_record,
    )
    monkeypatch.setattr(
        "ecg.data.dataset.preprocess_ecg",
        lambda signal, **kwargs: signal,
    )
    monkeypatch.setattr(
        "ecg.data.dataset.validate_signal",
        lambda signal, fields, sampling_rate: None,
    )

    return tmp_path


def test_dataset_keeps_record_without_superclass(fake_dataset_files):
    dataset = ECGDataset(fake_dataset_files, folds=(1,))

    assert len(dataset) == 2
    assert dataset.metadata.loc[2, "diagnostic_superclass"] == []


def test_unlabeled_record_returns_all_zero_target(fake_dataset_files):
    dataset = ECGDataset(fake_dataset_files, folds=(1,))

    signal, target = dataset[1]

    assert signal.shape == (12, 1000)
    assert target.shape == (5,)
    assert target.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def many_records(tmp_path, monkeypatch, fake_dataset_files):
    """Ghi đè metadata của fixture trên bằng 20 bản ghi NORM, chia đều fold 1 và 9."""
    metadata = pd.DataFrame(
        {
            "ecg_id": list(range(1, 21)),
            "scp_codes": [{"NORM": 100.0}] * 20,
            "strat_fold": [1] * 10 + [9] * 10,
            "filename_lr": [f"fake/record{i}" for i in range(1, 21)],
            "filename_hr": [f"fake/record{i}_hr" for i in range(1, 21)],
        }
    ).set_index("ecg_id")
    monkeypatch.setattr(
        "ecg.data.dataset.load_metadata",
        lambda data_dir: metadata.copy(),
    )
    return fake_dataset_files


def test_sample_n_keeps_requested_folds_and_is_reproducible(many_records):
    first = ECGDataset(many_records, folds=(9,), sample_n=4, seed=7)
    second = ECGDataset(many_records, folds=(9,), sample_n=4, seed=7)

    assert len(first) == 4
    assert set(first.metadata["strat_fold"]) == {9}
    assert list(first.metadata.index) == list(second.metadata.index)
    assert first.metadata.index.is_monotonic_increasing


def test_sample_n_larger_than_fold_keeps_all_records(many_records):
    dataset = ECGDataset(many_records, folds=(1,), sample_n=100)

    assert len(dataset) == 10


def test_sample_n_must_be_positive(many_records):
    with pytest.raises(ValueError, match="sample_n"):
        ECGDataset(many_records, folds=(1,), sample_n=0)
