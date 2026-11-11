"""Test cho tinh chỉnh tham số: các hook mới của train() (evaluate_test, log_model, epoch_callback, datasets) và phần không cần MLflow của tune.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from ecg.data.labels import SUPERCLASSES
from ecg.models.train import train
from ecg.train.tune import InMemoryDataset, ensure_study_config, parse_args


class FakeDataset(Dataset):
    """Dữ liệu giả cùng giao diện ECGDataset: (tín hiệu 12x1000, nhãn 5 lớp) + metadata."""

    def __init__(self, size: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.signals = torch.randn(size, 12, 1000, generator=generator)
        # Mỗi lớp có cả nhãn 0 và 1 để AUC xác định được.
        self.targets = torch.tensor(
            [
                [float((i + k) % 2) for k in range(len(SUPERCLASSES))]
                for i in range(size)
            ]
        )
        self.metadata = pd.DataFrame(
            {
                "diagnostic_superclass": [
                    [name for name, flag in zip(SUPERCLASSES, row) if flag]
                    for row in self.targets.tolist()
                ]
            }
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signals[index], self.targets[index]


class FakeTracker:
    """Ghi lại các lời gọi thay cho RunTracker, không cần MLflow."""

    run_id = "0" * 32

    def __init__(self) -> None:
        self.tags: dict = {}
        self.params: dict = {}
        self.epochs: list[int] = []
        self.summary: dict = {}
        self.artifacts: list[str] = []
        self.models = 0

    def set_tags(self, tags) -> None:
        self.tags.update(tags)

    def log_params(self, params) -> None:
        self.params.update(params)

    def log_epoch(self, epoch, metrics) -> None:
        self.epochs.append(epoch)

    def log_summary(self, metrics) -> None:
        self.summary.update(metrics)

    def log_artifact(self, path, artifact_dir=None) -> None:
        self.artifacts.append(Path(path).name)

    def log_model(self, model, input_shape) -> None:
        self.models += 1


def make_args(tmp_path: Path, epochs: int = 2) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=str(tmp_path / "khong-co-du-lieu"),
        output_dir=str(tmp_path / "out"),
        model="cnn1d",
        epochs=epochs,
        batch_size=8,
        learning_rate=1e-3,
        seed=42,
        sample_n=None,
    )


@pytest.fixture
def datasets() -> tuple[FakeDataset, FakeDataset]:
    return FakeDataset(32, seed=1), FakeDataset(16, seed=2)


def run_tuning_train(args, datasets, **kwargs) -> tuple[dict, FakeTracker]:
    tracker = FakeTracker()
    summary = train(
        args,
        tracker,
        evaluate_test=False,
        log_model=False,
        datasets=datasets,
        **kwargs,
    )
    return summary, tracker


def test_train_without_test_fold_returns_val_summary(tmp_path, datasets):
    summary, tracker = run_tuning_train(make_args(tmp_path), datasets)

    assert set(summary) == {
        "best_epoch",
        "best_val_loss",
        "best_val_auc_macro",
        "best_val_auc_micro",
    }
    assert not any(key.startswith("test_") for key in tracker.summary)
    assert tracker.params["n_test"] == "skipped"
    assert tracker.models == 0
    assert "test_metrics.json" not in tracker.artifacts
    assert not (tmp_path / "out" / "test_metrics.json").exists()


def test_epoch_callback_called_after_each_epoch(tmp_path, datasets):
    calls = []
    run_tuning_train(
        make_args(tmp_path, epochs=3),
        datasets,
        epoch_callback=lambda epoch, record: calls.append(
            (epoch, record["val_auc_macro"])
        ),
    )

    assert [epoch for epoch, _ in calls] == [1, 2, 3]
    assert all(0.0 <= value <= 1.0 for _, value in calls)


def test_exception_from_callback_stops_training(tmp_path, datasets):
    class Stop(Exception):
        pass

    def stop_at_two(epoch, record):
        if epoch == 2:
            raise Stop

    tracker = FakeTracker()
    with pytest.raises(Stop):
        train(
            make_args(tmp_path, epochs=4),
            tracker,
            evaluate_test=False,
            log_model=False,
            epoch_callback=stop_at_two,
            datasets=datasets,
        )
    # Metrics epoch 2 đã log trước khi callback dừng; không có epoch 3, 4.
    assert tracker.epochs == [1, 2]


def test_train_is_reproducible_in_same_process(tmp_path, datasets):
    first, _ = run_tuning_train(make_args(tmp_path), datasets)
    second, _ = run_tuning_train(make_args(tmp_path), datasets)

    assert first == pytest.approx(second, abs=1e-7)


def test_in_memory_dataset_matches_source(datasets):
    source = datasets[0]
    cached = InMemoryDataset(source)

    assert len(cached) == len(source)
    assert cached.metadata is source.metadata
    for index in range(len(source)):
        signal, target = cached[index]
        assert torch.equal(signal, source[index][0])
        assert torch.equal(target, source[index][1])


def test_train_same_result_with_in_memory_cache(tmp_path, datasets):
    plain, _ = run_tuning_train(make_args(tmp_path), datasets)
    cached, _ = run_tuning_train(
        make_args(tmp_path), tuple(InMemoryDataset(ds) for ds in datasets)
    )

    assert plain == pytest.approx(cached, abs=1e-7)


def test_study_config_saved_then_must_match():
    study = optuna.create_study(direction="maximize")
    config = {"model": "cnn1d", "epochs": 5, "batch_sizes": [16, 32, 64]}

    ensure_study_config(study, config)
    assert study.user_attrs == config
    ensure_study_config(study, dict(config))  # cùng cấu hình: không lỗi

    with pytest.raises(SystemExit, match="epochs"):
        ensure_study_config(study, {**config, "epochs": 3})


def test_default_study_name():
    full = parse_args(["run", "--model", "cnn1d", "--n-trials", "2"])
    sample = parse_args(
        ["run", "--model", "resnet1d", "--n-trials", "2", "--sample-n", "200"]
    )

    assert full.study_name == "cnn1d-aucmacro-v1"
    assert sample.study_name == "resnet1d-aucmacro-v1-s200"
    assert full.experiment == "ecg-tuning"


@pytest.mark.parametrize(
    "extra",
    [["--lr-min", "0.01", "--lr-max", "0.001"], ["--n-trials", "0"]],
)
def test_invalid_arguments_rejected(extra):
    with pytest.raises(SystemExit):
        parse_args(["run", "--model", "cnn1d", "--n-trials", "2", *extra])
