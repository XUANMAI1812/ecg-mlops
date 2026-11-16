"""Chạy thử study Optuna đầu-cuối với MLflow lưu SQLite trong thư mục tạm.

Không cần MLflow server và dữ liệu PTB-XL: dùng dữ liệu giả, 3 trial, 2 epoch.
Kiểm tra run cha và run con, không có metric test_* và không có model trong trial,
và lệnh summarize ghi được tóm tắt từ file study.
"""

from __future__ import annotations

import mlflow
import optuna
import pandas as pd
import pytest
import torch
from mlflow import MlflowClient
from torch.utils.data import Dataset

from ecg.data.labels import SUPERCLASSES
from ecg.train.tune import parse_args, run_study, summarize


class FakeDataset(Dataset):
    def __init__(self, size: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.signals = torch.randn(size, 12, 1000, generator=generator)
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


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    # Artifact của MLflow (backend SQLite) và file study ghi vào thư mục tạm.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    previous_uri = mlflow.get_tracking_uri()
    yield tmp_path
    while mlflow.active_run():
        mlflow.end_run()
    # Trả lại địa chỉ MLflow cũ để không ảnh hưởng test chạy sau.
    mlflow.set_tracking_uri(previous_uri)


def test_study_logs_parent_and_child_runs(workdir):
    uri = f"sqlite:///{workdir / 'mlflow.db'}"
    args = parse_args(
        [
            "run",
            "--model",
            "cnn1d",
            "--n-trials",
            "3",
            "--epochs",
            "2",
            "--n-startup-trials",
            "1",
            "--tracking-uri",
            uri,
        ]
    )
    datasets = (FakeDataset(32, seed=1), FakeDataset(16, seed=2))

    study = run_study(args, datasets=datasets)

    assert len(study.trials) == 3
    allowed = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    assert {trial.state for trial in study.trials} <= allowed
    assert (workdir / "optuna_data" / "cnn1d-aucmacro-v1.db").is_file()

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["ecg-tuning"], output_format="list")
    parents = [run for run in runs if run.data.tags.get("role") == "study"]
    children = [run for run in runs if "trial_number" in run.data.tags]
    assert len(parents) == 1 and len(children) == 3

    parent_id = parents[0].info.run_id
    for run in children:
        assert run.data.tags["mlflow.parentRunId"] == parent_id
        assert run.data.tags["trial_state"] in {"complete", "pruned"}
        assert run.info.status == "FINISHED"
        assert run.data.params["n_test"] == "skipped"
        assert not run.outputs.model_outputs
    for run in runs:
        assert not any(key.startswith("test_") for key in run.data.metrics)

    client = MlflowClient()
    names = [item.path for item in client.list_artifacts(parent_id, "study")]
    assert names == ["study/trials.csv"]

    summary_args = parse_args(
        [
            "summarize",
            "--storage-file",
            "optuna_data/cnn1d-aucmacro-v1.db",
            "--source",
            "kiem-thu",
            "--experiment",
            "ecg-tuning-import",
            "--tracking-uri",
            uri,
        ]
    )
    summarize(summary_args)

    imported = mlflow.search_runs(
        experiment_names=["ecg-tuning-import"], output_format="list"
    )
    assert len(imported) == 1
    run = imported[0]
    assert run.data.tags["role"] == "summary"
    assert run.data.tags["source"] == "kiem-thu"
    assert run.data.params["model"] == "cnn1d"
    names = sorted(
        item.path for item in client.list_artifacts(run.info.run_id, "study")
    )
    assert names == ["study/cnn1d-aucmacro-v1.db", "study/trials.csv"]


def test_diverged_trial_is_recorded_and_study_continues(workdir):
    uri = f"sqlite:///{workdir / 'mlflow.db'}"
    args = parse_args(
        [
            "run",
            "--model",
            "cnn1d",
            "--n-trials",
            "2",
            "--epochs",
            "1",
            "--study-name",
            "nan-study",
            "--tracking-uri",
            uri,
        ]
    )
    train_ds = FakeDataset(32, seed=1)
    train_ds.signals[0, 0, 0] = float("nan")  # loss thành NaN ngay epoch đầu

    study = run_study(args, datasets=(train_ds, FakeDataset(16, seed=2)))

    assert [trial.state for trial in study.trials] == [optuna.trial.TrialState.FAIL] * 2
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["ecg-tuning"], output_format="list")
    statuses = {run.data.tags.get("role", "trial"): run.info.status for run in runs}
    assert statuses == {"study": "FINISHED", "trial": "FAILED"}
