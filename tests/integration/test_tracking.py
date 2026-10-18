import mlflow
import numpy as np
import pytest
import torch
from torch import nn

from ecg.train.tracking import RunTracker, check_tracking_server


class ExportableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(12, 4, 3)
        self.fc = nn.Linear(4, 5)

    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=-1))


class DataDependentNet(ExportableNet):
    """Rẽ nhánh phụ thuộc dữ liệu: torch.export không xuất được."""

    def forward(self, x):
        if x.sum() > 1e9:
            return torch.zeros(x.shape[0], 5)
        return super().forward(x)


@pytest.fixture
def uri(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment("t", artifact_location=(tmp_path / "artifacts").as_uri())
    return tracking_uri


def test_tracker_logs_params_metrics_artifact_and_model(uri, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("x")
    with RunTracker(uri, "t", "run-a") as tracker:
        tracker.log_params({"lr": 0.1, "epochs": 2})
        tracker.log_epoch(1, {"train_loss": 1.0, "val_auc_macro": float("nan")})
        tracker.log_epoch(2, {"train_loss": 0.5, "val_auc_macro": 0.7})
        tracker.log_summary({"best_val_auc_macro": 0.7})
        tracker.log_artifact(str(note), "extra")
        model_uri = tracker.log_model(ExportableNet(), input_shape=(12, 100))
        run_id = tracker.run_id

    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params == {"lr": "0.1", "epochs": "2"}
    assert (
        run.data.metrics["train_loss"] == 0.5
        and run.data.metrics["best_val_auc_macro"] == 0.7
    )
    history = mlflow.MlflowClient().get_metric_history(run_id, "val_auc_macro")
    assert [m.step for m in history] == [2]  # NaN ở epoch 1 bị bỏ qua
    assert [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)] == ["extra"]
    assert model_uri == f"models:/{run.outputs.model_outputs[0].model_id}"

    loaded = mlflow.pyfunc.load_model(model_uri)
    for batch in (1, 5):
        assert loaded.predict(np.zeros((batch, 12, 100), dtype=np.float32)).shape == (
            batch,
            5,
        )


def test_model_that_cannot_be_exported_falls_back_to_pickle(uri):
    with (
        RunTracker(uri, "t", "run-b") as tracker,
        pytest.warns(UserWarning, match="pickle"),
    ):
        model_uri = tracker.log_model(DataDependentNet(), input_shape=(12, 100))
    loaded = mlflow.pyfunc.load_model(model_uri)
    assert loaded.predict(np.zeros((3, 12, 100), dtype=np.float32)).shape == (3, 5)


def test_run_is_marked_failed_on_error(uri):
    with pytest.raises(RuntimeError), RunTracker(uri, "t", "run-c") as tracker:
        run_id = tracker.run_id
        raise RuntimeError("lỗi giả")
    assert mlflow.get_run(run_id).info.status == "FAILED"


def test_unreachable_server_gives_clear_message():
    with pytest.raises(SystemExit, match="Không kết nối được MLflow"):
        check_tracking_server("http://127.0.0.1:5999")
