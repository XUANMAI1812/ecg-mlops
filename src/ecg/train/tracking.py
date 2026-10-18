"""Ghi log MLflow cho script huấn luyện: params, metrics theo epoch, artifact, model.

Cách dùng trong vòng lặp train:

    with RunTracker(uri, experiment, run_name) as tracker:
        tracker.log_params({"lr": 0.001, "batch_size": 64})
        for epoch in range(1, epochs + 1):
            tracker.log_epoch(epoch, {"train_loss": 0.5, "val_loss": 0.6})
        tracker.log_artifact("params.yaml", "config")
        tracker.log_model(model, input_shape=(12, 1000))
"""

from __future__ import annotations

import urllib.request
import warnings
from collections.abc import Mapping
from typing import Any, Self

import mlflow
import mlflow.pytorch
import numpy as np
from torch import nn


def check_tracking_server(uri: str) -> None:
    """Báo lỗi rõ ràng nếu MLflow server (http) chưa chạy, thay vì treo khi retry."""
    if not uri.startswith(("http://", "https://")):
        return
    try:
        urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=3)
    except Exception as exc:
        raise SystemExit(
            f"Không kết nối được MLflow tại {uri} ({exc}). "
            "Hãy chạy MLflow server trước (xem docs/setup.md)."
        ) from exc


class RunTracker:
    """Bọc một run MLflow. Dùng như context manager; run kết thúc FAILED nếu có lỗi."""

    def __init__(
        self, tracking_uri: str, experiment: str, run_name: str | None = None
    ) -> None:
        check_tracking_server(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._run_name = run_name
        self._run: Any = None
        self.model_uri: str | None = None

    def __enter__(self) -> Self:
        self._run = mlflow.start_run(run_name=self._run_name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

    @property
    def run_id(self) -> str:
        return self._run.info.run_id

    def set_tags(self, tags: Mapping[str, str]) -> None:
        mlflow.set_tags(dict(tags))

    def log_params(self, params: Mapping[str, Any]) -> None:
        mlflow.log_params(dict(params))

    def log_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """Ghi metrics của một epoch. Giá trị NaN bị bỏ qua (MLflow không nhận NaN)."""
        clean = {k: float(v) for k, v in metrics.items() if not np.isnan(v)}
        mlflow.log_metrics(clean, step=epoch)

    def log_summary(self, metrics: Mapping[str, float]) -> None:
        """Ghi metrics tổng kết một lần (ví dụ kết quả ở epoch tốt nhất)."""
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if not np.isnan(v)})

    def log_artifact(self, path: str, artifact_dir: str | None = None) -> None:
        mlflow.log_artifact(path, artifact_dir)

    def log_model(self, model: nn.Module, input_shape: tuple[int, ...]) -> str:
        """Log model PyTorch, trả về model_uri (dạng models:/m-<id>).

        input_shape là shape KHÔNG gồm batch, ví dụ (12, 1000). Ví dụ đầu vào dùng
        batch = 2: nếu dùng batch = 1, graph xuất ra bị cố định batch = 1 và model
        đã log sẽ lỗi khi predict với batch khác.
        Mặc định dùng định dạng pt2 (torch.export). Nếu model không export được thì
        chuyển sang pickle và cảnh báo.
        """
        model = model.to("cpu").eval()
        example = np.zeros((2, *input_shape), dtype=np.float32)
        try:
            info = mlflow.pytorch.log_model(
                model, name="model", input_example=example, serialization_format="pt2"
            )
        except Exception as exc:  # noqa: BLE001 - torch.export có nhiều loại lỗi khác nhau
            warnings.warn(
                f"Không export được pt2 ({exc!r}); dùng pickle.", stacklevel=2
            )
            info = mlflow.pytorch.log_model(
                model,
                name="model",
                input_example=example,
                serialization_format="pickle",
            )
        self.model_uri = info.model_uri
        return info.model_uri
