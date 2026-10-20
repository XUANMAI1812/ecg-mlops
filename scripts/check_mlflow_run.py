"""Kiểm tra một run MLflow có đủ params, metrics, artifact và model hay không.

Mặc định kiểm tra run mới nhất của experiment; dùng --run-id để chọn run cụ thể.
Mã thoát 0 khi mọi mục đạt, 1 khi có mục lỗi, 2 khi không kết nối được server
hoặc không tìm thấy experiment/run

Ví dụ (từ gốc repo, venv đã kích hoạt, MLflow server đang chạy):
    python scripts/check_mlflow_run.py
    python scripts/check_mlflow_run.py --experiment ecg-superdiagnostic --min-epochs 5
    python scripts/check_mlflow_run.py --run-id <run_id>

Kiểm tra dự đoán của model cần cài torch (model được lưu bằng PyTorch)
"""

from __future__ import annotations

import argparse
import http.client
import os
import sys
import urllib.request

import mlflow
import numpy as np

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT = "ecg-superdiagnostic"
REQUIRED_PARAMS = ("seed", "learning_rate", "batch_size", "epochs")
REQUIRED_METRICS = ("train_loss", "val_loss", "val_macro_f1")
MODEL_INPUT_SHAPE = (12, 1000)
NUM_CLASSES = 5
BATCH_SIZES = (1, 2, 5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="địa chỉ MLflow (mặc định: biến MLFLOW_TRACKING_URI, rồi %(default)s)",
    )
    parser.add_argument(
        "--experiment",
        default=os.environ.get("EXPERIMENT", DEFAULT_EXPERIMENT),
        help="tên experiment (mặc định: biến EXPERIMENT, rồi %(default)s)",
    )
    parser.add_argument(
        "--run-id", default=None, help="run cần kiểm tra (mặc định: mới nhất)"
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=2,
        help="số điểm tối thiểu của mỗi metric theo epoch (mặc định: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.min_epochs < 1:
        parser.error("--min-epochs phải >= 1")
    return args


def server_is_reachable(uri: str) -> bool:
    """Server http(s) phải trả lời /health; URI kiểu sqlite hoặc file thì bỏ qua bước này."""
    if not uri.startswith(("http://", "https://")):
        return True
    try:
        urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=3)
    except (OSError, http.client.HTTPException, ValueError):
        return False
    return True


def find_run(client: mlflow.MlflowClient, experiment_name: str, run_id: str | None):
    if run_id:
        return client.get_run(run_id)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise LookupError(f"không có experiment {experiment_name!r}")
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise LookupError(f"experiment {experiment_name!r} chưa có run nào")
    return runs[0]


def run_checks(client: mlflow.MlflowClient, run, min_epochs: int) -> dict[str, bool]:
    run_id = run.info.run_id
    params = set(run.data.params)
    metrics = set(run.data.metrics)
    history = {m: client.get_metric_history(run_id, m) for m in REQUIRED_METRICS}
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    models = run.outputs.model_outputs if run.outputs else []

    checks = {
        "run kết thúc FINISHED": run.info.status == "FINISHED",
        f"params có {list(REQUIRED_PARAMS)}": set(REQUIRED_PARAMS) <= params,
        f"metrics có {list(REQUIRED_METRICS)}": set(REQUIRED_METRICS) <= metrics,
        f"metrics có lịch sử theo epoch (>= {min_epochs} điểm)": all(
            len(points) >= min_epochs for points in history.values()
        ),
        "có ít nhất 1 artifact": len(artifacts) > 0,
        "có đúng 1 logged model": len(models) == 1,
    }
    if len(models) == 1:
        model = mlflow.pyfunc.load_model(f"models:/{models[0].model_id}")
        for batch in BATCH_SIZES:
            x = np.zeros((batch, *MODEL_INPUT_SHAPE), dtype=np.float32)
            shape = model.predict(x).shape
            checks[f"model predict batch={batch} -> {shape}"] = shape == (
                batch,
                NUM_CLASSES,
            )
    return checks


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not server_is_reachable(args.tracking_uri):
        print(f"LỖI không kết nối được MLflow tại {args.tracking_uri}", file=sys.stderr)
        return 2

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    try:
        run = find_run(client, args.experiment, args.run_id)
    except (LookupError, mlflow.exceptions.MlflowException) as exc:
        print(f"LỖI {exc}", file=sys.stderr)
        return 2
    print("run:", run.info.run_name, run.info.run_id, run.info.status)

    checks = run_checks(client, run, args.min_epochs)
    for name, ok in checks.items():
        print("OK  " if ok else "LỖI", name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
