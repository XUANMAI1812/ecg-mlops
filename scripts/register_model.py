"""Đăng ký model của một run MLflow vào Model Registry và gắn alias.

Các bước:
1. Đọc file ngưỡng D7 và kiểm tra trường run_id trùng --run-id.
2. Ghi ngưỡng vào chính run đó thành artifact serving/thresholds.json, để
   service nạp version nào thì đọc đúng bộ ngưỡng của version đó.
3. Đăng ký logged model duy nhất của run thành một version của --name. Nếu run
   này đã được đăng ký trước đó thì dùng lại version cũ, không tạo version mới.
4. Gắn tag thresholds_artifact cho version và chuyển alias --alias sang version đó.

Ví dụ (từ gốc repo, venv đã kích hoạt, MLflow server đang chạy):
    PYTHONPATH=src python scripts/register_model.py \
        --run-id <run_id> --thresholds <thư mục>/thresholds.json
    PYTHONPATH=src python scripts/register_model.py \
        --run-id <run_id> --thresholds thresholds.json --alias production
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow
from mlflow.entities.model_registry import ModelVersion

from ecg.models.train import DEFAULT_TRACKING_URI
from ecg.train.evaluate import _load_thresholds
from ecg.train.tracking import check_tracking_server

DEFAULT_MODEL_NAME = "ecg_classifier"
ALIASES = ("staging", "production")
THRESHOLDS_ARTIFACT_DIR = "serving"
THRESHOLDS_ARTIFACT = f"{THRESHOLDS_ARTIFACT_DIR}/thresholds.json"
THRESHOLDS_TAG = "thresholds_artifact"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--thresholds",
        required=True,
        help="File JSON ngưỡng D7 của run này (có trường run_id và thresholds).",
    )
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--alias", choices=ALIASES, default="staging")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    return parser.parse_args(argv)


def find_logged_model_id(client: mlflow.MlflowClient, run_id: str) -> str:
    """Trả về model_id của logged model duy nhất trong run."""
    run = client.get_run(run_id)
    models = run.outputs.model_outputs if run.outputs else []
    if len(models) != 1:
        raise SystemExit(f"Run {run_id} cần đúng 1 logged model, hiện có {len(models)}")
    return models[0].model_id


def log_thresholds(
    client: mlflow.MlflowClient,
    run_id: str,
    model_id: str,
    thresholds: dict[str, float],
) -> None:
    """Ghi ngưỡng vào run thành artifact THRESHOLDS_ARTIFACT (ghi đè nếu đã có)."""
    content = {"run_id": run_id, "model_id": model_id, "thresholds": thresholds}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(THRESHOLDS_ARTIFACT).name
        path.write_text(
            json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        client.log_artifact(run_id, str(path), THRESHOLDS_ARTIFACT_DIR)


def find_existing_version(
    client: mlflow.MlflowClient, name: str, run_id: str
) -> ModelVersion | None:
    """Version đã đăng ký từ run này (nếu có), để chạy lại script không tạo trùng."""
    versions = client.search_model_versions(f"name = '{name}' and run_id = '{run_id}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def register(
    client: mlflow.MlflowClient,
    run_id: str,
    thresholds_path: str | Path,
    name: str = DEFAULT_MODEL_NAME,
    alias: str = "staging",
) -> ModelVersion:
    """Chạy 4 bước ở đầu file; trả về version đang mang alias."""
    if alias not in ALIASES:
        raise ValueError(f"alias phải là một trong {ALIASES}, nhận {alias!r}")

    thresholds = _load_thresholds(thresholds_path, run_id)
    model_id = find_logged_model_id(client, run_id)
    log_thresholds(client, run_id, model_id, thresholds)

    version = find_existing_version(client, name, run_id)
    if version is None:
        version = mlflow.register_model(f"models:/{model_id}", name)
        print(f"Đã tạo version {version.version} của {name}")
    else:
        print(f"Run {run_id} đã có version {version.version}, dùng lại")

    client.set_model_version_tag(
        name, version.version, THRESHOLDS_TAG, THRESHOLDS_ARTIFACT
    )
    client.set_registered_model_alias(name, alias, version.version)
    return client.get_model_version_by_alias(name, alias)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    check_tracking_server(args.tracking_uri)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()

    version = register(client, args.run_id, args.thresholds, args.name, args.alias)

    print(f"Model: models:/{args.name}@{args.alias} -> version {version.version}")
    print(f"Run: {version.run_id}")
    print(f"Source: {version.source}")
    print(f"Ngưỡng: {version.tags.get(THRESHOLDS_TAG)} (artifact trong run)")


if __name__ == "__main__":
    main()
