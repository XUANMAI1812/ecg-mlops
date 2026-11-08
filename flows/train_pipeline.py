"""Prefect flow: kiểm tra dữ liệu -> huấn luyện -> đánh giá -> kiểm tra run MLflow.

Tiền xử lý tín hiệu (lọc, resample, chuẩn hoá) chạy trong ECGDataset khi đọc từng
bản ghi, nên bước đầu của flow chỉ kiểm tra dữ liệu PTB-XL có đủ trước khi train.
Mỗi bước gọi module của dự án bằng tiến trình con (PYTHONPATH=src), nên flow không
import code huấn luyện và không cần sửa train.py.

Deploy và chạy qua worker của pool ecg-local: xem docs/setup.md mục 13.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RECORDS = 21799  # PTB-XL v1.0.3 (D9)
RUN_ID_PATTERN = re.compile(r"^MLflow run: ([0-9a-f]{32})$", re.MULTILINE)


def _run_module(args: list[str]) -> str:
    """Chạy một lệnh Python của dự án từ gốc repo, ghi log từng dòng, trả về output."""
    logger = get_run_logger()
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYTHONUNBUFFERED": "1",
        "MLFLOW_DISABLE_TELEMETRY": "true",
    }
    command = [sys.executable, *args]
    logger.info("Chạy: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    for line in process.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        logger.info(line)
    if process.wait() != 0:
        raise RuntimeError(f"Lệnh lỗi (mã {process.returncode}): {' '.join(args)}")
    return "\n".join(lines)


def parse_run_id(output: str) -> str:
    """Lấy run_id từ dòng 'MLflow run: <id>' mà train.py in ra cuối cùng."""
    matches = RUN_ID_PATTERN.findall(output)
    if not matches:
        raise ValueError(
            "Không thấy dòng 'MLflow run: <run_id>' trong output của train"
        )
    return matches[-1]


@task
def check_data(data_dir: str) -> int:
    path = REPO_ROOT / data_dir
    database = path / "ptbxl_database.csv"
    if not database.is_file():
        raise FileNotFoundError(
            f"Không thấy {database}; chạy dvc pull (docs/setup.md mục 7)"
        )
    if not (path / "records100").is_dir():
        raise FileNotFoundError(f"Không thấy {path / 'records100'}")
    n_records = len(pd.read_csv(database, usecols=["ecg_id"]))
    if n_records != EXPECTED_RECORDS:
        raise ValueError(
            f"{database} có {n_records} bản ghi, kỳ vọng {EXPECTED_RECORDS}"
        )
    get_run_logger().info("Dữ liệu đủ: %d bản ghi", n_records)
    return n_records


@task
def train(
    model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    sample_n: int | None,
    data_dir: str,
    run_name: str,
) -> str:
    args = [
        "-m",
        "ecg.models.train",
        "--model",
        model,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--seed",
        str(seed),
        "--data-dir",
        data_dir,
        "--output-dir",
        f"artifacts/{run_name}",
        "--run-name",
        run_name,
    ]
    if sample_n is not None:
        args += ["--sample-n", str(sample_n)]
    return parse_run_id(_run_module(args))


@task
def evaluate(
    run_id: str, batch_size: int, seed: int, sample_n: int | None, data_dir: str
) -> None:
    # Cùng batch size, seed, sample_n với train để đánh giá đúng các bản ghi val
    # và ra đúng metric val đã log khi train.
    args = [
        "-m",
        "ecg.train.evaluate",
        "--run-id",
        run_id,
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--data-dir",
        data_dir,
    ]
    if sample_n is not None:
        args += ["--sample-n", str(sample_n)]
    _run_module(args)


@task
def check_run(run_id: str, epochs: int) -> None:
    _run_module(
        [
            "scripts/check_mlflow_run.py",
            "--run-id",
            run_id,
            "--min-epochs",
            str(epochs),
        ]
    )


@flow(name="ecg-train-pipeline", log_prints=True)
def train_pipeline(
    model: str = "cnn1d",
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    seed: int = 42,
    sample_n: int | None = 200,
    data_dir: str = "data/raw/ptb-xl",
) -> str:
    """sample_n=None: train và đánh giá trên toàn bộ fold."""
    # Tên run theo quy ước <model>-<số epoch>ep; chạy tập mẫu thêm hậu tố -s<N>.
    run_name = f"{model}-{epochs}ep"
    if sample_n is not None:
        run_name += f"-s{sample_n}"

    check_data(data_dir)
    run_id = train(
        model, epochs, batch_size, learning_rate, seed, sample_n, data_dir, run_name
    )
    evaluate(run_id, batch_size, seed, sample_n, data_dir)
    check_run(run_id, epochs)
    print(f"Xong: run {run_name} ({run_id})")
    return run_id


if __name__ == "__main__":
    train_pipeline()
