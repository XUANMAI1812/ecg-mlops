"""Đánh giá model đã log trong một run MLflow trên một fold của PTB-XL.

Mặc định đánh giá fold 9 (validation). Fold 10 (test) chỉ dùng cho đánh giá cuối,
nên phải truyền --allow-test. Kết quả (AUC, F1, confusion matrix theo từng lớp ở
ngưỡng 0,5) ghi thành artifact evaluation/<split>.json trong chính run đó; không
ghi thêm metric nào vào run.

Ví dụ (từ gốc repo, venv đã kích hoạt, MLflow server đang chạy):
    PYTHONPATH=src python -m ecg.train.evaluate --run-id <run_id>
    PYTHONPATH=src python -m ecg.train.evaluate --run-id <run_id> --sample-n 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import DataLoader

from ecg.data.dataset import ECGDataset
from ecg.data.labels import SUPERCLASSES
from ecg.models.train import DEFAULT_TRACKING_URI, calculate_metrics

SPLIT_FOLDS = {"val": (9,), "test": (10,)}
THRESHOLD = 0.5  # giống calculate_metrics; D7 (ngưỡng theo lớp) chưa chốt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_FOLDS), default="val")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Bắt buộc khi --split test (fold 10 chỉ dùng cho đánh giá cuối).",
    )
    parser.add_argument("--data-dir", default="data/raw/ptb-xl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Chỉ đánh giá N bản ghi, chọn theo --seed (mặc định: toàn bộ fold).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    args = parser.parse_args(argv)

    if args.split == "test" and not args.allow_test:
        parser.error("--split test cần thêm --allow-test")
    if args.batch_size < 1:
        parser.error("--batch-size phải >= 1")
    if args.sample_n is not None and args.sample_n < 1:
        parser.error("--sample-n phải >= 1")
    return args


def _clean(value: float) -> float | None:
    """NaN không hợp lệ trong JSON chuẩn; ghi thành null."""
    return None if math.isnan(value) else value


def evaluate_predictions(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    """Tính F1, AUC và confusion matrix theo từng lớp ở ngưỡng THRESHOLD."""
    if targets.shape != probabilities.shape or targets.shape[1:] != (
        len(SUPERCLASSES),
    ):
        raise ValueError(
            f"targets và probabilities phải cùng shape (n, {len(SUPERCLASSES)}), "
            f"nhận {targets.shape} và {probabilities.shape}"
        )

    metrics = calculate_metrics(targets, probabilities)
    predictions = (probabilities >= THRESHOLD).astype(int)
    matrices = multilabel_confusion_matrix(
        targets.astype(int), predictions, labels=list(range(len(SUPERCLASSES)))
    )
    confusion = {
        name: {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        }
        for name, matrix in zip(SUPERCLASSES, matrices)
    }

    return {
        "n_records": int(targets.shape[0]),
        "threshold": THRESHOLD,
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "auc_macro": _clean(metrics["macro_auc"]),
        "auc_micro": _clean(metrics["micro_auc"]),
        "per_class_auc": metrics["per_class_auc"],
        "confusion_matrix": confusion,
    }


def predict_probabilities(model, dataset: ECGDataset, batch_size: int):
    """Chạy model (đầu ra là logits) trên dataset, trả về targets và xác suất."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_targets = []
    all_probabilities = []
    for signals, targets in loader:
        # torch.sigmoid trên float32, giống run_epoch trong train.py, để kết quả
        # khớp đúng metric val đã log khi train.
        logits = torch.as_tensor(np.asarray(model.predict(signals.numpy())))
        all_targets.append(targets.numpy())
        all_probabilities.append(torch.sigmoid(logits).numpy())
    return np.concatenate(all_targets), np.concatenate(all_probabilities)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    run = client.get_run(args.run_id)
    models = run.outputs.model_outputs if run.outputs else []
    if len(models) != 1:
        raise SystemExit(
            f"Run {args.run_id} cần đúng 1 logged model, hiện có {len(models)}"
        )
    model = mlflow.pyfunc.load_model(f"models:/{models[0].model_id}")

    folds = SPLIT_FOLDS[args.split]
    dataset = ECGDataset(
        args.data_dir, folds=folds, sample_n=args.sample_n, seed=args.seed
    )
    targets, probabilities = predict_probabilities(model, dataset, args.batch_size)

    result = {
        "run_id": args.run_id,
        "split": args.split,
        "folds": list(folds),
        "sample_n": args.sample_n,
        "seed": args.seed,
        "superclasses": list(SUPERCLASSES),
        **evaluate_predictions(targets, probabilities),
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{args.split}.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        client.log_artifact(args.run_id, str(path), "evaluation")

    print(f"Run: {args.run_id} | split={args.split} | folds={list(folds)}")
    print(f"Records: {result['n_records']}")
    print(f"auc_macro={result['auc_macro']} | auc_micro={result['auc_micro']}")
    print(f"macro_f1={result['macro_f1']:.4f}")
    for name in SUPERCLASSES:
        cm = result["confusion_matrix"][name]
        print(
            f"  {name}: F1={result['per_class_f1'][name]:.4f} | "
            f"tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']}"
        )
    print(f"Artifact: evaluation/{args.split}.json")


if __name__ == "__main__":
    main()
