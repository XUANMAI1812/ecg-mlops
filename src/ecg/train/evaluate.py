"""Đánh giá model đã log trong một run MLflow trên một fold của PTB-XL.

Mặc định đánh giá fold 9 (validation). Fold 10 (test) chỉ dùng cho đánh giá cuối,
nên phải truyền --allow-test. Kết quả (AUC, F1, confusion matrix theo từng lớp)
ghi thành artifact evaluation/<split>.json trong chính run đó; không ghi thêm
metric nào vào run.

Ngưỡng mặc định 0,5 cho mọi lớp. Truyền --thresholds <file.json> để dùng ngưỡng
D7 riêng cho từng lớp (chọn trên fold 9); trường run_id trong file phải trùng
--run-id.

Ví dụ (từ gốc repo, venv đã kích hoạt, MLflow server đang chạy):
    PYTHONPATH=src python -m ecg.train.evaluate --run-id <run_id>
    PYTHONPATH=src python -m ecg.train.evaluate --run-id <run_id> --sample-n 200
    PYTHONPATH=src python -m ecg.train.evaluate --run-id <run_id> \
        --split test --allow-test --thresholds thresholds.json
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
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from torch.utils.data import DataLoader

from ecg.data.dataset import ECGDataset
from ecg.data.labels import SUPERCLASSES
from ecg.models.train import DEFAULT_TRACKING_URI, calculate_metrics

SPLIT_FOLDS = {"val": (9,), "test": (10,)}
THRESHOLD = 0.5  # ngưỡng mặc định khi không truyền --thresholds


def select_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold_grid: np.ndarray | None = None,
) -> dict[str, float]:
    """Chọn ngưỡng tối đa F1 riêng cho từng lớp.

    Nếu nhiều ngưỡng có F1 bằng nhau, chọn ngưỡng gần 0.5 nhất.
    Nếu khoảng cách bằng nhau, chọn ngưỡng thấp hơn.
    """
    if targets.shape != probabilities.shape or targets.shape[1:] != (
        len(SUPERCLASSES),
    ):
        raise ValueError(
            f"targets và probabilities phải cùng shape (n, {len(SUPERCLASSES)}), "
            f"nhận {targets.shape} và {probabilities.shape}"
        )

    if not np.all(np.isin(targets, [0, 1])):
        raise ValueError("targets chỉ được chứa giá trị 0 hoặc 1.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities phải chứa các giá trị hữu hạn.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities phải nằm trong khoảng [0, 1].")

    if targets.shape[0] == 0:
        raise ValueError("Không thể chọn ngưỡng từ tập dữ liệu rỗng.")

    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.05, 0.951, 0.01), 2)

    threshold_grid = np.asarray(threshold_grid, dtype=float)

    if (
        threshold_grid.ndim != 1
        or threshold_grid.size == 0
        or not np.all(np.isfinite(threshold_grid))
        or np.any(threshold_grid < 0)
        or np.any(threshold_grid > 1)
    ):
        raise ValueError("threshold_grid phải là mảng ngưỡng hợp lệ trong [0, 1].")

    selected = {}

    for index, class_name in enumerate(SUPERCLASSES):
        scores = []

        for threshold in threshold_grid:
            predictions = (probabilities[:, index] >= threshold).astype(int)
            score = f1_score(
                targets[:, index].astype(int),
                predictions,
                zero_division=0,
            )
            scores.append((float(threshold), float(score)))

        best_f1 = max(score for _, score in scores)
        candidates = [
            item for item in scores if np.isclose(item[1], best_f1, atol=1e-12, rtol=0)
        ]

        best_threshold, _ = min(
            candidates,
            key=lambda item: (abs(item[0] - 0.5), item[0]),
        )
        selected[class_name] = best_threshold

    return selected


def _validate_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    """Kiểm tra file ngưỡng có đủ lớp và giá trị hợp lệ."""
    if not isinstance(thresholds, dict):
        raise TypeError("thresholds phải là một JSON object.")

    expected_classes = set(SUPERCLASSES)
    actual_classes = set(thresholds)

    if actual_classes != expected_classes:
        missing = sorted(expected_classes - actual_classes)
        extra = sorted(actual_classes - expected_classes)
        raise ValueError(
            f"File ngưỡng phải có đúng các lớp {list(SUPERCLASSES)}. "
            f"Thiếu: {missing}; thừa: {extra}"
        )

    validated = {}

    for class_name in SUPERCLASSES:
        value = thresholds[class_name]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Ngưỡng của {class_name} phải là số.")

        value = float(value)

        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Ngưỡng của {class_name} phải nằm trong khoảng [0, 1].")

        validated[class_name] = value

    return validated


def _load_thresholds(path: str | Path, run_id: str | None = None) -> dict[str, float]:
    """Đọc ngưỡng D7 từ file JSON.

    Nếu truyền run_id, trường run_id trong file phải trùng để tránh dùng nhầm
    ngưỡng của model khác.
    """
    try:
        content = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không đọc được file ngưỡng {path}: {exc}") from exc

    if not isinstance(content, dict) or "thresholds" not in content:
        raise ValueError("File JSON phải chứa trường 'thresholds'.")

    if run_id is not None and content.get("run_id") != run_id:
        raise ValueError(
            f"run_id trong file ngưỡng ({content.get('run_id')}) khác --run-id ({run_id})."
        )

    return _validate_thresholds(content["thresholds"])


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
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Đường dẫn file JSON chứa ngưỡng D7 riêng cho từng lớp.",
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


def evaluate_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Tính F1, AUC và confusion matrix theo ngưỡng từng lớp."""
    if targets.shape != probabilities.shape or targets.shape[1:] != (
        len(SUPERCLASSES),
    ):
        raise ValueError(
            f"targets và probabilities phải cùng shape (n, {len(SUPERCLASSES)}), "
            f"nhận {targets.shape} và {probabilities.shape}"
        )

    if thresholds is None:
        thresholds = {name: THRESHOLD for name in SUPERCLASSES}
        use_default = True
    else:
        thresholds = _validate_thresholds(thresholds)
        use_default = False

    threshold_values = np.array(
        [thresholds[name] for name in SUPERCLASSES],
        dtype=float,
    )

    metrics = calculate_metrics(targets, probabilities)
    predictions = (probabilities >= threshold_values).astype(int)

    if use_default:
        per_class_f1 = metrics["per_class_f1"]
        macro_f1 = metrics["macro_f1"]
    else:
        per_class_f1 = {
            name: float(
                f1_score(
                    targets[:, index].astype(int),
                    predictions[:, index],
                    zero_division=0,
                )
            )
            for index, name in enumerate(SUPERCLASSES)
        }
        macro_f1 = float(np.mean(list(per_class_f1.values())))

    matrices = multilabel_confusion_matrix(
        targets.astype(int),
        predictions,
        labels=list(range(len(SUPERCLASSES))),
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
        "threshold": THRESHOLD if use_default else None,
        "thresholds": thresholds,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
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

    thresholds = (
        _load_thresholds(args.thresholds, args.run_id)
        if args.thresholds is not None
        else None
    )

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
        args.data_dir,
        folds=folds,
        sample_n=args.sample_n,
        seed=args.seed,
    )
    targets, probabilities = predict_probabilities(
        model,
        dataset,
        args.batch_size,
    )

    result = {
        "run_id": args.run_id,
        "split": args.split,
        "folds": list(folds),
        "sample_n": args.sample_n,
        "seed": args.seed,
        "superclasses": list(SUPERCLASSES),
        **evaluate_predictions(targets, probabilities, thresholds),
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{args.split}.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        client.log_artifact(args.run_id, str(path), "evaluation")

    print(f"Run: {args.run_id} | split={args.split} | folds={list(folds)}")
    print(f"Records: {result['n_records']}")
    print(f"auc_macro={result['auc_macro']} | auc_micro={result['auc_micro']}")
    print(f"macro_f1={result['macro_f1']:.4f}")

    for name in SUPERCLASSES:
        cm = result["confusion_matrix"][name]
        print(
            f"  {name}: threshold={result['thresholds'][name]:.2f} | "
            f"F1={result['per_class_f1'][name]:.4f} | "
            f"tp={cm['tp']} fp={cm['fp']} fn={cm['fn']} tn={cm['tn']}"
        )

    print(f"Artifact: evaluation/{args.split}.json")


if __name__ == "__main__":
    main()
