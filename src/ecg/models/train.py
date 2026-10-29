"""Train a baseline multi-label 1D-CNN on PTB-XL."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ecg.data.dataset import ECGDataset
from ecg.data.labels import SUPERCLASSES
from ecg.models.cnn1d import ECG1DCNN
from ecg.train.tracking import RunTracker

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
INPUT_SHAPE = (12, 1000)  # 12 chuyển đạo, 10 giây ở 100 Hz; không gồm batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_metrics(
    targets: np.ndarray, probabilities: np.ndarray
) -> dict:
    """Calculate per-class F1 and macro/micro AUC."""
    predictions = probabilities >= 0.5
    per_class_f1 = {}

    for index, name in enumerate(SUPERCLASSES):
        truth = targets[:, index].astype(bool)
        pred = predictions[:, index]

        tp = int(np.logical_and(truth, pred).sum())
        fp = int(np.logical_and(~truth, pred).sum())
        fn = int(np.logical_and(truth, ~pred).sum())

        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        per_class_f1[name] = f1

    # AUC của từng lớp không xác định nếu targets chỉ có một giá trị.
    valid_classes = [
        index
        for index in range(targets.shape[1])
        if np.unique(targets[:, index]).size == 2
    ]

    if valid_classes:
        per_class_auc = {
            SUPERCLASSES[index]: float(
                roc_auc_score(targets[:, index], probabilities[:, index])
            )
            for index in valid_classes
        }
        macro_auc = float(np.mean(list(per_class_auc.values())))
    else:
        per_class_auc = {}
        macro_auc = float("nan")

    # Micro AUC cần cả nhãn 0 và 1 trong toàn bộ ma trận.
    if np.unique(targets).size == 2:
        micro_auc = float(
            roc_auc_score(targets.ravel(), probabilities.ravel())
        )
    else:
        micro_auc = float("nan")

    return {
        "macro_f1": float(np.mean(list(per_class_f1.values()))),
        "per_class_f1": per_class_f1,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "per_class_auc": per_class_auc,
    }


def calculate_pos_weight(
    targets: np.ndarray,
) -> torch.Tensor:
    """Calculate BCE positive weights from training targets only."""
    targets = np.asarray(targets, dtype=np.float32)

    if targets.ndim != 2 or targets.shape[1] != len(SUPERCLASSES):
        raise ValueError(
            f"targets phải có shape (n_samples, {len(SUPERCLASSES)})"
        )
    if targets.shape[0] == 0:
        raise ValueError("Không thể tính pos_weight từ tập train rỗng")

    positive_counts = targets.sum(axis=0)
    negative_counts = targets.shape[0] - positive_counts

    # Nếu một lớp không có mẫu dương, đặt trọng số 1 để tránh chia cho 0.
    weights = np.divide(
        negative_counts,
        positive_counts,
        out=np.ones_like(negative_counts, dtype=np.float32),
        where=positive_counts > 0,
    )

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_records = 0
    all_targets = []
    all_probabilities = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for signals, targets in loader:
            signals = signals.to(device)
            targets = targets.to(device)

            logits = model(signals)
            loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = signals.shape[0]
            total_loss += loss.item() * batch_size
            total_records += batch_size

            all_targets.append(targets.cpu().numpy())
            all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())

    return (
        total_loss / total_records,
        np.concatenate(all_targets),
        np.concatenate(all_probabilities),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ptb-xl")
    parser.add_argument("--output-dir", default="artifacts/ecg-cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument("--experiment", default="ecg-superdiagnostic")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs và --batch-size phải >= 1")

    with RunTracker(args.tracking_uri, args.experiment, args.run_name) as tracker:
        train(args, tracker)


def train(args: argparse.Namespace, tracker: RunTracker) -> None:
    set_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cpu")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ECGDataset(data_dir, folds=tuple(range(1, 9)))
    val_ds = ECGDataset(data_dir, folds=(9,))
    test_ds = ECGDataset(data_dir, folds=(10,))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Device: {device}")
    print(f"Train records: {len(train_ds)}")
    print(f"Validation records: {len(val_ds)}")
    print(f"Test records: {len(test_ds)}")
    print(f"Batch size: {args.batch_size}; epochs: {args.epochs}")
    print(f"Superclass order: {SUPERCLASSES}")

    train_targets = np.stack(
        [
            np.array(
                [
                    1.0 if label in labels else 0.0
                    for label in SUPERCLASSES
                ],
                dtype=np.float32,
            )
            for labels in train_ds.metadata["diagnostic_superclass"]
        ]
    )
    pos_weight = calculate_pos_weight(train_targets).to(device)

    tracker.set_tags({"model": "cnn1d", "task": "superdiagnostic"})
    tracker.log_params(
        {
            "pos_weight": json.dumps(
               pos_weight.cpu().tolist()
            ),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "data_dir": str(data_dir),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "device": str(device),
            "torch_version": torch.__version__,
        }
    )
    # File DVC của dữ liệu (chứa hash phiên bản), nếu có; để truy vết run dùng dữ liệu nào.
    dvc_file = Path(f"{data_dir}.dvc")
    if dvc_file.exists():
        tracker.log_artifact(str(dvc_file), "data")

    model = ECG1DCNN().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )

    best_val_loss = float("inf")
    checkpoint_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_targets, val_probs = run_epoch(
            model, val_loader, criterion, None, device
        )
        val_metrics = calculate_metrics(val_targets, val_probs)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class_f1": val_metrics["per_class_f1"],
            "val_auc_macro": val_metrics["macro_auc"],
            "val_auc_micro": val_metrics["micro_auc"],
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        tracker.log_epoch(
            epoch,
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_metrics["macro_f1"],
                **{
                    f"val_f1_{name}": score
                    for name, score in val_metrics["per_class_f1"].items()
                },
               "val_auc_macro": val_metrics["macro_auc"],
               "val_auc_micro": val_metrics["micro_auc"],
            },
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "superclasses": list(SUPERCLASSES),
                    "sampling_rate": 100,
                    "seed": args.seed,
                    "val_auc_macro": val_metrics["macro_auc"],
                    "val_auc_micro": val_metrics["micro_auc"],
                },
                checkpoint_path,
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    tracker.log_artifact(str(output_dir / "history.json"), "metrics")

    # Evaluate the selected checkpoint on the test fold only after training.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_targets, test_probs = run_epoch(
        model, test_loader, criterion, None, device
    )
    test_metrics = calculate_metrics(test_targets, test_probs)

    results = {
        "test_loss": test_loss,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "checkpoint_epoch": checkpoint["epoch"],
        "superclasses": list(SUPERCLASSES),
        "test_fold": 10,
        "test_auc_macro": test_metrics["macro_auc"],
        "test_auc_micro": test_metrics["micro_auc"],
        "test_per_class_auc": test_metrics["per_class_auc"],
    }
    with (output_dir / "test_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    tracker.log_artifact(str(output_dir / "test_metrics.json"), "metrics")
    tracker.log_summary(
        {
            "best_epoch": checkpoint["epoch"],
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "test_auc_macro": test_metrics["macro_auc"],
            "test_auc_micro": test_metrics["micro_auc"],
            "best_val_auc_macro": checkpoint.get("val_auc_macro", float("nan")),
            "best_val_auc_micro": checkpoint.get("val_auc_micro", float("nan")),
            "test_macro_f1": test_metrics["macro_f1"],
            **{
                f"test_f1_{name}": score
                for name, score in test_metrics["per_class_f1"].items()
            },
        }
    )
    # Model đã nạp checkpoint tốt nhất ở trên; log lên MLflow ở dạng logits.
    tracker.log_model(model, input_shape=INPUT_SHAPE)

    print("\nTest results (fold 10):")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_macro_f1={test_metrics['macro_f1']:.4f}")
    for name, score in test_metrics["per_class_f1"].items():
        print(f"  {name}: F1={score:.4f}")

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"History: {output_dir / 'history.json'}")
    print(f"Test metrics: {output_dir / 'test_metrics.json'}")
    print(f"MLflow run: {tracker.run_id}")


if __name__ == "__main__":
    main()
