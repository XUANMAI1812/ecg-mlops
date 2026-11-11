"""Tinh chỉnh tham số bằng Optuna cho 1D-CNN và ResNet1D, chỉ dùng fold 1-9.

Hai lệnh:
- run: chạy một study. Mỗi trial gọi train() trong cùng tiến trình với fold 1-8
  (train) và fold 9 (val); không đánh giá fold 10, không log model. Sau mỗi epoch,
  trial báo val_auc_macro cho Optuna (step = số epoch) và MedianPruner dừng sớm
  trial kém, sớm nhất là sau epoch 2. Trên MLflow, mỗi lần chạy study là một run
  cha, mỗi trial là một run con.
- summarize: đọc file study (.db) đã chạy ở máy khác, ghi tóm tắt thành một run
  MLflow (cấu hình study, tham số tốt nhất, bảng mọi trial, file study); không train.

Mục tiêu tối ưu là best_val_auc_macro (AUC macro ở checkpoint chọn theo val loss),
còn pruner nhận val_auc_macro của từng epoch; hai giá trị có thể khác nhau.

Ví dụ (từ gốc repo, venv đã kích hoạt, MLflow server đang chạy):
    PYTHONPATH=src python -m ecg.train.tune run --model cnn1d --n-trials 20
    PYTHONPATH=src python -m ecg.train.tune summarize \\
        --storage-file optuna_data/cnn1d-aucmacro-v1.db --source may-klinh
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.trial import TrialState
from torch.utils.data import Dataset

from ecg.data.dataset import ECGDataset
from ecg.models.model_factory import SUPPORTED_MODELS
from ecg.models.train import DEFAULT_TRACKING_URI, train
from ecg.train.tracking import RunTracker

DEFAULT_EXPERIMENT = "ecg-tuning"
DEFAULT_STORAGE_DIR = "optuna_data"
DEFAULT_OUTPUT_ROOT = "artifacts/tuning"
OBJECTIVE = "best_val_auc_macro"
TRAIN_FOLDS = tuple(range(1, 9))
VAL_FOLDS = (9,)


class InMemoryDataset(Dataset):
    """Giữ trong RAM toàn bộ tín hiệu đã tiền xử lý của một dataset.

    Tiền xử lý không có bước ngẫu nhiên nên dữ liệu giống hệt khi đọc từ đĩa; chỉ
    nhanh hơn khi cùng dữ liệu được dùng lại qua nhiều epoch và nhiều trial.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.metadata = dataset.metadata
        size = len(dataset)
        if size == 0:
            raise ValueError("Dataset rỗng")
        first_signal, first_target = dataset[0]
        self.signals = torch.empty(
            (size, *first_signal.shape), dtype=first_signal.dtype
        )
        self.targets = torch.empty(
            (size, *first_target.shape), dtype=first_target.dtype
        )
        for index in range(size):
            signal, target = dataset[index]
            self.signals[index] = signal
            self.targets[index] = target

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signals[index], self.targets[index]


def load_datasets(
    data_dir: str, sample_n: int | None, seed: int, cache: bool
) -> tuple[Dataset, Dataset]:
    """Nạp fold 1-8 (train) và fold 9 (val) một lần cho cả study."""
    sample = {"sample_n": sample_n, "seed": seed}
    train_ds = ECGDataset(data_dir, folds=TRAIN_FOLDS, **sample)
    val_ds = ECGDataset(data_dir, folds=VAL_FOLDS, **sample)
    if not cache:
        return train_ds, val_ds
    print(f"Nạp vào RAM: {len(train_ds)} bản ghi train, {len(val_ds)} bản ghi val")
    return InMemoryDataset(train_ds), InMemoryDataset(val_ds)


def study_config(args: argparse.Namespace) -> dict[str, Any]:
    """Cấu hình cố định của một study; mọi lần chạy tiếp study phải giữ nguyên."""
    return {
        "model": args.model,
        "objective": OBJECTIVE,
        "direction": "maximize",
        "epochs": args.epochs,
        "seed": args.seed,
        "sample_n": args.sample_n if args.sample_n is not None else "all",
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "batch_sizes": list(args.batch_sizes),
        "sampler": "TPESampler",
        "pruner": "MedianPruner",
        "n_startup_trials": args.n_startup_trials,
        "n_warmup_steps": args.n_warmup_steps,
    }


def ensure_study_config(study: optuna.Study, config: dict[str, Any]) -> None:
    """Ghi cấu hình vào study mới; với study đã có thì bắt buộc cấu hình trùng."""
    stored = study.user_attrs
    if not stored:
        for key, value in config.items():
            study.set_user_attr(key, value)
        return
    different = sorted(key for key in config if stored.get(key) != config[key])
    if different:
        details = ", ".join(
            f"{key}: đã lưu {stored.get(key)!r}, lần này {config[key]!r}"
            for key in different
        )
        raise SystemExit(
            f"Study {study.study_name} đã có với cấu hình khác ({details}). "
            "Dùng --study-name khác hoặc giữ đúng cấu hình cũ."
        )


def create_study(
    storage_file: Path, study_name: str, seed: int, n_startup: int, n_warmup: int
) -> optuna.Study:
    storage_file.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_file.resolve()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup, n_warmup_steps=n_warmup
        ),
        load_if_exists=True,
    )


def make_objective(
    args: argparse.Namespace, datasets: tuple[Dataset, Dataset]
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float(
            "learning_rate", args.lr_min, args.lr_max, log=True
        )
        batch_size = trial.suggest_categorical("batch_size", list(args.batch_sizes))
        output_dir = Path(args.output_root) / args.study_name / f"t{trial.number}"
        train_args = argparse.Namespace(
            data_dir=args.data_dir,
            output_dir=str(output_dir),
            model=args.model,
            epochs=args.epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=args.seed,
            sample_n=args.sample_n,
        )

        def report(epoch: int, record: dict) -> None:
            value = record["val_auc_macro"]
            if math.isnan(value):
                return
            trial.report(value, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Dừng sớm sau epoch {epoch}")

        pruned = False
        run_name = f"{args.study_name}-t{trial.number}"
        with RunTracker(
            args.tracking_uri, args.experiment, run_name, nested=True
        ) as tracker:
            trial.set_user_attr("mlflow_run_id", tracker.run_id)
            tracker.set_tags(
                {"study": args.study_name, "trial_number": str(trial.number)}
            )
            try:
                summary = train(
                    train_args,
                    tracker,
                    evaluate_test=False,
                    log_model=False,
                    epoch_callback=report,
                    datasets=datasets,
                )
            except optuna.TrialPruned:
                tracker.set_tags({"trial_state": "pruned"})
                pruned = True
            else:
                tracker.set_tags({"trial_state": "complete"})
        if pruned:
            raise optuna.TrialPruned()
        return summary[OBJECTIVE]

    return objective


def count_states(study: optuna.Study) -> dict[str, int]:
    states = [trial.state for trial in study.trials]
    return {
        "complete": states.count(TrialState.COMPLETE),
        "pruned": states.count(TrialState.PRUNED),
        "failed": states.count(TrialState.FAIL),
    }


def log_study_summary(tracker: RunTracker, study: optuna.Study) -> None:
    """Ghi số trial theo trạng thái, trial tốt nhất và bảng trials.csv vào run."""
    counts = count_states(study)
    tracker.log_summary({f"n_trials_{state}": n for state, n in counts.items()})
    if counts["complete"]:
        best = study.best_trial
        tracker.log_summary({"study_best_value": best.value})
        tracker.log_params({f"best_{key}": value for key, value in best.params.items()})
        tracker.set_tags({"best_trial": str(best.number)})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trials.csv"
        study.trials_dataframe().to_csv(path, index=False)
        tracker.log_artifact(str(path), "study")


def print_summary(study: optuna.Study) -> None:
    counts = count_states(study)
    print(f"Study: {study.study_name}")
    print(
        f"Trials: complete={counts['complete']} pruned={counts['pruned']} "
        f"failed={counts['failed']}"
    )
    if counts["complete"]:
        best = study.best_trial
        print(f"Best trial: {best.number} | {OBJECTIVE}={best.value:.4f}")
        for key, value in best.params.items():
            print(f"  {key}={value}")


def run_study(
    args: argparse.Namespace, datasets: tuple[Dataset, Dataset] | None = None
) -> optuna.Study:
    storage_file = Path(args.storage_dir) / f"{args.study_name}.db"
    study = create_study(
        storage_file,
        args.study_name,
        args.seed,
        args.n_startup_trials,
        args.n_warmup_steps,
    )
    config = study_config(args)
    ensure_study_config(study, config)

    if datasets is None:
        datasets = load_datasets(
            args.data_dir, args.sample_n, args.seed, cache=not args.no_cache
        )

    with RunTracker(args.tracking_uri, args.experiment, args.study_name) as tracker:
        tracker.set_tags({"study": args.study_name, "role": "study"})
        tracker.log_params(
            {
                **config,
                "n_trials_requested": args.n_trials,
                "cache_in_memory": not args.no_cache,
                "storage_file": str(storage_file),
            }
        )
        # ValueError: loss hoặc xác suất thành NaN khi trial phân kỳ. Trial đó được
        # ghi FAIL (run con FAILED) và study chạy tiếp, không dừng cả study.
        study.optimize(
            make_objective(args, datasets), n_trials=args.n_trials, catch=(ValueError,)
        )
        log_study_summary(tracker, study)
        run_id = tracker.run_id

    print_summary(study)
    print(f"Study file: {storage_file}")
    print(f"MLflow run: {run_id}")
    return study


def summarize(args: argparse.Namespace) -> None:
    storage_file = Path(args.storage_file)
    if not storage_file.is_file():
        raise SystemExit(f"Không thấy file study {storage_file}")
    study_name = args.study_name or storage_file.stem
    study = optuna.load_study(
        study_name=study_name, storage=f"sqlite:///{storage_file.resolve()}"
    )

    run_name = f"{study_name}-summary"
    with RunTracker(args.tracking_uri, args.experiment, run_name) as tracker:
        tracker.set_tags(
            {"study": study_name, "role": "summary", "source": args.source}
        )
        tracker.log_params(study.user_attrs)
        log_study_summary(tracker, study)
        tracker.log_artifact(str(storage_file), "study")
        run_id = tracker.run_id

    print_summary(study)
    print(f"MLflow run: {run_id}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    common.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", parents=[common], help="Chạy một study.")
    run.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    run.add_argument("--n-trials", type=int, required=True)
    run.add_argument("--epochs", type=int, default=5)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Chỉ dùng N bản ghi mỗi tập train/val (mặc định: toàn bộ).",
    )
    run.add_argument("--data-dir", default="data/raw/ptb-xl")
    run.add_argument("--storage-dir", default=DEFAULT_STORAGE_DIR)
    run.add_argument(
        "--study-name",
        default=None,
        help="Mặc định <model>-aucmacro-v1, thêm -s<N> khi có --sample-n.",
    )
    run.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--lr-min", type=float, default=1e-4)
    run.add_argument("--lr-max", type=float, default=1e-2)
    run.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    run.add_argument(
        "--n-startup-trials",
        type=int,
        default=5,
        help="Số trial hoàn thành đầu tiên không bị dừng sớm.",
    )
    run.add_argument(
        "--n-warmup-steps",
        type=int,
        default=2,
        help="Chỉ xét dừng sớm từ epoch này trở đi (2: không dừng sau epoch 1).",
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="Không giữ dữ liệu trong RAM; đọc và tiền xử lý lại từ đĩa mỗi epoch.",
    )

    summary = commands.add_parser(
        "summarize", parents=[common], help="Ghi tóm tắt một file study lên MLflow."
    )
    summary.add_argument("--storage-file", required=True)
    summary.add_argument(
        "--study-name", default=None, help="Mặc định là tên file không có đuôi."
    )
    summary.add_argument(
        "--source", default="imported", help="Nơi đã chạy study, ví dụ may-klinh."
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        if args.n_trials < 1 or args.epochs < 1:
            parser.error("--n-trials và --epochs phải >= 1")
        if args.sample_n is not None and args.sample_n < 1:
            parser.error("--sample-n phải >= 1")
        if not 0 < args.lr_min < args.lr_max:
            parser.error("Cần 0 < --lr-min < --lr-max")
        if min(args.batch_sizes) < 1:
            parser.error("--batch-sizes phải >= 1")
        if args.study_name is None:
            args.study_name = f"{args.model}-aucmacro-v1"
            if args.sample_n is not None:
                args.study_name += f"-s{args.sample_n}"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "run":
        run_study(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
