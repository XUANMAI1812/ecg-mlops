"""Exploratory data analysis for the PTB-XL metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from ecg.data.labels import (
    add_superclass_column,
    summarize,
)
from ecg.data.load import load_metadata, load_scp_statements


def run_eda(data_dir: str | Path, output_dir: str | Path) -> dict:
    """Summarize PTB-XL metadata and save tables and plots."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(data_dir)
    statements = load_scp_statements(data_dir)
    meta = add_superclass_column(meta, statements)
    summary = summarize(meta)

    # Save numerical summary.
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save superclass counts as CSV.
    counts = pd.Series(summary["superclass_counts"], name="records")
    counts.index.name = "superclass"
    counts.to_csv(output_dir / "superclass_counts.csv")

    # Plot superclass distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("PTB-XL records by diagnostic superclass")
    ax.set_xlabel("Superclass")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "superclass_distribution.png", dpi=150)
    plt.close(fig)

    # Plot stratified folds.
    folds = pd.Series(summary["records_per_fold"], dtype="int64").sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    folds.plot(kind="bar", ax=ax)
    ax.set_title("PTB-XL records by strat_fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "fold_distribution.png", dpi=150)
    plt.close(fig)

    # Plot number of diagnostic superclasses assigned to each record.
    label_counts = meta["diagnostic_superclass"].map(len)
    per_record = label_counts.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    per_record.plot(kind="bar", ax=ax)
    ax.set_title("Number of diagnostic superclasses per record")
    ax.set_xlabel("Number of superclasses")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "labels_per_record.png", dpi=150)
    plt.close(fig)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exploratory data analysis on PTB-XL metadata."
    )
    parser.add_argument("--data-dir", default=os.environ.get("PTBXL_DIR"))
    parser.add_argument("--output-dir", default="reports/eda")
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("Cần --data-dir hoặc biến môi trường PTBXL_DIR")

    summary = run_eda(args.data_dir, args.output_dir)
    print(f"Số bản ghi: {summary['n_records']}")
    print(f"Số bệnh nhân: {summary['n_patients']}")
    print(f"Số bản ghi không có superclass: {summary['n_no_superclass']}")
    print(f"Số bản ghi đa nhãn: {summary['n_multi_label']}")
    print(f"Kết quả EDA đã lưu tại: {args.output_dir}")


if __name__ == "__main__":
    main()
