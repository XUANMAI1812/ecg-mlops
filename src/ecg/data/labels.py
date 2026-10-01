"""Gán nhãn superclass cho PTB-XL và thống kê phục vụ khảo sát dữ liệu

Cách gộp nhãn bám theo example_physionet.py đi kèm dataset: chỉ giữ các mã SCP có
diagnostic == 1 trong scp_statements.csv, rồi tra cột diagnostic_class. Không lọc theo
độ chắc chắn (likelihood) của từng mã
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from ecg.data.load import load_metadata, load_scp_statements

SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def build_code_to_superclass(scp_statements: pd.DataFrame) -> dict[str, str]:
    """Bảng tra: mã SCP chẩn đoán -> superclass"""
    diag = scp_statements[scp_statements["diagnostic"] == 1].dropna(
        subset=["diagnostic_class"]
    )
    if not diag.index.is_unique:
        raise ValueError("scp_statements.csv có mã SCP trùng trong nhóm diagnostic")
    return diag["diagnostic_class"].to_dict()


def add_superclass_column(
    meta: pd.DataFrame, scp_statements: pd.DataFrame
) -> pd.DataFrame:
    """Thêm cột diagnostic_superclass (list các superclass, đã sắp xếp, không trùng"""
    mapping = build_code_to_superclass(scp_statements)
    out = meta.copy()
    out["diagnostic_superclass"] = out["scp_codes"].apply(
        lambda codes: sorted({mapping[k] for k in codes if k in mapping})
    )
    return out


def count_superclasses(meta: pd.DataFrame) -> pd.Series:
    """Số bản ghi của mỗi superclass (bản ghi đa nhãn được đếm ở mọi lớp của nó)"""
    counts = meta["diagnostic_superclass"].explode().dropna().value_counts()
    return counts.reindex(list(SUPERCLASSES), fill_value=0).astype(int)


def summarize(meta: pd.DataFrame) -> dict:
    n_labels = meta["diagnostic_superclass"].apply(len)
    folds_per_patient = meta.groupby("patient_id")["strat_fold"].nunique()
    if "validated_by_human" in meta.columns:
        validated = meta["validated_by_human"].astype(str).str.lower().eq("true")
        validated_share = (
            validated.groupby(meta["strat_fold"]).mean().round(3).to_dict()
        )
    else:
        validated_share = None
    return {
        "n_records": len(meta),
        "n_patients": int(meta["patient_id"].nunique()),
        "records_per_fold": meta["strat_fold"].value_counts().sort_index().to_dict(),
        "superclass_counts": count_superclasses(meta).to_dict(),
        "n_no_superclass": int((n_labels == 0).sum()),
        "n_multi_label": int((n_labels > 1).sum()),
        "max_folds_per_patient": int(folds_per_patient.max()),
        "human_validated_share_per_fold": validated_share,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Thống kê nhãn superclass của PTB-XL.")
    parser.add_argument("--data-dir", default=os.environ.get("PTBXL_DIR"))
    args = parser.parse_args()
    if not args.data_dir:
        parser.error("Cần --data-dir hoặc biến môi trường PTBXL_DIR")

    meta = add_superclass_column(
        load_metadata(args.data_dir), load_scp_statements(args.data_dir)
    )
    s = summarize(meta)

    print(f"Số bản ghi (số dòng ptbxl_database.csv) : {s['n_records']}")
    print(f"Số bệnh nhân (patient_id khác nhau)     : {s['n_patients']}")
    print(f"Số fold tối đa mà 1 bệnh nhân xuất hiện  : {s['max_folds_per_patient']}")
    print("Số bản ghi theo strat_fold:")
    for fold, n in s["records_per_fold"].items():
        print(f"  fold {fold}: {n}")
    print("Số bản ghi theo superclass (đa nhãn, tổng có thể lớn hơn số bản ghi):")
    for name in SUPERCLASSES:
        print(f"  {name:<5}: {s['superclass_counts'][name]}")
    print(f"Bản ghi KHÔNG có superclass nào (D6)     : {s['n_no_superclass']}")
    print(f"Bản ghi có từ 2 superclass trở lên       : {s['n_multi_label']}")
    if s["human_validated_share_per_fold"] is not None:
        print("Tỉ lệ validated_by_human theo fold:")
        for fold, share in s["human_validated_share_per_fold"].items():
            print(f"  fold {fold}: {share}")


if __name__ == "__main__":
    main()
