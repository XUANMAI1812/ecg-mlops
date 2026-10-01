"""Đọc metadata và tín hiệu của PTB-XL (v1.0.3), kèm kiểm tra tín hiệu sau khi đọc"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

REQUIRED_DB_COLUMNS = (
    "scp_codes",
    "strat_fold",
    "patient_id",
    "filename_lr",
    "filename_hr",
)

# Theo trang dataset: mỗi bản ghi 10 giây, 12 chuyển đạo
RECORD_SECONDS = 10
N_LEADS = 12


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Đọc ptbxl_database.csv; chỉ mục là ecg_id, scp_codes được đổi thành dict"""
    path = Path(data_dir) / "ptbxl_database.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy {path}. Kiểm tra lại đường dẫn dữ liệu")
    header = pd.read_csv(path, nrows=0)
    if "ecg_id" not in header.columns:
        raise ValueError(
            f"{path} không có cột ecg_id; file có thể bị hỏng hoặc tải chưa đủ"
            f"Các cột đọc được: {list(header.columns)}"
        )
    df = pd.read_csv(path, index_col="ecg_id")
    missing = [c for c in REQUIRED_DB_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ptbxl_database.csv thiếu cột: {missing}")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def load_scp_statements(data_dir: str | Path) -> pd.DataFrame:
    """Đọc scp_statements.csv; chỉ mục là mã SCP (NORM, IMI, ...)"""
    path = Path(data_dir) / "scp_statements.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy {path}. Kiểm tra lại đường dẫn dữ liệu")
    return pd.read_csv(path, index_col=0)


def load_record(data_dir: str | Path, filename: str) -> tuple[np.ndarray, dict]:
    """Đọc một bản ghi WFDB (chưa kiểm tra nội dung tín hiệu)

    filename lấy từ cột filename_lr (100Hz) hoặc filename_hr (500Hz), không có đuôi
    .dat/.hea. Trả về (tín hiệu shape (số mẫu, 12), dict thông tin của wfdb)
    """
    try:
        signal, fields = wfdb.rdsamp(str(Path(data_dir) / filename))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Không thấy bản ghi {filename} trong {data_dir}; "
            "dữ liệu có thể tải chưa đủ (kiểm tra lại bước tải và SHA256)"
        ) from exc
    return signal, fields


def validate_signal(signal: np.ndarray, fields: dict, sampling_rate: int) -> None:
    """Kiểm tra một bản ghi đã đọc; sai thì raise ValueError với thông báo cụ thể.

    Kiểm tra: shape == (RECORD_SECONDS * sampling_rate, N_LEADS), tần số lấy mẫu trong
    header khớp sampling_rate, số tên chuyển đạo == N_LEADS, không có NaN/inf

    TODO: chưa kiểm tra tên và thứ tự chuyển đạo (cần xem header thật của PTB-XL ở
    output mục "Đọc thử một bản ghi" trước khi ghim danh sách)
    """
    expected = (RECORD_SECONDS * sampling_rate, N_LEADS)
    if signal.shape != expected:
        raise ValueError(f"shape {signal.shape}, kỳ vọng {expected}")
    if fields.get("fs") != sampling_rate:
        raise ValueError(
            f"fs trong header là {fields.get('fs')}, kỳ vọng {sampling_rate}"
        )
    n_names = len(fields.get("sig_name") or [])
    if n_names != N_LEADS:
        raise ValueError(f"header có {n_names} tên chuyển đạo, kỳ vọng {N_LEADS}")
    n_nan = int(np.isnan(signal).sum())
    n_inf = int(np.isinf(signal).sum())
    if n_nan or n_inf:
        raise ValueError(f"tín hiệu có {n_nan} NaN và {n_inf} inf")


def load_validated_record(
    data_dir: str | Path, filename: str, sampling_rate: int
) -> tuple[np.ndarray, dict]:
    """Đọc một bản ghi rồi kiểm tra bằng validate_signal"""
    signal, fields = load_record(data_dir, filename)
    try:
        validate_signal(signal, fields, sampling_rate)
    except ValueError as exc:
        raise ValueError(f"Bản ghi {filename} không hợp lệ: {exc}") from exc
    return signal, fields


def main() -> None:
    parser = argparse.ArgumentParser(description="In thông tin một bản ghi PTB-XL.")
    parser.add_argument("--data-dir", default=os.environ.get("PTBXL_DIR"))
    parser.add_argument("--ecg-id", type=int, default=1)
    parser.add_argument("--sampling-rate", type=int, choices=(100, 500), default=100)
    args = parser.parse_args()
    if not args.data_dir:
        parser.error("Cần --data-dir hoặc biến môi trường PTBXL_DIR")

    try:
        meta = load_metadata(args.data_dir)
        if args.ecg_id not in meta.index:
            raise ValueError(f"ecg_id={args.ecg_id} không có trong ptbxl_database.csv")
        row = meta.loc[args.ecg_id]
        column = "filename_lr" if args.sampling_rate == 100 else "filename_hr"
        signal, fields = load_validated_record(
            args.data_dir, row[column], args.sampling_rate
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"LỖI: {exc}") from exc

    print(f"ecg_id       : {args.ecg_id}")
    print(f"file         : {row[column]}")
    print(f"shape        : {signal.shape}")
    print(f"dtype        : {signal.dtype}")
    print(f"fs (header)  : {fields['fs']}")
    print(f"leads        : {fields['sig_name']}")
    print(
        f"NaN / inf    : {int(np.isnan(signal).sum())} / {int(np.isinf(signal).sum())}"
    )
    print(f"min / max    : {float(signal.min()):.3f} / {float(signal.max()):.3f}")
    print(f"scp_codes    : {row['scp_codes']}")
    print(f"strat_fold   : {row['strat_fold']}")
    print("validate     : OK (shape, fs, số chuyển đạo, không NaN/inf)")


if __name__ == "__main__":
    main()
