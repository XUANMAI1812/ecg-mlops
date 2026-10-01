"""Kiểm thử đọc metadata, đọc bản ghi WFDB và kiểm tra tín hiệu.

Toàn bộ dữ liệu ở đây là dữ liệu tổng hợp (giả lập), KHÔNG phải PTB-XL; các test này
không thay thế việc chạy khảo sát trên dữ liệu thật
TODO: bổ sung kiểm thử trên một mẫu nhỏ PTB-XL thật khi có pipeline tiền xử lý
"""

import sys

import numpy as np
import pandas as pd
import pytest
import wfdb

from ecg.data import load
from ecg.data.load import (
    load_metadata,
    load_record,
    load_scp_statements,
    load_validated_record,
    validate_signal,
)

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def make_fake_ptbxl(root, n_samples=1000):
    """Dựng thư mục giống PTB-XL với 1 bản ghi 100Hz gồm n_samples mẫu"""
    rec_dir = root / "records100" / "00000"
    rec_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 0.1, size=(n_samples, 12))
    wfdb.wrsamp(
        "00001_lr",
        fs=100,
        units=["mV"] * 12,
        sig_name=LEADS,
        p_signal=signal,
        fmt=["16"] * 12,
        write_dir=str(rec_dir),
    )
    pd.DataFrame(
        {
            "ecg_id": [1],
            "patient_id": [15709.0],
            "scp_codes": ["{'NORM': 100.0, 'SR': 0.0}"],
            "strat_fold": [3],
            "filename_lr": ["records100/00000/00001_lr"],
            "filename_hr": ["records500/00000/00001_hr"],
        }
    ).to_csv(root / "ptbxl_database.csv", index=False)
    pd.DataFrame(
        {"diagnostic": [1.0, None], "diagnostic_class": ["NORM", None]},
        index=["NORM", "SR"],
    ).to_csv(root / "scp_statements.csv")
    return root


@pytest.fixture()
def fake_ptbxl(tmp_path):
    return make_fake_ptbxl(tmp_path)


def test_load_metadata_parses_scp_codes(fake_ptbxl):
    meta = load_metadata(fake_ptbxl)
    assert meta.index.name == "ecg_id"
    assert meta.loc[1, "scp_codes"] == {"NORM": 100.0, "SR": 0.0}
    assert meta.loc[1, "strat_fold"] == 3


def test_load_scp_statements(fake_ptbxl):
    scp = load_scp_statements(fake_ptbxl)
    assert list(scp.index) == ["NORM", "SR"]


def test_load_record_shape_and_leads(fake_ptbxl):
    meta = load_metadata(fake_ptbxl)
    signal, fields = load_record(fake_ptbxl, meta.loc[1, "filename_lr"])
    assert signal.shape == (1000, 12)
    assert fields["fs"] == 100
    assert fields["sig_name"] == LEADS
    assert not np.isnan(signal).any()


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ptbxl_database.csv"):
        load_metadata(tmp_path)


def test_corrupt_metadata_gives_clear_error(tmp_path):
    (tmp_path / "ptbxl_database.csv").write_text("x\n")
    with pytest.raises(ValueError, match="ecg_id"):
        load_metadata(tmp_path)


def test_missing_record_gives_clear_error(fake_ptbxl):
    with pytest.raises(FileNotFoundError, match="bản ghi"):
        load_record(fake_ptbxl, "records500/00000/00001_hr")


def test_validate_accepts_valid_record(fake_ptbxl):
    signal, fields = load_record(fake_ptbxl, "records100/00000/00001_lr")
    validate_signal(signal, fields, 100)  # không raise


def test_validate_rejects_wrong_length(fake_ptbxl):
    signal, fields = load_record(fake_ptbxl, "records100/00000/00001_lr")
    with pytest.raises(ValueError, match="shape"):
        validate_signal(signal[:800], fields, 100)


def test_validate_rejects_wrong_lead_count(fake_ptbxl):
    signal, fields = load_record(fake_ptbxl, "records100/00000/00001_lr")
    with pytest.raises(ValueError, match="shape"):
        validate_signal(signal[:, :11], fields, 100)
    with pytest.raises(ValueError, match="tên chuyển đạo"):
        validate_signal(signal, dict(fields, sig_name=LEADS[:11]), 100)


def test_validate_rejects_wrong_fs(fake_ptbxl):
    signal, fields = load_record(fake_ptbxl, "records100/00000/00001_lr")
    with pytest.raises(ValueError, match="fs"):
        validate_signal(signal, dict(fields, fs=500), 100)


def test_validate_rejects_nan_and_inf(fake_ptbxl):
    signal, fields = load_record(fake_ptbxl, "records100/00000/00001_lr")
    bad = signal.copy()
    bad[5, 3] = np.nan
    bad[6, 4] = np.inf
    with pytest.raises(ValueError, match="1 NaN và 1 inf"):
        validate_signal(bad, fields, 100)


def test_load_validated_record_names_the_file(tmp_path):
    root = make_fake_ptbxl(tmp_path, n_samples=800)
    with pytest.raises(ValueError, match="00001_lr.*không hợp lệ"):
        load_validated_record(root, "records100/00000/00001_lr", 100)


def test_main_prints_ok_for_valid_record(fake_ptbxl, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["load", "--data-dir", str(fake_ptbxl)])
    load.main()
    out = capsys.readouterr().out
    assert "shape        : (1000, 12)" in out
    assert "validate     : OK" in out


def test_main_exits_with_error_for_invalid_record(tmp_path, monkeypatch):
    root = make_fake_ptbxl(tmp_path, n_samples=800)
    monkeypatch.setattr(sys, "argv", ["load", "--data-dir", str(root)])
    with pytest.raises(SystemExit) as exc:
        load.main()
    assert "không hợp lệ" in str(exc.value)
