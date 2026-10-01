"""Kiểm thử gộp nhãn superclass trên dữ liệu tổng hợp nhỏ (không cần tải PTB-XL)"""

import pandas as pd
import pytest

from ecg.data.labels import (
    SUPERCLASSES,
    add_superclass_column,
    build_code_to_superclass,
    count_superclasses,
    summarize,
)


@pytest.fixture()
def scp_statements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0, 1.0, None, 1.0],
            "diagnostic_class": ["NORM", "MI", "STTC", None, "CD"],
        },
        index=["NORM", "IMI", "NDT", "SR", "CRBBB"],
    )


@pytest.fixture()
def meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scp_codes": [
                {"NORM": 100.0, "SR": 0.0},  # NORM
                {"IMI": 80.0, "NDT": 50.0},  # MI + STTC (đa nhãn)
                {"SR": 0.0},  # không có mã chẩn đoán -> không có superclass
                {"IMI": 35.0, "CRBBB": 100.0, "NDT": 0.0},  # MI + CD + STTC
            ],
            "strat_fold": [1, 9, 10, 10],
            "patient_id": [1.0, 2.0, 3.0, 4.0],
            "validated_by_human": [True, False, True, True],
        },
        index=pd.Index([1, 2, 3, 4], name="ecg_id"),
    )


def test_mapping_only_keeps_diagnostic_codes(scp_statements):
    mapping = build_code_to_superclass(scp_statements)
    assert mapping == {"NORM": "NORM", "IMI": "MI", "NDT": "STTC", "CRBBB": "CD"}
    assert "SR" not in mapping


def test_superclass_column_is_sorted_unique_list(meta, scp_statements):
    out = add_superclass_column(meta, scp_statements)
    assert out.loc[1, "diagnostic_superclass"] == ["NORM"]
    assert out.loc[2, "diagnostic_superclass"] == ["MI", "STTC"]
    assert out.loc[3, "diagnostic_superclass"] == []
    assert out.loc[4, "diagnostic_superclass"] == ["CD", "MI", "STTC"]


def test_input_is_not_mutated(meta, scp_statements):
    add_superclass_column(meta, scp_statements)
    assert "diagnostic_superclass" not in meta.columns


def test_counts_cover_all_five_classes(meta, scp_statements):
    counts = count_superclasses(add_superclass_column(meta, scp_statements))
    assert list(counts.index) == list(SUPERCLASSES)
    assert counts.to_dict() == {"NORM": 1, "MI": 2, "STTC": 2, "CD": 1, "HYP": 0}


def test_summary(meta, scp_statements):
    s = summarize(add_superclass_column(meta, scp_statements))
    assert s["n_records"] == 4
    assert s["n_patients"] == 4
    assert s["n_no_superclass"] == 1
    assert s["n_multi_label"] == 2
    assert s["max_folds_per_patient"] == 1
    assert s["records_per_fold"] == {1: 1, 9: 1, 10: 2}
    assert s["human_validated_share_per_fold"] == {1: 1.0, 9: 0.0, 10: 1.0}
