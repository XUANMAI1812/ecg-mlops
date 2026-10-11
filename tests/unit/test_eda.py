import json

import pandas as pd

from ecg.data.eda import run_eda


def test_run_eda_creates_outputs(tmp_path, monkeypatch):
    metadata = pd.DataFrame(
        {
            "scp_codes": [
                {"NORM": 100},
                {"IMI": 100},
                {"NORM": 100, "IMI": 100},
            ],
            "strat_fold": [1, 1, 2],
            "patient_id": [101, 102, 103],
        },
        index=pd.Index([1, 2, 3], name="ecg_id"),
    )

    statements = pd.DataFrame(
        {
            "diagnostic": [1, 1],
            "diagnostic_class": ["NORM", "MI"],
        },
        index=["NORM", "IMI"],
    )

    monkeypatch.setattr(
        "ecg.data.eda.load_metadata",
        lambda data_dir: metadata.copy(),
    )
    monkeypatch.setattr(
        "ecg.data.eda.load_scp_statements",
        lambda data_dir: statements.copy(),
    )

    output_dir = tmp_path / "eda"
    summary = run_eda(tmp_path / "fake-data", output_dir)

    assert summary["n_records"] == 3
    assert summary["n_patients"] == 3
    assert summary["superclass_counts"]["NORM"] == 2
    assert summary["superclass_counts"]["MI"] == 2
    assert summary["n_multi_label"] == 1

    saved_summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert saved_summary["n_records"] == 3

    assert (output_dir / "superclass_counts.csv").is_file()
    assert (output_dir / "superclass_distribution.png").is_file()
    assert (output_dir / "fold_distribution.png").is_file()
    assert (output_dir / "labels_per_record.png").is_file()
