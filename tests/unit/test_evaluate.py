
import json

import numpy as np
import pytest

from ecg.data.labels import SUPERCLASSES
from ecg.train.evaluate import (
    _load_thresholds,
    _validate_thresholds,
    evaluate_predictions,
    parse_args,
    select_thresholds,
)


def test_evaluate_predictions_counts_confusion_matrix_per_class():
    targets = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    probabilities = np.array(
        [
            [0.9, 0.1, 0.1, 0.1, 0.1],
            [0.4, 0.8, 0.1, 0.1, 0.1],
            [0.6, 0.7, 0.1, 0.1, 0.1],
            [0.1, 0.2, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    result = evaluate_predictions(targets, probabilities)

    assert result["n_records"] == 4
    assert result["threshold"] == 0.5
    assert result["thresholds"] == {
        name: 0.5 for name in SUPERCLASSES
    }
    assert result["confusion_matrix"]["NORM"] == {
        "tn": 1, "fp": 1, "fn": 1, "tp": 1
    }
    assert result["confusion_matrix"]["MI"] == {
        "tn": 2, "fp": 0, "fn": 0, "tp": 2
    }
    assert result["confusion_matrix"]["HYP"] == {
        "tn": 4, "fp": 0, "fn": 0, "tp": 0
    }
    assert set(result["per_class_f1"]) == set(SUPERCLASSES)
    assert result["auc_micro"] is not None


def test_evaluate_predictions_writes_null_when_auc_is_undefined():
    targets = np.zeros((3, len(SUPERCLASSES)), dtype=np.float32)
    probabilities = np.full((3, len(SUPERCLASSES)), 0.2, dtype=np.float32)

    result = evaluate_predictions(targets, probabilities)

    assert result["auc_macro"] is None
    assert result["auc_micro"] is None


def test_evaluate_predictions_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        evaluate_predictions(np.zeros((2, 3)), np.zeros((2, 3)))


def test_test_split_requires_allow_test():
    with pytest.raises(SystemExit):
        parse_args(["--run-id", "abc", "--split", "test"])

    args = parse_args(["--run-id", "abc", "--split", "test", "--allow-test"])
    assert args.split == "test"


def test_default_split_is_validation():
    assert parse_args(["--run-id", "abc"]).split == "val"


def test_select_thresholds_returns_one_threshold_per_class():
    targets = np.zeros((4, len(SUPERCLASSES)), dtype=np.float32)
    probabilities = np.full_like(targets, 0.1)

    targets[:, 0] = [1, 1, 0, 0]
    probabilities[:, 0] = [0.8, 0.7, 0.2, 0.1]

    result = select_thresholds(
        targets,
        probabilities,
        threshold_grid=np.array([0.3, 0.5, 0.7]),
    )

    assert set(result) == set(SUPERCLASSES)
    assert result[SUPERCLASSES[0]] == 0.5


def test_select_thresholds_tie_prefers_threshold_closest_to_half():
    targets = np.zeros((2, len(SUPERCLASSES)), dtype=np.float32)
    probabilities = np.full_like(targets, 0.1)

    targets[:, 0] = [1, 0]
    probabilities[:, 0] = [0.8, 0.2]

    result = select_thresholds(
        targets,
        probabilities,
        threshold_grid=np.array([0.3, 0.5, 0.7]),
    )

    assert result[SUPERCLASSES[0]] == 0.5


def test_evaluate_predictions_uses_per_class_thresholds():
    targets = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    probabilities = np.array(
        [
            [0.6, 0.4, 0.1, 0.1, 0.1],
            [0.4, 0.6, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    thresholds = {name: 0.5 for name in SUPERCLASSES}
    thresholds["NORM"] = 0.7
    thresholds["MI"] = 0.3

    result = evaluate_predictions(targets, probabilities, thresholds)

    assert result["threshold"] is None
    assert result["thresholds"] == thresholds
    assert result["confusion_matrix"]["NORM"]["fn"] == 1
    assert result["confusion_matrix"]["MI"]["fp"] == 1


def test_validate_thresholds_rejects_missing_class():
    thresholds = {name: 0.5 for name in SUPERCLASSES}
    thresholds.pop(SUPERCLASSES[0])

    with pytest.raises(ValueError, match="Thiếu"):
        _validate_thresholds(thresholds)


@pytest.mark.parametrize("invalid_value", [-0.1, 1.1, float("nan"), "0.5"])
def test_validate_thresholds_rejects_invalid_values(invalid_value):
    thresholds = {name: 0.5 for name in SUPERCLASSES}
    thresholds[SUPERCLASSES[0]] = invalid_value

    with pytest.raises(ValueError):
        _validate_thresholds(thresholds)


def test_load_thresholds_reads_json(tmp_path):
    expected = {name: 0.5 for name in SUPERCLASSES}
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps({"thresholds": expected}),
        encoding="utf-8",
    )

    assert _load_thresholds(path) == expected


def test_parse_args_accepts_thresholds_file():
    args = parse_args(
        [
            "--run-id",
            "abc",
            "--thresholds",
            "thresholds.json",
        ]
    )

    assert args.thresholds == "thresholds.json"

def test_select_thresholds_rejects_non_binary_targets():
    targets = np.zeros((2, len(SUPERCLASSES)), dtype=np.float32)
    probabilities = np.full_like(targets, 0.5)
    targets[0, 0] = 2

    with pytest.raises(ValueError, match="0 hoặc 1"):
        select_thresholds(targets, probabilities)


@pytest.mark.parametrize(
    "invalid_probability",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_select_thresholds_rejects_invalid_probabilities(invalid_probability):
    targets = np.zeros((2, len(SUPERCLASSES)), dtype=np.float32)
    probabilities = np.full_like(targets, 0.5)
    probabilities[0, 0] = invalid_probability

    with pytest.raises(ValueError):
        select_thresholds(targets, probabilities)
