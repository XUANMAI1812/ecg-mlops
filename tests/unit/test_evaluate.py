import numpy as np
import pytest

from ecg.data.labels import SUPERCLASSES
from ecg.train.evaluate import evaluate_predictions, parse_args


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
    assert result["confusion_matrix"]["NORM"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["confusion_matrix"]["MI"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert result["confusion_matrix"]["HYP"] == {"tn": 4, "fp": 0, "fn": 0, "tp": 0}
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
