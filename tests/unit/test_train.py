import numpy as np
import pytest
import torch

from ecg.data.labels import SUPERCLASSES
from ecg.models.train import calculate_metrics, calculate_pos_weight


def test_calculate_pos_weight_uses_positive_negative_counts():
    targets = np.zeros((10, len(SUPERCLASSES)), dtype=np.float32)
    targets[:2, 0] = 1.0
    targets[:5, 1] = 1.0

    weights = calculate_pos_weight(targets)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (len(SUPERCLASSES),)
    assert weights[0].item() == pytest.approx(4.0)
    assert weights[1].item() == pytest.approx(1.0)
    assert torch.all(weights[2:] == 1.0)


def test_calculate_pos_weight_rejects_empty_targets():
    with pytest.raises(ValueError, match="rỗng"):
        calculate_pos_weight(np.empty((0, len(SUPERCLASSES))))


def test_calculate_pos_weight_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        calculate_pos_weight(np.zeros((4, 3)))


def test_calculate_metrics_preserves_f1_and_adds_auc():
    targets = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    probabilities = np.array(
        [
            [0.1, 0.2, 0.1, 0.1, 0.1],
            [0.2, 0.8, 0.1, 0.1, 0.1],
            [0.7, 0.3, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    metrics = calculate_metrics(targets, probabilities)

    assert "macro_f1" in metrics
    assert "per_class_f1" in metrics
    assert metrics["macro_auc"] == pytest.approx(1.0)
    assert metrics["micro_auc"] == pytest.approx(1.0)
    assert set(metrics["per_class_auc"]) == {
        SUPERCLASSES[0],
        SUPERCLASSES[1],
    }


def test_calculate_metrics_handles_classes_with_one_target_value():
    targets = np.zeros((4, len(SUPERCLASSES)), dtype=np.float32)
    targets[:, 0] = [0, 1, 0, 1]
    probabilities = np.full_like(targets, 0.2)
    probabilities[:, 0] = [0.1, 0.8, 0.2, 0.9]

    metrics = calculate_metrics(targets, probabilities)

    assert SUPERCLASSES[0] in metrics["per_class_auc"]
    assert len(metrics["per_class_auc"]) == 1
    assert not np.isnan(metrics["macro_auc"])
    assert not np.isnan(metrics["micro_auc"])


def test_calculate_pos_weight_uses_one_when_class_has_no_positives():
    targets = np.zeros((4, len(SUPERCLASSES)), dtype=np.float32)
    targets[:, 0] = 1.0

    weights = calculate_pos_weight(targets)

    assert weights[0].item() == pytest.approx(0.0)
    assert torch.all(weights[1:] == 1.0)


def test_calculate_metrics_returns_nan_auc_when_all_targets_are_zero():
    targets = np.zeros((4, len(SUPERCLASSES)), dtype=np.float32)
    probabilities = np.full_like(targets, 0.2)

    metrics = calculate_metrics(targets, probabilities)

    assert np.isnan(metrics["macro_auc"])
    assert np.isnan(metrics["micro_auc"])
    assert metrics["per_class_auc"] == {}
