import numpy as np
from src.evaluate import compute_metrics, find_optimal_threshold


def test_compute_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.85])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_find_optimal_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.8])
    best_thresh, best_f1 = find_optimal_threshold(y_true, y_proba)
    assert 0.2 < best_thresh < 0.7
    assert best_f1 == 1.0
