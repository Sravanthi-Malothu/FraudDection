import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.evaluate import compute_metrics, find_optimal_threshold
from src.ensemble import AveragedEnsemble


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


def test_averaged_ensemble_probabilities_and_predictions():
    # Create two dummy estimators with predictable output
    class MockModelA:
        def predict_proba(self, X):
            # Model A predicts 0.8 for all rows
            p1 = np.full(len(X), 0.8)
            return np.column_stack([1.0 - p1, p1])

    class MockModelB:
        def predict_proba(self, X):
            # Model B predicts 0.4 for all rows
            p1 = np.full(len(X), 0.4)
            return np.column_stack([1.0 - p1, p1])

    ensemble = AveragedEnsemble(
        model_names=["model_a", "model_b"],
        models={"model_a": MockModelA(), "model_b": MockModelB()},
        optimal_threshold=0.55,
    )

    X = np.zeros((4, 5))
    probas = ensemble.predict_proba(X)
    assert probas.shape == (4, 2)
    # Average of 0.8 and 0.4 is 0.6
    np.testing.assert_allclose(probas[:, 1], 0.6)
    np.testing.assert_allclose(probas[:, 0], 0.4)

    # 0.6 >= optimal_threshold (0.55) -> 1
    preds = ensemble.predict(X)
    assert np.all(preds == 1)

    # With high threshold 0.75 -> 0
    preds_high = ensemble.predict(X, threshold=0.75)
    assert np.all(preds_high == 0)
