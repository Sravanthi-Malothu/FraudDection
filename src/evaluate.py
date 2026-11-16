"""
evaluate.py — Model Evaluation & Metric Computation
Calculates ROC-AUC, PR-AUC, Precision, Recall, F1-Score, and Confusion Matrices.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_proba, threshold=0.5):
    """
    Computes comprehensive binary classification evaluation metrics.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float((tn + tp) / total) if total else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "support_positive": int((y_true == 1).sum()),
        "support_negative": int((y_true == 0).sum()),
    }


def find_optimal_threshold(y_true, y_proba, steps=100):
    """
    Finds the probability decision threshold that maximizes F1 score.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    best_thresh = 0.5
    best_f1 = -1.0

    for thresh in np.linspace(0.05, 0.95, steps):
        y_pred = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    return best_thresh, best_f1
