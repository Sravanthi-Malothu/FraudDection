"""
ensemble.py — Soft-Voting / Probability-Averaged Ensemble
Provides the AveragedEnsemble class for combining top-performing fraud detection models.
"""

import numpy as np


class AveragedEnsemble:
    """
    Soft-voting / probability-averaged ensemble of the top two models.
    Supports seamless prediction with or without feature scaling.
    """

    def __init__(self, model_names, models, scaler=None, optimal_threshold=0.5):
        self.model_names = model_names
        self.models = models  # dict: {name: fitted_model}
        self.scaler = scaler
        self.optimal_threshold = optimal_threshold

    def predict_proba(self, X):
        """Returns array of shape (N, 2) with [P(legit), P(fraud)]."""
        probas = []
        for name in self.model_names:
            model = self.models[name]
            if name == "logistic_regression" and self.scaler is not None:
                X_in = self.scaler.transform(X)
            else:
                X_in = X
            prob = model.predict_proba(X_in)[:, 1]
            probas.append(prob)
        avg_prob = np.mean(probas, axis=0)
        return np.column_stack([1.0 - avg_prob, avg_prob])

    def predict(self, X, threshold=None):
        thr = threshold if threshold is not None else self.optimal_threshold
        proba = self.predict_proba(X)[:, 1]
        return (proba >= thr).astype(int)
