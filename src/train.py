"""
train.py — Train Fraud Detection Models
Trains Random Forest and XGBoost classifiers on the transaction dataset.
Saves serialized model artifacts and metrics to the models/ directory.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features import engineer_features, FEATURE_COLS
from src.evaluate import compute_metrics, find_optimal_threshold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def load_dataset(max_rows=100000):
    """
    Loads dataset from data/ directory using available CSV files.
    Prefers fraudTrain.csv if present, otherwise uses fraudTest.csv.
    """
    train_path = DATA_DIR / "fraudTrain.csv"
    test_path = DATA_DIR / "fraudTest.csv"

    if train_path.exists():
        data_file = train_path
    elif test_path.exists():
        data_file = test_path
    else:
        raise FileNotFoundError(
            f"No dataset found in {DATA_DIR}. Please place fraudTest.csv in the data/ folder."
        )

    print(f"[Train] Loading up to {max_rows:,} rows from {data_file.name}...")
    df = pd.read_csv(data_file, nrows=max_rows)
    return df


def train_models(max_rows=100000, random_state=42):
    """
    End-to-end training pipeline for Random Forest and XGBoost models.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataset(max_rows=max_rows)
    print(f"[Train] Raw data shape: {df_raw.shape}, fraud rate: {df_raw['is_fraud'].mean()*100:.3f}%")

    print("[Train] Engineering features...")
    df_feat, le_category = engineer_features(df_raw, fit=True)

    X = df_feat[FEATURE_COLS]
    y = df_feat["is_fraud"].astype(int)

    # Time-ordered split if sorted, or stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"[Train] Split complete: Train={len(X_train):,}, Test={len(X_test):,}")

    neg_pos_ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    print(f"[Train] Class imbalance ratio: {neg_pos_ratio:.1f}:1")

    # 1. Random Forest
    print("[Train] Training Random Forest Classifier (balanced weights)...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = compute_metrics(y_test, rf_proba)
    print(f"  RF Test ROC-AUC: {rf_metrics['roc_auc']:.4f} | F1: {rf_metrics['f1']:.4f}")

    # 2. XGBoost
    print("[Train] Training XGBoost Classifier (scale_pos_weight)...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=neg_pos_ratio,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_proba)
    print(f"  XGB Test ROC-AUC: {xgb_metrics['roc_auc']:.4f} | F1: {xgb_metrics['f1']:.4f}")

    # 3. Ensemble
    ensemble_proba = (rf_proba + xgb_proba) / 2.0
    ensemble_metrics = compute_metrics(y_test, ensemble_proba)
    print(f"  Ensemble Test ROC-AUC: {ensemble_metrics['roc_auc']:.4f} | F1: {ensemble_metrics['f1']:.4f}")

    # Save artifacts
    print(f"[Train] Saving artifacts to {MODELS_DIR}...")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")
    joblib.dump(le_category, MODELS_DIR / "le_category.joblib")
    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.joblib")

    all_metrics = {
        "rf": rf_metrics,
        "xgb": xgb_metrics,
        "ensemble": ensemble_metrics,
    }
    joblib.dump(all_metrics, MODELS_DIR / "metrics.joblib")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("[Train] Training successfully completed!")
    return all_metrics


if __name__ == "__main__":
    train_models()
