"""
train.py — Train Fraud Detection Models with Time-Based Split
Enforces chronological train/validation/test partitioning, avoids data leakage,
fits all encoders on the training partition only, and handles severe class imbalance.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features import engineer_features, FEATURE_COLS, IDENTIFIER_COLS
from src.evaluate import compute_metrics, find_optimal_threshold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def load_full_dataset(data_dir=DATA_DIR):
    """
    Loads both fraudTrain.csv and fraudTest.csv if available.
    Sorts strictly chronologically by transaction timestamp.
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "fraudTrain.csv"
    test_path = data_dir / "fraudTest.csv"

    loaded_dfs = []
    if train_path.exists():
        print(f"[Data Loading] Found training file: {train_path.name}")
        df_tr = pd.read_csv(train_path)
        loaded_dfs.append(df_tr)
    if test_path.exists():
        print(f"[Data Loading] Found test file: {test_path.name}")
        df_te = pd.read_csv(test_path)
        loaded_dfs.append(df_te)

    if not loaded_dfs:
        raise FileNotFoundError(
            f"No dataset files found in {data_dir}. Expected fraudTrain.csv and/or fraudTest.csv."
        )

    # 1. Combine datasets if multiple files present
    if len(loaded_dfs) > 1:
        print(f"[Data Loading] Concatenating {len(loaded_dfs)} datasets...")
        df = pd.concat(loaded_dfs, ignore_index=True)
    else:
        df = loaded_dfs[0]

    # Convert timestamps and sort chronologically
    print("[Data Loading] Parsing transaction timestamps and sorting chronologically...")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    print(
        f"[Data Loading] Full dataset loaded: {len(df):,} transactions from "
        f"{df['trans_date_trans_time'].min()} to {df['trans_date_trans_time'].max()}"
    )
    return df


def time_based_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Performs a strict chronological (time-based) split without random shuffling:
    First 70% -> Train, Next 15% -> Validation, Last 15% -> Test.
    Asserts that max(train timestamp) < min(test timestamp).
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    n = len(df)
    train_end = int(round(train_ratio * n))
    val_end = int(round((train_ratio + val_ratio) * n))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    max_train_ts = df_train["trans_date_trans_time"].max()
    min_val_ts = df_val["trans_date_trans_time"].min()
    max_val_ts = df_val["trans_date_trans_time"].max()
    min_test_ts = df_test["trans_date_trans_time"].min()

    print("\n" + "=" * 70)
    print("  CHRONOLOGICAL (TIME-BASED) SPLIT VERIFICATION")
    print("=" * 70)
    print(f"  Training Window:   {df_train['trans_date_trans_time'].min()}  -->  {max_train_ts}")
    print(f"  Validation Window: {min_val_ts}  -->  {max_val_ts}")
    print(f"  Test Window:       {min_test_ts}  -->  {df_test['trans_date_trans_time'].max()}")

    # 5. Assertion: Train timestamp strictly precedes Test timestamp
    assert max_train_ts < min_test_ts, (
        f"Data leakage detected! max(train timestamp)={max_train_ts} is not strictly less than "
        f"min(test timestamp)={min_test_ts}."
    )
    assert max_train_ts <= min_val_ts, "Train timestamps overlap with validation timestamps."
    assert max_val_ts <= min_test_ts, "Validation timestamps overlap with test timestamps."
    print("  ✓ Temporal Leakage Check Passed: max(train timestamp) < min(test timestamp)")
    print("=" * 70 + "\n")

    return df_train, df_val, df_test


def print_class_balance(split_name, y_series):
    """
    Prints total records, legitimate count, fraud count, fraud rate,
    and the class imbalance ratio for a given partition.
    """
    total = len(y_series)
    fraud_count = int((y_series == 1).sum())
    legit_count = int((y_series == 0).sum())
    fraud_rate = (fraud_count / total * 100) if total else 0.0
    imbalance_ratio = (legit_count / fraud_count) if fraud_count > 0 else float("inf")

    print(f"[{split_name} Class Balance]")
    print(f"  Total records:         {total:>10,}")
    print(f"  Legitimate txs (0):    {legit_count:>10,} ({100.0 - fraud_rate:6.2f}%)")
    print(f"  Fraudulent txs (1):    {fraud_count:>10,} ({fraud_rate:6.3f}%)")
    print(f"  Imbalance ratio:       {imbalance_ratio:.1f}:1 (legitimate per fraud transaction)\n")


def train_models(data_dir=DATA_DIR, models_dir=MODELS_DIR, random_state=42):
    """
    End-to-end model training workflow:
    1. Loads full dataset (both fraudTrain.csv and fraudTest.csv if available).
    2. Sorts by transaction timestamp and performs a 70/15/15 time-based split.
    3. Fits encoders strictly on the training set; applies to validation and test.
    4. Removes identifier columns that leak (card number, transaction ID, names).
    5. Validates temporal integrity via assertion.
    6. Prints class balance for each split.
    7. Trains Random Forest and XGBoost with class imbalance handling.
    8. Evaluates on validation and held-out test splits, saving model artifacts.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full dataset loading and timestamp sort
    df_full = load_full_dataset(data_dir=data_dir)

    # 2. Time-based split: 70% Train, 15% Validation, 15% Test
    df_train, df_val, df_test = time_based_split(
        df_full, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # 6. Print class balance for each split
    print_class_balance("Training Split (70%)", df_train["is_fraud"])
    print_class_balance("Validation Split (15%)", df_val["is_fraud"])
    print_class_balance("Held-Out Test Split (15%)", df_test["is_fraud"])

    # 3. Fit encoders on training set ONLY, then transform validation and test
    print("[Feature Engineering] Fitting category encoders on TRAINING set only...")
    df_train, le_category = engineer_features(df_train, fit=True)

    print("[Feature Engineering] Applying fitted encoders to Validation and Test sets...")
    df_val, _ = engineer_features(df_val, le_category=le_category, fit=False)
    df_test, _ = engineer_features(df_test, le_category=le_category, fit=False)

    # 4. Remove identifier columns that leak
    # Ensure raw identifiers are strictly excluded from the feature matrices
    for col in IDENTIFIER_COLS:
        assert col not in FEATURE_COLS, f"Leaking identifier column found in FEATURE_COLS: {col}"

    X_train = df_train[FEATURE_COLS]
    y_train = df_train["is_fraud"].astype(int)

    X_val = df_val[FEATURE_COLS]
    y_val = df_val["is_fraud"].astype(int)

    X_test = df_test[FEATURE_COLS]
    y_test = df_test["is_fraud"].astype(int)

    print(f"[Model Features] Using {len(FEATURE_COLS)} generalized features (all leaking IDs excluded):")
    print(f"  {FEATURE_COLS}\n")

    # Calculate class imbalance weighting for XGBoost
    neg_pos_ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    # Train Random Forest (class_weight='balanced')
    print("[Train] Training Random Forest Classifier (class_weight='balanced')...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Train XGBoost (scale_pos_weight=neg_pos_ratio)
    print(f"[Train] Training XGBoost Classifier (scale_pos_weight={neg_pos_ratio:.1f})...")
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

    # Evaluate on Validation Split
    print("\n" + "=" * 70)
    print("  VALIDATION SET PERFORMANCE (15%)")
    print("=" * 70)
    rf_val_proba = rf.predict_proba(X_val)[:, 1]
    xgb_val_proba = xgb.predict_proba(X_val)[:, 1]
    ens_val_proba = (rf_val_proba + xgb_val_proba) / 2.0

    best_thresh_rf, _ = find_optimal_threshold(y_val, rf_val_proba)
    best_thresh_xgb, _ = find_optimal_threshold(y_val, xgb_val_proba)
    best_thresh_ens, _ = find_optimal_threshold(y_val, ens_val_proba)

    rf_val_metrics = compute_metrics(y_val, rf_val_proba, threshold=best_thresh_rf)
    xgb_val_metrics = compute_metrics(y_val, xgb_val_proba, threshold=best_thresh_xgb)
    ens_val_metrics = compute_metrics(y_val, ens_val_proba, threshold=best_thresh_ens)

    print(f"  Random Forest : ROC-AUC={rf_val_metrics['roc_auc']:.4f} | PR-AUC={rf_val_metrics['pr_auc']:.4f} | F1={rf_val_metrics['f1']:.4f} (thr={best_thresh_rf:.2f})")
    print(f"  XGBoost       : ROC-AUC={xgb_val_metrics['roc_auc']:.4f} | PR-AUC={xgb_val_metrics['pr_auc']:.4f} | F1={xgb_val_metrics['f1']:.4f} (thr={best_thresh_xgb:.2f})")
    print(f"  Ensemble (Avg): ROC-AUC={ens_val_metrics['roc_auc']:.4f} | PR-AUC={ens_val_metrics['pr_auc']:.4f} | F1={ens_val_metrics['f1']:.4f} (thr={best_thresh_ens:.2f})")

    # Evaluate on Held-Out Test Split
    print("\n" + "=" * 70)
    print("  HELD-OUT TEST SET PERFORMANCE (15%)")
    print("=" * 70)
    rf_test_proba = rf.predict_proba(X_test)[:, 1]
    xgb_test_proba = xgb.predict_proba(X_test)[:, 1]
    ens_test_proba = (rf_test_proba + xgb_test_proba) / 2.0

    rf_test_metrics = compute_metrics(y_test, rf_test_proba, threshold=best_thresh_rf)
    xgb_test_metrics = compute_metrics(y_test, xgb_test_proba, threshold=best_thresh_xgb)
    ens_test_metrics = compute_metrics(y_test, ens_test_proba, threshold=best_thresh_ens)

    print(f"  Random Forest : ROC-AUC={rf_test_metrics['roc_auc']:.4f} | PR-AUC={rf_test_metrics['pr_auc']:.4f} | Recall={rf_test_metrics['recall']:.4f} | F1={rf_test_metrics['f1']:.4f}")
    print(f"  XGBoost       : ROC-AUC={xgb_test_metrics['roc_auc']:.4f} | PR-AUC={xgb_test_metrics['pr_auc']:.4f} | Recall={xgb_test_metrics['recall']:.4f} | F1={xgb_test_metrics['f1']:.4f}")
    print(f"  Ensemble (Avg): ROC-AUC={ens_test_metrics['roc_auc']:.4f} | PR-AUC={ens_test_metrics['pr_auc']:.4f} | Recall={ens_test_metrics['recall']:.4f} | F1={ens_test_metrics['f1']:.4f}")
    print("=" * 70 + "\n")

    # Save trained models and artifacts
    print(f"[Artifacts] Saving serialized models to {models_dir}...")
    joblib.dump(rf, models_dir / "random_forest.joblib")
    joblib.dump(xgb, models_dir / "xgboost.joblib")
    joblib.dump(le_category, models_dir / "le_category.joblib")
    joblib.dump(FEATURE_COLS, models_dir / "feature_cols.joblib")

    all_metrics = {
        "rf": rf_test_metrics,
        "xgb": xgb_test_metrics,
        "ensemble": ens_test_metrics,
        "validation": {
            "rf": rf_val_metrics,
            "xgb": xgb_val_metrics,
            "ensemble": ens_val_metrics,
        },
        "split_summary": {
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
            "train_fraud_rate": float(y_train.mean()),
            "val_fraud_rate": float(y_val.mean()),
            "test_fraud_rate": float(y_test.mean()),
        },
    }
    joblib.dump(all_metrics, models_dir / "metrics.joblib")

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("[Train] Training pipeline finished successfully.")
    return all_metrics


if __name__ == "__main__":
    train_models()
