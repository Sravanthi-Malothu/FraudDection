"""
train_model.py — Fraud Detection Model Training Script
FraudLens | Fraud Intelligence Platform
"""

import pandas as pd
import numpy as np
import joblib
import os
import math
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

DATA_PATH = "/Users/sravanthimalothu/Desktop/FraudDecetion/fraudTest.csv"
MODELS_DIR = "/Users/sravanthimalothu/Desktop/FraudDecetion/models"
os.makedirs(MODELS_DIR, exist_ok=True)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def engineer_features(df, le_category=None, fit=False):
    df = df.copy()

    # Parse datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"] = df["trans_date_trans_time"].dt.month

    # Age from dob
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    ref_date = pd.Timestamp("2020-01-01")
    df["age"] = ((ref_date - df["dob"]).dt.days / 365.25).fillna(35)

    # Amount features
    df["amt_log"] = np.log1p(df["amt"])
    df["amt_per_city_pop"] = df["amt"] / (df["city_pop"] + 1)

    # Distance merchant to customer
    df["distance"] = haversine(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values
    )

    # Gender encode
    df["gender_enc"] = (df["gender"] == "M").astype(int)

    # Category encode
    if le_category is None:
        le_category = LabelEncoder()
    if fit:
        df["category_enc"] = le_category.fit_transform(df["category"].astype(str))
    else:
        # Handle unseen categories
        known = set(le_category.classes_)
        df["category_safe"] = df["category"].apply(lambda x: x if x in known else le_category.classes_[0])
        df["category_enc"] = le_category.transform(df["category_safe"].astype(str))

    return df, le_category


FEATURE_COLS = [
    "amt", "amt_log", "amt_per_city_pop",
    "city_pop", "distance",
    "hour", "day_of_week", "month",
    "age", "gender_enc", "category_enc",
    "lat", "long", "merch_lat", "merch_long"
]


def main():
    print("=" * 60)
    print("  FraudLens — Fraud Detection Model Training")
    print("=" * 60)
    print(f"\nLoading dataset (100,000 rows)...")
    df = pd.read_csv(DATA_PATH, nrows=100000)
    print(f"  Loaded: {len(df):,} rows | Columns: {list(df.columns)}")
    print(f"  Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

    print("\nEngineering features...")
    df_feat, le_category = engineer_features(df, fit=True)

    X = df_feat[FEATURE_COLS]
    y = df_feat["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Random Forest ──────────────────────────────────────────
    print("\nTraining Random Forest (class_weight='balanced')...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    print(f"  Random Forest AUC: {rf_auc:.4f}")
    print(classification_report(y_test, rf_preds, digits=4))

    # ── XGBoost ────────────────────────────────────────────────
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nTraining XGBoost (scale_pos_weight={fraud_ratio:.1f})...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=fraud_ratio,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    print(f"  XGBoost AUC: {xgb_auc:.4f}")
    print(classification_report(y_test, xgb_preds, digits=4))

    # ── Save ───────────────────────────────────────────────────
    print("\nSaving models...")
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost.joblib"))
    joblib.dump(le_category, os.path.join(MODELS_DIR, "le_category.joblib"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_cols.joblib"))

    # Save metrics
    metrics = {
        "rf": {
            "accuracy": float((rf_preds == y_test).mean()),
            "precision": float(precision_score(y_test, rf_preds, zero_division=0)),
            "recall": float(recall_score(y_test, rf_preds, zero_division=0)),
            "f1": float(f1_score(y_test, rf_preds, zero_division=0)),
            "auc": float(rf_auc),
        },
        "xgb": {
            "accuracy": float((xgb_preds == y_test).mean()),
            "precision": float(precision_score(y_test, xgb_preds, zero_division=0)),
            "recall": float(recall_score(y_test, xgb_preds, zero_division=0)),
            "f1": float(f1_score(y_test, xgb_preds, zero_division=0)),
            "auc": float(xgb_auc),
        }
    }
    joblib.dump(metrics, os.path.join(MODELS_DIR, "metrics.joblib"))
    print("  ✓ Models saved to models/")

    print("\n" + "=" * 60)
    print("  Training complete! Start the app with:")
    print("  python3 app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
