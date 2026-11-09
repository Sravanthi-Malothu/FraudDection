"""
train_model.py — Fraud Detection Model Training Script
FraudLens | Fraud Intelligence Platform

Leakage-safe evaluation:
  * Trains on fraudTrain.csv (full dataset, no subsampling).
  * fraudTest.csv is reserved strictly as the final held-out test set.
  * Time-based 85/15 train/validation split on trans_date_trans_time (never random).
  * Per-card behavioural features use only PAST transactions
    (shift/rolling with closed='left' — current row never included).
  * Test-set features are computed on train+test history so card history
    carries over without leaking future data.
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_TRAIN = os.path.join(BASE_DIR, "fraudTrain.csv")
DATA_TEST = os.path.join(BASE_DIR, "fraudTest.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# ── Base (original) feature set ────────────────────────────────────────
BASE_COLS = [
    "amt", "amt_log", "amt_per_city_pop",
    "city_pop", "distance",
    "hour", "day_of_week", "month",
    "age", "gender_enc", "category_enc",
    "lat", "long", "merch_lat", "merch_long",
]

# ── Leakage-safe per-card behavioural features (past only) ─────────────
BEHAVIORAL_COLS = [
    "txn_count_1h", "txn_count_24h",
    "amt_mean_card_past", "amt_std_card_past", "amt_zscore",
    "time_since_last_txn", "distance_from_last_txn_km",
    "is_new_merchant_for_card", "is_new_category_for_card",
]

FEATURE_COLS = BASE_COLS + BEHAVIORAL_COLS

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


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
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"] = df["trans_date_trans_time"].dt.month

    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    ref_date = pd.Timestamp("2020-01-01")
    df["age"] = ((ref_date - df["dob"]).dt.days / 365.25).fillna(35)

    df["amt_log"] = np.log1p(df["amt"])
    df["amt_per_city_pop"] = df["amt"] / (df["city_pop"] + 1)

    df["distance"] = haversine(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values
    )

    df["gender_enc"] = (df["gender"] == "M").astype(int)

    if le_category is None:
        le_category = LabelEncoder()
    if fit:
        df["category_enc"] = le_category.fit_transform(df["category"].astype(str))
    else:
        known = set(le_category.classes_)
        df["category_safe"] = df["category"].apply(
            lambda x: x if x in known else le_category.classes_[0]
        )
        df["category_enc"] = le_category.transform(df["category_safe"].astype(str))

    return df, le_category


def add_behavioral_features(df):
    """Add per-card behavioural features that only use PAST transactions.

    The frame is sorted by (cc_num, trans_date_trans_time) so that every
    groupby/shift/rolling looks strictly backward in time. The original row
    order is restored before returning so columns align with base features.
    """
    df = df.copy()
    if "__rid__" not in df.columns:
        df["__rid__"] = np.arange(len(df))
    order = df["__rid__"].values

    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    # ── New merchant / new category for this card (past-only) ──────────
    df["is_new_merchant_for_card"] = (
        ~df.groupby("cc_num")["merchant"].transform("duplicated")
    ).astype(int)
    df["is_new_category_for_card"] = (
        ~df.groupby("cc_num")["category"].transform("duplicated")
    ).astype(int)

    # ── Time since last txn (seconds) ──────────────────────────────────
    df["time_since_last_txn"] = (
        df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
    )

    # ── Distance from last txn (km): current vs previous merchant loc ──
    prev_lat = df.groupby("cc_num")["merch_lat"].shift(1)
    prev_long = df.groupby("cc_num")["merch_long"].shift(1)
    df["distance_from_last_txn_km"] = haversine(
        df["lat"].values, df["long"].values, prev_lat.values, prev_long.values
    )

    # ── Time-windowed txn counts + all-past amount stats (per card) ────
    count_1h, count_24h, amt_mean, amt_std = [], [], [], []
    for _, g in df.groupby("cc_num", sort=False):
        g = g.sort_values("trans_date_trans_time")
        g_idx = g.set_index("trans_date_trans_time")
        count_1h.append(g_idx["amt"].rolling("1h", closed="left").count().values)
        count_24h.append(
            g_idx["amt"].rolling("24h", closed="left").count().values
        )
        amt_mean.append(g_idx["amt"].expanding().mean().shift(1).values)
        amt_std.append(g_idx["amt"].expanding().std().shift(1).values)

    df["txn_count_1h"] = np.concatenate(count_1h)
    df["txn_count_24h"] = np.concatenate(count_24h)
    df["amt_mean_card_past"] = np.concatenate(amt_mean)
    df["amt_std_card_past"] = np.concatenate(amt_std)

    # ── amt_zscore: current amt vs card's past average/std ─────────────
    df["amt_zscore"] = (df["amt"] - df["amt_mean_card_past"]) / df[
        "amt_std_card_past"
    ].replace(0, np.nan)

    # ── Restore original row order ─────────────────────────────────────
    df = df.sort_values("__rid__").reset_index(drop=True).drop(columns="__rid__")

    # ── NaN guards (first txn of a card has no past) ───────────────────
    df["txn_count_1h"] = df["txn_count_1h"].fillna(0)
    df["txn_count_24h"] = df["txn_count_24h"].fillna(0)
    df["amt_mean_card_past"] = df["amt_mean_card_past"].fillna(0)
    df["amt_std_card_past"] = df["amt_std_card_past"].fillna(1)
    df["amt_zscore"] = df["amt_zscore"].fillna(0)
    df["time_since_last_txn"] = df["time_since_last_txn"].fillna(-1)
    df["distance_from_last_txn_km"] = df["distance_from_last_txn_km"].fillna(-1)

    return df


def select_threshold(y_true, y_proba):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.951, 0.01):
        f1 = f1_score(y_true, (y_proba >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "support": int((y_true == 1).sum()),
    }


def plot_roc(y_true, y_proba, label, ax):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, y_proba):.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Held-Out Test Set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)


def plot_pr(y_true, y_proba, label, ax):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax.plot(
        recall,
        precision,
        label=f"{label} (AP={average_precision_score(y_true, y_proba):.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Held-Out Test Set")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)


def plot_confusion(cm, label, ax):
    tick_labels = ["Legitimate", "Fraud"]
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix — {label}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{v:,}",
            ha="center",
            va="center",
            color="white" if v > cm.max() / 2 else "black",
        )


def train_xgb(Xtr, ytr, Xval, yval, fraud_ratio):
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=fraud_ratio,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xval, yval)],
        early_stopping_rounds=50,
        verbose=False,
    )
    val_proba = model.predict_proba(Xval)[:, 1]
    thr, _ = select_threshold(yval.values, val_proba)
    return model, thr, int(model.best_iteration)


def main():
    print("=" * 66)
    print("  FraudLens — Fraud Detection Training (leakage-safe, time-based)")
    print("=" * 66)

    if not os.path.exists(DATA_TRAIN):
        raise FileNotFoundError(
            f"Training data not found: {DATA_TRAIN}\n"
            "Place fraudTrain.csv next to this script. fraudTest.csv is "
            "reserved strictly as the final held-out test set."
        )
    if not os.path.exists(DATA_TEST):
        raise FileNotFoundError(f"Test data not found: {DATA_TEST}")

    # ── Load full training data (no subsampling) ───────────────────────
    print(f"\nLoading training data: {DATA_TRAIN} (full dataset)...")
    df_train_full = pd.read_csv(DATA_TRAIN)
    df_train_full = df_train_full.sort_values(
        "trans_date_trans_time"
    ).reset_index(drop=True)
    n = len(df_train_full)
    cut = int(round(0.85 * n))
    df_train = df_train_full.iloc[:cut].copy()
    df_val = df_train_full.iloc[cut:].copy()
    print(f"  Loaded: {n:,} rows | fraud rate: {df_train_full['is_fraud'].mean()*100:.3f}%")
    print(
        f"  Time split 85/15 on trans_date_trans_time: train={len(df_train):,} "
        f"val={len(df_val):,}"
    )

    # ── Held-out test data ─────────────────────────────────────────────
    print(f"\nLoading held-out test data: {DATA_TEST} ...")
    df_test = pd.read_csv(DATA_TEST)
    print(
        f"  Loaded: {len(df_test):,} rows | fraud rate: {df_test['is_fraud'].mean()*100:.3f}%"
    )

    # ── Base features (existing, fit on train only) ────────────────────
    print("\nEngineering base features (fit on train only)...")
    df_train_b, le_category = engineer_features(df_train, fit=True)
    df_val_b, _ = engineer_features(df_val, le_category=le_category, fit=False)
    df_test_b, _ = engineer_features(df_test, le_category=le_category, fit=False)

    # ── Behavioural features (history carries over, no future leakage) ──
    print("Engineering behavioural features (past-only, leakage-safe)...")

    # Train: history from train alone.
    df_train_bhv = add_behavioral_features(df_train.copy())

    # Val: concatenate train+val (train is entirely earlier in time within
    # fraudTrain, so val rows sit at the end after order restore).
    df_v = pd.concat([df_train, df_val], ignore_index=True)
    df_v = add_behavioral_features(df_v)
    df_val_bhv = df_v.iloc[len(df_train):].reset_index(drop=True)

    # Test: concatenate train+test so card history carries over.
    df_t = pd.concat([df_train, df_test], ignore_index=True)
    df_t = add_behavioral_features(df_t)
    df_test_bhv = df_t.iloc[len(df_train):].reset_index(drop=True)

    # ── Combine base + behavioural ─────────────────────────────────────
    df_train = pd.concat(
        [df_train_b.reset_index(drop=True), df_train_bhv[BEHAVIORAL_COLS].reset_index(drop=True)],
        axis=1,
    )
    df_val = pd.concat(
        [df_val_b.reset_index(drop=True), df_val_bhv[BEHAVIORAL_COLS].reset_index(drop=True)],
        axis=1,
    )
    df_test = pd.concat(
        [df_test_b.reset_index(drop=True), df_test_bhv[BEHAVIORAL_COLS].reset_index(drop=True)],
        axis=1,
    )

    X_train = df_train[FEATURE_COLS]
    y_train = df_train["is_fraud"]
    X_val = df_val[FEATURE_COLS]
    y_val = df_val["is_fraud"]
    X_test = df_test[FEATURE_COLS]
    y_test = df_test["is_fraud"]

    X_train_base = df_train[BASE_COLS]
    X_val_base = df_val[BASE_COLS]
    X_test_base = df_test[BASE_COLS]

    print(f"  Features: {len(FEATURE_COLS)} total "
        f"({len(BASE_COLS)} base + {len(BEHAVIORAL_COLS)} behavioural)")
    print(
        f"  Train fraud rate: {y_train.mean()*100:.3f}% | "
        f"Val: {y_val.mean()*100:.3f}% | Test: {y_test.mean()*100:.3f}%"
    )

    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    results = {
        "splits": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "train_fraud_rate": float(y_train.mean()),
            "validation_fraud_rate": float(y_val.mean()),
            "test_fraud_rate": float(y_test.mean()),
        },
        "ablation": {},
        "current": {},
    }

    # ── Ablation: XGBoost baseline features vs baseline + behavioural ──
    print("\n── XGBoost ablation: baseline features vs baseline + behavioural ──")

    print("\n[1/2] XGBoost — BASELINE features only ...")
    xgb_base, thr_base, base_iter = train_xgb(
        X_train_base, y_train, X_val_base, y_val, fraud_ratio
    )
    base_proba = xgb_base.predict_proba(X_test_base)[:, 1]
    base_metrics = compute_metrics(y_test.values, base_proba, thr_base)
    results["ablation"]["xgb_baseline_features"] = base_metrics
    print(
        f"  best_iteration={base_iter} threshold={thr_base:.2f} | "
        f"PR-AUC={base_metrics['pr_auc']:.4f} "
        f"ROC-AUC={base_metrics['roc_auc']:.4f} "
        f"P={base_metrics['precision']:.4f} R={base_metrics['recall']:.4f} "
        f"F1={base_metrics['f1']:.4f}"
    )

    print("\n[2/2] XGBoost — BASELINE + BEHAVIOURAL features ...")
    xgb_full, thr_full, full_iter = train_xgb(
        X_train, y_train, X_val, y_val, fraud_ratio
    )
    full_proba = xgb_full.predict_proba(X_test)[:, 1]
    full_metrics = compute_metrics(y_test.values, full_proba, thr_full)
    results["ablation"]["xgb_baseline_plus_behavioural"] = full_metrics
    results["current"]["xgboost"] = full_metrics
    print(
        f"  best_iteration={full_iter} threshold={thr_full:.2f} | "
        f"PR-AUC={full_metrics['pr_auc']:.4f} "
        f"ROC-AUC={full_metrics['roc_auc']:.4f} "
        f"P={full_metrics['precision']:.4f} R={full_metrics['recall']:.4f} "
        f"F1={full_metrics['f1']:.4f}"
    )

    # ── Random Forest on full feature set (for app compatibility) ─────
    print("\nTraining Random Forest on full feature set ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    print("  ✓ Random Forest trained")

    # ── Save models & artifacts ────────────────────────────────────────
    joblib.dump(xgb_full, os.path.join(MODELS_DIR, "xgboost.joblib"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(le_category, os.path.join(MODELS_DIR, "le_category.joblib"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_cols.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Models + metrics saved to {MODELS_DIR}/")

    # ── Plots to docs/ (chosen model = XGB + behavioural) ──────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_roc(y_test.values, full_proba, "XGBoost (+behavioural)", ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pr(y_test.values, full_proba, "XGBoost (+behavioural)", ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "pr_curve.png"), dpi=150)
    plt.close(fig)

    cm = np.array(
        [
            [
                full_metrics["confusion_matrix"]["tn"],
                full_metrics["confusion_matrix"]["fp"],
            ],
            [
                full_metrics["confusion_matrix"]["fn"],
                full_metrics["confusion_matrix"]["tp"],
            ],
        ]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_confusion(cm, "XGBoost (+behavioural)", ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"  Plots saved to {DOCS_DIR}/ (roc_curve.png, pr_curve.png, confusion_matrix.png)")

    # ── Results / ablation table ───────────────────────────────────────
    print("\n" + "=" * 66)
    print("  ABLATION TABLE — Held-Out Test Set (fraudTest.csv)")
    print("=" * 66)
    header = (
        f"{'Model':<34} {'PR-AUC':>9} {'ROC-AUC':>9} "
        f"{'Precision':>10} {'Recall':>9} {'F1':>8}"
    )
    print(header)
    print("-" * len(header))
    for label, key in [
        ("XGBoost — baseline features", "xgb_baseline_features"),
        ("XGBoost — baseline + behavioural", "xgb_baseline_plus_behavioural"),
    ]:
        m = results["ablation"][key]
        print(
            f"{label:<34} {m['pr_auc']:>9.4f} {m['roc_auc']:>9.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>9.4f} {m['f1']:>8.4f}"
        )
    print("-" * len(header))
    print("Note: Accuracy is not reported as a headline metric (imbalanced data).")
    print("=" * 66)
    print("  Training complete! Start the app with:")
    print("  python3 app.py")
    print("=" * 66)


if __name__ == "__main__":
    main()
