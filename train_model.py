"""
train_model.py — Fraud Detection Model Training Script
FraudLens | Fraud Intelligence Platform
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

FEATURE_COLS = [
    "amt", "amt_log", "amt_per_city_pop",
    "city_pop", "distance",
    "hour", "day_of_week", "month",
    "age", "gender_enc", "category_enc",
    "lat", "long", "merch_lat", "merch_long"
]

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


def select_threshold(y_true, y_proba):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.951, 0.01):
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
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
    auc = roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Held-Out Test Set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)


def plot_pr(y_true, y_proba, label, ax):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ax.plot(recall, precision, label=f"{label} (AP={ap:.4f})")
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
            j, i, f"{v:,}",
            ha="center", va="center",
            color="white" if v > cm.max() / 2 else "black"
        )


def main():
    print("=" * 64)
    print("  FraudLens — Fraud Detection Model Training (time-based split)")
    print("=" * 64)

    if not os.path.exists(DATA_TRAIN):
        raise FileNotFoundError(
            f"Training data not found: {DATA_TRAIN}\n"
            "Place fraudTrain.csv next to this script. fraudTest.csv is "
            "reserved strictly as the final held-out test set."
        )
    if not os.path.exists(DATA_TEST):
        raise FileNotFoundError(f"Test data not found: {DATA_TEST}")

    # ── Load full training data (no subsampling) ──────────────────
    print(f"\nLoading training data: {DATA_TRAIN} (full dataset)...")
    df_train_full = pd.read_csv(DATA_TRAIN)
    print(
        f"  Loaded: {len(df_train_full):,} rows | "
        f"fraud rate: {df_train_full['is_fraud'].mean() * 100:.3f}%"
    )

    # ── Time-based sort and split (never random) ──────────────────
    df_train_full = df_train_full.sort_values(
        "trans_date_trans_time"
    ).reset_index(drop=True)
    n = len(df_train_full)
    cut = int(round(0.85 * n))
    df_train = df_train_full.iloc[:cut].copy()
    df_val = df_train_full.iloc[cut:].copy()
    print(f"\nTime split (85/15) on sorted trans_date_trans_time:")
    print(
        f"  Train:   {len(df_train):,} rows | "
        f"{df_train['trans_date_trans_time'].min()} -> "
        f"{df_train['trans_date_trans_time'].max()}"
    )
    print(
        f"  Val:     {len(df_val):,} rows | "
        f"{df_val['trans_date_trans_time'].min()} -> "
        f"{df_val['trans_date_trans_time'].max()}"
    )

    # ── Held-out test data ────────────────────────────────────────
    print(f"\nLoading held-out test data: {DATA_TEST} ...")
    df_test = pd.read_csv(DATA_TEST)
    print(
        f"  Loaded: {len(df_test):,} rows | "
        f"fraud rate: {df_test['is_fraud'].mean() * 100:.3f}%"
    )

    # ── Feature engineering (fit on train only) ───────────────────
    print("\nEngineering features (fit on train only)...")
    df_train, le_category = engineer_features(df_train, fit=True)
    df_val, _ = engineer_features(df_val, le_category=le_category, fit=False)
    df_test, _ = engineer_features(df_test, le_category=le_category, fit=False)

    X_train = df_train[FEATURE_COLS]
    y_train = df_train["is_fraud"]
    X_val = df_val[FEATURE_COLS]
    y_val = df_val["is_fraud"]
    X_test = df_test[FEATURE_COLS]
    y_test = df_test["is_fraud"]

    print(f"  Features: {FEATURE_COLS}")
    print(
        f"  Train fraud rate: {y_train.mean() * 100:.3f}% | "
        f"Val: {y_val.mean() * 100:.3f}% | "
        f"Test: {y_test.mean() * 100:.3f}%"
    )

    results = {"baseline": {}, "current": {}, "splits": {}}

    # ── Baseline: evaluate existing saved models on held-out test ──
    baseline_rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    baseline_xgb_path = os.path.join(MODELS_DIR, "xgboost.joblib")
    baseline_models = {}
    if os.path.exists(baseline_rf_path):
        baseline_models["random_forest"] = joblib.load(baseline_rf_path)
    if os.path.exists(baseline_xgb_path):
        baseline_models["xgboost"] = joblib.load(baseline_xgb_path)

    if baseline_models:
        print("\n── Baseline (existing models) on held-out test set ──────────────")
        for name, model in baseline_models.items():
            proba = model.predict_proba(X_test)[:, 1]
            m = compute_metrics(y_test.values, proba, 0.5)
            results["baseline"][name] = m
            print(
                f"  {name}: ROC-AUC={m['roc_auc']:.4f} "
                f"PR-AUC={m['pr_auc']:.4f} "
                f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
            )

    # ── Train new models on time-based train split ─────────────────
    print("\n── Training new models on time-based train split ────────────────")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_val_proba = rf.predict_proba(X_val)[:, 1]
    rf_thr, rf_val_f1 = select_threshold(y_val.values, rf_val_proba)
    print(
        f"  Random Forest: best val threshold={rf_thr:.2f} "
        f"(val F1={rf_val_f1:.4f})"
    )

    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=fraud_ratio,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
    xgb_val_proba = xgb.predict_proba(X_val)[:, 1]
    xgb_thr, xgb_val_f1 = select_threshold(y_val.values, xgb_val_proba)
    print(
        f"  XGBoost: best val threshold={xgb_thr:.2f} "
        f"(val F1={xgb_val_f1:.4f}) | best_iteration={xgb.best_iteration}"
    )

    # ── Evaluate new models on held-out test set ───────────────────
    print("\n── New models on held-out test set ──────────────────────────────")
    for name, model, thr in [
        ("random_forest", rf, rf_thr),
        ("xgboost", xgb, xgb_thr),
    ]:
        proba = model.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test.values, proba, thr)
        results["current"][name] = m
        cm = m["confusion_matrix"]
        print(f"  {name} (threshold={thr:.2f}):")
        print(f"    ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f}")
        print(
            f"    Precision={m['precision']:.4f} | "
            f"Recall={m['recall']:.4f} | F1={m['f1']:.4f}"
        )
        print(
            f"    Confusion matrix: TN={cm['tn']:,} FP={cm['fp']:,} "
            f"FN={cm['fn']:,} TP={cm['tp']:,}"
        )

    # ── Save models & artifacts ────────────────────────────────────
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost.joblib"))
    joblib.dump(le_category, os.path.join(MODELS_DIR, "le_category.joblib"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_cols.joblib"))

    results["splits"] = {
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_fraud_rate": float(y_train.mean()),
        "validation_fraud_rate": float(y_val.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "train_start": str(df_train["trans_date_trans_time"].min()),
        "train_end": str(df_train["trans_date_trans_time"].max()),
        "val_start": str(df_val["trans_date_trans_time"].min()),
        "val_end": str(df_val["trans_date_trans_time"].max()),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics saved to {os.path.join(MODELS_DIR, 'metrics.json')}")

    # ── Plots to docs/ ─────────────────────────────────────────────
    rf_proba = rf.predict_proba(X_test)[:, 1]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_roc(y_test.values, rf_proba, "Random Forest", ax)
    plot_roc(y_test.values, xgb_proba, "XGBoost", ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pr(y_test.values, rf_proba, "Random Forest", ax)
    plot_pr(y_test.values, xgb_proba, "XGBoost", ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "pr_curve.png"), dpi=150)
    plt.close(fig)

    cm_rf = np.array([
        [
            results["current"]["random_forest"]["confusion_matrix"]["tn"],
            results["current"]["random_forest"]["confusion_matrix"]["fp"],
        ],
        [
            results["current"]["random_forest"]["confusion_matrix"]["fn"],
            results["current"]["random_forest"]["confusion_matrix"]["tp"],
        ],
    ])
    cm_xgb = np.array([
        [
            results["current"]["xgboost"]["confusion_matrix"]["tn"],
            results["current"]["xgboost"]["confusion_matrix"]["fp"],
        ],
        [
            results["current"]["xgboost"]["confusion_matrix"]["fn"],
            results["current"]["xgboost"]["confusion_matrix"]["tp"],
        ],
    ])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    plot_confusion(cm_rf, "Random Forest", axes[0])
    plot_confusion(cm_xgb, "XGBoost", axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(
        f"  Plots saved to {DOCS_DIR}/ "
        f"(roc_curve.png, pr_curve.png, confusion_matrix.png)"
    )

    # ── Results table ──────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  RESULTS TABLE — Held-Out Test Set (fraudTest.csv)")
    print("=" * 64)
    header = (
        f"{'Model':<18} {'Threshold':>10} {'ROC-AUC':>9} "
        f"{'PR-AUC':>9} {'Precision':>10} {'Recall':>9} {'F1':>8}"
    )
    print(header)
    print("-" * len(header))
    for section in ["baseline", "current"]:
        for name, m in results[section].items():
            print(
                f"{section.upper()}/{name:<12} {m['threshold']:>10.2f} "
                f"{m['roc_auc']:>9.4f} {m['pr_auc']:>9.4f} "
                f"{m['precision']:>10.4f} {m['recall']:>9.4f} {m['f1']:>8.4f}"
            )
    print("-" * len(header))
    print("\nNote: Accuracy is not reported as a headline metric (imbalanced data).")
    print("=" * 64)
    print("  Training complete! Start the app with:")
    print("  python3 app.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
