"""
train_model.py — Fraud Detection Model Training Script
FraudLens | Fraud Intelligence Platform

Leakage-safe, rigorous evaluation:
  * Trains on fraudTrain.csv (full dataset, no subsampling).
  * fraudTest.csv is reserved strictly as the final held-out test set.
  * Time-based 85/15 train/validation split on trans_date_trans_time (never random).
  * Per-card behavioural features use only PAST transactions
    (shift / rolling(closed='left') — the current row is never included).
  * Test-set features are computed on train+test history so card history
    carries over without leaking future data.
  * Evaluates Random Forest, XGBoost (Optuna-tuned), and an RF+XGB average
    ensemble on the SAME test set; picks the final model by validation PR-AUC.
  * Only the final model is saved for the API.
  * Fairness/ablation: retrains the final model without gender and age.
"""

import json
import os
import warnings

warnings.filterwarnings("ignore")

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

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:  # pragma: no cover - fallback when optuna is absent
    HAS_OPTUNA = False

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
FAIRNESS_DROP_COLS = ["gender_enc", "age"]
FEATURE_COLS_NO_FAIR = [c for c in FEATURE_COLS if c not in FAIRNESS_DROP_COLS]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


class AverageEnsemble:
    """Averages predict_proba of two fitted estimators. Picklable by joblib."""

    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b

    def predict_proba(self, X):
        pa = self.model_a.predict_proba(X)[:, 1]
        pb = self.model_b.predict_proba(X)[:, 1]
        p = (pa + pb) / 2.0
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


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
    """Per-card behavioural features that only use PAST transactions.

    Sorts by (cc_num, trans_date_trans_time) so every groupby/shift/rolling
    looks strictly backward in time, then restores the original row order.
    """
    df = df.copy()
    if "__rid__" not in df.columns:
        df["__rid__"] = np.arange(len(df))
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    # New merchant / new category for this card (past-only first occurrence)
    df["is_new_merchant_for_card"] = (
        ~df.groupby("cc_num")["merchant"].transform("duplicated")
    ).astype(int)
    df["is_new_category_for_card"] = (
        ~df.groupby("cc_num")["category"].transform("duplicated")
    ).astype(int)

    # Time since last txn (seconds)
    df["time_since_last_txn"] = (
        df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
    )

    # Distance from last txn (km): current customer vs previous merchant loc
    prev_lat = df.groupby("cc_num")["merch_lat"].shift(1)
    prev_long = df.groupby("cc_num")["merch_long"].shift(1)
    df["distance_from_last_txn_km"] = haversine(
        df["lat"].values, df["long"].values, prev_lat.values, prev_long.values
    )

    # Time-windowed txn counts + expanding amount stats (per card, past only)
    count_1h, count_24h, amt_mean, amt_std = [], [], [], []
    for _, g in df.groupby("cc_num", sort=False):
        g = g.sort_values("trans_date_trans_time")
        g_idx = g.set_index("trans_date_trans_time")
        count_1h.append(g_idx["amt"].rolling("1h", closed="left").count().values)
        count_24h.append(g_idx["amt"].rolling("24h", closed="left").count().values)
        amt_mean.append(g_idx["amt"].expanding().mean().shift(1).values)
        amt_std.append(g_idx["amt"].expanding().std().shift(1).values)

    df["txn_count_1h"] = np.concatenate(count_1h)
    df["txn_count_24h"] = np.concatenate(count_24h)
    df["amt_mean_card_past"] = np.concatenate(amt_mean)
    df["amt_std_card_past"] = np.concatenate(amt_std)

    # amt_zscore: current amt vs card's past average/std
    df["amt_zscore"] = (df["amt"] - df["amt_mean_card_past"]) / df[
        "amt_std_card_past"
    ].replace(0, np.nan)

    # Restore original row order
    df = df.sort_values("__rid__").reset_index(drop=True).drop(columns="__rid__")

    # NaN guards (first txn of a card has no past)
    df["txn_count_1h"] = df["txn_count_1h"].fillna(0)
    df["txn_count_24h"] = df["txn_count_24h"].fillna(0)
    df["amt_mean_card_past"] = df["amt_mean_card_past"].fillna(0)
    df["amt_std_card_past"] = df["amt_std_card_past"].fillna(1)
    df["amt_zscore"] = df["amt_zscore"].fillna(0)
    df["time_since_last_txn"] = df["time_since_last_txn"].fillna(-1)
    df["distance_from_last_txn_km"] = df["distance_from_last_txn_km"].fillna(-1)

    return df


def build_history_frames(df_train, df_val, df_test):
    """Compute behavioural features with history carried over, no future leak.

    train  -> behavioural from train alone.
    val    -> concatenate train+val (train entirely earlier in time).
    test   -> concatenate train+test (train is the card's past history).
    """
    df_train_bhv = add_behavioral_features(df_train.copy())

    df_v = pd.concat([df_train, df_val], ignore_index=True)
    df_v = add_behavioral_features(df_v)
    df_val_bhv = df_v.iloc[len(df_train):].reset_index(drop=True)

    df_t = pd.concat([df_train, df_test], ignore_index=True)
    df_t = add_behavioral_features(df_t)
    df_test_bhv = df_t.iloc[len(df_train):].reset_index(drop=True)

    return df_train_bhv, df_val_bhv, df_test_bhv


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
    total = tn + fp + fn + tp
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((tn + tp) / total) if total else 0.0,
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
    ax.set_title("ROC Curve — Held-Out Test Set (final model)")
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
    ax.set_title("Precision-Recall Curve — Held-Out Test Set (final model)")
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
            color="white" if v > cm.max() / 2 else "black",
        )


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=40):
    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "scale_pos_weight": fraud_ratio,
            "use_label_encoder": False,
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "n_estimators": 2000,
        }
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=40,
            verbose=False,
        )
        val_proba = model.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val.values, val_proba))

    if HAS_OPTUNA:
        study = optuna.create_study(direction="maximize", seed=42)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_val_pr_auc = float(study.best_value)
        trials = [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ]
        method = "optuna"
    else:  # pragma: no cover - deterministic fallback
        rng = np.random.default_rng(42)
        best_params, best_val_pr_auc = None, -1.0
        trials = []
        for i in range(n_trials):
            params = {
                "max_depth": int(rng.integers(3, 11)),
                "learning_rate": float(rng.uniform(0.01, 0.2)),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                "gamma": float(rng.uniform(0.0, 5.0)),
                "min_child_weight": int(rng.integers(1, 11)),
                "reg_lambda": float(rng.uniform(0.001, 10.0)),
                "reg_alpha": float(rng.uniform(0.001, 10.0)),
                "scale_pos_weight": fraud_ratio,
                "use_label_encoder": False,
                "eval_metric": "auc",
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
                "n_estimators": 2000,
            }
            model = XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=40,
                verbose=False,
            )
            val_proba = model.predict_proba(X_val)[:, 1]
            val_pr_auc = float(average_precision_score(y_val.values, val_proba))
            trials.append({"number": i, "value": val_pr_auc, "params": params})
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_params = params
        study = None
        method = "random_search"

    return {
        "method": method,
        "n_trials": n_trials,
        "best_params": best_params,
        "best_val_pr_auc": best_val_pr_auc,
        "trials": trials,
        "study": study,
    }


def train_xgb_final(X_train, y_train, X_val, y_val, best_params):
    params = dict(best_params)
    params.pop("n_estimators", None)
    model = XGBClassifier(
        n_estimators=2000,
        early_stopping_rounds=None,
        **params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
    val_proba = model.predict_proba(X_val)[:, 1]
    thr, _ = select_threshold(y_val.values, val_proba)
    return model, thr, int(model.best_iteration)


def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_on_test(model, X_test, y_test, threshold):
    proba = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test.values, proba, threshold), proba


def main():
    print("=" * 68)
    print("  FraudLens — Fraud Detection Training (leakage-safe, time-based)")
    print("=" * 68)

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
    print(
        f"  Loaded: {n:,} rows | fraud rate: {df_train_full['is_fraud'].mean()*100:.3f}%"
    )
    print(
        f"  Time split 85/15 on trans_date_trans_time: train={len(df_train):,} "
        f"val={len(df_val):,}"
    )

    # ── Held-out test data ─────────────────────────────────────────────
    print(f"\nLoading held-out test data: {DATA_TEST} ...")
    df_test = pd.read_csv(DATA_TEST)
    print(
        f"  Loaded: {len(df_test):,} rows | "
        f"fraud rate: {df_test['is_fraud'].mean()*100:.3f}%"
    )

    # ── Base features (fit on train only) ──────────────────────────────
    print("\nEngineering base features (fit on train only)...")
    df_train_b, le_category = engineer_features(df_train, fit=True)
    df_val_b, _ = engineer_features(df_val, le_category=le_category, fit=False)
    df_test_b, _ = engineer_features(
        df_test, le_category=le_category, fit=False
    )

    # ── Behavioural features (history carries over, no future leak) ────
    print("Engineering behavioural features (past-only, leakage-safe)...")
    df_train_bhv, df_val_bhv, df_test_bhv = build_history_frames(
        df_train, df_val, df_test
    )

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

    X_train_no_fair = df_train[FEATURE_COLS_NO_FAIR]
    X_val_no_fair = df_val[FEATURE_COLS_NO_FAIR]
    X_test_no_fair = df_test[FEATURE_COLS_NO_FAIR]

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
        "tuning": {},
        "model_comparison": {},
        "final_model": {},
        "fairness_ablation": {},
    }

    # ── 1. Tune XGBoost with Optuna (validation PR-AUC) ────────────────
    print("\n── Tuning XGBoost with Optuna (validation PR-AUC) ───────────────────")
    tuning = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=40)
    results["tuning"] = {
        "method": tuning["method"],
        "n_trials": tuning["n_trials"],
        "best_params": tuning["best_params"],
        "best_val_pr_auc": tuning["best_val_pr_auc"],
    }
    print(
        f"  Method={tuning['method']} trials={tuning['n_trials']} "
        f"best_val_PR-AUC={tuning['best_val_pr_auc']:.4f}"
    )
    print("  Best params:", json.dumps(tuning["best_params"]))

    # ── 2. Train candidate models ──────────────────────────────────────
    print("\n── Training candidate models ────────────────────────────────────────")

    print("[1/3] XGBoost (tuned) ...")
    xgb, xgb_thr, xgb_iter = train_xgb_final(
        X_train, y_train, X_val, y_val, tuning["best_params"]
    )
    print(f"  best_iteration={xgb_iter} val_threshold={xgb_thr:.2f}")

    print("[2/3] Random Forest ...")
    rf = train_rf(X_train, y_train)
    rf_val_proba = rf.predict_proba(X_val)[:, 1]
    rf_thr, _ = select_threshold(y_val.values, rf_val_proba)
    print(f"  val_threshold={rf_thr:.2f}")

    print("[3/3] RF+XGB average ensemble ...")
    ensemble_val_proba = (rf_val_proba + xgb.predict_proba(X_val)[:, 1]) / 2.0
    ens_thr, _ = select_threshold(y_val.values, ensemble_val_proba)
    print(f"  val_threshold={ens_thr:.2f}")

    # ── 3. Validation PR-AUC for model selection ───────────────────────
    rf_val_pr_auc = float(average_precision_score(y_val.values, rf_val_proba))
    xgb_val_proba = xgb.predict_proba(X_val)[:, 1]
    xgb_val_pr_auc = float(average_precision_score(y_val.values, xgb_val_proba))
    ens_val_pr_auc = float(average_precision_score(y_val.values, ensemble_val_proba))
    print("\nValidation PR-AUC:")
    print(f"  Random Forest : {rf_val_pr_auc:.4f}")
    print(f"  XGBoost       : {xgb_val_pr_auc:.4f}")
    print(f"  Ensemble      : {ens_val_pr_auc:.4f}")

    # ── 4. Evaluate all three on the SAME held-out test set ────────────
    print("\n── Held-out test set evaluation (same test set for all) ─────────────")
    rf_metrics, rf_proba = evaluate_on_test(rf, X_test, y_test, rf_thr)
    xgb_metrics, xgb_proba = evaluate_on_test(xgb, X_test, y_test, xgb_thr)
    ens_proba = (rf_proba + xgb_proba) / 2.0
    ens_metrics = compute_metrics(y_test.values, ens_proba, ens_thr)

    results["model_comparison"] = {
        "random_forest": {"val_pr_auc": rf_val_pr_auc, "test": rf_metrics},
        "xgboost": {"val_pr_auc": xgb_val_pr_auc, "test": xgb_metrics},
        "ensemble": {"val_pr_auc": ens_val_pr_auc, "test": ens_metrics},
    }

    print(
        f"{'Model':<18} {'Val PR-AUC':>11} {'Test PR-AUC':>12} "
        f"{'Precision':>10} {'Recall':>9} {'F1':>8}"
    )
    print("-" * 66)
    for name, key in [
        ("Random Forest", "random_forest"),
        ("XGBoost", "xgboost"),
        ("RF+XGB Ensemble", "ensemble"),
    ]:
        m = results["model_comparison"][key]["test"]
        print(
            f"{name:<18} {results['model_comparison'][key]['val_pr_auc']:>11.4f} "
            f"{m['pr_auc']:>12.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>9.4f} {m['f1']:>8.4f}"
        )
    print("-" * 66)

    # ── 5. Pick final model by validation PR-AUC ───────────────────────
    candidates = {
        "random_forest": (rf, rf_val_pr_auc),
        "xgboost": (xgb, xgb_val_pr_auc),
        "ensemble": (AverageEnsemble(rf, xgb), ens_val_pr_auc),
    }
    final_type = max(candidates, key=lambda k: candidates[k][1])
    final_model, final_val_pr_auc = candidates[final_type]
    final_test_metrics = results["model_comparison"][final_type]["test"]
    print(f"\nFinal model (by validation PR-AUC): {final_type.upper()} "
          f"(val PR-AUC={final_val_pr_auc:.4f})")

    results["final_model"] = {
        "type": final_type,
        "val_pr_auc": float(final_val_pr_auc),
        "test_metrics": final_test_metrics,
    }

    # ── 6. Fairness/ablation: final model without gender and age ───────
    print("\n── Fairness/ablation: final model without gender & age ──────────────")

    if final_type == "xgboost":
        fair_model, fair_thr, _ = train_xgb_final(
            X_train_no_fair, y_train, X_val_no_fair, y_val,
            tuning["best_params"],
        )
        fair_val_proba = fair_model.predict_proba(X_val_no_fair)[:, 1]
        fair_test_proba = fair_model.predict_proba(X_test_no_fair)[:, 1]
    elif final_type == "random_forest":
        fair_model = RandomForestClassifier(
            n_estimators=400, max_depth=14, class_weight="balanced",
            n_jobs=-1, random_state=42,
        )
        fair_model.fit(X_train_no_fair, y_train)
        fair_val_proba = fair_model.predict_proba(X_val_no_fair)[:, 1]
        fair_test_proba = fair_model.predict_proba(X_test_no_fair)[:, 1]
        fair_thr, _ = select_threshold(y_val.values, fair_val_proba)
    else:  # ensemble without gender/age
        rf_fair = RandomForestClassifier(
            n_estimators=400, max_depth=14, class_weight="balanced",
            n_jobs=-1, random_state=42,
        )
        rf_fair.fit(X_train_no_fair, y_train)
        xgb_fair, _, _ = train_xgb_final(
            X_train_no_fair, y_train, X_val_no_fair, y_val,
            tuning["best_params"],
        )
        fair_model = AverageEnsemble(rf_fair, xgb_fair)
        fair_val_proba = (
            rf_fair.predict_proba(X_val_no_fair)[:, 1]
            + xgb_fair.predict_proba(X_val_no_fair)[:, 1]
        ) / 2.0
        fair_thr, _ = select_threshold(y_val.values, fair_val_proba)
        fair_test_proba = fair_model.predict_proba(X_test_no_fair)[:, 1]

    fair_metrics = compute_metrics(y_test.values, fair_test_proba, fair_thr)
    full = final_test_metrics
    results["fairness_ablation"] = {
        "final_model_type": final_type,
        "full_features_test": full,
        "without_gender_age_test": fair_metrics,
        "delta": {
            "pr_auc": fair_metrics["pr_auc"] - full["pr_auc"],
            "precision": fair_metrics["precision"] - full["precision"],
            "recall": fair_metrics["recall"] - full["recall"],
            "f1": fair_metrics["f1"] - full["f1"],
        },
    }
    print(
        f"  Without gender/age: PR-AUC={fair_metrics['pr_auc']:.4f} "
        f"(delta {results['fairness_ablation']['delta']['pr_auc']:+.4f}) | "
        f"F1={fair_metrics['f1']:.4f} "
        f"(delta {results['fairness_ablation']['delta']['f1']:+.4f})"
    )

    # ── 7. Save final model + artifacts ────────────────────────────────
    joblib.dump(final_model, os.path.join(MODELS_DIR, "final_model.joblib"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_cols.joblib"))
    joblib.dump(le_category, os.path.join(MODELS_DIR, "le_category.joblib"))

    tuning_serializable = {
        "method": tuning["method"],
        "n_trials": tuning["n_trials"],
        "best_params": tuning["best_params"],
        "best_val_pr_auc": tuning["best_val_pr_auc"],
        "trials": tuning["trials"],
    }
    with open(os.path.join(MODELS_DIR, "tuning_results.json"), "w") as f:
        json.dump(tuning_serializable, f, indent=2)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Final model, features, tuning results + metrics saved to {MODELS_DIR}/")

    # ── 8. Plots for final model on test set ───────────────────────────
    if final_type == "ensemble":
        final_proba = ens_proba
    elif final_type == "xgboost":
        final_proba = xgb_proba
    else:
        final_proba = rf_proba

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_roc(y_test.values, final_proba, final_type.upper(), ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pr(y_test.values, final_proba, final_type.upper(), ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "pr_curve.png"), dpi=150)
    plt.close(fig)

    cm = np.array(
        [
            [final_test_metrics["confusion_matrix"]["tn"],
             final_test_metrics["confusion_matrix"]["fp"]],
            [final_test_metrics["confusion_matrix"]["fn"],
             final_test_metrics["confusion_matrix"]["tp"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_confusion(cm, final_type.upper(), ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"  Plots saved to {DOCS_DIR}/ (roc_curve.png, pr_curve.png, confusion_matrix.png)")

    # ── 9. Results table ───────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  FINAL RESULTS — Held-Out Test Set (fraudTest.csv)")
    print("=" * 68)
    header = (
        f"{'Model':<18} {'Val PR-AUC':>11} {'Test PR-AUC':>12} "
        f"{'Precision':>10} {'Recall':>9} {'F1':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, key in [
        ("Random Forest", "random_forest"),
        ("XGBoost (Optuna-tuned)", "xgboost"),
        ("RF+XGB Ensemble", "ensemble"),
    ]:
        entry = results["model_comparison"][key]
        m = entry["test"]
        print(
            f"{name:<18} {entry['val_pr_auc']:>11.4f} {m['pr_auc']:>12.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>9.4f} {m['f1']:>8.4f}"
        )
    print("-" * len(header))
    print(f"\nFinal model (API): {final_type.upper()} "
          f"| val PR-AUC={final_val_pr_auc:.4f} | test PR-AUC={final_test_metrics['pr_auc']:.4f}")
    print("Note: Accuracy is not reported as a headline metric (imbalanced data).")
    print("=" * 68)
    print("  Training complete! Start the app with:")
    print("  python3 app.py")
    print("=" * 68)


if __name__ == "__main__":
    main()
