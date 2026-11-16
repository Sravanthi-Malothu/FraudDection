"""
train.py — Train and Compare Fraud Detection Models with Time-Based Split
Enforces chronological train/validation/test partitioning, avoids data leakage,
fits all encoders/scalers on the training partition only, tunes hyperparameters
via randomized search using the validation set only, and builds an averaged
ensemble of the top two models.
"""

import sys
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import engineer_features, FEATURE_COLS, IDENTIFIER_COLS
from src.evaluate import compute_metrics, find_optimal_threshold
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


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

    # Assertion: Train timestamp strictly precedes Test timestamp
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


def tune_logistic_regression(X_train_s, y_train, X_val_s, y_val, random_state=42, n_iter=3):
    """
    Tunes Logistic Regression baseline (scaled features, class_weight='balanced')
    using randomized search strictly on the validation set.
    """
    print("[Tuning] Tuning Logistic Regression on Validation set...")
    rng = random.Random(random_state)
    c_candidates = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    sampled_c = rng.sample(c_candidates, min(n_iter, len(c_candidates)))

    best_score = -1.0
    best_model = None
    best_params = {}

    for c_val in sampled_c:
        params = {"C": c_val, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}
        lr = LogisticRegression(
            C=c_val,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )
        lr.fit(X_train_s, y_train)
        val_proba = lr.predict_proba(X_val_s)[:, 1]
        val_pr_auc = float(average_precision_score(y_val, val_proba))
        print(f"  LR candidate C={c_val:<6} -> Val PR-AUC = {val_pr_auc:.4f}")

        if val_pr_auc > best_score:
            best_score = val_pr_auc
            best_model = lr
            best_params = params

    print(f"  ✓ Best LR: C={best_params['C']} with Val PR-AUC={best_score:.4f}\n")
    return best_model, best_params, best_score


def tune_random_forest(X_train, y_train, X_val, y_val, random_state=42, n_iter=3):
    """
    Tunes Random Forest (class_weight='balanced_subsample')
    using randomized search strictly on the validation set.
    """
    print("[Tuning] Tuning Random Forest (class_weight='balanced_subsample') on Validation set...")
    rng = random.Random(random_state)
    param_grid = [
        {"n_estimators": 60, "max_depth": 10, "min_samples_split": 5, "max_features": "sqrt"},
        {"n_estimators": 80, "max_depth": 12, "min_samples_split": 2, "max_features": "sqrt"},
        {"n_estimators": 100, "max_depth": 14, "min_samples_split": 5, "max_features": 0.5},
        {"n_estimators": 75, "max_depth": 12, "min_samples_split": 10, "max_features": "log2"},
    ]
    sampled_params = rng.sample(param_grid, min(n_iter, len(param_grid)))

    best_score = -1.0
    best_model = None
    best_params = {}

    for params in sampled_params:
        rf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            max_features=params["max_features"],
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        val_proba = rf.predict_proba(X_val)[:, 1]
        val_pr_auc = float(average_precision_score(y_val, val_proba))
        print(
            f"  RF candidate n_est={params['n_estimators']}, depth={params['max_depth']}, "
            f"max_feat={params['max_features']} -> Val PR-AUC = {val_pr_auc:.4f}"
        )

        if val_pr_auc > best_score:
            best_score = val_pr_auc
            best_model = rf
            best_params = {**params, "class_weight": "balanced_subsample"}

    print(f"  ✓ Best RF: {best_params} with Val PR-AUC={best_score:.4f}\n")
    return best_model, best_params, best_score


def tune_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight, random_state=42, n_iter=3):
    """
    Tunes XGBoost with scale_pos_weight computed from the training set
    using randomized search strictly on the validation set.
    """
    print(f"[Tuning] Tuning XGBoost (scale_pos_weight={scale_pos_weight:.1f}) on Validation set...")
    rng = random.Random(random_state)
    param_grid = [
        {"n_estimators": 80, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.10, "subsample": 0.90, "colsample_bytree": 0.90},
        {"n_estimators": 100, "max_depth": 7, "learning_rate": 0.06, "subsample": 0.80, "colsample_bytree": 0.80},
        {"n_estimators": 140, "max_depth": 6, "learning_rate": 0.12, "subsample": 0.85, "colsample_bytree": 0.85},
    ]
    sampled_params = rng.sample(param_grid, min(n_iter, len(param_grid)))

    best_score = -1.0
    best_model = None
    best_params = {}

    for params in sampled_params:
        xgb = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        val_proba = xgb.predict_proba(X_val)[:, 1]
        val_pr_auc = float(average_precision_score(y_val, val_proba))
        print(
            f"  XGB candidate n_est={params['n_estimators']}, depth={params['max_depth']}, "
            f"lr={params['learning_rate']} -> Val PR-AUC = {val_pr_auc:.4f}"
        )

        if val_pr_auc > best_score:
            best_score = val_pr_auc
            best_model = xgb
            best_params = {**params, "scale_pos_weight": float(scale_pos_weight)}

    print(f"  ✓ Best XGBoost: {best_params} with Val PR-AUC={best_score:.4f}\n")
    return best_model, best_params, best_score


def tune_lightgbm(X_train, y_train, X_val, y_val, scale_pos_weight, random_state=42, n_iter=3):
    """
    Tunes LightGBM, if installed, with scale_pos_weight computed from the training set
    using randomized search strictly on the validation set.
    """
    if not HAS_LIGHTGBM:
        print("[Tuning] LightGBM is not installed. Skipping LightGBM.")
        return None, {}, 0.0

    print(f"[Tuning] Tuning LightGBM (scale_pos_weight={scale_pos_weight:.1f}) on Validation set...")
    rng = random.Random(random_state)
    param_grid = [
        {"n_estimators": 80, "num_leaves": 31, "learning_rate": 0.08, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 120, "num_leaves": 63, "learning_rate": 0.10, "subsample": 0.90, "colsample_bytree": 0.90},
        {"n_estimators": 100, "num_leaves": 45, "learning_rate": 0.06, "subsample": 0.80, "colsample_bytree": 0.80},
    ]
    sampled_params = rng.sample(param_grid, min(n_iter, len(param_grid)))

    best_score = -1.0
    best_model = None
    best_params = {}

    for params in sampled_params:
        lgbm = lgb.LGBMClassifier(
            n_estimators=params["n_estimators"],
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        lgbm.fit(X_train, y_train)
        val_proba = lgbm.predict_proba(X_val)[:, 1]
        val_pr_auc = float(average_precision_score(y_val, val_proba))
        print(
            f"  LGBM candidate n_est={params['n_estimators']}, leaves={params['num_leaves']}, "
            f"lr={params['learning_rate']} -> Val PR-AUC = {val_pr_auc:.4f}"
        )

        if val_pr_auc > best_score:
            best_score = val_pr_auc
            best_model = lgbm
            best_params = {**params, "scale_pos_weight": float(scale_pos_weight)}

    print(f"  ✓ Best LightGBM: {best_params} with Val PR-AUC={best_score:.4f}\n")
    return best_model, best_params, best_score


def train_models(data_dir=DATA_DIR, models_dir=MODELS_DIR, random_state=42, n_iter_search=3):
    """
    End-to-end model training, hyperparameter tuning, comparison, and ensembling:
    1. Loads full dataset and performs strict chronological 70/15/15 time-based split.
    2. Fits encoders, fraud rates, and feature scaler strictly on the training partition.
    3. Excludes leaking identifier columns.
    4. Evaluates:
       - Logistic Regression baseline (scaled features, class_weight='balanced')
       - Random Forest (class_weight='balanced_subsample')
       - XGBoost with scale_pos_weight from training set
       - LightGBM (if installed)
       - Soft-voting / averaged ensemble of the best two models
    5. Tunes with small randomized search using the validation set only (test set untouched).
    6. Finds optimal decision thresholds on the validation set.
    7. Evaluates all models on held-out test set once.
    8. Saves models with joblib and writes metadata.json.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full dataset loading and timestamp sort
    df_full = load_full_dataset(data_dir=data_dir)

    # 2. Time-based split: 70% Train, 15% Validation, 15% Test
    df_train, df_val, df_test = time_based_split(
        df_full, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # Print class balance for each split
    print_class_balance("Training Split (70%)", df_train["is_fraud"])
    print_class_balance("Validation Split (15%)", df_val["is_fraud"])
    print_class_balance("Held-Out Test Split (15%)", df_test["is_fraud"])

    # 3. Fit encoders on training set ONLY, then transform validation and test
    print("[Feature Engineering] Fitting category encoders & fraud rates on TRAINING set only...")
    df_train, le_category, cat_fraud_rates, glob_fraud_rate = engineer_features(df_train, fit=True)

    print("[Feature Engineering] Applying fitted encoders to Validation and Test sets...")
    df_val, _, _, _ = engineer_features(
        df_val,
        le_category=le_category,
        category_fraud_rates=cat_fraud_rates,
        global_fraud_rate=glob_fraud_rate,
        fit=False,
    )
    df_test, _, _, _ = engineer_features(
        df_test,
        le_category=le_category,
        category_fraud_rates=cat_fraud_rates,
        global_fraud_rate=glob_fraud_rate,
        fit=False,
    )

    # 4. Remove identifier columns that leak
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

    # Fit feature scaler on training set ONLY (used for Logistic Regression)
    print("[Preprocessing] Fitting StandardScaler on TRAINING set only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class imbalance weighting for XGBoost / LightGBM
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / max(1, pos_count))

    # --- Hyperparameter Tuning (Validation set ONLY; test set untouched) ---
    print("=" * 70)
    print("  HYPERPARAMETER TUNING VIA RANDOMIZED SEARCH (VALIDATION SET ONLY)")
    print("=" * 70)

    # 1. Logistic Regression
    lr_best, lr_params, lr_val_score = tune_logistic_regression(
        X_train_scaled, y_train, X_val_scaled, y_val, random_state=random_state, n_iter=n_iter_search
    )

    # 2. Random Forest
    rf_best, rf_params, rf_val_score = tune_random_forest(
        X_train, y_train, X_val, y_val, random_state=random_state, n_iter=n_iter_search
    )

    # 3. XGBoost
    xgb_best, xgb_params, xgb_val_score = tune_xgboost(
        X_train, y_train, X_val, y_val, scale_pos_weight=scale_pos_weight, random_state=random_state, n_iter=n_iter_search
    )

    # 4. LightGBM (if installed)
    lgb_best, lgb_params, lgb_val_score = tune_lightgbm(
        X_train, y_train, X_val, y_val, scale_pos_weight=scale_pos_weight, random_state=random_state, n_iter=n_iter_search
    )

    candidate_models = {
        "logistic_regression": lr_best,
        "random_forest": rf_best,
        "xgboost": xgb_best,
    }
    tuned_hyperparams = {
        "logistic_regression": lr_params,
        "random_forest": rf_params,
        "xgboost": xgb_params,
    }
    val_pr_scores = {
        "logistic_regression": lr_val_score,
        "random_forest": rf_val_score,
        "xgboost": xgb_val_score,
    }

    if HAS_LIGHTGBM and lgb_best is not None:
        candidate_models["lightgbm"] = lgb_best
        tuned_hyperparams["lightgbm"] = lgb_params
        val_pr_scores["lightgbm"] = lgb_val_score

    # Select the BEST TWO models based on validation PR-AUC
    sorted_models = sorted(val_pr_scores.items(), key=lambda item: item[1], reverse=True)
    best_two_names = [sorted_models[0][0], sorted_models[1][0]]
    print("=" * 70)
    print(f"  MODEL RANKING BY VALIDATION PR-AUC:")
    for rank, (m_name, score) in enumerate(sorted_models, 1):
        marker = "★ [SELECTED FOR ENSEMBLE]" if m_name in best_two_names else ""
        print(f"    {rank}. {m_name:<20}: PR-AUC = {score:.4f} {marker}")
    print(f"\n  Top two models selected for Averaged Ensemble: {best_two_names[0]} + {best_two_names[1]}")
    print("=" * 70 + "\n")

    # 5. Build Soft-Voting / Averaged Ensemble of top two models
    best_two_dict = {name: candidate_models[name] for name in best_two_names}
    ensemble = AveragedEnsemble(
        model_names=best_two_names,
        models=best_two_dict,
        scaler=scaler if "logistic_regression" in best_two_names else None,
    )

    # --- Validation Evaluation & Optimal Threshold Tuning ---
    print("=" * 70)
    print("  VALIDATION SET PERFORMANCE & THRESHOLD OPTIMIZATION (15%)")
    print("=" * 70)

    val_probas = {}
    val_metrics = {}
    optimal_thresholds = {}

    # Evaluate individual models on validation set
    for name, model in candidate_models.items():
        if name == "logistic_regression":
            proba = model.predict_proba(X_val_scaled)[:, 1]
        else:
            proba = model.predict_proba(X_val)[:, 1]
        val_probas[name] = proba

        opt_thr, _ = find_optimal_threshold(y_val, proba)
        optimal_thresholds[name] = opt_thr
        val_metrics[name] = compute_metrics(y_val, proba, threshold=opt_thr)

    # Evaluate ensemble on validation set
    ens_val_proba = ensemble.predict_proba(X_val)[:, 1]
    val_probas["ensemble"] = ens_val_proba
    opt_thr_ens, _ = find_optimal_threshold(y_val, ens_val_proba)
    optimal_thresholds["ensemble"] = opt_thr_ens
    ensemble.optimal_threshold = opt_thr_ens
    val_metrics["ensemble"] = compute_metrics(y_val, ens_val_proba, threshold=opt_thr_ens)

    for m_name in list(candidate_models.keys()) + ["ensemble"]:
        m = val_metrics[m_name]
        thr = optimal_thresholds[m_name]
        print(
            f"  {m_name:<20}: ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} | "
            f"Recall={m['recall']:.4f} | F1={m['f1']:.4f} (thr={thr:.2f})"
        )

    # --- Test Set Evaluation (Held-Out, Test Set Never Touched During Tuning) ---
    print("\n" + "=" * 70)
    print("  HELD-OUT TEST SET PERFORMANCE (15%) [Thresholds Fixed From Validation]")
    print("=" * 70)

    test_metrics = {}
    for name, model in candidate_models.items():
        if name == "logistic_regression":
            t_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            t_proba = model.predict_proba(X_test)[:, 1]
        test_metrics[name] = compute_metrics(y_test, t_proba, threshold=optimal_thresholds[name])

    ens_test_proba = ensemble.predict_proba(X_test)[:, 1]
    test_metrics["ensemble"] = compute_metrics(y_test, ens_test_proba, threshold=optimal_thresholds["ensemble"])

    for m_name in list(candidate_models.keys()) + ["ensemble"]:
        m = test_metrics[m_name]
        print(
            f"  {m_name:<20}: ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} | "
            f"Precision={m['precision']:.4f} | Recall={m['recall']:.4f} | F1={m['f1']:.4f}"
        )
    print("=" * 70 + "\n")

    # --- Save Models with joblib ---
    print(f"[Artifacts] Saving serialized models to {models_dir}...")
    joblib.dump(lr_best, models_dir / "logistic_regression.joblib")
    joblib.dump(rf_best, models_dir / "random_forest.joblib")
    joblib.dump(xgb_best, models_dir / "xgboost.joblib")
    if HAS_LIGHTGBM and lgb_best is not None:
        joblib.dump(lgb_best, models_dir / "lightgbm.joblib")
    joblib.dump(ensemble, models_dir / "ensemble.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(le_category, models_dir / "le_category.joblib")
    joblib.dump(FEATURE_COLS, models_dir / "feature_cols.joblib")
    joblib.dump(cat_fraud_rates, models_dir / "category_fraud_rates.joblib")

    all_metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "optimal_thresholds": optimal_thresholds,
        "best_two_models": best_two_names,
    }
    joblib.dump(all_metrics, models_dir / "metrics.joblib")

    # --- Metadata JSON ---
    metadata = {
        "metadata_version": "1.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "split_dates": {
            "train_start": str(df_train["trans_date_trans_time"].min()),
            "train_end": str(df_train["trans_date_trans_time"].max()),
            "val_start": str(df_val["trans_date_trans_time"].min()),
            "val_end": str(df_val["trans_date_trans_time"].max()),
            "test_start": str(df_test["trans_date_trans_time"].min()),
            "test_end": str(df_test["trans_date_trans_time"].max()),
        },
        "split_sizes": {
            "train_samples": int(len(df_train)),
            "val_samples": int(len(df_val)),
            "test_samples": int(len(df_test)),
            "train_fraud_count": int(pos_count),
            "val_fraud_count": int((y_val == 1).sum()),
            "test_fraud_count": int((y_test == 1).sum()),
            "train_fraud_rate": float(y_train.mean()),
            "val_fraud_rate": float(y_val.mean()),
            "test_fraud_rate": float(y_test.mean()),
            "scale_pos_weight": float(scale_pos_weight),
        },
        "feature_list": FEATURE_COLS,
        "feature_count": len(FEATURE_COLS),
        "library_versions": {
            "python": sys.version,
            "scikit-learn": sklearn.__version__,
            "xgboost": getattr(sys.modules.get("xgboost"), "__version__", "unknown"),
            "lightgbm": getattr(sys.modules.get("lightgbm"), "__version__", "not installed") if HAS_LIGHTGBM else "not installed",
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "models_evaluated": list(candidate_models.keys()),
        "best_two_models": best_two_names,
        "tuned_hyperparameters": tuned_hyperparams,
        "optimal_thresholds": optimal_thresholds,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Artifacts] Saved metadata JSON to {metadata_path}")

    # Backwards compatibility metrics.json
    metrics_json_path = models_dir / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("[Train] End-to-end model training and comparison completed successfully.")
    return metadata


if __name__ == "__main__":
    train_models()
