"""
evaluate.py — Comprehensive Model Evaluation on Held-Out Test Set
Computes PR-AUC (primary metric), ROC-AUC, F1-optimal and targeted thresholds (90% precision, 90% recall),
cost analysis with configurable FN/FP costs, bootstrap 95% confidence intervals, and generates publication-grade plots.
Outputs detailed evaluation results to docs/results.json.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.calibration import calibration_curve

# Use non-interactive backend for headless plot generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import engineer_features, FEATURE_COLS, IDENTIFIER_COLS
from src.ensemble import AveragedEnsemble

# Ensure pickle compatibility across script invocations
setattr(sys.modules["__main__"], "AveragedEnsemble", AveragedEnsemble)

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"


def compute_metrics(y_true, y_proba, threshold=0.5):
    """
    Computes comprehensive binary classification evaluation metrics.
    Does not report accuracy as a headline metric; prioritizes PR-AUC and Recall/Precision.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
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
        "support_positive": int((y_true == 1).sum()),
        "support_negative": int((y_true == 0).sum()),
    }


def find_optimal_threshold(y_true, y_proba, steps=1000):
    """
    Finds the probability decision threshold that maximizes F1 score.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    best_thresh = 0.5
    best_f1 = -1.0

    for thresh in np.linspace(0.01, 0.99, steps):
        y_pred = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    return best_thresh, best_f1


def find_target_precision_threshold(y_true, y_proba, target_precision=0.90, steps=1000):
    """
    Finds the threshold that achieves >= target_precision (e.g. 90%)
    while maximizing recall on the validation set.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    best_thresh = 0.95
    best_recall = -1.0
    highest_precision = -1.0
    fallback_thresh = 0.95

    for thresh in np.linspace(0.01, 0.99, steps):
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        if prec > highest_precision and rec > 0:
            highest_precision = prec
            fallback_thresh = float(thresh)

        if prec >= target_precision:
            if rec > best_recall:
                best_recall = rec
                best_thresh = float(thresh)

    if best_recall < 0:
        # Fallback to threshold that achieved highest precision
        return fallback_thresh, highest_precision

    return best_thresh, target_precision


def find_target_recall_threshold(y_true, y_proba, target_recall=0.90, steps=1000):
    """
    Finds the threshold that achieves >= target_recall (e.g. 90%)
    while maximizing precision on the validation set.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    best_thresh = 0.05
    best_prec = -1.0
    highest_recall = -1.0
    fallback_thresh = 0.05

    for thresh in np.linspace(0.01, 0.99, steps):
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        if rec > highest_recall:
            highest_recall = rec
            fallback_thresh = float(thresh)

        if rec >= target_recall:
            if prec > best_prec:
                best_prec = prec
                best_thresh = float(thresh)

    if best_prec < 0:
        return fallback_thresh, highest_recall

    return best_thresh, target_recall


def compute_cost_breakdown(tn, fp, fn, tp, fn_cost_per_case, fp_cost_per_case):
    """
    Computes financial cost breakdown for a given confusion matrix:
    - False Negative Cost: unprevented fraud loss (average fraud amount)
    - False Positive Cost: manual investigation overhead (e.g. $5 review cost)
    - Total Cost: FN * fn_cost + FP * fp_cost
    - Baseline Cost (no ML, all fraud allowed): (FN + TP) * fn_cost
    - Net Savings: Baseline Cost - Total Cost
    """
    fn_cost_total = float(fn * fn_cost_per_case)
    fp_cost_total = float(fp * fp_cost_per_case)
    total_cost = fn_cost_total + fp_cost_total

    total_frauds = fn + tp
    baseline_cost = float(total_frauds * fn_cost_per_case)
    savings = baseline_cost - total_cost
    savings_pct = (savings / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

    return {
        "fn_cost_total": round(fn_cost_total, 2),
        "fp_cost_total": round(fp_cost_total, 2),
        "total_cost": round(total_cost, 2),
        "baseline_cost_no_model": round(baseline_cost, 2),
        "net_savings": round(savings, 2),
        "savings_percentage": round(savings_pct, 2),
    }


def bootstrap_confidence_intervals(
    y_true, y_proba, threshold, n_bootstraps=500, random_state=42
):
    """
    Computes 95% bootstrap confidence intervals for PR-AUC, Recall, Precision, and ROC-AUC.
    Returns (2.5th percentile, 97.5th percentile) for each metric.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    rng = np.random.default_rng(random_state)
    boot_pr_auc = []
    boot_roc_auc = []
    boot_recall = []
    boot_precision = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p_b = y_proba[idx]

        # Ensure both classes are present in the resample
        if y_b.sum() == 0 or y_b.sum() == n:
            continue

        boot_pr_auc.append(average_precision_score(y_b, p_b))
        boot_roc_auc.append(roc_auc_score(y_b, p_b))

        pred_b = (p_b >= threshold).astype(int)
        boot_recall.append(recall_score(y_b, pred_b, zero_division=0))
        boot_precision.append(precision_score(y_b, pred_b, zero_division=0))

    def get_ci(arr):
        if not arr:
            return [0.0, 0.0]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "pr_auc_95ci": get_ci(boot_pr_auc),
        "roc_auc_95ci": get_ci(boot_roc_auc),
        "recall_95ci": get_ci(boot_recall),
        "precision_95ci": get_ci(boot_precision),
    }


def generate_evaluation_plots(
    models_probas, y_test, chosen_model_name, optimal_thresholds, docs_dir=DOCS_DIR
):
    """
    Generates publication-quality evaluation figures saved to docs/:
    1. precision_recall_curve.png
    2. roc_curve.png
    3. confusion_matrix_heatmap.png
    4. threshold_vs_precision_recall.png
    5. calibration_curve.png
    """
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    palette = {
        "xgboost": "#2563EB",            # Vibrant Blue
        "ensemble": "#7C3AED",           # Deep Purple
        "random_forest": "#059669",      # Emerald Green
        "logistic_regression": "#DC2626",# Red
        "lightgbm": "#D97706",           # Amber
    }

    # Set overall visual style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    # 1. Precision-Recall Curve (All Models)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    baseline_rate = float((y_test == 1).mean())
    ax.axhline(
        y=baseline_rate,
        color="#9CA3AF",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline Prevalence ({baseline_rate * 100:.2f}%)",
    )

    for name, proba in models_probas.items():
        pr_score = average_precision_score(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        color = palette.get(name, "#4B5563")
        label_name = name.replace("_", " ").title()
        ax.plot(rec, prec, color=color, linewidth=2.2, label=f"{label_name} (PR-AUC = {pr_score:.4f})")

    ax.set_title("Precision-Recall Curve (Held-Out Test Set)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Fraud Detected)", fontsize=12, labelpad=8)
    ax.set_ylabel("Precision (True Fraud / Flagged)", fontsize=12, labelpad=8)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    plt.tight_layout()
    pr_path = docs_dir / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Precision-Recall Curve: {pr_path}")

    # 2. ROC Curve (All Models)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")

    for name, proba in models_probas.items():
        auc_score = roc_auc_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        color = palette.get(name, "#4B5563")
        label_name = name.replace("_", " ").title()
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{label_name} (ROC-AUC = {auc_score:.4f})")

    ax.set_title("Receiver Operating Characteristic (ROC Curve)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (Fall-out)", fontsize=12, labelpad=8)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12, labelpad=8)
    ax.set_xlim([-0.01, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    plt.tight_layout()
    roc_path = docs_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved ROC Curve: {roc_path}")

    # 3. Confusion Matrix Heatmaps (Multi-panel for key models)
    model_keys = [k for k in ["ensemble", "xgboost", "random_forest", "logistic_regression"] if k in models_probas]
    cols = 2
    rows = (len(model_keys) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.5 * rows), dpi=300)
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(model_keys):
        ax = axes[i]
        thr = optimal_thresholds[name]["f1_optimal"]
        y_pred = (models_probas[name] >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Legit (0)", "Fraud (1)"],
            yticklabels=["Legit (0)", "Fraud (1)"],
        )
        label_name = name.replace("_", " ").title()
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ax.set_title(
            f"{label_name} (Threshold = {thr:.2f})\nPrec: {prec:.1%} | Rec: {rec:.1%} | F1: {f1:.3f}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("Actual Label", fontsize=11)

    for j in range(len(model_keys), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    cm_path = docs_dir / "confusion_matrix_heatmap.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Confusion Matrix Heatmap: {cm_path}")

    # 4. Threshold vs Precision, Recall & F1 (For Best Model / Ensemble)
    best_proba = models_probas[chosen_model_name]
    thresholds = np.linspace(0.02, 0.98, 200)
    precs = []
    recs = []
    f1s = []

    for t in thresholds:
        yp = (best_proba >= t).astype(int)
        precs.append(precision_score(y_test, yp, zero_division=0))
        recs.append(recall_score(y_test, yp, zero_division=0))
        f1s.append(f1_score(y_test, yp, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.plot(thresholds, precs, color="#2563EB", linewidth=2.2, label="Precision")
    ax.plot(thresholds, recs, color="#059669", linewidth=2.2, label="Recall")
    ax.plot(thresholds, f1s, color="#7C3AED", linewidth=2.2, linestyle="--", label="F1-Score")

    # Mark validation-selected key thresholds
    t_f1 = optimal_thresholds[chosen_model_name]["f1_optimal"]
    t_p90 = optimal_thresholds[chosen_model_name]["target_precision_90"]
    t_r90 = optimal_thresholds[chosen_model_name]["target_recall_90"]

    ax.axvline(x=t_f1, color="#7C3AED", linestyle=":", linewidth=1.8, label=f"F1-Optimal ({t_f1:.2f})")
    ax.axvline(x=t_p90, color="#2563EB", linestyle=":", linewidth=1.8, label=f"90% Precision ({t_p90:.2f})")
    ax.axvline(x=t_r90, color="#059669", linestyle=":", linewidth=1.8, label=f"90% Recall ({t_r90:.2f})")

    ax.set_title(
        f"Threshold vs Precision, Recall & F1 ({chosen_model_name.replace('_', ' ').title()})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Probability Decision Threshold", fontsize=12, labelpad=8)
    ax.set_ylabel("Metric Value", fontsize=12, labelpad=8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="center left", frameon=True, framealpha=0.95)
    plt.tight_layout()
    tpr_path = docs_dir / "threshold_vs_precision_recall.png"
    plt.savefig(tpr_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Threshold vs Precision/Recall: {tpr_path}")

    # 5. Calibration Curve (Reliability Diagram)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.5, label="Perfectly Calibrated")

    for name, proba in models_probas.items():
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
        color = palette.get(name, "#4B5563")
        label_name = name.replace("_", " ").title()
        ax.plot(prob_pred, prob_true, marker="o", markersize=4, color=color, linewidth=2.0, label=label_name)

    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12, labelpad=8)
    ax.set_ylabel("Fraction of True Frauds", fontsize=12, labelpad=8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    plt.tight_layout()
    cal_path = docs_dir / "calibration_curve.png"
    plt.savefig(cal_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Calibration Curve: {cal_path}")


def evaluate_all(
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
    docs_dir=DOCS_DIR,
    fp_cost_usd=5.0,
    n_bootstraps=500,
):
    """
    Main evaluation pipeline on held-out test set only:
    1. Loads dataset, applies strict time-based split (70/15/15).
    2. Featurizes validation and test sets using fitted preprocessors.
    3. Selects thresholds on the VALIDATION SET ONLY:
       - F1-optimal threshold
       - 90% target precision threshold
       - 90% target recall threshold
    4. Evaluates all models on the HELD-OUT TEST SET at those thresholds.
    5. Computes cost analysis (dynamic FN cost = average test fraud amount).
    6. Computes bootstrap 95% confidence intervals for PR-AUC and recall.
    7. Generates plots and saves comprehensive results to docs/results.json.
    """
    models_dir = Path(models_dir)
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    from src.train import AveragedEnsemble, load_full_dataset, time_based_split

    print("=" * 75)
    print("  SENTRA FRAUD DETECTION: HELD-OUT TEST EVALUATION PIPELINE")
    print("=" * 75)

    # 1. Load full dataset and split chronologically
    df_full = load_full_dataset(data_dir=data_dir)
    df_train, df_val, df_test = time_based_split(df_full, 0.70, 0.15, 0.15)

    # Load fitted encoders and category fraud rates from training
    le_category = joblib.load(models_dir / "le_category.joblib")
    cat_fraud_rates = joblib.load(models_dir / "category_fraud_rates.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")

    # Featurize validation and test sets using training parameters (fit=False)
    print("\n[Features] Applying pre-fitted feature engineering to Validation and Test sets...")
    df_val_f, _, _, _ = engineer_features(
        df_val,
        le_category=le_category,
        category_fraud_rates=cat_fraud_rates,
        fit=False,
    )
    df_test_f, _, _, _ = engineer_features(
        df_test,
        le_category=le_category,
        category_fraud_rates=cat_fraud_rates,
        fit=False,
    )

    X_val = df_val_f[FEATURE_COLS]
    y_val = df_val_f["is_fraud"].astype(int).values

    X_test = df_test_f[FEATURE_COLS]
    y_test = df_test_f["is_fraud"].astype(int).values

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Calculate financial parameters
    test_fraud_mask = y_test == 1
    avg_fraud_amount = float(df_test.loc[test_fraud_mask, "amt"].mean())
    avg_legit_amount = float(df_test.loc[~test_fraud_mask, "amt"].mean())
    total_fraud_exposure = float(df_test.loc[test_fraud_mask, "amt"].sum())
    fn_cost_usd = avg_fraud_amount  # Configurable FN cost: average unprevented fraud loss

    test_total = len(y_test)
    test_frauds = int(test_fraud_mask.sum())
    test_legits = int((~test_fraud_mask).sum())
    test_fraud_rate = float(test_frauds / test_total)

    print("\n[Test Set Summary]")
    print(f"  Total test transactions:   {test_total:>10,}")
    print(f"  Legitimate transactions:   {test_legits:>10,} ({100 - test_fraud_rate * 100:.2f}%)")
    print(f"  Fraudulent transactions:   {test_frauds:>10,} ({test_fraud_rate * 100:.3f}%)")
    print(f"  Average fraud amount (FN): ${avg_fraud_amount:.2f}")
    print(f"  Manual review cost (FP):   ${fp_cost_usd:.2f}")
    print(f"  Total fraud exposure:      ${total_fraud_exposure:,.2f}\n")

    # 2. Load trained models
    models = {}
    if (models_dir / "logistic_regression.joblib").exists():
        models["logistic_regression"] = joblib.load(models_dir / "logistic_regression.joblib")
    if (models_dir / "random_forest.joblib").exists():
        models["random_forest"] = joblib.load(models_dir / "random_forest.joblib")
    if (models_dir / "xgboost.joblib").exists():
        models["xgboost"] = joblib.load(models_dir / "xgboost.joblib")
    if (models_dir / "lightgbm.joblib").exists():
        models["lightgbm"] = joblib.load(models_dir / "lightgbm.joblib")
    if (models_dir / "ensemble.joblib").exists():
        models["ensemble"] = joblib.load(models_dir / "ensemble.joblib")

    print(f"[Models] Loaded {len(models)} model artifacts: {list(models.keys())}")

    # 3. Generate probability predictions on validation and test sets
    val_probas = {}
    test_probas = {}

    for name, model in models.items():
        if name == "logistic_regression":
            val_probas[name] = model.predict_proba(X_val_scaled)[:, 1]
            test_probas[name] = model.predict_proba(X_test_scaled)[:, 1]
        else:
            val_probas[name] = model.predict_proba(X_val)[:, 1]
            test_probas[name] = model.predict_proba(X_test)[:, 1]

    # 4. Select thresholds strictly on VALIDATION SET
    print("\n" + "=" * 75)
    print("  STEP 1: THRESHOLD SELECTION (STRICTLY ON VALIDATION SET)")
    print("=" * 75)

    threshold_catalog = {}
    for name in models:
        v_proba = val_probas[name]
        t_f1, _ = find_optimal_threshold(y_val, v_proba)
        t_p90, _ = find_target_precision_threshold(y_val, v_proba, target_precision=0.90)
        t_r90, _ = find_target_recall_threshold(y_val, v_proba, target_recall=0.90)

        threshold_catalog[name] = {
            "f1_optimal": float(t_f1),
            "target_precision_90": float(t_p90),
            "target_recall_90": float(t_r90),
        }
        print(
            f"  {name:<20}: F1-Optimal Thr={t_f1:.2f} | "
            f"90% Prec Thr={t_p90:.2f} | 90% Recall Thr={t_r90:.2f}"
        )

    # 5. Evaluate on HELD-OUT TEST SET
    print("\n" + "=" * 75)
    print("  STEP 2: EVALUATION ON HELD-OUT TEST SET (NEVER TOUCHED IN TUNING)")
    print("=" * 75)

    results_data = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "test_set_summary": {
            "total_transactions": test_total,
            "fraud_cases": test_frauds,
            "legitimate_cases": test_legits,
            "fraud_rate_pct": round(test_fraud_rate * 100, 4),
            "average_fraud_amount": round(avg_fraud_amount, 2),
            "average_legit_amount": round(avg_legit_amount, 2),
            "total_fraud_exposure": round(total_fraud_exposure, 2),
        },
        "cost_parameters": {
            "false_negative_cost_usd": round(fn_cost_usd, 2),
            "false_positive_cost_usd": round(fp_cost_usd, 2),
            "note": "FN cost is average fraud amount loss; FP cost is manual analyst review cost",
        },
        "baseline_cost_no_model": round(test_frauds * fn_cost_usd, 2),
        "models": {},
    }

    # Print tabular comparison header
    print(
        f"\n{'Model':<20} | {'PR-AUC (Primary)':<18} | {'ROC-AUC':<10} | {'Threshold':<9} | "
        f"{'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Net Savings':<12}"
    )
    print("-" * 115)

    for name, model in models.items():
        t_proba = test_probas[name]
        pr_auc = float(average_precision_score(y_test, t_proba))
        roc_auc = float(roc_auc_score(y_test, t_proba))

        chosen_thr = threshold_catalog[name]["f1_optimal"]

        # Compute Bootstrap 95% Confidence Intervals at F1-optimal threshold
        print(f"  [Bootstrapping {name:<20}] computing 95% CIs ({n_bootstraps} resamples)...", end="\r")
        boot_ci = bootstrap_confidence_intervals(
            y_test, t_proba, threshold=chosen_thr, n_bootstraps=n_bootstraps, random_state=42
        )

        model_entry = {
            "pr_auc": round(pr_auc, 4),
            "pr_auc_95ci": [round(boot_ci["pr_auc_95ci"][0], 4), round(boot_ci["pr_auc_95ci"][1], 4)],
            "roc_auc": round(roc_auc, 4),
            "roc_auc_95ci": [round(boot_ci["roc_auc_95ci"][0], 4), round(boot_ci["roc_auc_95ci"][1], 4)],
            "thresholds": {},
        }

        # Evaluate at each validation-selected threshold
        for thr_label, thr_val in threshold_catalog[name].items():
            y_pred = (t_proba >= thr_val).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))

            cost_res = compute_cost_breakdown(tn, fp, fn, tp, fn_cost_usd, fp_cost_usd)

            thr_entry = {
                "threshold": round(thr_val, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                "cost_analysis": cost_res,
            }
            if thr_label == "f1_optimal":
                thr_entry["recall_95ci"] = [round(boot_ci["recall_95ci"][0], 4), round(boot_ci["recall_95ci"][1], 4)]
                thr_entry["precision_95ci"] = [round(boot_ci["precision_95ci"][0], 4), round(boot_ci["precision_95ci"][1], 4)]

            model_entry["thresholds"][thr_label] = thr_entry

        results_data["models"][name] = model_entry

        # Tabular line for F1-optimal threshold
        opt_data = model_entry["thresholds"]["f1_optimal"]
        savings_str = f"${opt_data['cost_analysis']['net_savings']:,.0f}"
        pr_str = f"{pr_auc:.4f} [{boot_ci['pr_auc_95ci'][0]:.3f}, {boot_ci['pr_auc_95ci'][1]:.3f}]"
        rec_str = f"{opt_data['recall']:.1%} [{boot_ci['recall_95ci'][0]:.1%}, {boot_ci['recall_95ci'][1]:.1%}]"
        print(
            f"{name:<20} | {pr_str:<18} | {roc_auc:<10.4f} | {chosen_thr:<9.2f} | "
            f"{opt_data['precision']:<10.1%} | {rec_str:<10} | {opt_data['f1']:<10.4f} | {savings_str:<12}"
        )

    print("-" * 115)

    # 6. Save results to docs/results.json
    results_path = docs_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n[Artifacts] Saved comprehensive evaluation results: {results_path}")

    # 7. Generate publication-grade plots into docs/
    print("\n[Plots] Generating evaluation plots into docs/...")
    chosen_best = "ensemble" if "ensemble" in models else ("xgboost" if "xgboost" in models else list(models.keys())[0])
    generate_evaluation_plots(test_probas, y_test, chosen_best, threshold_catalog, docs_dir=docs_dir)

    print("\n[Evaluate] Evaluation pipeline completed successfully.")
    return results_data


if __name__ == "__main__":
    evaluate_all()
