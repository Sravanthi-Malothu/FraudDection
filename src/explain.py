"""
explain.py — Model Explainability & Risk Attribution
Provides rule-based domain signals and SHAP feature attributions.
"""

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def identify_risk_triggers(feature_dict, confidence=0.5):
    """
    Identifies plain-English risk factors and AML signals based on feature thresholds.
    """
    factors = []
    amt = float(feature_dict.get("amt", 0.0))
    dist = float(feature_dict.get("distance", 0.0))
    hour = int(feature_dict.get("hour", 12))
    city_pop = float(feature_dict.get("city_pop", 10000.0))
    amt_per_pop = float(feature_dict.get("amt_per_city_pop", 0.0))

    if amt >= 5000:
        factors.append({
            "title": "Unusual High-Value Transfer",
            "desc": f"Transaction amount of €{amt:,.2f} is significantly above historical baseline.",
            "type": "Triggered rule",
            "tag_class": "badge-triggered",
            "link_text": "View transactions",
        })
    elif amt >= 800:
        factors.append({
            "title": "Substantial Transaction Amount",
            "desc": f"Amount of €{amt:,.2f} exceeds 95th percentile for category.",
            "type": "Triggered rule",
            "tag_class": "badge-triggered",
            "link_text": "View transactions",
        })

    if hour < 6 or hour >= 22:
        factors.append({
            "title": "Off-Hours Activity",
            "desc": f"Initiated at {hour:02d}:00 outside standard cardholder activity windows.",
            "type": "Risk signal",
            "tag_class": "badge-signal",
            "link_text": "View session log",
        })

    if dist > 80:
        factors.append({
            "title": "Cross-Border / Long Distance Velocity",
            "desc": f"Terminal/merchant located {dist:.1f} km from cardholder billing address.",
            "type": "Risk signal",
            "tag_class": "badge-signal",
            "link_text": "View geo telemetry",
        })

    if city_pop < 1500 and amt > 200:
        factors.append({
            "title": "High Amount to Population Density Ratio",
            "desc": f"Large spend in low-density population area (pop. {int(city_pop):,}).",
            "type": "Risk signal",
            "tag_class": "badge-signal",
            "link_text": "View demographic log",
        })

    if not factors:
        factors.append({
            "title": "Transaction within Normal Baseline",
            "desc": "Amount, geographic distance, and timing align with legitimate consumer patterns.",
            "type": "Clear signal",
            "tag_class": "badge-clear",
            "link_text": "View audit record",
        })

    return factors


def get_tree_shap_explainer(model):
    """
    Initializes SHAP TreeExplainer if SHAP is available.
    """
    if not HAS_SHAP:
        return None
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return None


def explain_instance_shap(explainer, X_row, feature_names, top_k=5):
    """
    Returns top-k feature contributions for a single instance using SHAP.
    """
    if explainer is None:
        return []

    try:
        shap_values = explainer.shap_values(X_row)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class (fraud)

        row_shap = np.array(shap_values[0])
        order = np.argsort(-np.abs(row_shap))[:top_k]

        explanations = []
        for idx in order:
            feat = feature_names[idx]
            val = float(X_row.iloc[0, idx]) if hasattr(X_row, "iloc") else float(X_row[0][idx])
            s_val = float(row_shap[idx])
            direction = "increases" if s_val > 0 else "decreases"
            explanations.append({
                "feature": feat,
                "value": val,
                "shap_value": s_val,
                "summary": f"{feat} ({val:.2f}) {direction} fraud risk ({s_val:+.3f})",
            })
        return explanations
    except Exception:
        return []
