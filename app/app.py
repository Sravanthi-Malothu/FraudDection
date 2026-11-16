"""
app.py — FraudLens Fraud Intelligence Platform
Flask Backend
"""

import os
import math
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATA_PATH  = "/Users/sravanthimalothu/Desktop/FraudDecetion/fraudTest.csv"
MODELS_DIR = "/Users/sravanthimalothu/Desktop/FraudDecetion/models"

# ── Load final model at startup ────────────────────────────────────────────────
final_model       = None
final_model_type  = None
le_category       = None
feature_cols      = None
model_metrics     = None

def load_models():
    global final_model, final_model_type, le_category, feature_cols, model_metrics
    final_path = os.path.join(MODELS_DIR, "final_model.joblib")
    le_path    = os.path.join(MODELS_DIR, "le_category.joblib")
    fc_path    = os.path.join(MODELS_DIR, "feature_cols.joblib")
    mt_path    = os.path.join(MODELS_DIR, "metrics.json")

    if os.path.exists(final_path) and os.path.exists(le_path) and os.path.exists(fc_path):
        final_model    = joblib.load(final_path)
        le_category    = joblib.load(le_path)
        feature_cols   = joblib.load(fc_path)
        if os.path.exists(mt_path):
            with open(mt_path) as f:
                model_metrics = json.load(f)
            final_model_type = model_metrics.get("final_model", {}).get("type")
        print("[FraudLens] Final model loaded successfully.")
    else:
        print("[FraudLens] WARNING: Final model files not found. Run train_model.py first.")

load_models()

# ── Pre-compute stats from first 50k rows ─────────────────────────────────────
_stats_cache = None

def compute_stats():
    global _stats_cache
    if _stats_cache:
        return _stats_cache

    print("[FraudLens] Computing dataset statistics...")
    df = pd.read_csv(DATA_PATH, nrows=50000)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour

    total         = int(len(df))
    fraud_count   = int(df["is_fraud"].sum())
    fraud_rate    = round(float(df["is_fraud"].mean() * 100), 2)
    avg_amount    = round(float(df["amt"].mean()), 2)
    max_amount    = round(float(df["amt"].max()), 2)

    # Fraud by category
    fbc = df.groupby("category")["is_fraud"].sum().sort_values(ascending=False)
    fraud_by_category = {k: int(v) for k, v in fbc.items()}

    # Fraud by hour
    fbh = df.groupby("hour")["is_fraud"].sum()
    fraud_by_hour = {str(int(k)): int(v) for k, v in fbh.items()}

    # Legit by category
    lbc = df[df["is_fraud"] == 0].groupby("category").size()
    legit_by_category = {k: int(v) for k, v in lbc.items()}

    _stats_cache = {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "legitimate_count": total - fraud_count,
        "fraud_rate": fraud_rate,
        "avg_amount": avg_amount,
        "max_amount": max_amount,
        "fraud_by_category": fraud_by_category,
        "legit_by_category": legit_by_category,
        "fraud_by_hour": fraud_by_hour,
    }
    print("[FraudLens] Stats computed.")
    return _stats_cache


# ── Feature engineering (mirrors train_model.py) ──────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda= math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_feature_row(data):
    """Build a single-row feature dict from API input."""
    amt         = float(data.get("amt", 50))
    category    = str(data.get("category", "misc_net"))
    city_pop    = float(data.get("city_pop", 10000))
    lat         = float(data.get("lat", 36.0))
    lon         = float(data.get("long", -86.0))
    merch_lat   = float(data.get("merch_lat", lat + 0.05))
    merch_long  = float(data.get("merch_long", lon + 0.05))
    hour        = int(data.get("hour", 12))
    gender      = str(data.get("gender", "M"))
    age         = float(data.get("age", 35))
    day_of_week = int(data.get("day_of_week", 2))
    month       = int(data.get("month", 6))

    amt_log         = math.log1p(amt)
    amt_per_city_pop= amt / (city_pop + 1)
    distance        = haversine(lat, lon, merch_lat, merch_long)
    gender_enc      = 1 if gender == "M" else 0

    # Category encode
    if le_category is not None:
        known = set(le_category.classes_)
        cat_safe = category if category in known else le_category.classes_[0]
        category_enc = int(le_category.transform([cat_safe])[0])
    else:
        category_enc = 0

    return {
        "amt": amt,
        "amt_log": amt_log,
        "amt_per_city_pop": amt_per_city_pop,
        "city_pop": city_pop,
        "distance": distance,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "age": age,
        "gender_enc": gender_enc,
        "category_enc": category_enc,
        "lat": lat,
        "long": lon,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        # Behavioural features for a single transaction with no prior card
        # history (defaults match the NaN guards in train_model.py):
        "txn_count_1h": 0,
        "txn_count_24h": 0,
        "amt_mean_card_past": 0,
        "amt_std_card_past": 1,
        "amt_zscore": 0,
        "time_since_last_txn": -1,
        "distance_from_last_txn_km": -1,
        "is_new_merchant_for_card": 1,
        "is_new_category_for_card": 1,
    }


def identify_risk_factors(row, confidence):
    factors = []
    if row["amt"] > 500:
        factors.append(f"High transaction amount (${row['amt']:.2f})")
    if row["distance"] > 100:
        factors.append(f"Large distance to merchant ({row['distance']:.1f} km)")
    if row["hour"] < 6 or row["hour"] > 22:
        factors.append(f"Unusual transaction hour ({row['hour']}:00)")
    if row["city_pop"] < 1000:
        factors.append("Low population area transaction")
    if row["amt_per_city_pop"] > 0.01:
        factors.append("High amount relative to city size")
    if not factors:
        factors.append("Transaction within normal parameters")
    return factors


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    try:
        stats = compute_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-metrics")
def api_model_metrics():
    if model_metrics is None:
        # Return fallback if models not trained yet
        return jsonify({
            "rf":  {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0},
            "xgb": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0},
            "status": "models_not_trained"
        })
    comparison = model_metrics.get("model_comparison", {})
    legacy = {}
    for key in ["random_forest", "xgboost", "ensemble"]:
        m = comparison.get(key, {}).get("test", {})
        cm = m.get("confusion_matrix", {})
        total = cm.get("tn", 0) + cm.get("fp", 0) + cm.get("fn", 0) + cm.get("tp", 0)
        accuracy = (cm.get("tn", 0) + cm.get("tp", 0)) / total if total else 0.0
        legacy[key] = {
            "accuracy": round(accuracy, 4),
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "f1": m.get("f1", 0),
            "auc": m.get("roc_auc", 0),
            "pr_auc": m.get("pr_auc", 0),
            "val_pr_auc": comparison.get(key, {}).get("val_pr_auc", 0),
        }
    legacy["final_model"] = model_metrics.get("final_model", {})
    legacy["status"] = "ok"
    return jsonify(legacy)


@app.route("/api/fraud-by-category")
def api_fraud_by_category():
    stats = compute_stats()
    return jsonify(stats["fraud_by_category"])


@app.route("/api/fraud-by-hour")
def api_fraud_by_hour():
    stats = compute_stats()
    return jsonify(stats["fraud_by_hour"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if final_model is None or feature_cols is None:
        return jsonify({
            "error": "Final model not loaded. Run train_model.py first.",
            "status": 503
        }), 503

    try:
        data = request.get_json(force=True)
        row  = build_feature_row(data)
        X    = pd.DataFrame([row])[feature_cols]

        proba = final_model.predict_proba(X)[0][1]
        confidence = float(proba)
        is_fraud   = int(confidence >= 0.5)

        if confidence >= 0.75:
            risk_level = "HIGH"
        elif confidence >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Expose component scores when the final model is an ensemble
        rf_score = xgb_score = confidence
        if (final_model_type == "ensemble"
                and hasattr(final_model, "model_a")
                and hasattr(final_model, "model_b")):
            rf_score  = float(final_model.model_a.predict_proba(X)[0][1])
            xgb_score = float(final_model.model_b.predict_proba(X)[0][1])

        risk_factors = identify_risk_factors(row, confidence)

        return jsonify({
            "is_fraud":     is_fraud,
            "confidence":   round(confidence * 100, 2),
            "rf_score":     round(rf_score * 100, 2),
            "xgb_score":    round(xgb_score * 100, 2),
            "risk_level":   risk_level,
            "risk_factors": risk_factors,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    compute_stats()   # warm up cache
    app.run(debug=True, host="0.0.0.0", port=8080)
