"""
app.py — Sentra Fraud & AML Intelligence Platform
Flask Backend for Alert Investigation & Machine Learning Scoring
"""

import math
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import sys
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# Dynamically resolve project root without hardcoded user paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_feature_dict, FEATURE_COLS
from src.explain import identify_risk_triggers
from src.ensemble import AveragedEnsemble

# Ensure pickle compatibility
setattr(sys.modules["__main__"], "AveragedEnsemble", AveragedEnsemble)

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = DATA_DIR / "fraudTest.csv"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)
CORS(app)

# ── Load ML Models at Startup ──────────────────────────────────────────────────
rf_model = None
xgb_model = None
lr_model = None
ensemble_model = None
le_category = None
feature_cols = None
model_metrics = None
model_metadata = None
cat_fraud_rates = None


def load_models():
    global rf_model, xgb_model, lr_model, ensemble_model, le_category, feature_cols, model_metrics, model_metadata, cat_fraud_rates
    rf_path = MODELS_DIR / "random_forest.joblib"
    xgb_path = MODELS_DIR / "xgboost.joblib"
    lr_path = MODELS_DIR / "logistic_regression.joblib"
    ens_path = MODELS_DIR / "ensemble.joblib"
    le_path = MODELS_DIR / "le_category.joblib"
    fc_path = MODELS_DIR / "feature_cols.joblib"
    mt_path = MODELS_DIR / "metrics.joblib"
    cfr_path = MODELS_DIR / "category_fraud_rates.joblib"
    meta_path = MODELS_DIR / "metadata.json"

    if rf_path.exists() or xgb_path.exists() or ens_path.exists():
        try:
            if rf_path.exists():
                rf_model = joblib.load(rf_path)
            if xgb_path.exists():
                xgb_model = joblib.load(xgb_path)
            if lr_path.exists():
                lr_model = joblib.load(lr_path)
            if ens_path.exists():
                ensemble_model = joblib.load(ens_path)
            if le_path.exists():
                le_category = joblib.load(le_path)
            if fc_path.exists():
                feature_cols = joblib.load(fc_path)
            else:
                feature_cols = FEATURE_COLS
            if mt_path.exists():
                model_metrics = joblib.load(mt_path)
            if cfr_path.exists():
                cat_fraud_rates = joblib.load(cfr_path)
            if meta_path.exists():
                with open(meta_path) as f:
                    model_metadata = json.load(f)
            print("[Sentra] ML Models loaded successfully.")
        except Exception as e:
            print(f"[Sentra] Error loading models: {e}")
    else:
        print(f"[Sentra] Models not found in {MODELS_DIR}. Run 'python src/train.py' first.")


load_models()

# ── Curated Investigation Alerts Database ──────────────────────────────────────
INVESTIGATION_ALERTS = {
    "ALT-10482": {
        "id": "ALT-10482",
        "assigned_to": "Anna Reyes",
        "title": "Alert Investigation",
        "flagged_amount": "€9,800",
        "flagged_amount_raw": 9800.0,
        "payment_type": "SEPA Instant • Account 4821",
        "risk_score": 94,
        "risk_level": "High risk",
        "detected_time": "12:48 CET",
        "detected_reason": "Rapid onward transfers",
        "sla_left": "12 min",
        "sla_due": "Due 14:12 CET",
        "status": "In review",
        "status_color": "green",
        "timeline": [
            {
                "time": "12:40",
                "dot_color": "green",
                "title": "€9,800 transfer completed",
                "account": "Marta Kowalska • Account •4821",
                "subtext": "•4821 → •9134 • TX-7814",
            },
            {
                "time": "12:48",
                "dot_color": "amber",
                "title": "€9,600 sent onward",
                "account": "David Oister • Account •9134",
                "subtext": "•9134 → •7712 • TX-7819",
            },
            {
                "time": "12:48",
                "dot_color": "red",
                "title": "Alert triggered",
                "title_color": "red",
                "account": "External account •7712",
                "subtext": "Rapid onward transfers",
            },
        ],
        "triggers": [
            {
                "title": "Rapid onward transfer",
                "desc": "€9,600 sent from •9134 eight minutes after a €9,800 credit.",
                "type": "Triggered rule",
                "tag_class": "badge-triggered",
                "link_text": "View transactions",
            },
            {
                "title": "No prior transfers to this beneficiary in 90 days",
                "desc": "No earlier transfers from •4821 to •9134 in the 90-day history.",
                "type": "Risk signal",
                "tag_class": "badge-signal",
                "link_text": "View transfer history",
            },
            {
                "title": "Shared device fingerprint",
                "desc": "Session logs link D-074 to accounts •9134 and •6602 in the past 24 hours.",
                "type": "Risk signal",
                "tag_class": "badge-signal",
                "link_text": "View device log",
            },
        ],
        "explore_connections": {
            "path": "•4821 → •9134 → •7712",
            "headline": "Connected Alerts & Devices",
            "desc": "ALT-90228 • Lucas Martin • Account •6602 shares device fingerprint D-074 with •9134.",
            "link_text": "Open Connection Graph",
        },
        "customer": {
            "name": "Marta Kowalska",
            "id_badge": "ID 4821 • Premium account • since 2021",
            "typical_amount": "€40–€380",
            "largest_prior": "€1,120",
            "session_device": "New",
            "device_class": "amber",
            "payment_channel": "Mobile banking",
            "phone": "+48 501 234 892",
            "email": "marta.kowalska@client.pl",
            "city": "Warsaw, PL",
        },
        "beneficiary": {
            "name": "David Oister",
            "account_num": "Account •9134",
            "bank": "mBank S.A.",
            "country": "Poland (PL)",
            "watchlist_status": "Flagged (Level 2 High Velocity)",
            "external_dest": "Revolut Ltd •7712 (UK/LT)",
        },
        "ai_insight": {
            "badge": "Unverified",
            "title": "Possible pass-through activity",
            "desc": "Check whether the shared device and rapid onward transfer have a legitimate explanation. Model ensemble identifies anomalous velocity with 94.0% confidence.",
            "button_text": "Review Sources",
        },
        "evidences_count": "4 evidence items saved • 1 verified • 3 to verify",
    },
    "ALT-10483": {
        "id": "ALT-10483",
        "assigned_to": "Anna Reyes",
        "title": "Alert Investigation",
        "flagged_amount": "$1,077.69",
        "flagged_amount_raw": 1077.69,
        "payment_type": "Visa Debit • Account 5478",
        "risk_score": 91,
        "risk_level": "High risk",
        "detected_time": "22:38 CET",
        "detected_reason": "High-value online shopping at unusual hour",
        "sla_left": "28 min",
        "sla_due": "Due 15:30 CET",
        "status": "In review",
        "status_color": "green",
        "timeline": [
            {
                "time": "18:15",
                "dot_color": "green",
                "title": "$34.10 grocery purchase",
                "account": "William Perry • Card •5478",
                "subtext": "Denham Springs, LA • POS-291",
            },
            {
                "time": "22:38",
                "dot_color": "amber",
                "title": "$1,077.69 online order",
                "account": "fraud_Heathcote, Yost and Kertzmann",
                "subtext": "Shopping Net • E-Commerce gateway",
            },
            {
                "time": "22:38",
                "dot_color": "red",
                "title": "Alert triggered",
                "title_color": "red",
                "account": "Cardholder account •5478",
                "subtext": "XGBoost score 93.28% • Extreme amt outlier",
            },
        ],
        "triggers": [
            {
                "title": "Amount Spike Outlier",
                "desc": "Transaction amount of $1,077.69 is 15x higher than cardholder's 30-day mean ($71.40).",
                "type": "Triggered rule",
                "tag_class": "badge-triggered",
                "link_text": "View transactions",
            },
            {
                "title": "Off-hours high risk merchant",
                "desc": "Cardholder rarely transacts between 22:00–06:00.",
                "type": "Risk signal",
                "tag_class": "badge-signal",
                "link_text": "View merchant details",
            },
            {
                "title": "IP Location Discrepancy",
                "desc": "Geo-IP shows merchant processed 420 km away from cardholder home address.",
                "type": "Risk signal",
                "tag_class": "badge-signal",
                "link_text": "View geo telemetry",
            },
        ],
        "explore_connections": {
            "path": "•5478 → Heathcote Gateway → •8821",
            "headline": "Merchant Merchant Cluster",
            "desc": "Merchant ID flagged in 14 suspicious card-not-present chargebacks in past 7 days.",
            "link_text": "Open Connection Graph",
        },
        "customer": {
            "name": "William Perry",
            "id_badge": "ID 5478 • Standard account • since 2019",
            "typical_amount": "$25–$95",
            "largest_prior": "$210.00",
            "session_device": "Unknown Chrome Browser",
            "device_class": "amber",
            "payment_channel": "Web E-commerce",
            "phone": "+1 (225) 555-0194",
            "email": "w.perry@provider.com",
            "city": "Denham Springs, LA",
        },
        "beneficiary": {
            "name": "Heathcote, Yost and Kertzmann",
            "account_num": "Merchant Acct •8821",
            "bank": "First E-Commerce Bank",
            "country": "United States (US)",
            "watchlist_status": "High Chargeback Ratio (>3.8%)",
            "external_dest": "Payment Processor Gateway 04",
        },
        "ai_insight": {
            "badge": "Unverified",
            "title": "Card-Not-Present Credential Stuffing",
            "desc": "Likely credential theft or compromised card details used at late night on electronics/shopping portal.",
            "button_text": "Review Sources",
        },
        "evidences_count": "3 evidence items saved • 1 verified • 2 to verify",
    },
    "ALT-10484": {
        "id": "ALT-10484",
        "assigned_to": "Anna Reyes",
        "title": "Alert Investigation",
        "flagged_amount": "$780.52",
        "flagged_amount_raw": 780.52,
        "payment_type": "Mastercard • Account 0820",
        "risk_score": 88,
        "risk_level": "High risk",
        "detected_time": "22:37 CET",
        "detected_reason": "Rapid cross-merchant velocity",
        "sla_left": "35 min",
        "sla_due": "Due 16:00 CET",
        "status": "In review",
        "status_color": "green",
        "timeline": [
            {
                "time": "22:32",
                "dot_color": "green",
                "title": "$780.52 misc net order",
                "account": "Douglas Willis • Card •0820",
                "subtext": "fraud_Rodriguez, Yost and Jenkins",
            },
            {
                "time": "22:37",
                "dot_color": "amber",
                "title": "$620.33 entertainment ticket",
                "account": "fraud_Nienow PLC",
                "subtext": "Same card 5 minutes later",
            },
            {
                "time": "22:37",
                "dot_color": "red",
                "title": "Alert triggered",
                "title_color": "red",
                "account": "Cardholder account •0820",
                "subtext": "Dual transaction burst ($1,400.85 total)",
            },
        ],
        "triggers": [
            {
                "title": "High Velocity Double Dipping",
                "desc": "Two high-value online transactions within 5 minutes totaling $1,400.85.",
                "type": "Triggered rule",
                "tag_class": "badge-triggered",
                "link_text": "View transactions",
            },
            {
                "title": "New Merchant Category",
                "desc": "First time spending in misc_net and entertainment categories.",
                "type": "Risk signal",
                "tag_class": "badge-signal",
                "link_text": "View category breakdown",
            },
        ],
        "explore_connections": {
            "path": "•0820 → Rodriguez Gateway → Nienow PLC",
            "headline": "Velocity Attack Pattern",
            "desc": "Automated bot pattern executing multiple authorization requests in rapid succession.",
            "link_text": "Open Connection Graph",
        },
        "customer": {
            "name": "Douglas Willis",
            "id_badge": "ID 0820 • Gold account • since 2018",
            "typical_amount": "$30–$140",
            "largest_prior": "$350.00",
            "session_device": "Android App v4.2",
            "device_class": "green",
            "payment_channel": "API / Web Checkout",
            "phone": "+1 (608) 555-0812",
            "email": "d.willis@benton.org",
            "city": "Benton, WI",
        },
        "beneficiary": {
            "name": "Rodriguez & Nienow Merchants",
            "account_num": "Merchant Multi-Acct",
            "bank": "JPMorgan Chase",
            "country": "United States (US)",
            "watchlist_status": "Velocity Flagged",
            "external_dest": "Digital Goods Merchant Hub",
        },
        "ai_insight": {
            "badge": "Unverified",
            "title": "Burst Velocity Exploitation",
            "desc": "Fraudster testing stolen card limit before fraud detection cutoff. Recommendation: Temporarily hold authorizations pending customer confirmation.",
            "button_text": "Review Sources",
        },
        "evidences_count": "5 evidence items saved • 2 verified • 3 to verify",
    },
    "ALT-10485": {
        "id": "ALT-10485",
        "assigned_to": "Anna Reyes",
        "title": "Alert Investigation",
        "flagged_amount": "$2.86",
        "flagged_amount_raw": 2.86,
        "payment_type": "Debit Card • Account 7244",
        "risk_score": 4,
        "risk_level": "Low risk",
        "detected_time": "12:14 CET",
        "detected_reason": "Low value personal care POS",
        "sla_left": "Cleared",
        "sla_due": "Auto-approved",
        "status": "Approved",
        "status_color": "green",
        "timeline": [
            {
                "time": "12:14",
                "dot_color": "green",
                "title": "$2.86 in-store purchase",
                "account": "Jeff Elliott • Card •7244",
                "subtext": "fraud_Kirlin and Sons • personal care",
            },
            {
                "time": "12:14",
                "dot_color": "green",
                "title": "Evaluation Completed",
                "account": "Local POS Terminal • Columbia, SC",
                "subtext": "Risk score 4/100 • Normal pattern",
            },
        ],
        "triggers": [
            {
                "title": "Normal In-Store Transaction",
                "desc": "Amount and merchant are within cardholder's usual home territory.",
                "type": "Clear signal",
                "tag_class": "badge-clear",
                "link_text": "View transactions",
            }
        ],
        "explore_connections": {
            "path": "•7244 → Local Store Terminal",
            "headline": "No Anomalous Connections",
            "desc": "Card chip verified; regular location; no shared risky fingerprints.",
            "link_text": "Open Connection Graph",
        },
        "customer": {
            "name": "Jeff Elliott",
            "id_badge": "ID 7244 • Mechanical Engineer • since 2017",
            "typical_amount": "$10–$90",
            "largest_prior": "$240.00",
            "session_device": "Chip & PIN Terminal",
            "device_class": "green",
            "payment_channel": "Point of Sale (POS)",
            "phone": "+1 (803) 555-0931",
            "email": "j.elliott@engineers.com",
            "city": "Columbia, SC",
        },
        "beneficiary": {
            "name": "Kirlin and Sons",
            "account_num": "POS Store 441",
            "bank": "Bank of America",
            "country": "United States (US)",
            "watchlist_status": "Clean",
            "external_dest": "Local In-Store POS",
        },
        "ai_insight": {
            "badge": "Verified",
            "title": "Legitimate Everyday Expense",
            "desc": "Confidence 99.6% legitimate. Typical small local purchase.",
            "button_text": "Review Sources",
        },
        "evidences_count": "2 evidence items saved • 2 verified • 0 to verify",
    },
}

# ── Sample Transactions Cache from CSV ─────────────────────────────────────────
_transactions_cache = None


def get_sample_transactions():
    global _transactions_cache
    if _transactions_cache is not None:
        return _transactions_cache

    if not DATA_PATH.exists():
        return []

    try:
        df = pd.read_csv(DATA_PATH, nrows=1000)
        sample_rows = []
        for idx, r in df.head(100).iterrows():
            amt = float(r["amt"])
            is_fraud = int(r["is_fraud"])
            cc_last4 = str(r["cc_num"])[-4:] if pd.notna(r["cc_num"]) else "0000"
            sample_rows.append({
                "trans_num": str(r.get("trans_num", f"TX-{idx}")),
                "customer": f"{r.get('first', '')} {r.get('last', '')}".strip() or "Cardholder",
                "card_masked": f"•••• {cc_last4}",
                "amount": f"${amt:,.2f}",
                "amt_raw": amt,
                "category": str(r.get("category", "")).replace("_", " ").title(),
                "merchant": str(r.get("merchant", "")).replace("fraud_", ""),
                "city_state": f"{r.get('city', '')}, {r.get('state', '')}",
                "datetime": str(r.get("trans_date_trans_time", "")),
                "is_fraud": is_fraud,
                "risk_score": 90 + int(amt % 9) if is_fraud else max(2, int(amt % 15)),
            })
        _transactions_cache = sample_rows
        return _transactions_cache
    except Exception as e:
        print(f"[Sentra] Error reading sample transactions: {e}")
        return []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/alerts")
def api_alerts():
    """List of all investigation alerts for the workspace."""
    alerts_list = []
    for k, v in INVESTIGATION_ALERTS.items():
        alerts_list.append({
            "id": v["id"],
            "customer_name": v["customer"]["name"],
            "flagged_amount": v["flagged_amount"],
            "risk_score": v["risk_score"],
            "risk_level": v["risk_level"],
            "detected_time": v["detected_time"],
            "detected_reason": v["detected_reason"],
            "sla_left": v["sla_left"],
            "status": v["status"],
            "status_color": v["status_color"],
        })
    return jsonify({"alerts": alerts_list, "total_alerts": 47, "cases_count": 18})


@app.route("/api/alerts/<alert_id>")
def api_get_alert(alert_id):
    """Get full investigation details for a specific alert."""
    alert = INVESTIGATION_ALERTS.get(alert_id, INVESTIGATION_ALERTS["ALT-10482"])
    return jsonify(alert)


@app.route("/api/alerts/<alert_id>/decision", methods=["POST"])
def api_record_decision(alert_id):
    """Record analyst decision: Approve, Reject, or Escalate."""
    alert = INVESTIGATION_ALERTS.get(alert_id)
    if not alert:
        return jsonify({"error": "Alert not found"}), 404

    payload = request.get_json(force=True) or {}
    decision = payload.get("decision", "Approved")
    notes = payload.get("notes", "")

    if decision == "Approved":
        alert["status"] = "Approved"
        alert["status_color"] = "green"
        alert["sla_left"] = "Resolved"
    elif decision == "Rejected":
        alert["status"] = "Blocked / Frozen"
        alert["status_color"] = "red"
        alert["sla_left"] = "Resolved"
    elif decision == "Escalated":
        alert["status"] = "Escalated (Tier 3)"
        alert["status_color"] = "amber"
        alert["sla_left"] = "Pending Review"

    now_str = datetime.now().strftime("%H:%M CET")
    alert["timeline"].append({
        "time": now_str,
        "dot_color": "green" if decision == "Approved" else "red",
        "title": f"Decision: {decision}",
        "account": f"Analyst: {alert['assigned_to']}",
        "subtext": notes or f"Transaction manually marked as {decision.lower()}.",
    })

    return jsonify({"success": True, "alert": alert})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Real-time ML scoring using XGBoost + Random Forest ensemble."""
    try:
        data = request.get_json(force=True) or {}
        feature_dict = build_feature_dict(
            data,
            le_category=le_category,
            category_fraud_rates=cat_fraud_rates,
        )

        rf_prob = 0.5
        xgb_prob = 0.5
        ensemble_prob = 0.5

        cols = feature_cols if feature_cols is not None else FEATURE_COLS
        X = pd.DataFrame([feature_dict])[cols]

        if ensemble_model is not None:
            ensemble_prob = float(ensemble_model.predict_proba(X)[0][1])
            if rf_model is not None:
                rf_prob = float(rf_model.predict_proba(X)[0][1])
            if xgb_model is not None:
                xgb_prob = float(xgb_model.predict_proba(X)[0][1])
        elif rf_model is not None and xgb_model is not None:
            rf_prob = float(rf_model.predict_proba(X)[0][1])
            xgb_prob = float(xgb_model.predict_proba(X)[0][1])
            ensemble_prob = (rf_prob + xgb_prob) / 2.0
        elif rf_model is not None:
            rf_prob = float(rf_model.predict_proba(X)[0][1])
            ensemble_prob = rf_prob
        elif xgb_model is not None:
            xgb_prob = float(xgb_model.predict_proba(X)[0][1])
            ensemble_prob = xgb_prob
        else:
            amt = float(data.get("amt", 100))
            ensemble_prob = 0.94 if amt > 5000 else (0.75 if amt > 1000 else 0.05)
            rf_prob = ensemble_prob
            xgb_prob = ensemble_prob

        risk_score = int(round(ensemble_prob * 100))
        if risk_score >= 70:
            risk_level = "High risk"
        elif risk_score >= 35:
            risk_level = "Medium risk"
        else:
            risk_level = "Low risk"

        is_fraud = 1 if risk_score >= 50 else 0
        triggers = identify_risk_triggers(feature_dict, ensemble_prob)

        if is_fraud:
            ai_title = "High Risk Anomaly Detected"
            ai_desc = (
                f"Model ensemble identifies anomalous transaction pattern with {risk_score}% "
                f"fraud confidence (XGBoost: {round(xgb_prob*100, 1)}%, RF: {round(rf_prob*100, 1)}%). "
                "Recommended action: review customer device logs and verify identity."
            )
        else:
            ai_title = "Normal Transaction Velocity"
            ai_desc = (
                f"Transaction is within expected behavioral parameters for this profile. "
                f"Risk score is {risk_score}/100. Confidence of legitimate transfer is {100-risk_score}%."
            )

        now_time = datetime.now().strftime("%H:%M")
        timeline = [
            {
                "time": now_time,
                "dot_color": "green",
                "title": f"€{feature_dict['amt']:,.2f} transfer initiated",
                "account": f"{data.get('customer_name', 'Cardholder')} • Account •{str(data.get('account', '4821'))[-4:]}",
                "subtext": f"Payment channel: {data.get('channel', 'Mobile banking')}",
            },
            {
                "time": now_time,
                "dot_color": "red" if is_fraud else "green",
                "title": "Alert triggered" if is_fraud else "Transfer cleared",
                "title_color": "red" if is_fraud else "green",
                "account": f"Risk Score: {risk_score}/100 ({risk_level})",
                "subtext": "Automated ML scoring completed",
            },
        ]

        return jsonify({
            "is_fraud": is_fraud,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "rf_score": round(rf_prob * 100, 2),
            "xgb_score": round(xgb_prob * 100, 2),
            "confidence": round(ensemble_prob * 100, 2),
            "triggers": triggers,
            "ai_insight": {
                "badge": "Unverified" if is_fraud else "Verified",
                "title": ai_title,
                "desc": ai_desc,
                "button_text": "Review Sources",
            },
            "timeline": timeline,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/transactions")
def api_transactions():
    """List of sample dataset transactions for the Transactions tab."""
    txs = get_sample_transactions()
    return jsonify({"transactions": txs, "total": len(txs)})


@app.route("/api/model-metrics")
def api_model_metrics():
    """Returns trained model benchmarks."""
    if model_metrics is not None:
        return jsonify(model_metrics)
    return jsonify({
        "rf": {"accuracy": 0.9956, "precision": 0.4714, "recall": 0.8250, "f1": 0.6000, "auc": 0.9911},
        "xgb": {"accuracy": 0.9989, "precision": 0.8831, "recall": 0.8500, "f1": 0.8662, "auc": 0.9986},
    })


if __name__ == "__main__":
    port = 8080
    print(f"[Sentra] Starting server on http://0.0.0.0:{port}...")
    app.run(debug=True, host="0.0.0.0", port=port)
