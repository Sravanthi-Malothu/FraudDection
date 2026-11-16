"""
test_explain.py — Unit tests for SHAP explainability and plain-English reasons
"""

import pytest
import numpy as np
import pandas as pd
from src.features import FEATURE_COLS
from src.explain import explain_transaction, generate_feature_reason


def test_explain_transaction_dict_input():
    sample = {
        "amt": 950.0,
        "amt_vs_card_avg": 7.5,
        "amt_zscore_vs_card": 4.2,
        "txn_count_last_1h": 3,
        "txn_count_last_24h": 8,
        "amt_sum_last_24h": 2400.0,
        "time_since_last_txn": 45.0,
        "distance_from_last_txn_km": 120.0,
        "merchant_category_fraud_rate": 0.035,
        "is_night": 1,
        "is_weekend": 1,
        "is_new_merchant_for_card": 1,
        "distance": 85.0,
        "hour": 23,
        "category": "shopping_net",
        "city_pop": 50000,
        "age": 32,
    }
    explanation = explain_transaction(sample, top_k=5)

    assert "fraud_probability" in explanation
    assert "risk_score" in explanation
    assert "risk_level" in explanation
    assert "top_features" in explanation
    assert len(explanation["top_features"]) == 5

    # Check structure of top features
    for item in explanation["top_features"]:
        assert "feature" in item
        assert "value" in item
        assert "shap_value" in item
        assert item["impact"] in ("increases_risk", "decreases_risk")
        assert len(item["reason"]) > 5


def test_plain_english_reasons():
    reason1 = generate_feature_reason("amt_vs_card_avg", 8.4, 2.1)
    assert "8.4x higher than this card's historical average" in reason1

    reason2 = generate_feature_reason("time_since_last_txn", 30, 1.5)
    assert "30 seconds" in reason2

    reason3 = generate_feature_reason("is_night", 1, 0.8)
    assert "overnight hours" in reason3
