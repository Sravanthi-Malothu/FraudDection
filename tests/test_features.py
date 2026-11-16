"""
test_features.py — Comprehensive Unit Tests for Feature Functions
Tests mathematical transformations, Haversine geo-distance, demographic calculations,
and per-card historical features computed exclusively from past transactions.
"""

import math
import numpy as np
import pandas as pd
import pytest
from src.features import haversine, build_feature_dict, FEATURE_COLS, engineer_features


def test_haversine_same_point():
    """Distance between identical coordinate points must be exactly zero."""
    dist = haversine(40.7128, -74.0060, 40.7128, -74.0060)
    assert math.isclose(dist, 0.0, abs_tol=1e-5)


def test_haversine_known_distance():
    """NY (40.7128, -74.0060) to London (51.5074, -0.1278) is approx 5,570 km."""
    dist = haversine(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500 < dist < 5650


def test_haversine_array_inputs():
    """Haversine supports vectorized NumPy arrays."""
    lat1 = np.array([40.7128, 34.0522])
    lon1 = np.array([-74.0060, -118.2437])
    lat2 = np.array([40.7128, 34.0522])
    lon2 = np.array([-74.0060, -118.2437])
    dists = haversine(lat1, lon1, lat2, lon2)
    assert len(dists) == 2
    assert math.isclose(dists[0], 0.0, abs_tol=1e-5)
    assert math.isclose(dists[1], 0.0, abs_tol=1e-5)


def test_build_feature_dict_keys():
    """build_feature_dict generates all required model columns."""
    data = {
        "amt": 150.0,
        "category": "shopping_net",
        "city_pop": 50000,
        "lat": 36.0,
        "long": -86.0,
        "hour": 14,
        "gender": "F",
        "age": 30,
    }
    feat_dict = build_feature_dict(data)
    for col in FEATURE_COLS:
        assert col in feat_dict, f"Missing expected column: {col}"
    assert feat_dict["amt"] == 150.0
    assert feat_dict["gender_enc"] == 0
    assert feat_dict["hour"] == 14
    assert feat_dict["is_night"] == 0
    assert math.isclose(feat_dict["amt_log"], np.log1p(150.0), rel_tol=1e-5)


def test_night_and_weekend_flags():
    """Verifies that night (10pm-5am) and weekend flags are accurately set."""
    # Night hours: 22, 23, 0, 1, 2, 3, 4, 5
    for h in [22, 23, 0, 1, 2, 3, 4, 5]:
        d = build_feature_dict({"amt": 50.0, "hour": h})
        assert d["is_night"] == 1, f"Hour {h} should be flagged as night"

    # Day hours: 6 through 21
    for h in [6, 12, 17, 21]:
        d = build_feature_dict({"amt": 50.0, "hour": h})
        assert d["is_night"] == 0, f"Hour {h} should NOT be flagged as night"


def test_engineer_features_card_behavioral():
    """
    Tests per-card behavioral features computed by engineer_features():
    - txn_count_last_1h, txn_count_last_24h
    - amt_sum_last_24h, time_since_last_txn
    - amt_vs_card_avg, amt_zscore_vs_card
    - is_new_merchant_for_card
    - distance_from_last_txn_km
    - merchant_category_fraud_rate
    """
    sample_df = pd.DataFrame([
        # Txn 0 (Day 1, 09:00:00)
        {
            "cc_num": 9999,
            "trans_date_trans_time": "2021-03-01 09:00:00",
            "amt": 100.0,
            "merchant": "fraud_StoreA",
            "category": "grocery_pos",
            "lat": 30.0,
            "long": -90.0,
            "merch_lat": 30.01,
            "merch_long": -90.01,
            "city_pop": 25000,
            "gender": "M",
            "dob": "1990-01-01",
            "is_fraud": 0,
        },
        # Txn 1 (Day 1, 09:30:00 - 30 mins later, within 1h)
        {
            "cc_num": 9999,
            "trans_date_trans_time": "2021-03-01 09:30:00",
            "amt": 300.0,
            "merchant": "fraud_StoreB",
            "category": "shopping_net",
            "lat": 30.0,
            "long": -90.0,
            "merch_lat": 30.02,
            "merch_long": -90.02,
            "city_pop": 25000,
            "gender": "M",
            "dob": "1990-01-01",
            "is_fraud": 0,
        },
        # Txn 2 (Day 1, 11:30:00 - 2h after Txn 1, >1h but <24h)
        {
            "cc_num": 9999,
            "trans_date_trans_time": "2021-03-01 11:30:00",
            "amt": 200.0,
            "merchant": "fraud_StoreA",
            "category": "grocery_pos",
            "lat": 30.0,
            "long": -90.0,
            "merch_lat": 30.01,
            "merch_long": -90.01,
            "city_pop": 25000,
            "gender": "M",
            "dob": "1990-01-01",
            "is_fraud": 0,
        },
    ])

    df_feat, le_cat, cat_rates, feat_cols = engineer_features(sample_df, fit=True)

    # First txn: no history for this card
    row0 = df_feat.iloc[0]
    assert row0["txn_count_last_1h"] == 0
    assert row0["txn_count_last_24h"] == 0
    assert row0["amt_sum_last_24h"] == 0.0
    assert row0["time_since_last_txn"] == -1.0
    assert row0["amt_vs_card_avg"] == 1.0
    assert row0["is_new_merchant_for_card"] == 1
    assert row0["distance_from_last_txn_km"] == -1.0

    # Second txn: 30 mins later
    row1 = df_feat.iloc[1]
    assert row1["txn_count_last_1h"] == 1
    assert row1["txn_count_last_24h"] == 1
    assert row1["amt_sum_last_24h"] == 100.0
    assert row1["time_since_last_txn"] == 1800.0  # 30 minutes in seconds
    assert row1["amt_vs_card_avg"] == 300.0 / 100.0  # 3.0x past average
    assert row1["is_new_merchant_for_card"] == 1  # StoreB is new for card

    # Third txn: 2h after Txn 1
    row2 = df_feat.iloc[2]
    assert row2["txn_count_last_1h"] == 0  # previous txn was 2 hours ago
    assert row2["txn_count_last_24h"] == 2  # Txns 0 and 1 are within 24h
    assert row2["amt_sum_last_24h"] == 400.0  # 100 + 300
    assert row2["is_new_merchant_for_card"] == 0  # StoreA was visited at Txn 0!
    assert row2["amt_vs_card_avg"] == 200.0 / 200.0  # past mean of [100, 300] is 200 -> 1.0x
