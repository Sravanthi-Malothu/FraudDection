"""
test_no_future_leakage.py
Unit tests proving that per-card features use ONLY past transactions
and are strictly invariant to any future transactions or modifications.
"""

import numpy as np
import pandas as pd
import pytest
from src.features import engineer_features, FEATURE_COLS


def create_sample_transaction_history():
    """
    Creates a realistic sequential multi-card transaction history.
    Card 101 has 5 transactions over 2 days.
    Card 202 has 3 transactions interspersed.
    """
    data = [
        # Card 101: Txn 0 (Day 1, 10:00)
        {"cc_num": 101, "trans_date_trans_time": "2021-01-01 10:00:00", "amt": 50.0,
         "merchant": "M_GROCERY", "category": "grocery_pos", "lat": 40.71, "long": -74.00,
         "merch_lat": 40.72, "merch_long": -74.01, "city_pop": 50000, "gender": "F", "dob": "1985-05-15", "is_fraud": 0},
        # Card 202: Txn 1 (Day 1, 10:15)
        {"cc_num": 202, "trans_date_trans_time": "2021-01-01 10:15:00", "amt": 30.0,
         "merchant": "M_COFFEE", "category": "food_dining", "lat": 34.05, "long": -118.24,
         "merch_lat": 34.06, "merch_long": -118.25, "city_pop": 100000, "gender": "M", "dob": "1990-08-20", "is_fraud": 0},
        # Card 101: Txn 2 (Day 1, 10:45 - 45 min after Txn 0, within 1h)
        {"cc_num": 101, "trans_date_trans_time": "2021-01-01 10:45:00", "amt": 150.0,
         "merchant": "M_GAS", "category": "gas_transport", "lat": 40.71, "long": -74.00,
         "merch_lat": 40.75, "merch_long": -74.05, "city_pop": 50000, "gender": "F", "dob": "1985-05-15", "is_fraud": 0},
        # Card 101: Txn 3 (Day 1, 12:30 - 2.5h after Txn 0, outside 1h, inside 24h)
        {"cc_num": 101, "trans_date_trans_time": "2021-01-01 12:30:00", "amt": 200.0,
         "merchant": "M_GROCERY", "category": "grocery_pos", "lat": 40.71, "long": -74.00,
         "merch_lat": 40.72, "merch_long": -74.01, "city_pop": 50000, "gender": "F", "dob": "1985-05-15", "is_fraud": 0},
        # Card 202: Txn 4 (Day 1, 18:00)
        {"cc_num": 202, "trans_date_trans_time": "2021-01-01 18:00:00", "amt": 90.0,
         "merchant": "M_RESTAURANT", "category": "food_dining", "lat": 34.05, "long": -118.24,
         "merch_lat": 34.10, "merch_long": -118.30, "city_pop": 100000, "gender": "M", "dob": "1990-08-20", "is_fraud": 0},
        # Card 101: Txn 5 (Day 2, 09:00 - 23h after Txn 0, within 24h of Txns 2 and 3)
        {"cc_num": 101, "trans_date_trans_time": "2021-01-02 09:00:00", "amt": 80.0,
         "merchant": "M_ONLINE", "category": "shopping_net", "lat": 40.71, "long": -74.00,
         "merch_lat": 41.50, "merch_long": -73.50, "city_pop": 50000, "gender": "F", "dob": "1985-05-15", "is_fraud": 0},
        # Card 101: Txn 6 (Day 2, 15:00 - over 24h after Day 1 morning txns)
        {"cc_num": 101, "trans_date_trans_time": "2021-01-02 15:00:00", "amt": 300.0,
         "merchant": "M_ELECTRONICS", "category": "shopping_pos", "lat": 40.71, "long": -74.00,
         "merch_lat": 40.80, "merch_long": -74.10, "city_pop": 50000, "gender": "F", "dob": "1985-05-15", "is_fraud": 0},
        # Card 202: Txn 7 (Day 2, 23:30 - Night txn)
        {"cc_num": 202, "trans_date_trans_time": "2021-01-02 23:30:00", "amt": 500.0,
         "merchant": "M_HOTEL", "category": "travel", "lat": 34.05, "long": -118.24,
         "merch_lat": 36.10, "merch_long": -115.15, "city_pop": 100000, "gender": "M", "dob": "1990-08-20", "is_fraud": 1},
    ]
    return pd.DataFrame(data)


def test_per_card_features_mathematical_correctness():
    """
    Verifies that rolling and expanding window values match expected ground truth.
    """
    df = create_sample_transaction_history()
    df_feat, _, _, _ = engineer_features(df, fit=True)

    # Card 101 rows: indices 0, 2, 3, 5, 6
    row0 = df_feat.iloc[0]  # Card 101, Txn 0 (Day 1, 10:00, $50, M_GROCERY)
    assert row0["txn_count_last_1h"] == 0
    assert row0["txn_count_last_24h"] == 0
    assert row0["amt_sum_last_24h"] == 0.0
    assert row0["time_since_last_txn"] == -1.0
    assert row0["amt_vs_card_avg"] == 1.0
    assert row0["amt_zscore_vs_card"] == 0.0
    assert row0["is_new_merchant_for_card"] == 1
    assert row0["distance_from_last_txn_km"] == -1.0

    row2 = df_feat.iloc[2]  # Card 101, Txn 2 (Day 1, 10:45, $150, M_GAS - 45m later)
    assert row2["txn_count_last_1h"] == 1
    assert row2["txn_count_last_24h"] == 1
    assert row2["amt_sum_last_24h"] == 50.0
    assert row2["time_since_last_txn"] == 45 * 60  # 2700s
    assert row2["amt_vs_card_avg"] == 150.0 / 50.0  # 3.0x past average
    assert row2["is_new_merchant_for_card"] == 1
    assert row2["distance_from_last_txn_km"] > 0

    row3 = df_feat.iloc[3]  # Card 101, Txn 3 (Day 1, 12:30, $200, M_GROCERY)
    assert row3["txn_count_last_1h"] == 0  # 10:45 is > 1h ago
    assert row3["txn_count_last_24h"] == 2  # 10:00 and 10:45 are within 24h
    assert row3["amt_sum_last_24h"] == 50.0 + 150.0  # $200.0 past sum
    assert row3["is_new_merchant_for_card"] == 0  # M_GROCERY was seen at Txn 0!
    assert row3["amt_vs_card_avg"] == 200.0 / 100.0  # past mean of [50, 150] is 100 -> 2.0x

    # Night & weekend test
    row7 = df_feat.iloc[7]  # 2021-01-02 23:30 (Saturday night 11:30 PM)
    assert row7["is_night"] == 1
    assert row7["is_weekend"] == 1


def test_strict_invariance_to_future_data():
    """
    PROVES ZERO FUTURE DATA LEAKAGE:
    Features computed for historical transactions (up to index T) must be
    100% invariant to any modifications, deletions, or additions of future rows.
    """
    df_original = create_sample_transaction_history()

    # Pre-compute features on original data
    df_feat_orig, le, cat_rates, glob_rate = engineer_features(df_original, fit=True)

    # Cutoff at index T = 4 (evaluate rows 0 through 4)
    T = 4
    past_features_before = df_feat_orig.iloc[: T + 1][FEATURE_COLS].copy()

    # Create modified scenario: alter FUTURE rows (5, 6, 7) drastically
    df_altered = df_original.copy()
    # Drastic future changes: $10,000,000 spend, new future merchants, new coordinates
    df_altered.loc[T + 1 :, "amt"] = [10_000_000.0, 50_000_000.0, 99_000_000.0]
    df_altered.loc[T + 1 :, "merchant"] = ["FUTURE_MULE_1", "FUTURE_MULE_2", "FUTURE_MULE_3"]
    df_altered.loc[T + 1 :, "merch_lat"] = [89.0, 89.5, 89.9]
    df_altered.loc[T + 1 :, "is_fraud"] = [1, 1, 1]

    # Append 3 extra future transactions days later
    extra_future_rows = pd.DataFrame([
        {"cc_num": 101, "trans_date_trans_time": "2021-01-10 10:00:00", "amt": 5000.0,
         "merchant": "FUTURE_STORE", "category": "shopping_net", "lat": 40.71, "long": -74.00,
         "merch_lat": 40.72, "merch_long": -74.01, "city_pop": 50000, "gender": "F", "dob": "1985-05-15", "is_fraud": 0},
        {"cc_num": 202, "trans_date_trans_time": "2021-01-10 12:00:00", "amt": 7500.0,
         "merchant": "FUTURE_STORE", "category": "shopping_net", "lat": 34.05, "long": -118.24,
         "merch_lat": 34.06, "merch_long": -118.25, "city_pop": 100000, "gender": "M", "dob": "1990-08-20", "is_fraud": 0},
    ])
    df_altered = pd.concat([df_altered, extra_future_rows], ignore_index=True)

    # Compute features on the altered dataset using the same training encoder/rates
    df_feat_altered, _, _, _ = engineer_features(
        df_altered, le_category=le, category_fraud_rates=cat_rates, global_fraud_rate=glob_rate, fit=False
    )
    past_features_after = df_feat_altered.iloc[: T + 1][FEATURE_COLS].copy()

    # PROOF: Every single feature column for every past row <= T must be strictly equal
    for col in FEATURE_COLS:
        col_before = past_features_before[col].values
        col_after = past_features_after[col].values

        if np.issubdtype(col_before.dtype, np.floating):
            np.testing.assert_allclose(
                col_before,
                col_after,
                rtol=1e-7,
                atol=1e-7,
                err_msg=f"Future data leakage detected in feature '{col}'! Past values changed when future data changed.",
            )
        else:
            assert np.array_equal(
                col_before, col_after
            ), f"Future data leakage detected in feature '{col}'! Past values changed when future data changed."


def test_isolated_prediction_without_card_history():
    """
    Tests that single-record prediction without cc_num works gracefully
    with baseline defaults and returns all 26 feature columns.
    """
    df_single = pd.DataFrame([{
        "trans_date_trans_time": "2021-05-01 14:30:00",
        "amt": 95.0,
        "category": "grocery_pos",
        "lat": 36.1627,
        "long": -86.7816,
        "merch_lat": 36.1700,
        "merch_long": -86.7900,
        "city_pop": 25000,
        "gender": "F",
        "dob": "1980-01-01",
    }])

    df_feat, _, _, _ = engineer_features(df_single, fit=False)

    for col in FEATURE_COLS:
        assert col in df_feat.columns, f"Missing feature: {col}"
        assert not df_feat[col].isna().any(), f"NaN found in feature: {col}"

    assert df_feat["txn_count_last_1h"].iloc[0] == 0
    assert df_feat["time_since_last_txn"].iloc[0] == -1.0
    assert df_feat["amt_vs_card_avg"].iloc[0] == 1.0
    assert df_feat["is_night"].iloc[0] == 0
    assert df_feat["is_weekend"].iloc[0] == 1  # 2021-05-01 is Saturday
