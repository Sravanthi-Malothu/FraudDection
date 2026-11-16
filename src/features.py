"""
features.py — Feature Engineering for Fraud Detection
Handles mathematical transformations, Haversine geo-distance,
demographic extraction, categorical encoding, and strictly past-only
per-card behavioral features (zero future leakage).
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Explicit feature columns used for model input (excludes leaking identifiers)
FEATURE_COLS = [
    # Existing baseline features
    "amt",
    "amt_log",
    "amt_per_city_pop",
    "city_pop",
    "distance",
    "hour",
    "day_of_week",
    "month",
    "age",
    "gender_enc",
    "category_enc",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    # Per-card behavioral features (past transactions only)
    "txn_count_last_1h",
    "txn_count_last_24h",
    "amt_sum_last_24h",
    "time_since_last_txn",
    "amt_vs_card_avg",
    "amt_zscore_vs_card",
    "is_new_merchant_for_card",
    "merchant_category_fraud_rate",
    "distance_from_last_txn_km",
    "is_night",
    "is_weekend",
]

# Sensitive or leaking identifier columns explicitly excluded from model features
IDENTIFIER_COLS = [
    "cc_num",
    "trans_num",
    "first",
    "last",
    "street",
    "city",
    "state",
    "zip",
    "job",
    "dob",
    "trans_date_trans_time",
    "unix_time",
    "merchant",
    "Unnamed: 0",
]

KNOWN_CATEGORIES = [
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
]


def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between coordinate pairs in kilometers.
    Accepts floats or numpy arrays.
    """
    R = 6371.0  # Earth radius in kilometers

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def add_past_only_card_features(df):
    """
    Computes per-card behavioral features using ONLY past transactions.
    Uses shift and rolling windows with closed='left' so that the current
    transaction row is strictly excluded from its own historical statistics.
    """
    df = df.copy()

    # Preserve original row ordering
    has_orig_idx = "__orig_idx__" in df.columns
    if not has_orig_idx:
        df["__orig_idx__"] = np.arange(len(df))

    # Sort strictly by (cc_num, trans_date_trans_time)
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    # 3. time_since_last_txn (seconds) - diff between consecutive card transactions
    df["time_since_last_txn"] = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .diff()
        .dt.total_seconds()
        .fillna(-1.0)
    )

    # 8. distance_from_last_txn_km (between current location and immediately preceding merchant location)
    if "merch_lat" in df.columns and "merch_long" in df.columns and "lat" in df.columns and "long" in df.columns:
        prev_merch_lat = df.groupby("cc_num")["merch_lat"].shift(1)
        prev_merch_long = df.groupby("cc_num")["merch_long"].shift(1)
        dist_last = haversine(
            df["lat"].values, df["long"].values, prev_merch_lat.values, prev_merch_long.values
        )
        df["distance_from_last_txn_km"] = np.nan_to_num(dist_last, nan=-1.0)
    else:
        df["distance_from_last_txn_km"] = -1.0

    # 6. is_new_merchant_for_card (1 if first time this card used this merchant, 0 otherwise)
    if "merchant" in df.columns:
        df["is_new_merchant_for_card"] = (
            ~df.duplicated(subset=["cc_num", "merchant"], keep="first")
        ).astype(int)
    else:
        df["is_new_merchant_for_card"] = 1

    # 4 & 5. Card expanding average and std (shifted by 1 to exclude current row)
    # Using groupby transform/apply
    past_mean = (
        df.groupby("cc_num")["amt"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    past_std = (
        df.groupby("cc_num")["amt"]
        .apply(lambda s: s.expanding().std().shift(1))
        .reset_index(level=0, drop=True)
    )

    # 4. amt_vs_card_avg: current amount / past average
    # When past_mean is NaN or 0 (first txn for card), defaults to 1.0
    amt_ratio = df["amt"] / past_mean
    df["amt_vs_card_avg"] = amt_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # 5. amt_zscore_vs_card: (current amount - past mean) / past std
    # When past_std is NaN or 0 (fewer than 2 past txns), defaults to 0.0
    zscore = (df["amt"] - past_mean) / past_std
    df["amt_zscore_vs_card"] = zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 1 & 2. Rolling time windows per card: strictly closed='left' to exclude current row
    counts_1h = []
    counts_24h = []
    sums_24h = []

    for _, g in df.groupby("cc_num", sort=False):
        g_idx = g.set_index("trans_date_trans_time")
        counts_1h.append(g_idx["amt"].rolling("1h", closed="left").count().values)
        counts_24h.append(g_idx["amt"].rolling("24h", closed="left").count().values)
        sums_24h.append(g_idx["amt"].rolling("24h", closed="left").sum().values)

    df["txn_count_last_1h"] = np.nan_to_num(np.concatenate(counts_1h), nan=0.0).astype(int)
    df["txn_count_last_24h"] = np.nan_to_num(np.concatenate(counts_24h), nan=0.0).astype(int)
    df["amt_sum_last_24h"] = np.nan_to_num(np.concatenate(sums_24h), nan=0.0)

    # Restore original row ordering
    if not has_orig_idx:
        df = df.sort_values("__orig_idx__").reset_index(drop=True).drop(columns=["__orig_idx__"])

    return df


def engineer_features(df, le_category=None, category_fraud_rates=None, global_fraud_rate=None, fit=False):
    """
    Transforms raw credit card transaction DataFrame into model feature matrix.
    When fit=True, encoders and fraud rate lookups are fit on the training split only.
    When fit=False, previously fitted encoders are applied to prevent any leakage.
    """
    df = df.copy()

    # Temporal features (derived from timestamp)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"] = df["trans_date_trans_time"].dt.month

    # 9. is_night (10pm to 5am inclusive: 22, 23, 0, 1, 2, 3, 4, 5) and is_weekend
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Age calculation from date of birth
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        ref_date = pd.Timestamp("2020-01-01")
        df["age"] = ((ref_date - df["dob"]).dt.days / 365.25).fillna(35).astype(float)
    else:
        df["age"] = 35.0

    # Amount transformations
    df["amt_log"] = np.log1p(df["amt"])
    df["amt_per_city_pop"] = df["amt"] / (df["city_pop"] + 1)

    # Geospatial distance
    df["distance"] = haversine(
        df["lat"].values,
        df["long"].values,
        df["merch_lat"].values,
        df["merch_long"].values,
    )

    # Demographics & Category Encoding
    df["gender_enc"] = (df["gender"] == "M").astype(int)

    if fit:
        le_category = LabelEncoder()
        df["category_enc"] = le_category.fit_transform(df["category"].astype(str))
    else:
        if le_category is None:
            le_category = LabelEncoder()
            le_category.fit(KNOWN_CATEGORIES)
        known_set = set(le_category.classes_)
        safe_cats = df["category"].apply(lambda c: c if c in known_set else le_category.classes_[0])
        df["category_enc"] = le_category.transform(safe_cats.astype(str))

    # 7. merchant_category_fraud_rate, computed from the training set only
    if fit:
        if "is_fraud" in df.columns:
            category_fraud_rates = df.groupby("category")["is_fraud"].mean().to_dict()
            global_fraud_rate = float(df["is_fraud"].mean())
        else:
            category_fraud_rates = {c: 0.005 for c in KNOWN_CATEGORIES}
            global_fraud_rate = 0.005

    if category_fraud_rates is None:
        category_fraud_rates = {c: 0.005 for c in KNOWN_CATEGORIES}
    if global_fraud_rate is None:
        global_fraud_rate = 0.005

    df["merchant_category_fraud_rate"] = (
        df["category"].map(category_fraud_rates).fillna(global_fraud_rate).astype(float)
    )

    # Add per-card past-only behavioral features (if cc_num is present in df)
    if "cc_num" in df.columns:
        df = add_past_only_card_features(df)
    else:
        # Fallbacks for isolated records without card history
        df["txn_count_last_1h"] = 0
        df["txn_count_last_24h"] = 0
        df["amt_sum_last_24h"] = 0.0
        df["time_since_last_txn"] = -1.0
        df["amt_vs_card_avg"] = 1.0
        df["amt_zscore_vs_card"] = 0.0
        df["is_new_merchant_for_card"] = 1
        df["distance_from_last_txn_km"] = -1.0

    return df, le_category, category_fraud_rates, global_fraud_rate


def build_feature_dict(data, le_category=None, category_fraud_rates=None, global_fraud_rate=0.005):
    """
    Build a single feature dictionary from incoming API request dictionary.
    Excludes all raw identifiers (card numbers, transaction hashes, person names).
    """
    amt = float(data.get("amt", 50.0))
    category = str(data.get("category", "shopping_net")).lower()
    city_pop = float(data.get("city_pop", 10000.0))
    lat = float(data.get("lat", 36.1627))
    lon = float(data.get("long", -86.7816))
    merch_lat = float(data.get("merch_lat", lat + 0.05))
    merch_long = float(data.get("merch_long", lon + 0.05))
    hour = int(data.get("hour", 12))
    day_of_week = int(data.get("day_of_week", 2))
    month = int(data.get("month", 6))
    gender = str(data.get("gender", "F")).upper()
    age = float(data.get("age", 35.0))

    amt_log = math.log1p(amt)
    amt_per_city_pop = amt / (city_pop + 1.0)
    distance = float(haversine(lat, lon, merch_lat, merch_long))
    gender_enc = 1 if gender == "M" else 0

    if le_category is not None and hasattr(le_category, "classes_"):
        known = set(le_category.classes_)
        safe_cat = category if category in known else le_category.classes_[0]
        category_enc = int(le_category.transform([safe_cat])[0])
    else:
        category_enc = 0

    # Categorical fraud rate
    if category_fraud_rates and category in category_fraud_rates:
        merchant_category_fraud_rate = float(category_fraud_rates[category])
    else:
        merchant_category_fraud_rate = float(global_fraud_rate)

    is_night = 1 if (hour >= 22 or hour <= 5) else 0
    is_weekend = 1 if day_of_week >= 5 else 0

    # Behavioral values (from user payload if provided, else baseline defaults)
    txn_count_last_1h = int(data.get("txn_count_last_1h", 0))
    txn_count_last_24h = int(data.get("txn_count_last_24h", 0))
    amt_sum_last_24h = float(data.get("amt_sum_last_24h", 0.0))
    time_since_last_txn = float(data.get("time_since_last_txn", -1.0))
    amt_vs_card_avg = float(data.get("amt_vs_card_avg", 1.0))
    amt_zscore_vs_card = float(data.get("amt_zscore_vs_card", 0.0))
    is_new_merchant_for_card = int(data.get("is_new_merchant_for_card", 1))
    distance_from_last_txn_km = float(data.get("distance_from_last_txn_km", -1.0))

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
        "txn_count_last_1h": txn_count_last_1h,
        "txn_count_last_24h": txn_count_last_24h,
        "amt_sum_last_24h": amt_sum_last_24h,
        "time_since_last_txn": time_since_last_txn,
        "amt_vs_card_avg": amt_vs_card_avg,
        "amt_zscore_vs_card": amt_zscore_vs_card,
        "is_new_merchant_for_card": is_new_merchant_for_card,
        "merchant_category_fraud_rate": merchant_category_fraud_rate,
        "distance_from_last_txn_km": distance_from_last_txn_km,
        "is_night": is_night,
        "is_weekend": is_weekend,
    }
