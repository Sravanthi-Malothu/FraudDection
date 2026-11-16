"""
features.py — Feature Engineering for Fraud Detection
Handles mathematical transformations, Haversine geo-distance,
demographic extraction, and categorical encoding.
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
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
    Compute great-circle distance between pairs of coordinates in kilometers.
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


def engineer_features(df, le_category=None, fit=False):
    """
    Transforms raw credit card transaction DataFrame into model feature matrix.
    """
    df = df.copy()

    # Temporal features
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"] = df["trans_date_trans_time"].dt.month

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

    if le_category is None:
        le_category = LabelEncoder()
        le_category.fit(KNOWN_CATEGORIES)

    if fit:
        df["category_enc"] = le_category.fit_transform(df["category"].astype(str))
    else:
        known_set = set(le_category.classes_)
        safe_cats = df["category"].apply(lambda c: c if c in known_set else le_category.classes_[0])
        df["category_enc"] = le_category.transform(safe_cats.astype(str))

    return df, le_category


def build_feature_dict(data, le_category=None):
    """
    Build a single 15-feature dictionary from incoming API request dictionary.
    """
    amt = float(data.get("amt", 50.0))
    category = str(data.get("category", "shopping_net")).lower()
    city_pop = float(data.get("city_pop", 10000.0))
    lat = float(data.get("lat", 36.1627))
    lon = float(data.get("long", -86.7816))
    merch_lat = float(data.get("merch_lat", lat + 0.05))
    merch_long = float(data.get("merch_long", lon + 0.05))
    hour = int(data.get("hour", 12))
    gender = str(data.get("gender", "F")).upper()
    age = float(data.get("age", 35.0))
    day_of_week = int(data.get("day_of_week", 2))
    month = int(data.get("month", 6))

    amt_log = math.log1p(amt)
    amt_per_city_pop = amt / (city_pop + 1.0)
    distance = float(haversine(lat, lon, merch_lat, merch_long))
    gender_enc = 1 if gender == "M" else 0

    if le_category is not None:
        known = set(le_category.classes_)
        safe_cat = category if category in known else le_category.classes_[0]
        category_enc = int(le_category.transform([safe_cat])[0])
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
    }
