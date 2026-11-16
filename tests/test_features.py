import math
import numpy as np
import pandas as pd
from src.features import haversine, build_feature_dict, FEATURE_COLS, engineer_features


def test_haversine_same_point():
    # Distance between identical points is 0
    dist = haversine(40.7128, -74.0060, 40.7128, -74.0060)
    assert math.isclose(dist, 0.0, abs_tol=1e-5)


def test_haversine_known_distance():
    # NY (40.7128, -74.0060) to London (51.5074, -0.1278) ~ 5570 km
    dist = haversine(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500 < dist < 5650


def test_build_feature_dict_keys():
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
