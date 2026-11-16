import pandas as pd
import numpy as np
from src.train import time_based_split
from src.features import FEATURE_COLS, IDENTIFIER_COLS


def test_time_based_split_ratios_and_assertion():
    # Synthetic time-series dataframe of 100 transactions
    timestamps = pd.date_range("2021-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "trans_date_trans_time": timestamps,
        "is_fraud": [0] * 90 + [1] * 10,
        "amt": np.random.uniform(5, 500, size=100),
    })

    train, val, test = time_based_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # Assertion 5 check
    assert train["trans_date_trans_time"].max() < test["trans_date_trans_time"].min()
    assert train["trans_date_trans_time"].max() <= val["trans_date_trans_time"].min()
    assert val["trans_date_trans_time"].max() <= test["trans_date_trans_time"].min()


def test_leaking_identifiers_excluded_from_features():
    for id_col in IDENTIFIER_COLS:
        assert id_col not in FEATURE_COLS, f"Leaking identifier column '{id_col}' found in model features!"
