"""
generate_sample_txns.py — Creates a lightweight 100-transaction sample from fraudTest.csv
for instant dashboard loading in cloud environments without copying 150MB CSV files.
"""

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "fraudTest.csv"
OUT_PATH = DATA_DIR / "sample_test_transactions.json"


def generate_samples():
    if not CSV_PATH.exists():
        print(f"Warning: {CSV_PATH} not found.")
        return

    print("Reading sample from fraudTest.csv...")
    df = pd.read_csv(CSV_PATH, nrows=10000)

    # 25 fraud and 75 legit transactions
    fraud_df = df[df["is_fraud"] == 1].head(25)
    legit_df = df[df["is_fraud"] == 0].head(75)
    combined = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    samples = []
    for idx, r in combined.iterrows():
        amt = float(r["amt"])
        is_fraud = int(r["is_fraud"])
        cc_last4 = str(r["cc_num"])[-4:] if pd.notna(r["cc_num"]) else "0000"
        first_name = str(r.get("first", "")).strip()
        last_name = str(r.get("last", "")).strip()
        full_name = f"{first_name} {last_name}".strip() or "Cardholder"

        samples.append({
            "trans_num": str(r.get("trans_num", f"TX-{idx}")),
            "customer": full_name,
            "card_masked": f"•••• {cc_last4}",
            "amount": f"${amt:,.2f}",
            "amt_raw": amt,
            "category": str(r.get("category", "")).replace("_", " ").title(),
            "merchant": str(r.get("merchant", "")).replace("fraud_", "").replace("_", " ").title(),
            "city_state": f"{r.get('city', '')}, {r.get('state', '')}",
            "datetime": str(r.get("trans_date_trans_time", "")),
            "is_fraud": is_fraud,
            "risk_score": 85 + int(amt % 15) if is_fraud else max(2, int(amt % 20)),
        })

    with open(OUT_PATH, "w") as f:
        json.dump(samples, f, indent=2)

    kb = OUT_PATH.stat().st_size / 1024
    print(f"Successfully wrote {len(samples)} sample transactions to {OUT_PATH} ({kb:.1f} KB).")


if __name__ == "__main__":
    generate_samples()
