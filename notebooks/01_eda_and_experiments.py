"""
Exploratory Data Analysis (EDA) & Feature Engineering Research Script.
Run standalone to inspect dataset distributions, smoker/BMI interactions, and baseline correlations.
"""
from pathlib import Path
import pandas as pd


def run_eda():
    raw_csv = Path("artifacts/data_ingestion/insurance_raw.csv")
    if not raw_csv.exists():
        print("Raw dataset not yet ingested. Run `python main.py` first.")
        return

    df = pd.read_csv(raw_csv)
    print("=== 1. DATASET SHAPE & DTYPES ===")
    print(f"Shape: {df.shape}")
    print(df.dtypes)

    print("\n=== 2. SUMMARY STATISTICS ===")
    print(df.describe().round(2))

    print("\n=== 3. MEAN RISK CHARGES BY SMOKER & OBESITY (BMI >= 30) ===")
    df["is_obese"] = df["bmi"] >= 30.0
    grouped = df.groupby(["smoker", "is_obese"])["charges"].agg(["count", "mean", "median", "std"])
    print(grouped.round(2))


if __name__ == "__main__":
    run_eda()
