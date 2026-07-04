"""
Phase 1: Dataset Inspection
Inspect both raw datasets and print schema, dtypes, ranges, missing values,
duplicates, categorical values, and descriptive statistics.
No assumptions are made about column names beyond what is verified here.
"""
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ------------------------------------------------------------------
# 1. Fear & Greed sentiment dataset
# ------------------------------------------------------------------
section("FEAR & GREED INDEX DATASET")
fg = pd.read_csv(RAW_DIR / "fear_greed_index.csv")

print("Shape:", fg.shape)
print("\nColumns:", list(fg.columns))
print("\nDtypes:\n", fg.dtypes)
print("\nFirst 5 rows:\n", fg.head())
print("\nLast 5 rows:\n", fg.tail())
print("\nMissing values:\n", fg.isna().sum())
print("\nDuplicate rows (full):", fg.duplicated().sum())
print("\nDuplicate 'date' values:", fg.duplicated(subset=["date"]).sum() if "date" in fg.columns else "N/A")
print("\nUnique classification values:\n", fg["classification"].value_counts() if "classification" in fg.columns else "N/A")
print("\nDescribe (numeric):\n", fg.describe())
if "date" in fg.columns:
    fg["_date_parsed"] = pd.to_datetime(fg["date"], errors="coerce")
    print("\nDate range:", fg["_date_parsed"].min(), "to", fg["_date_parsed"].max())
    print("Unparseable dates:", fg["_date_parsed"].isna().sum())
if "timestamp" in fg.columns:
    print("\nTimestamp sample values:", fg["timestamp"].head(3).tolist())
    print("Timestamp dtype:", fg["timestamp"].dtype)
    print("Timestamp min/max:", fg["timestamp"].min(), fg["timestamp"].max())

# ------------------------------------------------------------------
# 2. Historical trader data (Hyperliquid)
# ------------------------------------------------------------------
section("HISTORICAL TRADER DATA (Hyperliquid) - basic scan")
# Read only header + small sample first since file is 45MB+
sample = pd.read_csv(RAW_DIR / "historical_data.csv", nrows=5)
print("Columns:", list(sample.columns))
print("Sample dtypes:\n", sample.dtypes)
print("\nSample rows:\n", sample)

section("HISTORICAL TRADER DATA - full load")
hd = pd.read_csv(RAW_DIR / "historical_data.csv")
print("Shape:", hd.shape)
print("\nColumns:", list(hd.columns))
print("\nDtypes:\n", hd.dtypes)
print("\nFirst 5 rows:\n", hd.head())
print("\nLast 5 rows:\n", hd.tail())
print("\nMissing values:\n", hd.isna().sum())
print("\nDuplicate rows (full):", hd.duplicated().sum())

# Categorical columns
for col in ["Coin", "Side", "Direction", "Crossed"]:
    if col in hd.columns:
        print(f"\nUnique values in '{col}' (top 20):\n", hd[col].value_counts().head(20))
        print(f"Number of unique '{col}':", hd[col].nunique())

if "Account" in hd.columns:
    print("\nNumber of unique Accounts:", hd["Account"].nunique())

section("HISTORICAL TRADER DATA - numeric describe")
print(hd.describe(include=[np.number]))

section("HISTORICAL TRADER DATA - timestamp columns inspection")
for col in ["Timestamp", "Timestamp IST"]:
    if col in hd.columns:
        print(f"\nColumn: {col}")
        print("dtype:", hd[col].dtype)
        print("sample values:", hd[col].head(5).tolist())
        print("nulls:", hd[col].isna().sum())

section("DONE - PHASE 1 RAW INSPECTION")