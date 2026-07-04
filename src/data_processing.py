"""
Phase 2 & 3: Data Cleaning and Merging
========================================
Loads the two raw datasets, cleans them, and merges them on a common
calendar date. All cleaning decisions are documented via print statements
so the pipeline is auditable end-to-end.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------------
# SENTIMENT DATA
# ------------------------------------------------------------------
def load_and_clean_sentiment():
    section("CLEANING: Fear & Greed Sentiment Data")
    fg = pd.read_csv(RAW_DIR / "fear_greed_index.csv")
    before_rows = len(fg)

    # Standardize column names
    fg.columns = [c.strip().lower().replace(" ", "_") for c in fg.columns]

    # Parse date (already ISO-formatted YYYY-MM-DD)
    fg["date"] = pd.to_datetime(fg["date"], errors="coerce").dt.normalize()
    bad_dates = fg["date"].isna().sum()
    if bad_dates:
        print(f"Dropping {bad_dates} rows with unparseable dates")
        fg = fg.dropna(subset=["date"])

    # Standardize classification text (strip whitespace, title-case)
    fg["classification"] = fg["classification"].astype(str).str.strip().str.title()

    valid_classes = {"Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"}
    unexpected = set(fg["classification"].unique()) - valid_classes
    if unexpected:
        print("WARNING - unexpected classification values found:", unexpected)
    else:
        print("All classification values match expected 5-category scheme:", sorted(valid_classes))

    # Duplicate dates (sentiment should be one row per day)
    dup_dates = fg.duplicated(subset=["date"]).sum()
    print(f"Duplicate calendar dates: {dup_dates}")
    if dup_dates:
        fg = fg.drop_duplicates(subset=["date"], keep="first")

    # Ordered category for correct plotting/grouping order
    order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    fg["classification"] = pd.Categorical(fg["classification"], categories=order, ordered=True)

    print(f"Rows before cleaning: {before_rows} -> after cleaning: {len(fg)}")
    print(f"Date range: {fg['date'].min().date()} to {fg['date'].max().date()}")

    return fg[["date", "value", "classification"]].rename(
        columns={"value": "sentiment_value", "classification": "sentiment"}
    )


# ------------------------------------------------------------------
# TRADER DATA
# ------------------------------------------------------------------
def load_and_clean_trades():
    section("CLEANING: Historical Trader Data (Hyperliquid)")
    hd = pd.read_csv(RAW_DIR / "historical_data.csv")
    before_rows = len(hd)
    print(f"Rows loaded: {before_rows}")

    # Standardize column names -> snake_case
    hd.columns = [c.strip().lower().replace(" ", "_") for c in hd.columns]

    # Strip whitespace from string columns
    str_cols = hd.select_dtypes(include=["object", "str"]).columns
    for c in str_cols:
        hd[c] = hd[c].astype(str).str.strip()

    # --- Timestamp handling -------------------------------------------------
    # DATA QUALITY ISSUE: the raw 'timestamp' column (epoch ms) is heavily
    # rounded in the source file (only 7 distinct values across 211k rows,
    # e.g. 1.73e+12), so it cannot be used for date/time reconstruction.
    # 'timestamp_ist' (format DD-MM-YYYY HH:MM) carries full minute-level
    # precision and is used as the authoritative time field instead.
    n_distinct_raw_ts = hd["timestamp"].nunique()
    print(f"Raw 'timestamp' column has only {n_distinct_raw_ts} distinct values "
          f"(precision-loss artifact) -> NOT used for time reconstruction.")

    hd["datetime_ist"] = pd.to_datetime(
        hd["timestamp_ist"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    bad_ts = hd["datetime_ist"].isna().sum()
    print(f"Unparseable 'timestamp_ist' values: {bad_ts}")
    if bad_ts:
        hd = hd.dropna(subset=["datetime_ist"])

    hd["trade_date"] = hd["datetime_ist"].dt.normalize()
    hd["trade_hour"] = hd["datetime_ist"].dt.hour
    hd["day_of_week"] = hd["datetime_ist"].dt.day_name()

    # --- Numeric columns: coerce safely ------------------------------------
    numeric_cols = ["execution_price", "size_tokens", "size_usd",
                     "start_position", "closed_pnl", "fee"]
    for c in numeric_cols:
        hd[c] = pd.to_numeric(hd[c], errors="coerce")
    na_numeric = hd[numeric_cols].isna().sum()
    print("Missing values after numeric coercion:\n", na_numeric[na_numeric > 0])

    # --- Side / Direction standardization ----------------------------------
    hd["side"] = hd["side"].str.upper().str.strip()
    valid_sides = {"BUY", "SELL"}
    unexpected_sides = set(hd["side"].unique()) - valid_sides
    if unexpected_sides:
        print("WARNING - unexpected side values:", unexpected_sides)

    # --- Duplicates ---------------------------------------------------------
    full_dupes = hd.duplicated().sum()
    print(f"Fully duplicated rows: {full_dupes}")
    if full_dupes:
        hd = hd.drop_duplicates()

    # Near-duplicate check: same account/coin/order/trade-id combo repeats
    # are expected (partial fills against one order) and are NOT removed.
    same_order_multi = hd.groupby("order_id").size()
    print(f"Orders with >1 fill (partial fills, expected/legit): "
          f"{(same_order_multi > 1).sum()} of {len(same_order_multi)} orders")

    # --- Zero-size / dust conversion rows -----------------------------------
    zero_size = (hd["size_usd"] <= 0).sum()
    print(f"Rows with size_usd <= 0: {zero_size} "
          f"(mostly 'Spot Dust Conversion' events; kept, but excluded from "
          f"size-dependent ratio calcs to avoid divide-by-zero)")

    # --- Outliers: DO NOT remove, just report ------------------------------
    q99 = hd["closed_pnl"].abs().quantile(0.99)
    print(f"99th percentile of |closed_pnl|: {q99:.2f} "
          f"(extreme values retained - legitimate large trades)")

    print(f"Rows before cleaning: {before_rows} -> after cleaning: {len(hd)}")
    print(f"Date range (trade_date): {hd['trade_date'].min().date()} to "
          f"{hd['trade_date'].max().date()}")
    print(f"Unique accounts: {hd['account'].nunique()}, unique coins: {hd['coin'].nunique()}")

    return hd


# ------------------------------------------------------------------
# MERGE
# ------------------------------------------------------------------
def merge_datasets(trades: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    section("MERGE: Trades <> Sentiment (on calendar date)")
    rows_before = len(trades)

    sent_min, sent_max = sentiment["date"].min(), sentiment["date"].max()
    trade_min, trade_max = trades["trade_date"].min(), trades["trade_date"].max()
    overlap_start = max(sent_min, trade_min)
    overlap_end = min(sent_max, trade_max)
    print(f"Sentiment date range: {sent_min.date()} to {sent_max.date()}")
    print(f"Trade date range:     {trade_min.date()} to {trade_max.date()}")
    print(f"Overlapping range:    {overlap_start.date()} to {overlap_end.date()}")

    merged = trades.merge(sentiment, left_on="trade_date", right_on="date", how="left")

    matched = merged["sentiment"].notna().sum()
    unmatched = merged["sentiment"].isna().sum()
    print(f"Rows before merge: {rows_before}")
    print(f"Rows after merge:  {len(merged)}")
    print(f"Matched:   {matched} ({matched/len(merged)*100:.2f}%)")
    print(f"Unmatched: {unmatched} ({unmatched/len(merged)*100:.2f}%)")

    if unmatched:
        unmatched_dates = merged.loc[merged["sentiment"].isna(), "trade_date"].dt.date.unique()
        print(f"Unmatched trade dates (first 10 of {len(unmatched_dates)}): "
              f"{sorted(unmatched_dates)[:10]}")

    # Validate no row-count inflation from merge (sentiment has 1 row/date)
    assert len(merged) == rows_before, "Merge unexpectedly duplicated trade rows!"
    print("Validation passed: merge did not duplicate trade rows (1:1 date join).")

    merged = merged.drop(columns=["date"])
    return merged


def run():
    sentiment = load_and_clean_sentiment()
    trades = load_and_clean_trades()
    merged = merge_datasets(trades, sentiment)

    sentiment.to_csv(PROC_DIR / "sentiment_clean.csv", index=False)
    trades.to_parquet(PROC_DIR / "trades_clean.parquet", index=False)
    merged.to_parquet(PROC_DIR / "merged.parquet", index=False)
    print(f"\nSaved cleaned/merged data to {PROC_DIR}")
    return merged


if __name__ == "__main__":
    run()