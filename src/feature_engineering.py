"""
Phase 4: Feature Engineering
==============================
Builds trade-level analytical features and trader-level aggregates from
the cleaned/merged dataset. Only features that are mathematically and
financially defensible given the actual available columns are created.

NOTE: No 'leverage' column exists in the source data, so all
leverage-dependent features/analyses are skipped (documented here and
in the README).
"""
import numpy as np
import pandas as pd

MIN_TRADES_THRESHOLD = 30  # minimum trades for a trader to be included in
                            # trader-level comparative analysis (Phase 9/10)


def add_trade_level_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Profitability flags (only meaningful for rows where a position was
    # actually closed, i.e. closed_pnl != 0; open-only fills carry 0 PnL
    # by construction and are neither a win nor a loss)
    df["is_closing_trade"] = df["closed_pnl"] != 0
    df["is_profitable"] = df["closed_pnl"] > 0
    df["is_loss"] = df["closed_pnl"] < 0

    df["absolute_pnl"] = df["closed_pnl"].abs()

    # Safe ratios (avoid divide-by-zero on dust-conversion / zero-size rows)
    df["fee_to_size_ratio"] = np.where(
        df["size_usd"] > 0, df["fee"] / df["size_usd"], np.nan
    )
    df["pnl_to_size_ratio"] = np.where(
        df["size_usd"] > 0, df["closed_pnl"] / df["size_usd"], np.nan
    )

    # Position size buckets (quantile-based, computed on size_usd > 0 only)
    positive_size = df.loc[df["size_usd"] > 0, "size_usd"]
    bins = positive_size.quantile([0, .25, .5, .75, 1.0]).values
    bins = np.unique(bins)
    labels = ["Small", "Medium", "Large", "Very Large"][: len(bins) - 1]
    df["size_bucket"] = pd.cut(df["size_usd"], bins=bins, labels=labels, include_lowest=True)

    # Net trade direction proxy from Side (BUY/SELL); Direction gives
    # open/close + long/short context, both retained.
    df["is_buy"] = df["side"] == "BUY"

    return df


def trader_level_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per account with performance/behavior metrics."""
    closing = df[df["is_closing_trade"]]

    g = df.groupby("account")
    gc = closing.groupby("account")

    agg = pd.DataFrame({
        "total_trades": g.size(),
        "closing_trades": gc.size(),
        "total_pnl": gc["closed_pnl"].sum(),
        "avg_pnl": gc["closed_pnl"].mean(),
        "median_pnl": gc["closed_pnl"].median(),
        "pnl_std": gc["closed_pnl"].std(),
        "win_rate": gc["is_profitable"].mean(),
        "avg_win": gc.apply(lambda x: x.loc[x["is_profitable"], "closed_pnl"].mean()),
        "avg_loss": gc.apply(lambda x: x.loc[x["is_loss"], "closed_pnl"].mean()),
        "avg_trade_size_usd": g["size_usd"].mean(),
        "total_volume_usd": g["size_usd"].sum(),
        "total_fees": g["fee"].sum(),
        "buy_pct": g["is_buy"].mean(),
        "active_days": g["trade_date"].nunique(),
    })

    agg["closing_trades"] = agg["closing_trades"].fillna(0).astype(int)
    agg["sell_pct"] = 1 - agg["buy_pct"]
    agg["trades_per_active_day"] = agg["total_trades"] / agg["active_days"].replace(0, np.nan)

    # Risk-adjusted performance proxy: mean PnL / std PnL (Sharpe-style,
    # NOT annualized - a simple cross-trade consistency measure).
    agg["risk_adjusted_pnl"] = np.where(
        agg["pnl_std"] > 0, agg["avg_pnl"] / agg["pnl_std"], np.nan
    )

    agg["fee_to_volume_ratio"] = np.where(
        agg["total_volume_usd"] > 0, agg["total_fees"] / agg["total_volume_usd"], np.nan
    )

    return agg.reset_index()


def trader_sentiment_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Trader performance broken out by sentiment category (Phase 9/10)."""
    closing = df[df["is_closing_trade"] & df["sentiment"].notna()]
    out = (
        closing.groupby(["account", "sentiment"], observed=True)
        .agg(trades=("closed_pnl", "size"), total_pnl=("closed_pnl", "sum"),
             mean_pnl=("closed_pnl", "mean"), win_rate=("is_profitable", "mean"))
        .reset_index()
    )
    return out


def trader_day_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Trader-day level aggregates for overtrading analysis (Phase 11)."""
    closing = df[df["is_closing_trade"]]
    day_g = df.groupby(["account", "trade_date"])
    day_gc = closing.groupby(["account", "trade_date"])

    out = pd.DataFrame({
        "n_trades": day_g.size(),
        "total_pnl": day_gc["closed_pnl"].sum(),
        "avg_pnl": day_gc["closed_pnl"].mean(),
        "total_volume": day_g["size_usd"].sum(),
        "total_fees": day_g["fee"].sum(),
        "win_rate": day_gc["is_profitable"].mean(),
    }).reset_index()

    sentiment_lookup = df[["trade_date", "sentiment", "sentiment_value"]].drop_duplicates("trade_date")
    out = out.merge(sentiment_lookup, on="trade_date", how="left")
    out["pnl_per_trade"] = np.where(out["n_trades"] > 0, out["total_pnl"] / out["n_trades"], np.nan)
    return out