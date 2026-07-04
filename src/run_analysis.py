"""
Master analysis script: Phases 5-21.
Run from project root: python3 src/run_analysis.py
Produces console output (redirect to a log for the executive summary),
saved tables (outputs/tables/*.csv) and figures (outputs/figures/*.png).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_processing import run as clean_and_merge
from feature_engineering import (
    add_trade_level_features, trader_level_aggregates,
    trader_sentiment_performance, trader_day_aggregates, MIN_TRADES_THRESHOLD
)
from visualization import save_fig, SENTIMENT_ORDER, SENTIMENT_PALETTE, format_large_numbers
import statistics_helpers as sh

TABLE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
np.random.seed(42)


def section(title):
    print("\n" + "#" * 78)
    print("# " + title)
    print("#" * 78)


def save_table(df, name):
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=True if df.index.name else False)
    print(f"Saved table: {path}")


# ====================================================================
# LOAD
# ====================================================================
section("PHASE 5: LOADING CLEANED/MERGED DATA + BASIC EDA")
df = clean_and_merge()
df = add_trade_level_features(df)
closing = df[df["is_closing_trade"]].copy()
closing_sent = closing[closing["sentiment"].notna()].copy()

print(f"\nTotal trade fills: {len(df):,}")
print(f"Closing fills (non-zero realized PnL): {len(closing):,} "
      f"({len(closing)/len(df)*100:.1f}% of all fills)")
print(f"Unique traders: {df['account'].nunique()}")
print(f"Unique assets: {df['coin'].nunique()}")
print(f"\nSentiment distribution (trade-fill level):")
sent_dist = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER)
print(sent_dist)
save_table(sent_dist.rename("n_trades"), "sentiment_distribution")

print(f"\nBUY vs SELL distribution:")
print(df["side"].value_counts())

print(f"\nClosed PnL distribution (closing trades only):")
print(closing["closed_pnl"].describe())

print(f"\nTrade size (USD) distribution:")
print(df["size_usd"].describe())

print(f"\nFee distribution:")
print(df["fee"].describe())

# --- EDA visuals ---
fig, ax = plt.subplots(figsize=(9, 5))
sns.countplot(x="sentiment", data=df, order=SENTIMENT_ORDER,
              palette=SENTIMENT_PALETTE, ax=ax)
ax.set_title("Number of Trade Fills by Market Sentiment")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Number of trades")
format_large_numbers(ax)
save_fig(fig, "01_trade_count_by_sentiment")

fig, ax = plt.subplots(figsize=(9, 5))
daily = df.groupby("trade_date").size()
daily.plot(ax=ax, color="#333333", linewidth=1)
ax.set_title("Daily Trading Activity Over Time")
ax.set_xlabel("Date"); ax.set_ylabel("Number of trades")
save_fig(fig, "02_daily_trading_activity")

fig, ax = plt.subplots(figsize=(9, 5))
daily_pnl = closing.groupby("trade_date")["closed_pnl"].sum()
daily_pnl.plot(ax=ax, color="#1f6f4f", linewidth=1)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Daily Total Closed PnL Over Time")
ax.set_xlabel("Date"); ax.set_ylabel("Total closed PnL (USD)")
format_large_numbers(ax)
save_fig(fig, "03_daily_pnl_over_time")

fig, ax = plt.subplots(figsize=(9, 5))
clipped_pnl = closing["closed_pnl"].clip(closing["closed_pnl"].quantile(0.01),
                                          closing["closed_pnl"].quantile(0.99))
sns.histplot(clipped_pnl, bins=80, ax=ax, color="#4C72B0")
ax.set_title("Closed PnL Distribution (clipped at 1st/99th pct for readability)")
ax.set_xlabel("Closed PnL (USD)"); ax.set_ylabel("Frequency")
save_fig(fig, "04_pnl_distribution")

# ====================================================================
# PHASE 6: PERFORMANCE VS SENTIMENT
# ====================================================================
section("PHASE 6: TRADER PERFORMANCE VS MARKET SENTIMENT")

def profit_factor(x):
    wins = x.loc[x > 0].sum()
    losses = x.loc[x < 0].abs().sum()
    return wins / losses if losses > 0 else np.nan

perf_by_sentiment = closing_sent.groupby("sentiment", observed=True).agg(
    total_pnl=("closed_pnl", "sum"),
    mean_pnl=("closed_pnl", "mean"),
    median_pnl=("closed_pnl", "median"),
    pnl_std=("closed_pnl", "std"),
    n_trades=("closed_pnl", "size"),
    n_profitable=("is_profitable", "sum"),
    n_losing=("is_loss", "sum"),
    win_rate=("is_profitable", "mean"),
    total_fees=("fee", "sum"),
    avg_trade_size=("size_usd", "mean"),
).reindex(SENTIMENT_ORDER)

perf_by_sentiment["avg_win"] = closing_sent[closing_sent["is_profitable"]].groupby(
    "sentiment", observed=True)["closed_pnl"].mean().reindex(SENTIMENT_ORDER)
perf_by_sentiment["avg_loss"] = closing_sent[closing_sent["is_loss"]].groupby(
    "sentiment", observed=True)["closed_pnl"].mean().reindex(SENTIMENT_ORDER)
perf_by_sentiment["risk_adjusted"] = perf_by_sentiment["mean_pnl"] / perf_by_sentiment["pnl_std"]
perf_by_sentiment["profit_factor"] = closing_sent.groupby(
    "sentiment", observed=True)["closed_pnl"].apply(profit_factor).reindex(SENTIMENT_ORDER)

print(perf_by_sentiment.round(3))
save_table(perf_by_sentiment.round(4), "performance_by_sentiment")

most_profitable = perf_by_sentiment["total_pnl"].idxmax()
least_profitable = perf_by_sentiment["total_pnl"].idxmin()
most_profitable_median = perf_by_sentiment["median_pnl"].idxmax()
print(f"\nMost profitable sentiment by TOTAL PnL: {most_profitable} "
      f"(${perf_by_sentiment.loc[most_profitable,'total_pnl']:,.0f})")
print(f"Least profitable sentiment by TOTAL PnL: {least_profitable} "
      f"(${perf_by_sentiment.loc[least_profitable,'total_pnl']:,.0f})")
print(f"Most profitable sentiment by MEDIAN PnL: {most_profitable_median} "
      f"(${perf_by_sentiment.loc[most_profitable_median,'median_pnl']:,.2f})")
same_story = most_profitable == most_profitable_median
print(f"Total PnL and median PnL point to the {'SAME' if same_story else 'DIFFERENT'} "
      f"sentiment as most favorable -> "
      f"{'consistent' if same_story else 'total PnL is likely skewed by a few large winners'}")

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=perf_by_sentiment.index, y=perf_by_sentiment["total_pnl"],
            order=SENTIMENT_ORDER, palette=SENTIMENT_PALETTE, ax=ax)
ax.set_title("Total Closed PnL by Market Sentiment")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Total Closed PnL (USD)")
format_large_numbers(ax)
save_fig(fig, "05_total_pnl_by_sentiment")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=perf_by_sentiment.index, y=perf_by_sentiment["mean_pnl"],
            order=SENTIMENT_ORDER, palette=SENTIMENT_PALETTE, ax=ax[0])
ax[0].set_title("Mean PnL by Sentiment"); ax[0].set_xlabel(""); ax[0].set_ylabel("Mean PnL (USD)")
ax[0].tick_params(axis='x', rotation=30)
sns.barplot(x=perf_by_sentiment.index, y=perf_by_sentiment["median_pnl"],
            order=SENTIMENT_ORDER, palette=SENTIMENT_PALETTE, ax=ax[1])
ax[1].set_title("Median PnL by Sentiment"); ax[1].set_xlabel(""); ax[1].set_ylabel("Median PnL (USD)")
ax[1].tick_params(axis='x', rotation=30)
save_fig(fig, "06_mean_vs_median_pnl_by_sentiment")

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=perf_by_sentiment.index, y=perf_by_sentiment["win_rate"] * 100,
            order=SENTIMENT_ORDER, palette=SENTIMENT_PALETTE, ax=ax)
ax.set_title("Win Rate by Market Sentiment")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Win rate (%)")
save_fig(fig, "07_win_rate_by_sentiment")

fig, ax = plt.subplots(figsize=(9, 6))
plot_data = closing_sent.copy()
plot_data["pnl_clip"] = plot_data["closed_pnl"].clip(
    plot_data["closed_pnl"].quantile(0.02), plot_data["closed_pnl"].quantile(0.98))
sns.boxplot(x="sentiment", y="pnl_clip", data=plot_data, order=SENTIMENT_ORDER,
            palette=SENTIMENT_PALETTE, ax=ax, showfliers=False)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Closed PnL Distribution by Sentiment (2nd-98th pct, outliers hidden)")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Closed PnL (USD)")
save_fig(fig, "08_pnl_boxplot_by_sentiment")

# Statistical test: PnL across sentiment groups
section("PHASE 15 (partial): Statistical test - PnL across sentiment groups")
groups = [closing_sent.loc[closing_sent["sentiment"] == s, "closed_pnl"].values for s in SENTIMENT_ORDER]
kw_result = sh.kruskal_test(groups, SENTIMENT_ORDER, "closed_pnl", "sentiment categories")

# ====================================================================
# PHASE 7: BEHAVIOR VS SENTIMENT
# ====================================================================
section("PHASE 7: TRADING BEHAVIOR VS MARKET SENTIMENT")

behavior_by_sentiment = df[df["sentiment"].notna()].groupby("sentiment", observed=True).agg(
    n_trades=("size_usd", "size"),
    avg_trade_size=("size_usd", "mean"),
    median_trade_size=("size_usd", "median"),
    total_volume=("size_usd", "sum"),
    avg_fee=("fee", "mean"),
    n_active_traders=("account", "nunique"),
    buy_pct=("is_buy", "mean"),
).reindex(SENTIMENT_ORDER)
behavior_by_sentiment["sell_pct"] = 1 - behavior_by_sentiment["buy_pct"]

# Trading frequency: trades per active trader per active day within each sentiment
days_per_sentiment = df[df["sentiment"].notna()].groupby("sentiment", observed=True)["trade_date"].nunique().reindex(SENTIMENT_ORDER)
behavior_by_sentiment["calendar_days"] = days_per_sentiment
behavior_by_sentiment["trades_per_trader_per_day"] = (
    behavior_by_sentiment["n_trades"] /
    (behavior_by_sentiment["n_active_traders"] * behavior_by_sentiment["calendar_days"])
)

print(behavior_by_sentiment.round(4))
save_table(behavior_by_sentiment.round(4), "behavior_by_sentiment")

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=behavior_by_sentiment.index, y=behavior_by_sentiment["avg_trade_size"],
            order=SENTIMENT_ORDER, palette=SENTIMENT_PALETTE, ax=ax)
ax.set_title("Average Trade Size (USD) by Sentiment")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Avg trade size (USD)")
format_large_numbers(ax)
save_fig(fig, "09_avg_trade_size_by_sentiment")

fig, ax = plt.subplots(figsize=(10, 6))
buy_sell = df[df["sentiment"].notna()].groupby(["sentiment", "side"], observed=True).size().unstack()
buy_sell = buy_sell.reindex(SENTIMENT_ORDER)
buy_sell_pct = buy_sell.div(buy_sell.sum(axis=1), axis=0) * 100
buy_sell_pct.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_title("BUY vs SELL Share by Sentiment")
ax.set_xlabel("Sentiment"); ax.set_ylabel("% of trades")
ax.tick_params(axis='x', rotation=30)
ax.legend(title="Side")
save_fig(fig, "10_buy_sell_share_by_sentiment")

most_active_sentiment = behavior_by_sentiment["trades_per_trader_per_day"].idxmax()
largest_size_sentiment = behavior_by_sentiment["avg_trade_size"].idxmax()
print(f"\nHighest trading frequency (trades/trader/day): {most_active_sentiment}")
print(f"Largest average position size: {largest_size_sentiment}")

# ====================================================================
# PHASE 8: DIRECTION / SIDE ANALYSIS
# ====================================================================
section("PHASE 8: BUY VS SELL PERFORMANCE BY SENTIMENT")

direction_perf = closing_sent.groupby(["sentiment", "side"], observed=True).agg(
    n_trades=("closed_pnl", "size"),
    total_pnl=("closed_pnl", "sum"),
    mean_pnl=("closed_pnl", "mean"),
    median_pnl=("closed_pnl", "median"),
    win_rate=("is_profitable", "mean"),
    avg_trade_size=("size_usd", "mean"),
    pnl_std=("closed_pnl", "std"),
).reindex(pd.MultiIndex.from_product([SENTIMENT_ORDER, ["BUY", "SELL"]],
                                       names=["sentiment", "side"]))
print(direction_perf.round(3))
save_table(direction_perf.round(4), "direction_performance_by_sentiment")

fig, ax = plt.subplots(figsize=(10, 6))
dp = direction_perf.reset_index()
sns.barplot(x="sentiment", y="mean_pnl", hue="side", data=dp, order=SENTIMENT_ORDER, ax=ax)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Mean Closed PnL: BUY vs SELL by Sentiment")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Mean Closed PnL (USD)")
ax.tick_params(axis='x', rotation=30)
save_fig(fig, "11_buy_sell_pnl_by_sentiment")

# Are apparent BUY/SELL-sentiment patterns driven by a few extreme trades?
section("Robustness check: BUY/SELL pattern sensitivity to extreme trades")
for s in SENTIMENT_ORDER:
    sub = closing_sent[closing_sent["sentiment"] == s]
    if len(sub) < 30:
        continue
    trimmed = sub[sub["closed_pnl"].abs() <= sub["closed_pnl"].abs().quantile(0.95)]
    full_mean_diff = (sub[sub.side == "BUY"]["closed_pnl"].mean() -
                       sub[sub.side == "SELL"]["closed_pnl"].mean())
    trimmed_mean_diff = (trimmed[trimmed.side == "BUY"]["closed_pnl"].mean() -
                          trimmed[trimmed.side == "SELL"]["closed_pnl"].mean())
    print(f"{s}: BUY-SELL mean PnL gap = {full_mean_diff:.2f} (all data) vs "
          f"{trimmed_mean_diff:.2f} (top 5% extreme trades excluded)")

# Statistical test: chi-square sentiment vs profitability
section("PHASE 15 (partial): Chi-square - sentiment vs profitability")
ct = pd.crosstab(closing_sent["sentiment"], closing_sent["is_profitable"])
chi_result = sh.chi_square_test(ct, "sentiment", "is_profitable")

print("\nSaving intermediate results and continuing to trader-level analysis...")
df.to_parquet(Path(__file__).resolve().parent.parent / "data" / "processed" / "features.parquet", index=False)

# ====================================================================
# PHASE 9: TRADER-LEVEL ANALYSIS
# ====================================================================
section("PHASE 9: TRADER-LEVEL AGGREGATED ANALYSIS")

trader_agg = trader_level_aggregates(df)
print(f"Total traders: {len(trader_agg)}")
print(f"Minimum-trade threshold used for comparative analysis: "
      f"{MIN_TRADES_THRESHOLD} closing trades (documented per Phase 9 guidance)")

qualified = trader_agg[trader_agg["closing_trades"] >= MIN_TRADES_THRESHOLD].copy()
print(f"Traders meeting the {MIN_TRADES_THRESHOLD}-trade threshold: {len(qualified)} of {len(trader_agg)}")

print("\nTrader-level summary statistics (qualified traders only):")
print(qualified[["total_trades", "total_pnl", "avg_pnl", "median_pnl", "win_rate",
                  "avg_trade_size_usd", "total_fees", "risk_adjusted_pnl"]].describe().round(2))
save_table(trader_agg.round(4).set_index("account"), "trader_level_aggregates_all")
save_table(qualified.round(4).set_index("account"), "trader_level_aggregates_qualified")

trader_sent_perf = trader_sentiment_performance(df)
save_table(trader_sent_perf.round(4), "trader_sentiment_performance")

# ====================================================================
# PHASE 10: TRADER SEGMENTATION
# ====================================================================
section("PHASE 10: TRADER SEGMENTATION (Top 20% / Middle 60% / Bottom 20% by total PnL)")

q_top = qualified["total_pnl"].quantile(0.80)
q_bottom = qualified["total_pnl"].quantile(0.20)
def segment(pnl):
    if pnl >= q_top:
        return "Top 20%"
    elif pnl <= q_bottom:
        return "Bottom 20%"
    return "Middle 60%"
qualified["segment"] = qualified["total_pnl"].apply(segment)
print(f"Top-20% PnL threshold: >= ${q_top:,.2f}")
print(f"Bottom-20% PnL threshold: <= ${q_bottom:,.2f}")
print(qualified["segment"].value_counts())

seg_compare = qualified.groupby("segment").agg(
    n_traders=("account", "count"),
    avg_total_trades=("total_trades", "mean"),
    avg_trade_size=("avg_trade_size_usd", "mean"),
    avg_win_rate=("win_rate", "mean"),
    avg_buy_pct=("buy_pct", "mean"),
    avg_fees=("total_fees", "mean"),
    avg_pnl_std=("pnl_std", "mean"),
    avg_risk_adjusted=("risk_adjusted_pnl", "mean"),
    avg_trades_per_day=("trades_per_active_day", "mean"),
).reindex(["Top 20%", "Middle 60%", "Bottom 20%"])
print(seg_compare.round(3))
save_table(seg_compare.round(4), "trader_segment_comparison")

seg_map = qualified.set_index("account")["segment"]
tsp = trader_sent_perf.copy()
tsp["segment"] = tsp["account"].map(seg_map)
tsp_seg = tsp.dropna(subset=["segment"])
seg_sentiment = tsp_seg.groupby(["segment", "sentiment"], observed=True).agg(
    avg_mean_pnl=("mean_pnl", "mean"), avg_win_rate=("win_rate", "mean"),
    total_trades=("trades", "sum")
).reindex(pd.MultiIndex.from_product([["Top 20%", "Middle 60%", "Bottom 20%"], SENTIMENT_ORDER],
                                       names=["segment", "sentiment"]))
print("\nSegment performance by sentiment:")
print(seg_sentiment.round(3))
save_table(seg_sentiment.round(4), "segment_performance_by_sentiment")

fig, ax = plt.subplots(figsize=(9, 5))
seg_compare["avg_win_rate"].mul(100).plot(kind="bar", ax=ax, color=["#2ca02c", "#7f7f7f", "#d62728"])
ax.set_title("Average Win Rate by Trader Segment")
ax.set_ylabel("Win rate (%)"); ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
save_fig(fig, "12_win_rate_by_segment")

fig, ax = plt.subplots(figsize=(9, 5))
seg_compare["avg_trades_per_day"].plot(kind="bar", ax=ax, color=["#2ca02c", "#7f7f7f", "#d62728"])
ax.set_title("Average Trades per Active Day by Trader Segment")
ax.set_ylabel("Trades / active day"); ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
save_fig(fig, "13_trade_frequency_by_segment")

top_freq = seg_compare.loc["Top 20%", "avg_trades_per_day"]
bottom_freq = seg_compare.loc["Bottom 20%", "avg_trades_per_day"]
print(f"\nTop 20% traders average {top_freq:.2f} trades/active day vs "
      f"{bottom_freq:.2f} for Bottom 20% -> "
      f"{'Top traders trade MORE' if top_freq > bottom_freq else 'Bottom traders trade MORE (possible overtrading signal)'}")

section("PHASE 15 (partial): Mann-Whitney - win rate, Top 20% vs Bottom 20%")
sh.mannwhitney_test(
    qualified.loc[qualified["segment"] == "Top 20%", "win_rate"].values,
    qualified.loc[qualified["segment"] == "Bottom 20%", "win_rate"].values,
    "Top 20%", "Bottom 20%", "win_rate"
)

# ====================================================================
# PHASE 11: OVERTRADING ANALYSIS
# ====================================================================
section("PHASE 11: OVERTRADING ANALYSIS (trader-day level)")

trader_day = trader_day_aggregates(df)
numeric_day_cols = trader_day.select_dtypes(include=[np.number]).columns
trader_day_to_save = trader_day.copy()
trader_day_to_save[numeric_day_cols] = trader_day_to_save[numeric_day_cols].round(4)
save_table(trader_day_to_save, "trader_day_aggregates")

sh.spearman_corr(trader_day["n_trades"], trader_day["total_pnl"], "n_trades", "total_pnl")
res_freq_pnlpertrade = sh.spearman_corr(trader_day["n_trades"], trader_day["pnl_per_trade"], "n_trades", "pnl_per_trade")
sh.spearman_corr(trader_day["n_trades"], trader_day["win_rate"], "n_trades", "win_rate")
sh.spearman_corr(trader_day["n_trades"], trader_day["total_fees"], "n_trades", "total_fees")

fig, ax = plt.subplots(figsize=(9, 5))
bucket_edges = [0, 5, 10, 20, 50, trader_day["n_trades"].max() + 1]
trader_day["freq_bucket"] = pd.cut(trader_day["n_trades"], bins=bucket_edges,
                                     labels=["1-5", "6-10", "11-20", "21-50", "50+"])
freq_perf = trader_day.groupby("freq_bucket", observed=True)["pnl_per_trade"].mean()
freq_perf.plot(kind="bar", ax=ax, color="#4C72B0")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Average PnL per Trade by Daily Trade-Frequency Bucket")
ax.set_xlabel("Trades per trader per day"); ax.set_ylabel("Avg PnL per trade (USD)")
ax.tick_params(axis='x', rotation=0)
save_fig(fig, "14_pnl_per_trade_by_frequency_bucket")

rho_val = res_freq_pnlpertrade["rho"] if res_freq_pnlpertrade else float("nan")
print(f"\nSpearman corr(trade frequency, PnL per trade) = {rho_val:.4f} -> "
      f"{'higher frequency associated with LOWER per-trade PnL' if rho_val < 0 else 'no clear negative relationship'} "
      f"(correlation only, not causation).")

freq_sent = trader_day.dropna(subset=["sentiment"]).groupby("sentiment", observed=True).agg(
    avg_daily_trades=("n_trades", "mean"), avg_pnl_per_trade=("pnl_per_trade", "mean")
).reindex(SENTIMENT_ORDER)
print("\nDaily trade frequency and PnL/trade by sentiment:")
print(freq_sent.round(3))
save_table(freq_sent.round(4), "frequency_vs_performance_by_sentiment")

# ====================================================================
# PHASE 12: RISK ANALYSIS
# ====================================================================
section("PHASE 12: RISK ANALYSIS")

risk_by_sentiment = closing_sent.groupby("sentiment", observed=True).agg(
    avg_position_size=("size_usd", "mean"),
    pnl_volatility=("closed_pnl", "std"),
    max_loss=("closed_pnl", "min"),
    avg_loss=("closed_pnl", lambda x: x[x < 0].mean()),
    avg_fee_burden=("fee", "mean"),
).reindex(SENTIMENT_ORDER)
risk_by_sentiment["risk_adjusted_perf"] = perf_by_sentiment["risk_adjusted"]
print(risk_by_sentiment.round(3))
save_table(risk_by_sentiment.round(4), "risk_by_sentiment")

print("\nNOTE: No 'leverage' column exists in the raw dataset -> leverage-based "
      "risk metrics are not computed (documented limitation).")

sh.spearman_corr(closing_sent["size_usd"], closing_sent["closed_pnl"], "size_usd", "closed_pnl")

extreme_greed_loss = risk_by_sentiment.loc["Extreme Greed", "max_loss"]
fear_loss = risk_by_sentiment.loc["Fear", "max_loss"]
print(f"\nMax single loss - Extreme Greed: ${extreme_greed_loss:,.2f} vs Fear: ${fear_loss:,.2f}")

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=risk_by_sentiment.index, y=risk_by_sentiment["pnl_volatility"],
            order=SENTIMENT_ORDER, hue=risk_by_sentiment.index, palette=SENTIMENT_PALETTE,
            legend=False, ax=ax)
ax.set_title("Closed PnL Volatility (Std Dev) by Sentiment")
ax.set_xlabel("Sentiment"); ax.set_ylabel("PnL std dev (USD)")
format_large_numbers(ax)
save_fig(fig, "15_pnl_volatility_by_sentiment")

# ====================================================================
# PHASE 13: ASSET / COIN ANALYSIS
# ====================================================================
section("PHASE 13: ASSET / COIN ANALYSIS")

MIN_ASSET_OBS = 100
asset_stats = closing.groupby("coin").agg(
    n_trades=("closed_pnl", "size"),
    total_pnl=("closed_pnl", "sum"),
    win_rate=("is_profitable", "mean"),
    total_volume=("size_usd", "sum"),
).sort_values("n_trades", ascending=False)
qualified_assets = asset_stats[asset_stats["n_trades"] >= MIN_ASSET_OBS]
print(f"Assets with >= {MIN_ASSET_OBS} closing trades: {len(qualified_assets)} of {len(asset_stats)} total assets")
print("\nTop 10 most frequently traded assets:")
print(asset_stats.head(10).round(2))
print("\nTop 10 most profitable assets (min sample size applied):")
print(qualified_assets.sort_values("total_pnl", ascending=False).head(10).round(2))
save_table(qualified_assets.round(4), "asset_performance")

top_assets = qualified_assets.sort_values("n_trades", ascending=False).head(8).index.tolist()
asset_sentiment = closing_sent[closing_sent["coin"].isin(top_assets)].groupby(
    ["coin", "sentiment"], observed=True)["closed_pnl"].mean().unstack().reindex(columns=SENTIMENT_ORDER)
print("\nMean PnL by top asset x sentiment:")
print(asset_sentiment.round(2))
save_table(asset_sentiment.round(4), "asset_sentiment_performance")

fig, ax = plt.subplots(figsize=(10, 6))
top10_by_vol = asset_stats.sort_values("total_volume", ascending=False).head(10)
sns.barplot(x=top10_by_vol["total_volume"], y=top10_by_vol.index, ax=ax, color="#4C72B0")
ax.set_title("Top 10 Assets by Trading Volume (USD)")
ax.set_xlabel("Total volume (USD)"); ax.set_ylabel("Asset")
format_large_numbers(ax, axis="x")
save_fig(fig, "16_top_assets_by_volume")

# ====================================================================
# PHASE 14: TIME-BASED ANALYSIS
# ====================================================================
section("PHASE 14: TIME-BASED ANALYSIS")

hour_perf = closing.groupby("trade_hour")["closed_pnl"].agg(["mean", "sum", "count"])
dow_perf = closing.groupby("day_of_week")["closed_pnl"].agg(["mean", "sum", "count"]).reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
print("Mean/total PnL by hour of day (IST):")
print(hour_perf.round(2))
print("\nMean/total PnL by day of week:")
print(dow_perf.round(2))
save_table(hour_perf.round(4), "pnl_by_hour")
save_table(dow_perf.round(4), "pnl_by_day_of_week")

fig, ax = plt.subplots(figsize=(10, 5))
hour_perf["sum"].plot(kind="bar", ax=ax, color="#4C72B0")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Total Closed PnL by Hour of Day (IST)")
ax.set_xlabel("Hour (IST)"); ax.set_ylabel("Total PnL (USD)")
format_large_numbers(ax)
save_fig(fig, "17_pnl_by_hour")

section("Sentiment transition analysis")
sent_daily = df[["trade_date", "sentiment"]].drop_duplicates("trade_date").sort_values("trade_date")
sent_daily["prev_sentiment"] = sent_daily["sentiment"].shift(1)
transitions_of_interest = {
    "Fear -> Greed": (sent_daily["prev_sentiment"].isin(["Fear", "Extreme Fear"])) &
                      (sent_daily["sentiment"].isin(["Greed", "Extreme Greed"])),
    "Greed -> Fear": (sent_daily["prev_sentiment"].isin(["Greed", "Extreme Greed"])) &
                      (sent_daily["sentiment"].isin(["Fear", "Extreme Fear"])),
}
for label, mask in transitions_of_interest.items():
    trans_dates = sent_daily.loc[mask, "trade_date"]
    n_days = len(trans_dates)
    if n_days == 0:
        print(f"{label}: no qualifying transition days found in trading date range.")
        continue
    trans_day_trades = df[df["trade_date"].isin(trans_dates)]
    print(f"{label}: {n_days} transition days, "
          f"avg trades/day = {len(trans_day_trades)/n_days:.1f}, "
          f"avg trade size = ${trans_day_trades['size_usd'].mean():,.2f}")

# ====================================================================
# PHASE 16: CORRELATION ANALYSIS
# ====================================================================
section("PHASE 16: CORRELATION ANALYSIS (numeric features)")

corr_cols = ["closed_pnl", "size_usd", "execution_price", "fee", "sentiment_value"]
corr_data = closing_sent[corr_cols].copy()
spearman_corr_matrix = corr_data.corr(method="spearman")
print(spearman_corr_matrix.round(3))
save_table(spearman_corr_matrix.round(4), "correlation_matrix_spearman")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(spearman_corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
ax.set_title("Spearman Correlation Heatmap (Trade-Level Features)")
save_fig(fig, "18_correlation_heatmap")

strongest = spearman_corr_matrix.abs().where(
    ~np.eye(len(corr_cols), dtype=bool)).stack().idxmax()
print(f"\nStrongest pairwise correlation: {strongest[0]} <-> {strongest[1]} "
      f"(rho = {spearman_corr_matrix.loc[strongest]:.3f})")

# ====================================================================
# PHASE 17: OUTLIER / ROBUSTNESS ANALYSIS
# ====================================================================
section("PHASE 17: OUTLIER AND ROBUSTNESS ANALYSIS")

trader_pnl_sorted = trader_agg.sort_values("total_pnl", ascending=False)
top5_share = trader_pnl_sorted.head(5)["total_pnl"].sum() / trader_agg["total_pnl"].sum() * 100
print(f"Top 5 traders account for {top5_share:.1f}% of total realized PnL across all {len(trader_agg)} traders")

trade_pnl_sorted = closing.reindex(closing["closed_pnl"].abs().sort_values(ascending=False).index)
top1pct_n = max(1, int(len(closing) * 0.01))
top1pct_share = trade_pnl_sorted.head(top1pct_n)["closed_pnl"].abs().sum() / closing["closed_pnl"].abs().sum() * 100
print(f"Top 1% of trades by |PnL| account for {top1pct_share:.1f}% of total absolute PnL")
save_table(pd.DataFrame({"metric": ["top5_traders_pnl_share_pct", "top1pct_trades_abs_pnl_share_pct"],
                          "value": [top5_share, top1pct_share]}).set_index("metric"),
           "outlier_concentration")

winsor_lower = closing_sent["closed_pnl"].quantile(0.01)
winsor_upper = closing_sent["closed_pnl"].quantile(0.99)
closing_sent["pnl_winsorized"] = closing_sent["closed_pnl"].clip(winsor_lower, winsor_upper)
winsor_perf = closing_sent.groupby("sentiment", observed=True)["pnl_winsorized"].agg(["mean", "median"]).reindex(SENTIMENT_ORDER)
print("\nWinsorized (1st/99th pct) mean/median PnL by sentiment (robustness check):")
print(winsor_perf.round(3))
save_table(winsor_perf.round(4), "winsorized_performance_by_sentiment")

winsor_leader = winsor_perf["mean"].idxmax()
original_leader = perf_by_sentiment["mean_pnl"].idxmax()
print(f"\nMost profitable sentiment by winsorized mean PnL: {winsor_leader} "
      f"(original unwinsorized mean-PnL leader was {original_leader}) -> "
      f"conclusion {'UNCHANGED' if winsor_leader == original_leader else 'CHANGES under winsorization, indicating mean PnL by sentiment is sensitive to extreme trades'}")

section("PHASES 5-17 COMPLETE")
print("Proceed to hidden-pattern synthesis and executive summary (see outputs/tables and outputs/figures).")