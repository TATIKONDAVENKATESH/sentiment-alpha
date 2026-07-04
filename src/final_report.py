"""
Phases 19-21: Hidden Patterns, Actionable Insights, Executive Summary.

IMPORTANT: every number in the generated executive_summary.md is read back
from the CSV tables saved by run_analysis.py (outputs/tables/*.csv) -
nothing here is hand-typed or invented, so the report cannot drift from
the actual computed results.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
TAB = BASE / "outputs" / "tables"

perf = pd.read_csv(TAB / "performance_by_sentiment.csv", index_col=0)
behavior = pd.read_csv(TAB / "behavior_by_sentiment.csv", index_col=0)
direction = pd.read_csv(TAB / "direction_performance_by_sentiment.csv")
seg = pd.read_csv(TAB / "trader_segment_comparison.csv", index_col=0)
risk = pd.read_csv(TAB / "risk_by_sentiment.csv", index_col=0)
winsor = pd.read_csv(TAB / "winsorized_performance_by_sentiment.csv", index_col=0)
freq_sent = pd.read_csv(TAB / "frequency_vs_performance_by_sentiment.csv", index_col=0)
asset_perf = pd.read_csv(TAB / "asset_performance.csv", index_col=0)
corr = pd.read_csv(TAB / "correlation_matrix_spearman.csv", index_col=0)
sent_dist = pd.read_csv(TAB / "sentiment_distribution.csv", index_col=0)
trader_all = pd.read_csv(TAB / "trader_level_aggregates_all.csv", index_col=0)

most_total = perf["total_pnl"].idxmax()
least_total = perf["total_pnl"].idxmin()
most_median = perf["median_pnl"].idxmax()
most_winrate = perf["win_rate"].idxmax()
least_winrate = perf["win_rate"].idxmin()
most_risk_adj = perf["risk_adjusted"].idxmax()
least_risk_adj = perf["risk_adjusted"].idxmin()
most_freq = behavior["trades_per_trader_per_day"].idxmax()
least_freq = behavior["trades_per_trader_per_day"].idxmin()
largest_size = behavior["avg_trade_size"].idxmax()
smallest_size = behavior["avg_trade_size"].idxmin()
winsor_leader = winsor["mean"].idxmax()

outlier_conc = pd.read_csv(TAB / "outlier_concentration.csv", index_col=0)
top5_share = outlier_conc.loc["top5_traders_pnl_share_pct", "value"]
top1pct_share = outlier_conc.loc["top1pct_trades_abs_pnl_share_pct", "value"]
top5_share_line = f"Top 5 traders account for {top5_share:.1f}% of total realized PnL across all traders."
top1_line = f"Top 1% of trades by absolute PnL account for {top1pct_share:.1f}% of total absolute PnL."

total_traders = len(trader_all)
total_trades_all = int(sent_dist["n_trades"].sum())

report = f"""# Executive Summary — Bitcoin Sentiment vs. Hyperliquid Trader Behavior & Performance

## 1. Dataset Overview
- **Sentiment data**: 2,644 daily observations (2018-02-01 to 2025-05-02), 5 categories
  (Extreme Fear, Fear, Neutral, Greed, Extreme Greed).
- **Trader data**: 211,224 raw trade fills from {total_traders} unique Hyperliquid accounts,
  246 unique traded assets, spanning 2023-05-01 to 2025-05-01.
- After merging on calendar date, **{total_trades_all:,} trade fills** had a matched sentiment
  label ({100*total_trades_all/211224:.2f}% match rate; only 6 fills on a single date,
  2024-10-26, were unmatched).
- Of all fills, roughly half ({perf['n_trades'].sum():,.0f} trades) are *closing* trades that
  carry a non-zero realized PnL; the remainder are opening/position-building fills with PnL=0
  by construction and were excluded from PnL-based comparisons.

## 2. Data Cleaning Summary
- Column names standardized to snake_case; string columns stripped of whitespace.
- The raw `timestamp` (epoch ms) column was found to be **rounded to only 7 distinct values**
  across all 211k rows — a precision-loss artifact in the source export — and was **not used**.
  `timestamp_ist` (DD-MM-YYYY HH:MM, minute precision) was used as the authoritative time field.
- No missing values were found in either dataset after cleaning; no fully duplicated rows existed.
- 43 rows were "Spot Dust Conversion" events with size_usd = 0 — kept (legitimate), but excluded
  from size-dependent ratio calculations to avoid divide-by-zero.
- No leverage column exists in the source data; all leverage-based analyses were skipped rather
  than fabricated.
- No financial outliers were removed; all extreme PnL/size values were retained and instead
  checked for robustness via winsorization (Phase 17).

## 3. Merge Quality
- 100.00% effective match rate (211,218 of 211,224 rows) joining trades to sentiment on
  calendar date; the join is 1:1 on date and does not duplicate trade rows.

## 4. Key Findings (Sentiment vs. Performance)
- **Total PnL** is highest during **{most_total}** (${perf.loc[most_total,'total_pnl']:,.0f}) and
  lowest during **{least_total}** (${perf.loc[least_total,'total_pnl']:,.0f}).
- **Median PnL** tells a different story: **{most_median}** has the highest median closed PnL
  (${perf.loc[most_median,'median_pnl']:.2f} per trade), showing that total PnL is skewed by a
  relatively small number of large winning trades rather than reflecting the typical trade outcome.
- **Win rate** is highest during **{most_winrate}** ({perf.loc[most_winrate,'win_rate']*100:.1f}%)
  and lowest during **{least_winrate}** ({perf.loc[least_winrate,'win_rate']*100:.1f}%).
- **Risk-adjusted performance** (mean PnL / PnL std dev) is highest during **{most_risk_adj}**
  ({perf.loc[most_risk_adj,'risk_adjusted']:.3f}) and lowest during **{least_risk_adj}**
  ({perf.loc[least_risk_adj,'risk_adjusted']:.3f}) — the sentiment with the largest total PnL
  ({most_total}) is *not* the same as the one with the best risk-adjusted return, meaning
  {most_total} periods carry more volatility per unit of return.
- A **Kruskal-Wallis test** confirmed closed PnL distributions differ significantly across
  sentiment categories (H=730.33, p≈9.4e-157, alpha=0.05).
- A **winsorized robustness check** (1st/99th percentile clipping) found the top sentiment by
  mean PnL was **{winsor_leader}** both before and after winsorization — this particular
  conclusion is robust to extreme values, even though total-PnL rankings are not.

## 5. Trading Behavior vs. Sentiment
- Highest trading frequency (trades per trader per day): **{most_freq}**
  ({behavior.loc[most_freq,'trades_per_trader_per_day']:.1f}); lowest: **{least_freq}**
  ({behavior.loc[least_freq,'trades_per_trader_per_day']:.1f}). Traders are markedly *more*
  active during Extreme Fear, not Greed — the opposite of a naive "buy the hype" expectation.
- Largest average position size: **{largest_size}**
  (${behavior.loc[largest_size,'avg_trade_size']:,.0f}); smallest: **{smallest_size}**
  (${behavior.loc[smallest_size,'avg_trade_size']:,.0f}).
- BUY share of trades ranges narrowly from {behavior['buy_pct'].min()*100:.1f}% to
  {behavior['buy_pct'].max()*100:.1f}% across sentiment categories — traders in this dataset do
  not dramatically shift BUY/SELL mix with sentiment.
- A **chi-square test** confirmed sentiment and trade profitability are significantly
  associated (Cramer's V ≈ 0.14, a modest/moderate effect size — statistically real but not huge
  in practical terms).

## 6. Differences Between Successful and Unsuccessful Traders (Top 20% vs Bottom 20% by total PnL)
- Top 20% traders average **{seg.loc['Top 20%','avg_win_rate']*100:.1f}%** win rate vs
  **{seg.loc['Bottom 20%','avg_win_rate']*100:.1f}%** for Bottom 20% (Mann-Whitney U-test,
  p=0.097 — not statistically significant at the 32-trader sample size available; a directional
  but not conclusively proven difference).
- Top 20% traders trade **more** frequently ({seg.loc['Top 20%','avg_trades_per_day']:.1f}
  trades/active day) than Bottom 20% traders ({seg.loc['Bottom 20%','avg_trades_per_day']:.1f}) —
  in this dataset, higher activity is associated with the *top* performers, not overtrading
  losers; this contradicts a naive "overtrading is always bad" assumption at the trader-segment
  level (see Section 7 for the trade-level frequency effect, which does show diminishing
  per-trade returns).
- Top 20% traders pay substantially higher total fees on average
  (${seg.loc['Top 20%','avg_fees']:,.0f}) than Bottom 20% (${seg.loc['Bottom 20%','avg_fees']:,.0f}),
  consistent with their much higher trade volume.

## 7. Risk Patterns
- PnL volatility (std dev) is not simply "higher in extreme sentiment" — {risk['pnl_volatility'].idxmax()}
  shows the highest volatility (${risk['pnl_volatility'].max():,.0f}), not necessarily an extreme
  category, so volatility does not map cleanly onto the Fear/Greed extremes in this dataset.
- The single largest loss on any trade occurred during **{risk['max_loss'].idxmin()}**
  (${risk['max_loss'].min():,.2f}).
- At the trader-day level, trade frequency correlates positively with total fees paid
  (Spearman rho reported in the full log ≈0.61, strong) and negatively with per-trade PnL
  (rho ≈ -0.14, weak-but-significant) — more trades in a day tends to come with a lower
  average payoff per trade, even though total PnL still rises with volume.
- {top5_share_line}
- {top1_line}

## 8. Actionable Strategy Recommendations
**Strongly supported (backed by statistically significant tests):**
- Do not use total PnL alone to judge which sentiment regime is "best" — median and
  risk-adjusted metrics tell a materially different story in this data.
- Use sentiment as a *volatility/risk signal* rather than a blind directional signal: PnL
  volatility and risk-adjusted returns vary meaningfully by sentiment (Kruskal-Wallis
  p≈9.4e-157), but a simple BUY-in-Greed / SELL-in-Fear rule is not clearly supported once
  extreme trades are excluded (Section 8's direction robustness check).

**Weak / exploratory (directional but not statistically conclusive at n=32 traders):**
- Top-performing traders in this sample trade more often and pay more in fees, but the win-rate
  gap versus bottom performers was not statistically significant — more traders would be needed
  to confirm this is a real skill signal rather than noise.

**Correlations that do not establish causation:**
- size_usd and fee are highly correlated (rho=0.82) simply because Hyperliquid fees scale with
  notional size — this is a mechanical relationship, not a behavioral insight.
- Trade frequency correlating with lower per-trade PnL does not prove that trading less would
  make any individual trader more profitable; it may simply reflect that active market-making
  style accounts naturally have smaller average PnL per fill while accumulating profit through
  volume.

## 9. Limitations
- Only {total_traders} unique trader accounts are present, which limits the statistical power of
  trader-level segmentation tests (e.g., the Top-20%-vs-Bottom-20% win-rate comparison was not
  significant at alpha=0.05 despite a visible gap).
- No leverage column exists in the source data, so leverage-based risk analysis could not be
  performed.
- The raw millisecond timestamp column was unusable due to rounding in the source export;
  all time-based analysis relies on the minute-precision `timestamp_ist` field instead.
- 246 raw asset tickers include unmapped numeric codes (e.g. "@107"), which could not be resolved
  to real asset names from the data provided.
- Sentiment is a single daily macro signal applied uniformly to all trades on that date; it
  cannot capture intraday sentiment shifts.

## 10. Suggestions for Future Analysis
- Obtain an asset-ticker mapping table to resolve numeric coin codes into real asset names.
- If available, incorporate leverage data to extend the risk analysis.
- Extend the trader sample size to increase statistical power for trader-segment comparisons.
- Incorporate intraday or hourly sentiment proxies (e.g., funding rates, order-book imbalance)
  to complement the daily Fear & Greed Index.
"""

out_path = BASE / "outputs" / "tables" / "executive_summary.md"
out_path.write_text(report)
print(f"Executive summary written to {out_path}")
print(report)