"""Builds notebooks/analysis.ipynb as an executable notebook that runs
the full pipeline (src/ modules) and displays results inline. Run once
to (re)generate the notebook file."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Bitcoin Market Sentiment vs. Hyperliquid Trader Behavior & Performance

**Objective:** Explore and analyze the relationship between Bitcoin market sentiment
(Fear & Greed Index) and trader behavior/performance using historical Hyperliquid trade data.

This notebook orchestrates the full pipeline implemented in `src/`:
1. Data cleaning & merging (`src/data_processing.py`)
2. Feature engineering (`src/feature_engineering.py`)
3. Full analysis - EDA, sentiment performance, behavior, direction, trader-level,
   segmentation, overtrading, risk, asset, time-based, statistical testing,
   correlation, outlier/robustness (`src/run_analysis.py`)
4. Hidden patterns + executive summary synthesis (`src/final_report.py`)

Run all cells top-to-bottom. Figures are saved to `outputs/figures/` and tables to
`outputs/tables/` as a side effect, and also displayed inline below.""")

code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
pd.set_option("display.max_columns", 30)""")

md("## Phase 1-3: Data Inspection, Cleaning & Merging")
code("""from data_processing import load_and_clean_sentiment, load_and_clean_trades, merge_datasets

sentiment = load_and_clean_sentiment()
trades = load_and_clean_trades()
merged = merge_datasets(trades, sentiment)
merged.head()""")

md("## Phase 4: Feature Engineering")
code("""from feature_engineering import add_trade_level_features, trader_level_aggregates, MIN_TRADES_THRESHOLD

df = add_trade_level_features(merged)
closing = df[df["is_closing_trade"]].copy()
closing_sent = closing[closing["sentiment"].notna()].copy()
print(f"Total fills: {len(df):,} | Closing fills: {len(closing):,} | Unique traders: {df['account'].nunique()}")
df.head()""")

md("""## Phases 5-17: Full Analysis

The full analysis (EDA, sentiment performance, behavior analysis, direction analysis,
trader-level aggregation, segmentation, overtrading, risk, asset, time-based,
statistical testing, correlation, and outlier/robustness checks) is implemented in
`src/run_analysis.py`. We execute it here as a script so all printed output, tables,
and figures are generated in one reproducible pass.""")

code("""import subprocess, sys
result = subprocess.run(
    [sys.executable, str(Path.cwd().parent / "src" / "run_analysis.py")],
    cwd=str(Path.cwd().parent), capture_output=True, text=True
)
print(result.stdout[-6000:])  # tail of console output
if result.returncode != 0:
    print(result.stderr)""")

md("### Key tables produced")
code("""tables_dir = Path.cwd().parent / "outputs" / "tables"
perf_by_sentiment = pd.read_csv(tables_dir / "performance_by_sentiment.csv", index_col=0)
perf_by_sentiment""")

code("""behavior_by_sentiment = pd.read_csv(tables_dir / "behavior_by_sentiment.csv", index_col=0)
behavior_by_sentiment""")

code("""trader_segment_comparison = pd.read_csv(tables_dir / "trader_segment_comparison.csv", index_col=0)
trader_segment_comparison""")

code("""risk_by_sentiment = pd.read_csv(tables_dir / "risk_by_sentiment.csv", index_col=0)
risk_by_sentiment""")

code("""asset_performance = pd.read_csv(tables_dir / "asset_performance.csv", index_col=0)
asset_performance.sort_values("total_pnl", ascending=False).head(10)""")

md("### Key figures produced")
code("""figs_dir = Path.cwd().parent / "outputs" / "figures"
for name in ["05_total_pnl_by_sentiment", "06_mean_vs_median_pnl_by_sentiment",
             "07_win_rate_by_sentiment", "08_pnl_boxplot_by_sentiment"]:
    display(Image(filename=str(figs_dir / f"{name}.png")))""")

code("""for name in ["12_win_rate_by_segment", "13_trade_frequency_by_segment",
             "15_pnl_volatility_by_sentiment", "18_correlation_heatmap"]:
    display(Image(filename=str(figs_dir / f"{name}.png")))""")

md("## Phases 19-21: Hidden Patterns, Strategy Insights & Executive Summary")
code("""result = subprocess.run(
    [sys.executable, str(Path.cwd().parent / "src" / "final_report.py")],
    cwd=str(Path.cwd().parent), capture_output=True, text=True
)
if result.returncode != 0:
    print(result.stderr)
summary_path = Path.cwd().parent / "outputs" / "tables" / "executive_summary.md"
display(Markdown(summary_path.read_text()))""")

nb["cells"] = cells
out_path = Path("notebooks/analysis.ipynb")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written to {out_path.resolve()}")