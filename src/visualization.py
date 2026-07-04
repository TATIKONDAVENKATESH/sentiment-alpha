"""
Phase 18: Visualization helpers.
Consistent styling, saves all figures to outputs/figures/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_PALETTE = {
    "Extreme Fear": "#8B0000",
    "Fear": "#E9967A",
    "Neutral": "#B0B0B0",
    "Greed": "#90EE90",
    "Extreme Greed": "#006400",
}

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"


def format_large_numbers(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if abs(x) >= 1e6
                                 else f"{x/1e3:.0f}K" if abs(x) >= 1e3 else f"{x:.0f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def save_fig(fig, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path