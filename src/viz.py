"""Shared matplotlib styling: one validated categorical palette, thin marks, quiet axes."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated for colour-vision deficiency).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#8a8986"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def label_line_end(ax, x, y, text, color, dx=4):
    """Direct label at the right end of a line (text stays in ink, marker carries colour)."""
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)
    ax.plot([x], [y], "o", ms=4, color=color)
