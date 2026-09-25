"""Shared look for the SLB figures: IBM Plex Sans, thin ink axes, one palette across figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent
WIDE = 7.2  # inches, a full text width

INK = "#18212b"
MUTED = "#6b7682"
FAINT = "#c9d0d6"
GRID = "#e9ecef"
PAPER = "#ffffff"

TEST = "#e8a33d"   # amber: test split, SL calls
DEV = "#23a08c"    # teal: dev split
TRAIN = "#7f95ab"  # slate: train split
CORAL = "#d6573b"  # thresholds, rejections
NAVY = "#123a56"
BLUE = "#2c7bb6"

SPECIES = {"human": "H. sapiens", "scer": "S. cerevisiae", "spom": "S. pombe", "bsub": "B. subtilis",
           "cele": "C. elegans", "dmel": "D. melanogaster", "mmus": "M. musculus"}
SPECIES_COLOR = {"human": "#2c7bb6", "scer": "#23a08c", "spom": "#8a63b8", "bsub": "#e8a33d",
                 "cele": "#d6573b", "dmel": "#7a9a3c", "mmus": "#b8657f"}

SWIRL = LinearSegmentedColormap.from_list(
    "swirl", ["#f7faf9", "#d6ebee", "#9dcdd6", "#58a3b8", "#2b7292", "#154b6a", "#0a2438"])
EMBER = LinearSegmentedColormap.from_list(
    "ember", ["#000000", "#2a1203", "#6b2e05", "#b85c0c", "#e8a33d", "#f7d9a0", "#fffaf0"])


def use() -> None:
    plt.rcParams.update({
        "font.family": "IBM Plex Sans", "font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.6,
        "mathtext.fontset": "custom", "mathtext.rm": "IBM Plex Sans", "mathtext.it": "IBM Plex Sans:italic",
        "mathtext.bf": "IBM Plex Sans:bold",
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "lines.linewidth": 1.2, "lines.solid_capstyle": "round", "axes.titlepad": 4,
    })


def panel(ax, letter: str, x: float = -0.02, y: float = 1.02, color: str = INK, **kw) -> None:
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=color,
            ha="right", va="bottom", **kw)


def save(fig, stem: str, dpi: int = 300, facecolor: str | None = None) -> Path:
    path = OUT / f"{stem}.png"
    fig.savefig(path, dpi=dpi, facecolor=facecolor or fig.get_facecolor())
    plt.close(fig)
    print("wrote", path)
    return path
