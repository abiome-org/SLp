"""Rebuild the three manuscript figures from the cited table and SLB result JSON.

Usage: uv run python figures/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/models/slb1.3"

# Feng et al., Nature Communications 15:9058 (2024), Table 3, complete
# SynLethDB dataset, random negatives, positive:negative ratio 1:1.
# Columns are CV1/2/3 F1 then CV1/2/3 NDCG@10; "-" is missing.
FENG_TABLE_3 = {
    "GRSMF": (.849, .750, .677, .284, .104, .000),
    "SL2MF": (.766, .667, .667, .280, .005, .000),
    "CMFW": (.717, .668, .667, .239, .116, .000),
    "SLMGAE": (.883, .779, .738, .270, .101, .039),
    "NSF4SL": (.869, .709, .685, .228, .104, .004),
    "PTGNN": (.869, .733, .670, .236, .120, .010),
    "PiLSL": (.863, .723, .670, np.nan, np.nan, np.nan),
    "KG4SL": (.878, .740, .667, .251, .108, .000),
    "SLGNN": (.859, .685, .668, .147, .045, .000),
    "DDGCN": (.839, .743, .667, .157, .008, .005),
    "GCATSL": (.883, .775, .692, .264, .122, .002),
    "MGE4SL": (.697, .670, .668, .003, .004, .004),
}

INK = "#192b3a"
MUTED = "#627789"
GRID = "#dbe4e8"
PAPER = "#fbfcf8"
BLUE = "#216e99"
TEAL = "#218d82"
CORAL = "#db6958"
PURPLE = "#7656a4"


def theme() -> None:
    plt.rcParams.update({
        "font.family": "IBM Plex Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png", "pdf"):
        path = OUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.22)
        if suffix == "svg":
            # Matplotlib emits spaces before path-command newlines. They have
            # no rendering role and make the generated asset fail git's check.
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def figure_1() -> None:
    """The published seen-gene to unseen-gene generalization gradient."""
    vals = np.array(list(FENG_TABLE_3.values()), dtype=float)
    fig, axarr = plt.subplots(1, 2, figsize=(12.5, 5.5), gridspec_kw={"wspace": .29})
    fig.subplots_adjust(left=.085, right=.98, top=.68, bottom=.22)
    fig.text(.085, .94, "When the genes become unfamiliar", fontsize=22, fontweight="bold", color=INK)
    fig.text(.085, .885, "12 published SL predictors  ·  Same study, three ways to hold out gene pairs",
             fontsize=11.5, color=MUTED)
    x = np.arange(3)
    labels = ["CV1\nboth genes seen", "CV2\none gene unseen", "CV3\nboth genes unseen"]
    for k, ax in enumerate(axarr):
        dat = vals[:, k * 3:(k + 1) * 3]
        for row in dat:
            ax.plot(x, row, color="#b4c3ca", lw=1.25, alpha=.85, zorder=1)
            ax.scatter(x, row, s=16, color="#b4c3ca", zorder=2)
        med = np.nanmedian(dat, axis=0)
        ax.plot(x, med, color=CORAL, lw=3.1, marker="o", ms=7, zorder=4)
        slm = dat[list(FENG_TABLE_3).index("SLMGAE")]
        ax.plot(x, slm, color=PURPLE, lw=2.2, marker="o", ms=5, zorder=3)
        ax.set_xlim(-.14, 2.18)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", length=0, pad=10, labelsize=10)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title("A  ·  F1 classification" if k == 0 else "B  ·  NDCG@10 ranking",
                     loc="left", fontweight="bold", fontsize=13, pad=17)
        ax.text(2.1, med[2] + (.007 if k == 0 else .014),
                f"median {med[2]:.3f}", color=CORAL, ha="right", fontsize=10, fontweight="bold")
        if k == 0:
            ax.set_ylim(.62, .925)
            ax.axhline(2 / 3, ls=(0, (3, 3)), lw=1, color=MUTED)
            ax.text(.03, .674, "all-positive F1 = 0.667", color=MUTED, fontsize=8.5)
        else:
            ax.set_ylim(-.017, .315)
            ax.set_yticks([0, .1, .2, .3])
    legend = [
        Line2D([0], [0], color="#b4c3ca", lw=1.5, label="individual methods"),
        Line2D([0], [0], color=CORAL, lw=3, marker="o", label="median"),
        Line2D([0], [0], color=PURPLE, lw=2.2, marker="o", label="SLMGAE"),
    ]
    fig.legend(handles=legend, loc="lower right", bbox_to_anchor=(.98, .765), frameon=False,
               ncol=3, fontsize=9.5)
    fig.text(.085, .035,
             "Source: Feng et al. (2024), Table 3. SynLethDB, 1:1 random negatives. "
             "Tested pair labels are held out in every CV split. F1 and NDCG are not AUROC.",
             color=MUTED, fontsize=9)
    save(fig, "01_gene_exposure")


def read_result(name: str) -> dict:
    with (RESULTS / f"{name}_test.json").open() as f:
        return json.load(f)


def figure_2() -> None:
    """SLB test benchmark: whole-benchmark performance and human-specific score."""
    specs = [
        ("slp_fusion__loss", "SLp Fusion", "ours", "all"),
        ("ontotype", "Ontotype", "mechanistic", "all"),
        ("go_ppi_gbm", "GO/PPI GBM", "mechanistic", "all"),
        ("dekegel2021__allspecies", "De Kegel 2021", "published", "all"),
        ("musl__allspecies", "MuSL 2026", "published", "all"),
        ("synleaf__allspecies", "SynLeaF 2026", "published", "all"),
        ("musl__human", "MuSL 2026 · human only", "published", "human"),
        ("synleaf__human", "SynLeaF 2026 · human only", "published", "human"),
    ]
    # Reproducible, clean 2026 adapters appear when their test score exists.
    specs += [
        (name, label, "published", scope)
        for name, label, scope in [
            ("sl_predict_2026__mae", "SL-Predict 2026 · MAE", "human"),
            ("pagan__clean_n2p", "PAGAN 2026 · genes-to-pairs", "two"),
            ("cilantro_sl", "Cilantro-SL 2026", "human"),
            ("ryan2026_context__slbtrain_full", "Ryan 2026 · paralog RF", "human"),
            ("gigcn__binary_go", "GiGCN 2026 · binary GO", "human"),
        ]
        if (RESULTS / f"{name}_test.json").exists()
    ]
    rows = [(name, label, kind, scope, read_result(name)) for name, label, kind, scope in specs]
    rows.sort(key=lambda row: row[4]["slb_score"], reverse=True)
    n = len(rows)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, max(6.1, .60 * n + 2.2)),
                                  gridspec_kw={"width_ratios": [1.1, 1], "wspace": .15})
    fig.subplots_adjust(left=.29, right=.97, top=.74, bottom=.14)
    fig.text(.05, .94, "Unseen gene families change the leaderboard", fontsize=21,
             fontweight="bold", color=INK)
    fig.text(.05, .89, "SLB-1.3 held-out test  ·  Fitness-balanced AUROC  ·  Measured negatives",
             fontsize=11.5, color=MUTED)
    fig.text(.05, .83, "Points show scores; horizontal whiskers are family-bootstrap 95% intervals.",
             fontsize=9.7, color=MUTED)
    yy = np.arange(n)[::-1]
    colors = {"ours": CORAL, "mechanistic": TEAL, "published": BLUE}
    for y, (name, label, kind, scope, r) in zip(yy, rows):
        score = r["slb_score"]
        ci = r["slb_score_ci95"]
        color = colors[kind]
        ax.plot([ci[0], ci[1]], [y, y], color=color, lw=2.2, zorder=2)
        ax.plot([ci[0], ci[0]], [y - .075, y + .075], color=color, lw=1.5)
        ax.plot([ci[1], ci[1]], [y - .075, y + .075], color=color, lw=1.5)
        marker = "D" if scope == "two" else "o"
        face = PAPER if scope == "human" else color
        ax.scatter(score, y, s=75, marker=marker, color=face, edgecolor=color,
                   linewidth=2, zorder=3)
        bx.scatter(r["species_scores"]["human"], y, s=74, marker=marker,
                   color=face, edgecolor=color, linewidth=2, zorder=3)
        ax.text(.72, y, f"{score:.3f}", va="center", color=color, fontsize=9.6,
                fontweight="bold")
        bx.text(.78, y, f"{r['species_scores']['human']:.3f}", va="center", color=color,
                fontsize=9.6, fontweight="bold")
    for panel, title, xmax in [(ax, "A  ·  SLB: mean of 3 species", .75),
                               (bx, "B  ·  Human component", .81)]:
        panel.axvline(.5, color=MUTED, lw=1.2, ls=(0, (3, 4)))
        panel.grid(axis="x", color=GRID, linewidth=.75)
        panel.set_axisbelow(True)
        panel.set_xlim(.44, xmax)
        panel.set_ylim(-.65, n - .35)
        panel.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=18)
        panel.spines["left"].set_visible(False)
        panel.spines["bottom"].set_visible(False)
        panel.tick_params(axis="y", length=0)
        panel.tick_params(axis="x", length=0)
    ax.set_yticks(yy, [label for _, label, _, _, _ in rows], fontsize=10)
    bx.set_yticks(yy, [""] * n)
    ax.set_xticks([.5, .55, .6, .65, .7])
    bx.set_xticks([.5, .55, .6, .65, .7, .75])
    fig.text(.05, .04,
             "Open circles: human only. Diamond: human + budding yeast. Unscored species tie at 0.500. "
             "Intervals describe uncertainty in the three-species score.", color=MUTED, fontsize=9)
    save(fig, "02_slb_held_out_test")


def figure_3() -> None:
    """Where the signal comes from across the three headline species."""
    specs = [
        ("slp_fusion__loss", "SLp Fusion"),
        ("ontotype", "Ontotype"),
        ("go_ppi_gbm", "GO/PPI GBM"),
        ("dekegel2021__allspecies", "De Kegel 2021"),
        ("musl__allspecies", "MuSL 2026"),
        ("synleaf__allspecies", "SynLeaF 2026"),
    ]
    specs += [(name, label) for name, label in [
        ("slxgo2026__allspecies", "SLxGO 2026"),
        ("pagan__allspecies", "PAGAN 2026"),
        ("mckg_sl__allspecies", "MCKG-SL 2026"),
    ] if (RESULTS / f"{name}_test.json").exists()]
    data = np.array([[read_result(name)["species_scores"][sp]
                      for sp in ("human", "scer", "spom")] for name, _ in specs])
    labels = [label for _, label in specs]
    fig, ax = plt.subplots(figsize=(8.7, max(5.1, .54 * len(labels) + 1.7)))
    fig.subplots_adjust(left=.29, right=.83, top=.78, bottom=.15)
    fig.text(.07, .94, "The signal is uneven across species", fontsize=20,
             fontweight="bold", color=INK)
    fig.text(.07, .875, "SLB-1.3 test  ·  Fitness-balanced AUROC  ·  Both gene families unseen",
             fontsize=11, color=MUTED)
    cmap = LinearSegmentedColormap.from_list("slb", ["#eef1ed", "#b9deda", "#54aca9", "#1c697e"])
    norm = Normalize(vmin=.5, vmax=.72)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(3), ["Human", "Budding yeast", "Fission yeast"], fontsize=10.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=10.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=11)
    for i in range(len(labels)):
        for j in range(3):
            val = data[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12,
                    fontweight="bold", color=PAPER if val >= .65 else INK)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=PAPER, linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=.045, pad=.08)
    bar.set_ticks([.5, .55, .6, .65, .7])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=9)
    fig.text(.07, .055, "Each cell is scored within screen and context after balancing single-gene fitness.",
             fontsize=9, color=MUTED)
    save(fig, "03_species_signal")


if __name__ == "__main__":
    theme()
    figure_1()
    figure_2()
    figure_3()
