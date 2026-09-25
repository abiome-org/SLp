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
        "svg.hashsalt": "slb1.3",
        "pdf.fonttype": 42,
    })


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "png", "pdf"):
        path = OUT / f"{stem}.{suffix}"
        metadata = {"Date": None} if suffix == "svg" else (
            {"CreationDate": None} if suffix == "pdf" else None)
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.22,
                    metadata=metadata)
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
        ("dev_rank_ensemble__loss", "Dev Rank Ensemble", "ensemble", "all"),
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
    colors = {"ensemble": CORAL, "mechanistic": TEAL, "published": BLUE}
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
        ("dev_rank_ensemble__loss", "Dev Rank Ensemble"),
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


def figure_4() -> None:
    """Exploratory cell-line ancestry readout of frozen human predictions."""
    path = ROOT / "reference/slb1.3_ancestry_audit.json"
    if not path.exists():
        return
    audit = json.loads(path.read_text())
    records = {m["name"]: m for m in audit["models"]}
    selected = [
        ("SL-Predict 2026 (MAE branch)", "SL-Predict 2026 · MAE"),
        ("Ryan 2026 (full clean refit)", "Ryan 2026 · paralog RF"),
        ("Dev Rank Ensemble (loss)", "Dev Rank Ensemble"),
        ("GO/PPI GBM", "GO/PPI GBM"),
        ("Ontotype", "Ontotype"),
        ("SynLeaF (all-species)", "SynLeaF 2026"),
    ]
    groups = [("AFR", PURPLE), ("EAS", TEAL), ("EUR", CORAL)]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 6.8), sharey=True,
                             gridspec_kw={"wspace": .08})
    fig.subplots_adjust(left=.235, right=.965, top=.76, bottom=.20)
    fig.text(.04, .945, "Held-out human SL screens by cell-line ancestry",
             color=INK, fontsize=20, fontweight="bold")
    fig.text(.04, .885, "Frozen SLB-1.3 test predictions  ·  Fitness-balanced AUROC within cell line and screen",
             color=MUTED, fontsize=11)
    fig.text(.04, .825, "Points show scores; whiskers are 95% gene-family bootstrap intervals.",
             color=MUTED, fontsize=10)
    ys = np.arange(len(selected))[::-1]
    for ax, (group, color) in zip(axes, groups):
        s = audit["support"][group]
        ax.set_title(f"{group}  ·  {s['cell_lines']} lines, {s['sl']} SL pairs",
                     loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=20)
        ax.set_xlim(.35, 1.07)
        ax.set_ylim(-.5, len(selected) - .5)
        ax.set_xticks([.4, .5, .6, .7, .8, .9])
        ax.grid(axis="x", color=GRID, linewidth=.8, zorder=0)
        ax.axvline(.5, ls="--", lw=1.1, color=MUTED, zorder=1)
        ax.tick_params(axis="y", length=0, pad=11)
        ax.tick_params(axis="x", length=0, pad=7)
        for (name, _), y in zip(selected, ys):
            rec = records[name]["by_group"][group]
            score, (lo, hi) = rec["score"], rec["ci95"]
            ax.plot([lo, hi], [y, y], color=color, lw=2.6, solid_capstyle="round", zorder=2)
            ax.plot([lo, lo], [y - .10, y + .10], color=color, lw=1.5, zorder=2)
            ax.plot([hi, hi], [y - .10, y + .10], color=color, lw=1.5, zorder=2)
            ax.scatter([score], [y], s=66, color=color, edgecolors=PAPER, linewidths=1.5, zorder=3)
            ax.text(1.03, y, f"{score:.3f}", ha="right", va="center",
                    color=color, fontsize=9.5, fontweight="bold")
    axes[0].set_yticks(ys, [label for _, label in selected], fontsize=10.5)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.text(.04, .105, "AFR has 3 cell lines and 32 positive pairs; groups use different cancer types, screens and pair panels.",
             color=MUTED, fontsize=9.5)
    fig.text(.04, .065, "Genotype estimates: Dutil et al. (2019) via Cellosaurus; one EAS line is self-reported. Cell lines, not patients.",
             color=MUTED, fontsize=9.5)
    save(fig, "04_human_ancestry")


def _ancestry_benchmark() -> dict:
    return json.loads((ROOT / "reference/slb1.3_ancestry_benchmark.json").read_text())


def _block_label(block: dict) -> str:
    site = block["cancer_site"].replace("_", " ").title()
    source = "Flister" if block["sources"] == "flister2025" else "Harle"
    return f"{site} · {source}"


def figure_5() -> None:
    """Matched support exposes the donor count behind ancestry point estimates."""
    if not (ROOT / "reference/slb1.3_ancestry_benchmark.json").exists():
        return
    result = _ancestry_benchmark()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2), gridspec_kw={"wspace": .20})
    fig.subplots_adjust(left=.12, right=.97, top=.73, bottom=.15)
    fig.text(.055, .95, "One African-ancestry donor per matched block",
             fontsize=22, fontweight="bold", color=INK)
    fig.text(.055, .89, "SLB-ANC-1.0  ·  Same cancer site, screen and gene pair  ·  Gene-family-held-out test",
             fontsize=11, color=MUTED)
    fig.text(.055, .84, "Bars count evaluable cell lines. Dashed lines mark the five-line minimum; pair-count and SL-count gates are additional.",
             fontsize=10, color=MUTED)
    for ax, target, color in zip(axes, ("AFR", "EAS"), (PURPLE, TEAL)):
        blocks = [b for b in result["panel_support"][target]["blocks"] if b["scorable"]]
        y = np.arange(len(blocks))[::-1]
        for i, (b, yi) in enumerate(zip(blocks, y)):
            for group, offset, c in ((target, .17, color), ("EUR", -.17, "#a6b7bf")):
                n = b["evaluable_lines"][group]
                positive = sum(r["sl"] for r in b["lines"] if r["group"] == group)
                ax.barh(yi + offset, n, height=.27, color=c, edgecolor=PAPER, linewidth=1, zorder=3)
                ax.text(n + .13, yi + offset, f"{n} {'line' if n == 1 else 'lines'} · {positive} SL", va="center",
                        color=INK if group == target else MUTED, fontsize=9.2,
                        fontweight="bold" if group == target else "normal")
        ax.axvline(5, color=CORAL, ls=(0, (3, 3)), lw=1.3, zorder=1)
        ax.text(5.08, len(blocks) - .36, "gate: 5", color=CORAL, fontsize=9, va="top")
        ax.set_yticks(y, [_block_label(b) for b in blocks], fontsize=10.5)
        ax.set_xlim(0, 11.7)
        ax.set_ylim(-.65, len(blocks) - .25)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.grid(axis="x", color=GRID, linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"{target} versus EUR  ·  {len(blocks)} scorable blocks",
                     loc="left", fontsize=12, fontweight="bold", pad=17)
        ax.set_xlabel("Evaluable cell lines", fontsize=10.5, labelpad=10)
    legend = [Line2D([0], [0], color=PURPLE, lw=7, label="AFR"),
              Line2D([0], [0], color=TEAL, lw=7, label="EAS"),
              Line2D([0], [0], color="#a6b7bf", lw=7, label="EUR comparator")]
    fig.legend(handles=legend, ncol=3, loc="upper right", bbox_to_anchor=(.97, .81),
               frameon=False, fontsize=9.5)
    fig.text(.055, .055, "SL counts refer to retained matched test pairs. The two panels use different pair panels."
             " Cell lines are donor proxies, not independent gene-pair rows.", color=MUTED, fontsize=9.2)
    save(fig, "05_ancestry_support")


def figure_6() -> None:
    """Primary-model cell-line scores on matched panels, without false donor CIs."""
    if not (ROOT / "reference/slb1.3_ancestry_benchmark.json").exists():
        return
    result = _ancestry_benchmark()
    primary = next(m for m in result["models"] if m["name"] == result["protocol"]["primary_model"])
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.4), gridspec_kw={"wspace": .21})
    fig.subplots_adjust(left=.13, right=.97, top=.72, bottom=.17)
    fig.text(.055, .95, "Matched-panel scores change the ancestry contrast",
             fontsize=21.5, fontweight="bold", color=INK)
    fig.text(.055, .89, "Frozen Dev Rank Ensemble  ·  Fitness-balanced AUROC within each cell line and screen",
             fontsize=11.2, color=MUTED)
    fig.text(.055, .835, "Small dots are cell lines; diamonds average cell lines within each site × screen block.",
             fontsize=10.2, color=MUTED)
    for ax, target, color in zip(axes, ("AFR", "EAS"), (PURPLE, TEAL)):
        rec = primary["comparisons"][target]
        blocks = [b for b in rec["blocks"] if b["scorable"]]
        y = np.arange(len(blocks))[::-1]
        for b, yi in zip(blocks, y):
            for group, offset, c in ((target, .17, color), ("EUR", -.17, "#a6b7bf")):
                vals = [r for r in b["lines"] if r["group"] == group and r["auroc"] is not None]
                if not vals:
                    continue
                xs = np.array([r["auroc"] for r in vals])
                # Deterministic spread keeps coincident donors visible.
                jitter = np.linspace(-.045, .045, len(vals)) if len(vals) > 1 else np.zeros(1)
                ax.scatter(xs, yi + offset + jitter, s=37, color=c, alpha=.75,
                           edgecolors=PAPER, linewidths=.7, zorder=3)
                ax.scatter([xs.mean()], [yi + offset], s=102, marker="D", color=c,
                           edgecolors=INK if group == target else "#8299a5", linewidths=.85, zorder=4)
        ax.axvline(.5, color=MUTED, ls=(0, (3, 4)), lw=1.1, zorder=1)
        ax.set_xlim(.33, 1.035)
        ax.set_ylim(-.58, len(blocks) - .28)
        ax.set_yticks(y, [_block_label(b) for b in blocks], fontsize=10.5)
        ax.set_xticks([.4, .5, .6, .7, .8, .9, 1.0])
        ax.grid(axis="x", color=GRID, linewidth=.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Per-line balanced AUROC", fontsize=10.5, labelpad=10)
        ax.set_title(f"{target} − EUR  ·  point gap {rec['gap_auroc']:+.3f}",
                     loc="left", fontsize=12, fontweight="bold", pad=17)
    legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor=PURPLE,
                     markeredgecolor=PAPER, markersize=8, label="AFR line"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL,
                     markeredgecolor=PAPER, markersize=8, label="EAS line"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor="#a6b7bf",
                     markeredgecolor=PAPER, markersize=8, label="EUR line"),
              Line2D([0], [0], marker="D", color="none", markerfacecolor=INK,
                     markeredgecolor=INK, markersize=7, label="block mean")]
    fig.legend(handles=legend, ncol=4, loc="upper right", bbox_to_anchor=(.97, .805),
               frameon=False, fontsize=9)
    fig.text(.055, .065, "The matched AFR panel has RKO (19 SL) and NCI-H23 (4 SL): one line in each block."
             " Population-level intervals and fairness verdicts are withheld by the adequacy gate.",
             color=MUTED, fontsize=9.15)
    save(fig, "06_ancestry_line_scores")


if __name__ == "__main__":
    theme()
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
