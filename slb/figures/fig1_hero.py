"""Figure 1 (hero): the held-out homology universe of SLB, and four of the measured screens it is built from."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter

import style as S

D = S.OUT / "data"
ROOT = S.OUT.parent
ORDER = ["human", "mmus", "dmel", "cele", "bsub", "spom", "scer"]
BUCKETS = ["test", "dev", "train"]
EDGE = {"test": "#f2a93b", "dev": "#2fc4a9", "train": "#4d6f93"}
BG = "#000000"
WHITE = "#f4f1ea"
GREY = "#8d98a3"


def layout(nodes: pl.DataFrame) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Angle of every gene: species sectors (sized ~ n^0.6), then test | dev | train blocks, then families."""
    size = nodes.group_by("family").len().rename({"len": "fsize"})
    nodes = nodes.join(size, on="family")
    counts = {sp: nodes.filter(pl.col("species") == sp).height for sp in ORDER}
    w = {sp: max(counts[sp], 400) ** 0.6 for sp in ORDER}
    gap = np.deg2rad(2.2)
    total = 2 * np.pi - gap * len(ORDER)
    theta, sectors = {}, {}
    start = np.pi / 2 + np.deg2rad(8)
    for sp in ORDER:
        span = total * w[sp] / sum(w.values())
        g = nodes.filter(pl.col("species") == sp).with_columns(
            pl.col("bucket").replace_strict({b: i for i, b in enumerate(BUCKETS)}).alias("bi")
        ).sort(["bi", "fsize", "family", "gene"], descending=[False, True, False, False])
        ang = start - np.linspace(0, span, g.height, endpoint=False) - span / g.height / 2
        theta.update(zip(g["species"] + ":" + g["gene"], ang))
        sectors[sp] = (start, start - span)
        start -= span + gap
    return theta, sectors


def curves(e: pl.DataFrame, theta: dict[str, float], n: int = 24) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    a = np.array([theta[u] for u in e["u"]]); b = np.array([theta[v] for v in e["v"]])
    p0 = np.stack([np.cos(a), np.sin(a)], 1); p2 = np.stack([np.cos(b), np.sin(b)], 1)
    same = (e["u"].str.split(":").list.first() == e["v"].str.split(":").list.first()).to_numpy()
    mid = (p0 + p2) / 2
    ctrl = np.where(same[:, None], mid / np.maximum(np.linalg.norm(mid, axis=1, keepdims=True), 1e-9) * 0.72,
                    mid * 0.12)
    pts = (1 - t)[None] ** 2 * p0[:, None] + 2 * ((1 - t) * t)[None] * ctrl[:, None] + (t ** 2)[None] * p2[:, None]
    return pts * 0.985


def degree() -> dict[str, int]:
    parts = []
    for f in ("train", "dev", "dev_semi", "test_inputs", "test_semi_inputs"):
        x = pl.read_parquet(ROOT / "data/slb" / f"{f}.parquet", columns=["species", "gene_a", "gene_b"])
        parts += [x.select((pl.col("species") + ":" + pl.col(c)).alias("n")) for c in ("gene_a", "gene_b")]
    v = pl.concat(parts)["n"].value_counts()
    return dict(zip(v["n"], v["count"]))


def draw_map(ax, fig) -> None:
    nodes = pl.read_parquet(D / "hero_nodes.parquet")
    edges = pl.read_parquet(D / "hero_edges.parquet")
    theta, sectors = layout(nodes)
    deg = degree()
    for b, (lw, a, glow) in (("train", (0.2, 0.075, 0.0)), ("dev", (0.18, 0.15, 0.0)), ("test", (0.2, 0.2, 0.0))):
        seg = curves(edges.filter(pl.col("bucket") == b), theta)
        if glow:
            ax.add_collection(LineCollection(seg, colors=EDGE[b], linewidths=2.6, alpha=glow * 0.6, zorder=2))
            ax.add_collection(LineCollection(seg, colors=EDGE[b], linewidths=1.0, alpha=glow * 1.6, zorder=2))
        ax.add_collection(LineCollection(seg, colors=EDGE[b], linewidths=lw, alpha=a, zorder=3))
    # rim: bucket ticks and measured-pair histogram
    key = nodes.select((pl.col("species") + ":" + pl.col("gene")).alias("n"), "species", "bucket")
    ang = np.array([theta[k] for k in key["n"]])
    col = [EDGE[b] for b in key["bucket"]]
    r0, r1 = 1.0, 1.035
    ax.add_collection(LineCollection(np.stack([np.stack([r0 * np.cos(ang), r0 * np.sin(ang)], 1),
                                               np.stack([r1 * np.cos(ang), r1 * np.sin(ang)], 1)], 1),
                                     colors=col, linewidths=0.35, alpha=0.95, zorder=4))
    d = np.log10(np.array([deg.get(k, 1) for k in key["n"]]) + 1)
    h = 1.06 + 0.16 * d / d.max()
    ax.add_collection(LineCollection(np.stack([np.stack([1.06 * np.cos(ang), 1.06 * np.sin(ang)], 1),
                                               np.stack([h * np.cos(ang), h * np.sin(ang)], 1)], 1),
                                     colors=[S.SPECIES_COLOR[s] for s in key["species"]], linewidths=0.3,
                                     alpha=0.55, zorder=4))
    counts = nodes.group_by("species").len()
    for sp, (a0, a1) in sectors.items():
        am = (a0 + a1) / 2
        arc = np.linspace(a0, a1, 60)
        ax.plot(1.25 * np.cos(arc), 1.25 * np.sin(arc), color=S.SPECIES_COLOR[sp], lw=1.2, alpha=0.9, solid_capstyle="butt")
        n = counts.filter(pl.col("species") == sp)["len"][0]
        c, s_ = np.cos(am), np.sin(am)
        x0, y0 = 1.31 * c, 1.31 * s_
        ha = "left" if c > 0.15 else "right" if c < -0.15 else "center"
        va = "bottom" if s_ > 0.3 else "top" if s_ < -0.3 else "center"
        dy = {"bottom": 0.055, "top": 0.0, "center": 0.03}[va]
        ax.text(x0, y0 + dy, S.SPECIES[sp], ha=ha, va="center", color=WHITE, fontsize=7.4 if n > 400 else 6.6,
                style="italic")
        ax.text(x0, y0 + dy - 0.07, f"{n:,} genes", ha=ha, va="center", color=GREY, fontsize=5.5)
    ax.set_xlim(-1.62, 1.62); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.axis("off")


def screen(ax, sp: str, letter: str) -> None:
    z = np.load(D / "screens.npz")
    meta = json.loads((D / f"screen_{sp}.json").read_text())
    m, p = z[f"{sp}_meas"], z[f"{sp}_sl"]
    mn = np.log1p(m) / np.log1p(np.percentile(m[m > 0], 99.5))
    pn = np.log1p(gaussian_filter(p, 0.7) * 2) / np.log1p(np.percentile(p[p > 0], 99.7) * 2 if (p > 0).any() else 1)
    mn, pn = np.clip(mn, 0, 1), np.clip(pn, 0, 1)
    base = np.array([0.10, 0.32, 0.50])
    rgb = mn[..., None] ** 0.9 * base * 0.95
    rgb = rgb + (pn[..., None] ** 1.05) * np.array([1.0, 0.64, 0.22]) * 0.95
    hot = np.clip(pn - 0.75, 0, 1)[..., None] * np.array([0.0, 0.3, 0.55]) * 2
    rgb = np.clip(rgb + hot, 0, 1)
    ax.imshow(rgb, interpolation="bilinear", origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1d2733"); s.set_linewidth(0.6)
    ax.text(0.04, 0.96, f"({letter})", transform=ax.transAxes, color=WHITE, fontsize=7.4, fontweight="bold", va="top")
    ax.text(0.96, 0.96, meta["species"], transform=ax.transAxes, color=WHITE, fontsize=7.0, style="italic",
            ha="right", va="top")
    caption(ax, meta["subtitle"], meta["pairs"], meta["sl"])


def caption(ax, sub: str, pairs: int, sl: int, box: bool = False) -> None:
    pairs_s = f"{pairs / 1e6:.1f} M" if pairs >= 1e6 else f"{pairs / 1e3:.0f} k"
    kw = dict(bbox=dict(boxstyle="round,pad=0.25", fc="#000000", ec="none", alpha=0.72)) if box else {}
    ax.text(0.04, 0.04, f"{sub}\n{pairs_s} measured pairs · {sl:,} SL", transform=ax.transAxes, color="#c7ced5",
            fontsize=5.5, va="bottom", linespacing=1.55, **kw)
    ax.text(0.04, 0.04, f"\n{pairs_s} measured pairs · {sl:,} SL", transform=ax.transAxes, color="#f2c27a",
            fontsize=5.5, va="bottom", linespacing=1.55)


def human_panel(ax, letter: str) -> None:
    mat = np.load(D / "human_ctx.npy")
    lines = pl.read_csv(D / "human_ctx_lines.csv")
    meas = ~np.isnan(mat)
    rgb = np.zeros(mat.shape + (3,))
    rgb[meas] = np.array([0.05, 0.15, 0.24])
    rgb[mat == 0] = np.array([0.08, 0.24, 0.37])
    rgb[mat == 1] = np.array([1.0, 0.70, 0.30])
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#1d2733"); s.set_linewidth(0.6)
    groups = lines["ancestry_group"].to_list()
    for g, c in (("AFR", S.CORAL), ("EAS", S.TEST), ("EUR", S.BLUE)):
        idx = [i for i, x in enumerate(groups) if x == g]
        if idx:
            ax.plot([-8, -8], [min(idx) - 0.4, max(idx) + 0.4], color=c, lw=2.2, clip_on=False, solid_capstyle="butt")
    ax.set_xlim(-12, mat.shape[1] - 0.5)
    box = dict(boxstyle="round,pad=0.2", fc="#000000", ec="none", alpha=0.7)
    ax.text(0.04, 0.96, f"({letter})", transform=ax.transAxes, color=WHITE, fontsize=7.4, fontweight="bold", va="top", bbox=box)
    ax.text(0.96, 0.96, "H. sapiens", transform=ax.transAxes, color=WHITE, fontsize=7.0, style="italic", ha="right",
            va="top", bbox=box)
    meta = json.loads((D / "screen_human.json").read_text())
    ax.text(0.97, 0.20, "rows: 50 cell lines\n(AFR · EAS · EUR)\ncols: 1,400 most-\nscreened pairs", transform=ax.transAxes,
            color="#c7ced5", fontsize=5.0, ha="right", va="bottom", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.25", fc="#000000", ec="none", alpha=0.6))
    caption(ax, "8 paralog screens", meta["pairs"], meta["sl"], box=True)


def main() -> None:
    S.use()
    fig = plt.figure(figsize=(S.WIDE, 7.2), facecolor=BG)
    axm = fig.add_axes([-0.01, 0.275, 0.73, 0.715], facecolor=BG)
    draw_map(axm, fig)
    axm.text(0.045, 0.975, "(a)", transform=axm.transAxes, color=WHITE, fontsize=8.2, fontweight="bold", va="top")

    nodes = pl.read_parquet(D / "hero_nodes.parquet")
    fam = nodes.select("family", "bucket").unique()
    share = {b: fam.filter(pl.col("bucket") == b).height / fam.height for b in BUCKETS}
    axt = fig.add_axes([0.715, 0.285, 0.28, 0.69], facecolor=BG)
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.axis("off")
    axt.text(0, 0.97, "held-out gene families", color=WHITE, fontsize=8.4, fontweight="semibold", va="top")
    axt.text(0, 0.905, f"{nodes.height:,} genes · {fam.height:,} families\n7 scored species · 12 proteomes\nin the homology graph",
             color=GREY, fontsize=6.2, va="top", linespacing=1.5)
    y = 0.74
    for b in BUCKETS:
        axt.plot([0, 0.12], [y, y], color=EDGE[b] if b != "train" else "#7f9cbd", lw=2.4, solid_capstyle="butt")
        axt.text(0.17, y, b, color=EDGE[b] if b != "train" else "#9fb5cc", fontsize=7.4, fontweight="semibold", va="center")
        axt.text(0.46, y, f"{share[b]:.0%} of families", color=GREY, fontsize=6.2, va="center")
        y -= 0.058
    notes = [("curves", "homology between benchmark genes.\nOrthologs cross the circle; paralogs\nand overlapping ORFs loop at the rim.\nColour: the family's split."),
             ("inner ring", "every gene, coloured by its split"),
             ("outer ring", "measured pairs per gene (log)")]
    y = 0.53
    for head, body in notes:
        axt.text(0, y, head, color=WHITE, fontsize=6.5, fontweight="semibold", va="top")
        axt.text(0, y - 0.036, body, color=GREY, fontsize=6.0, va="top", linespacing=1.45)
        y -= 0.1 + 0.036 * body.count("\n")
    axt.text(0, y - 0.02, "No test gene has a paralog (≥ 30%\nidentity), an ortholog or an\noverlapping ORF in train,\nin any species.",
             color="#f2c27a", fontsize=6.4, va="top", linespacing=1.5, style="italic")

    for i, (sp, letter) in enumerate((("scer", "b"), ("spom", "c"), ("bsub", "d"), ("human", "e"))):
        ax = fig.add_axes([0.012 + i * 0.2475, 0.03, 0.232, 0.232 * S.WIDE / 7.2], facecolor=BG)
        human_panel(ax, letter) if sp == "human" else screen(ax, sp, letter)
    fig.text(0.5, 0.004, "(b–d) every measured pair of three screens, genes ordered by number of SL partners;  "
             "(e) human cell lines × most-screened gene pairs;  blue = measured, amber = SL call",
             color=GREY, fontsize=5.8, ha="center", va="bottom")
    S.save(fig, "fig1_hero", dpi=360, facecolor=BG)


if __name__ == "__main__":
    main()
