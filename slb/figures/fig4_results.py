"""Figure 4: the held-out leaderboard, its per-species anatomy, dev-test agreement and human ancestry."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import style as S

D = S.OUT / "data"
ROOT = S.OUT.parent

PRETTY = {"lgbm": "LightGBM baseline", "fitness_lgbm": "Fitness-only LightGBM", "fitness": "Fitness sum",
          "paralog_identity": "Paralog identity", "codependency": "DepMap co-dependency", "random": "Random",
          "SL-Predict MAE vectors only (exploratory test ablation)": "SL-Predict · MAE vectors only (ablation)",
          "SL-Predict coessentiality only (exploratory test ablation)": "SL-Predict · co-essentiality only (ablation)",
          "Ryan 2026 (released external-GEMINI weights, leakage diagnostic)": "Ryan 2026 · released weights (leaky)",
          "SLxGO 2026 (GO-PCA branch, input provenance unresolved)": "SLxGO 2026 · GO-PCA (provenance unclear)",
          "SLp-1.1 (decoder)": "SLp-1.1 · SL decoder (leaky)", "SLp-1.1 (label-free)": "SLp-1.1 · label-free (leaky)"}
FAMILY_COLOR = {"ensemble": S.CORAL, "mechanistic": S.NAVY, "feature": S.DEV, "graph": "#8a63b8",
                "baseline": "#9aa5b1", "world model": S.TEST}
FAMILY_LABEL = {"ensemble": "dev-selected ensemble", "mechanistic": "ontology / mechanistic",
                "feature": "feature / foundation model", "graph": "graph / knowledge graph",
                "baseline": "baseline", "world model": "SLp-1.1 world model"}
SCORE = LinearSegmentedColormap.from_list(
    "score", [(0, "#b8432a"), (0.35, "#f1cbbd"), (0.5, "#ffffff"), (0.62, "#cfe8ea"), (0.8, "#4f9bb2"), (1, "#0e3a57")])


def families() -> dict[str, str]:
    bat = yaml.safe_load((ROOT / "battery.yaml").read_text())["models"]
    fam = {}
    for e in yaml.safe_load((ROOT / "leaderboard.yaml").read_text()):
        stem = Path(e["predictions"]).name.removesuffix("_test.parquet")
        key = stem.split("__")[0]
        fam[e["name"]] = ("world model" if key.startswith("slp11") else "feature" if "sl_predict" in e["predictions"]
                          else bat.get(key, {}).get("family", "baseline"))
    return fam


def main() -> None:
    S.use()
    lb = pl.read_csv(D / "leaderboard.csv")
    fam = families()
    lb = lb.with_columns(pl.col("name").replace_strict(fam).alias("family"),
                         pl.col("name").replace(PRETTY).alias("label"),
                         ((pl.col("leaky") == "False") & (pl.col("ranked").cast(pl.String) != "false")).alias("ranked_"))
    ranked = lb.filter(pl.col("ranked_")).sort("slb", descending=True)
    other = lb.filter(~pl.col("ranked_")).sort("slb", descending=True)
    rows = pl.concat([ranked, other])
    n_r, n = ranked.height, rows.height

    fig = plt.figure(figsize=(S.WIDE, 8.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[5.05, 2.3], hspace=0.36, wspace=0.34,
                          left=0.035, right=0.985, top=0.965, bottom=0.07)
    top = gs[0, :].subgridspec(1, 3, width_ratios=[2.2, 1.55, 1.4], wspace=0.04)
    axn, axf, axh = fig.add_subplot(top[0]), fig.add_subplot(top[1]), fig.add_subplot(top[2])

    y = np.arange(n, dtype=float)
    y[n_r:] += 0.9  # gap before unranked entries
    for ax in (axn, axf, axh):
        ax.set_ylim(y[-1] + 0.7, -0.8)
    # names
    axn.axis("off")
    for yi, r in zip(y, rows.iter_rows(named=True)):
        ranked_ = r["ranked_"]
        rank = f"{int(yi) + 1}" if ranked_ else "–"
        axn.text(0.0, yi, rank, ha="left", va="center", color=S.MUTED, fontsize=6.4)
        axn.text(0.075, yi, r["label"], ha="left", va="center", fontsize=6.7,
                 color=S.INK if ranked_ else S.MUTED, fontweight="semibold" if yi < 3 else "normal")
        axn.add_patch(plt.Rectangle((0.965, yi - 0.28), 0.03, 0.56, color=FAMILY_COLOR[r["family"]],
                                    transform=axn.get_yaxis_transform(), clip_on=False))
    axn.text(0.075, y[n_r] - 1.15, "unranked: leaky, provenance unresolved, or exploratory", fontsize=6.2,
             color=S.MUTED, style="italic", va="center")
    S.panel(axn, "a", x=0.0, y=1.012)
    axn.texts[-1].set_ha("left")

    # forest
    axf.axvspan(0.44, 0.5, color="#f6f7f8", zorder=0)
    axf.axvline(0.5, color=S.MUTED, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for x in (0.55, 0.6, 0.65, 0.7):
        axf.axvline(x, color=S.GRID, lw=0.6, zorder=0)
    for yi, r in zip(y, rows.iter_rows(named=True)):
        c = FAMILY_COLOR[r["family"]] if r["ranked_"] else "#b3bcc5"
        axf.plot([r["lo"], r["hi"]], [yi, yi], color=c, lw=1.5, alpha=0.55, solid_capstyle="round", zorder=2)
        axf.plot(r["slb"], yi, "o", ms=4.2 if r["ranked_"] else 3.4, mfc=c if r["ranked_"] else "white",
                 mec=c, mew=1.0, zorder=3)
        axf.text(0.735, yi, f"{r['slb']:.3f}", ha="right", va="center", fontsize=6.3,
                 color=S.INK if r["ranked_"] else S.MUTED, fontfamily="IBM Plex Mono")
    axf.set_xlim(0.44, 0.74)
    axf.set_xticks([0.5, 0.55, 0.6, 0.65, 0.7])
    axf.tick_params(axis="y", left=False, labelleft=False)
    axf.spines["left"].set_visible(False)
    axf.set_xlabel("SLB score, held-out test (95% family-bootstrap CI)", labelpad=3)
    axf.text(0.497, -0.95, "chance", fontsize=6, color=S.MUTED, ha="right", va="center")

    # heatmap of components
    cols = [("sp_human", "Hsa"), ("sp_scer", "Sce"), ("sp_spom", "Spo"), ("sp_bsub", "Bsu"),
            ("sp_cele", "Cel"), (None, None), ("paralog", "paralog"), ("nonparalog", "other")]
    norm = TwoSlopeNorm(vmin=0.36, vcenter=0.5, vmax=0.74)
    for j, (c, lab) in enumerate(cols):
        if c is None:
            continue
        for yi, r in zip(y, rows.iter_rows(named=True)):
            v = r[c]
            tied = v is None or abs(v - 0.5) < 1e-9
            face = "#f3f4f5" if tied else SCORE(norm(v))
            axh.add_patch(plt.Rectangle((j - 0.46, yi - 0.43), 0.92, 0.86, color=face, lw=0, zorder=1))
            if tied:
                axh.text(j, yi, "tie", ha="center", va="center", fontsize=5.2, color="#a3acb5")
            else:
                dark = norm(v) > 0.78 or norm(v) < 0.12
                axh.text(j, yi, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.6,
                         color="white" if dark else S.INK, fontfamily="IBM Plex Mono")
    axh.set_xlim(-0.55, len(cols) - 0.45)
    axh.set_xticks([j for j, (c, _) in enumerate(cols) if c])
    axh.set_xticklabels([lab for c, lab in cols if c], fontsize=5.9)
    for t, (c, _) in zip(axh.get_xticklabels(), [x for x in cols if x[0]]):
        if c.startswith("sp_"):
            t.set_fontstyle("italic")
    axh.xaxis.tick_top()
    axh.tick_params(axis="x", length=0, pad=2)
    axh.tick_params(axis="y", left=False, labelleft=False)
    for s in axh.spines.values():
        s.set_visible(False)
    for x0, x1, lab in ((-0.45, 2.45, "headline"), (2.55, 4.45, "auxiliary"), (5.55, 7.45, "pair type")):
        axh.plot([x0, x1], [-2.0, -2.0], color=S.MUTED, lw=0.5, clip_on=False)
        axh.text((x0 + x1) / 2, -2.25, lab, ha="center", va="bottom", fontsize=6.0, color=S.MUTED)
    cax = axh.inset_axes([0.08, -0.058, 0.84, 0.018])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SCORE)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0.4, 0.5, 0.6, 0.7])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.8, length=1.5)
    cb.set_label("component AUROC (tie = species not scored)", fontsize=6.0, labelpad=1.5)

    handles = [Line2D([], [], marker="o", ls="", ms=4.2, mfc=c, mec=c, label=FAMILY_LABEL[k])
               for k, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, 0.405), ncol=3, fontsize=6.1,
               handletextpad=0.1, columnspacing=1.0, borderpad=0.2, labelspacing=0.35)

    # (b) dev vs test
    axb = fig.add_subplot(gs[1, 0])
    d = lb.filter(pl.col("dev").is_not_null())
    lo, hi = 0.46, 0.68
    axb.fill_between([lo, hi], [lo - 0.03, hi - 0.03], [lo + 0.03, hi + 0.03], color="#f1f3f5", lw=0, zorder=0)
    axb.plot([lo, hi], [lo, hi], color=S.MUTED, lw=0.7, ls=(0, (3, 3)), zorder=1)
    for r in d.iter_rows(named=True):
        c = FAMILY_COLOR[r["family"]]
        axb.plot(r["dev"], r["slb"], "o", ms=4.4 if r["ranked_"] else 3.6, mfc=c if r["ranked_"] else "white",
                 mec=c, mew=0.9, alpha=0.95, zorder=3)
    for name, dx, dy, ha in (("Dev Rank Ensemble (loss)", -0.004, 0.009, "right"), ("Ontotype", 0.005, -0.012, "left"),
                             ("De Kegel 2021 (all-species)", 0.006, -0.002, "left"),
                             ("Paralog identity", 0.004, -0.009, "left"), ("Random", 0.006, -0.004, "left")):
        r = d.filter(pl.col("label") == name)
        if r.height:
            axb.annotate(name.replace(" (all-species)", "").replace(" (loss)", ""), (r["dev"][0], r["slb"][0]),
                         (r["dev"][0] + dx, r["slb"][0] + dy), fontsize=5.9, color=S.INK, ha=ha, va="center")
    rho = spearmanr(d.filter(pl.col("ranked_"))["dev"], d.filter(pl.col("ranked_"))["slb"])[0]
    axb.text(0.03, 0.96, f"Spearman ρ = {rho:.2f}  (ranked models, n = {d.filter(pl.col('ranked_')).height})",
             transform=axb.transAxes, fontsize=6.4, va="top")
    axb.text(0.97, 0.05, "band: ±0.03, the dev noise scale", transform=axb.transAxes, fontsize=5.9,
             color=S.MUTED, ha="right")
    axb.set_xlim(lo, hi); axb.set_ylim(lo, hi)
    axb.set_xlabel("SLB score, dev"); axb.set_ylabel("SLB score, test")
    axb.set_aspect("equal")
    S.panel(axb, "b", x=-0.16)

    # (c) ancestry
    axc = fig.add_subplot(gs[1, 1])
    anc = {m["name"]: m for m in json.loads((D / "ancestry.json").read_text())}
    names = [r for r in ranked["name"].to_list() if r in anc][:7]
    groups = [("AFR", S.CORAL, "African (3 lines)"), ("EAS", S.TEST, "East Asian (7)"), ("EUR", S.BLUE, "European (38)")]
    for i, nm in enumerate(names):
        for k, (g, c, _) in enumerate(groups):
            b = anc[nm]["by_group"][g]
            yy = i + (k - 1) * 0.24
            axc.plot(b["ci95"], [yy, yy], color=c, lw=1.2, alpha=0.45)
            axc.plot(b["score"], yy, "o", ms=3.6, color=c, mec="white", mew=0.4)
    axc.axvline(0.5, color=S.MUTED, lw=0.7, ls=(0, (3, 3)))
    axc.set_yticks(range(len(names)))
    axc.set_yticklabels([PRETTY.get(nm, nm).replace(" (all-species)", "").replace(" (full clean refit)", " (full)")
                         for nm in names], fontsize=6.2)
    axc.set_ylim(len(names) - 0.5, -0.6)
    axc.tick_params(axis="y", length=0)
    axc.spines["left"].set_visible(False)
    axc.set_xlim(0.38, 0.9)
    axc.set_xlabel("human test AUROC within cell line and screen")
    axc.legend(handles=[Line2D([], [], marker="o", ls="-", lw=1, ms=3.6, color=c, label=lab) for _, c, lab in groups],
               loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=6.0, handlelength=1.3,
               columnspacing=0.9, borderpad=0.1)
    for x in (0.6, 0.7, 0.8):
        axc.axvline(x, color=S.GRID, lw=0.6, zorder=0)
    S.panel(axc, "c", x=-0.02)
    axc.texts[-1].set_position((-0.33, 1.02))
    S.save(fig, "fig4_results", dpi=320)


if __name__ == "__main__":
    main()
