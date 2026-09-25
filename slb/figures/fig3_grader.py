"""Figure 3: the grader removes shortcuts, rejects cheats, and has a measurable noise floor."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

import style as S

D = S.OUT / "data"
COV = {"f_lo": "fitness, sicker gene", "f_hi": "fitness, healthier gene", "pan_lo": "pan-line fitness, sicker",
       "pan_hi": "pan-line fitness, healthier", "deg_lo": "row count, rarer gene", "deg_hi": "row count, commoner gene"}
PRETTY = {"lgbm": "LightGBM baseline", "fitness_lgbm": "Fitness-only LightGBM", "fitness": "Fitness sum",
          "paralog_identity": "Paralog identity", "codependency": "DepMap co-dependency", "random": "Random"}


def love(ax) -> None:
    b = pl.read_csv(D / "balance.csv")
    rows = []
    for sp in ("human", "scer", "spom"):
        for c in COV:
            if sp != "human" and c.startswith("pan"):
                continue
            r = b.filter((pl.col("species") == sp) & (pl.col("covariate") == c))
            if r.height:
                rows.append((sp, c, r["raw"][0], r["balanced"][0]))
        rows.append(None)
    ax.axvspan(1e-5, 0.03, color="#eef6f4", zorder=0)
    ax.axvline(0.03, color=S.DEV, lw=0.7, ls=(0, (3, 2)))
    y, yt, yl = 0, [], []
    for r in rows[:-1]:
        if r is None:
            y += 0.7
            continue
        sp, c, raw, bal = r
        col = S.SPECIES_COLOR[sp]
        raw_, bal_ = max(raw, 2e-4), max(bal, 2e-4)
        ax.annotate("", xy=(bal_, y), xytext=(raw_, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, alpha=0.55, mutation_scale=6, shrinkA=2, shrinkB=2))
        ax.plot(raw_, y, "o", ms=3.4, mfc="white", mec=col, mew=0.9)
        ax.plot(bal_, y, "o", ms=3.4, color=col)
        yt.append(y); yl.append(COV[c])
        y += 1
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.0)
    ax.set_ylim(y - 0.3, -0.8)
    ax.set_xscale("log"); ax.set_xlim(2.5e-4, 2.5)
    ax.set_xticks([1e-3, 1e-2, 0.1, 1]); ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("|standardised mean difference|, SL vs non-SL (dev)")
    # species brackets
    i = 0
    for sp, n in (("human", 6), ("scer", 4), ("spom", 4)):
        y0 = yt[i]; y1 = yt[i + n - 1]
        ax.text(2.35, (y0 + y1) / 2, S.SPECIES[sp], rotation=270, ha="center", va="center", fontsize=6.2,
                style="italic", color=S.SPECIES_COLOR[sp])
        ax.plot([1.95, 1.95], [y0 - 0.3, y1 + 0.3], color=S.SPECIES_COLOR[sp], lw=1.2)
        i += n
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.4, mfc="white", mec=S.INK, label="unweighted"),
                       Line2D([], [], marker="o", ls="", ms=3.4, color=S.INK, label="SLB weights")],
              loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=5.9, handletextpad=0.1, borderpad=0.1)
    ax.text(3.0e-4, y - 0.45, "< 0.03: balanced", fontsize=5.6, color=S.DEV, ha="left", va="bottom")


def probes(ax) -> None:
    p = json.loads((D / "probes.json").read_text())
    dv, rc = p["dev"], p["row_count"]
    lb = pl.read_csv(D / "leaderboard.csv")
    best = lb.filter(pl.col("name") == "Dev Rank Ensemble (loss)")["dev"][0]
    rows = [("exact labels", dv["label_oracle_accept"], None, "accept"),
            ("inverted labels", dv["inverse_oracle_reject"], None, "reject"),
            ("constant score", dv["constant_reject"], None, "reject"),
            ("per-stratum hit-rate prior", dv["stratum_hit_rate_prior_reject"], None, "reject"),
            ("random, 20 seeds", dv["random_mean"], dv["random_sd"], "reject"),
            ("fitness-only LightGBM", dv["fitness_only_probe"], None, "reject"),
            ("−(gene row counts), dev", rc["dev_new"], rc["dev_old"], "leak"),
            ("−(gene row counts), test", rc["test_new"], rc["test_old"], "leak")]
    ax.axvline(0.5, color=S.MUTED, lw=0.7, ls=(0, (3, 3)))
    ax.axvline(best, color=S.CORAL, lw=0.7, ls=(0, (1, 1.5)))
    ax.text(best + 0.012, -0.45, f"best model, dev {best:.3f}", fontsize=5.6, color=S.CORAL, va="center")
    for i, (lab, v, extra, kind) in enumerate(rows):
        if kind == "leak":
            ax.annotate("", xy=(v, i), xytext=(extra, i),
                        arrowprops=dict(arrowstyle="-|>", color=S.CORAL, lw=1.0, mutation_scale=7, shrinkA=3, shrinkB=3))
            ax.plot(extra, i, "o", ms=4.2, mfc="white", mec=S.CORAL, mew=1.0)
            ax.plot(v, i, "o", ms=4.2, color=S.DEV)
            ax.text(extra + 0.025, i, f"{extra:.3f} before", fontsize=5.6, color=S.CORAL, va="center", ha="left")
            ax.text(v - 0.025, i, f"now {v:.3f}", fontsize=5.6, color=S.DEV, va="center", ha="right")
        else:
            c = S.NAVY if kind == "accept" else S.DEV
            if extra:
                ax.plot([v - 2 * extra, v + 2 * extra], [i, i], color=c, lw=2.2, alpha=0.35, solid_capstyle="butt")
            ax.plot(v, i, "o", ms=4.2, color=c)
            ax.text(v + (-0.03 if v > 0.9 else -0.03 if v > 0.4 else 0.03), i, f"{v:.3f}", fontsize=5.6, color=c,
                    va="center", ha="left" if v < 0.4 else "right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.set_ylim(len(rows) - 0.35, -0.6)
    ax.set_xlim(-0.03, 1.03)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("SLB score of the probe")
    ax.axhspan(5.5, 7.65, color="#fbf0ec", zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.text(1.02, 7.52, "fixed leak: labels were readable from row counts", fontsize=5.6, color=S.CORAL,
            ha="right", va="bottom", style="italic")


def noise(ax) -> None:
    n = pl.read_csv(D / "noise.csv")
    lb = pl.read_csv(D / "leaderboard.csv").filter(pl.col("dev").is_not_null())
    bins = np.linspace(0.44, 0.56, 25)
    for split, c in (("test", S.TEST), ("dev", S.DEV)):
        v = n.filter(pl.col("split") == split)["slb"].to_numpy()
        ax.hist(v, bins=bins, color=c, alpha=0.55, lw=0, label=f"{split}: SD {v.std(ddof=1):.3f}")
        ax.hist(v, bins=bins, histtype="step", color=c, lw=0.9)
    ax.set_xlim(0.44, 0.67)
    ax.set_xlabel("SLB score")
    ax.set_ylabel("random per-gene scorings (of 60)")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.55)
    ranked = lb.filter(pl.col("leaky") == "False").sort("dev", descending=True)
    for i, r in enumerate(ranked.head(8).iter_rows(named=True)):
        ax.plot([r["dev"], r["dev"]], [top * 1.08, top * 1.28], color=S.NAVY, lw=0.9, alpha=0.8)
    ax.text(0.595, top * 1.34, "dev scores of the 8 best models", fontsize=5.8, color=S.NAVY, ha="center", va="bottom")
    sd = n.filter(pl.col("split") == "dev")["slb"].std()
    ax.annotate("", xy=(0.5 + 2 * sd, top * 0.95), xytext=(0.5 - 2 * sd, top * 0.95),
                arrowprops=dict(arrowstyle="<->", color=S.INK, lw=0.6, mutation_scale=5))
    ax.text(0.5, top * 0.99, "±2 SD", fontsize=5.8, ha="center", va="bottom")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=5.9, handlelength=1.0, borderpad=0.1)


def robustness(ax) -> None:
    r = pl.read_csv(D / "robustness.csv")
    order = r.filter(pl.col("split") == "official").sort("slb", descending=True)["model"].to_list()
    ax.axvline(0.5, color=S.MUTED, lw=0.7, ls=(0, (3, 3)))
    for i, m in enumerate(order):
        x = r.filter(pl.col("model") == m)
        alt = x.filter(pl.col("split") != "official")["slb"].to_numpy()
        off = x.filter(pl.col("split") == "official")["slb"][0]
        ax.plot([alt.min(), alt.max()], [i, i], color=S.TRAIN, lw=2.6, alpha=0.3, solid_capstyle="round")
        ax.plot(alt, np.full(len(alt), i), "o", ms=3.0, mfc="white", mec=S.TRAIN, mew=0.8)
        ax.plot(off, i, "D", ms=3.6, color=S.TEST)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([PRETTY[m] for m in order], fontsize=6.2)
    ax.set_ylim(len(order) - 0.5, -0.7)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.47, 0.58)
    ax.set_xlabel("SLB score, test")
    ax.legend(handles=[Line2D([], [], marker="D", ls="", ms=3.6, color=S.TEST, label="official split"),
                       Line2D([], [], marker="o", ls="", ms=3.0, mfc="white", mec=S.TRAIN, label="4 other family splits")],
              loc="lower right", fontsize=5.9, handletextpad=0.1, borderpad=0.2)


def main() -> None:
    S.use()
    fig = plt.figure(figsize=(S.WIDE, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.25, 1], wspace=0.62, hspace=0.42,
                          left=0.19, right=0.965, top=0.965, bottom=0.08)
    a = fig.add_subplot(gs[0, 0]); love(a); S.panel(a, "a", x=-0.52)
    b = fig.add_subplot(gs[0, 1]); probes(b); S.panel(b, "b", x=-0.62)
    c = fig.add_subplot(gs[1, 0]); noise(c); S.panel(c, "c", x=-0.52)
    d = fig.add_subplot(gs[1, 1]); robustness(d); S.panel(d, "d", x=-0.62)
    S.save(fig, "fig3_grader", dpi=320)


if __name__ == "__main__":
    main()
