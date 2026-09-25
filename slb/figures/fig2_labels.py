"""Figure 2: labels are measured outcomes, admitted only if they reproduce, and cleaned of artefacts."""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style as S

D = S.OUT / "data"
POS, NEG, AMB = S.TEST, "#3d8fb0", "#c9cfd5"

NAMES = {"scer_emaps": "9 S. cer. E-MAPs", "dualcrispri2025": "Dual CRISPRi-seq 2025", "crisprtnseq2024": "CRISPRi-TnSeq 2024",
         "dualtnseq2025": "Dual Tn-seq 2025", "spidr2025": "SPIDR 2025", "chymera2020": "CHyMErA 2020"}
SP = {"babu2011": "ecol", "billmann2016": "dmel", "byrne2007": "cele", "chou2025": "human", "chymera2020": "human",
      "costanzo2016": "scer", "costanzo2021": "scer", "cote2016": "ecol", "crisprtnseq2024": "spne", "dede2020": "human",
      "dualcrispri2025": "spne", "dualtnseq2025": "spne", "fischer2015": "dmel", "flister2025": "human", "frost2012": "spom",
      "gagarinova2016": "ecol", "gier2020": "mmus", "han2017": "human", "harle2025": "human", "heigwer2023": "dmel",
      "horlbeck2018": "human", "horn2011": "dmel", "ito2021": "human", "koo2025": "bsub", "kumar2016": "ecol",
      "kuzmin2018": "scer", "kuzmin2020": "scer", "lehner2006": "cele", "parrish2021": "human", "roguev2013": "mmus",
      "ryan2012": "spom", "scer_emaps": "scer", "shen2017": "human", "spidr2025": "human", "thompson2021": "human",
      "wong2016": "human", "zhao2018": "human"}
SP_SHORT = {"human": "Hsa", "scer": "Sce", "spom": "Spo", "bsub": "Bsu", "cele": "Cel", "dmel": "Dme", "mmus": "Mmu",
            "ecol": "Eco", "spne": "Spn"}


def name(s: str) -> str:
    if s in NAMES:
        return NAMES[s]
    i = next(i for i, ch in enumerate(s) if ch.isdigit())
    return f"{s[:i].capitalize()} {s[i:]}"


def reason(r: str) -> str:
    r = r.lower()
    return ("contradicted by other screens" if "contradict" in r else "unverifiable" if "unverifiable" in r
            else "hit list, no measured negatives" if "hit list" in r else "implausible hit rate" if "56%" in r
            else "own replicates disagree")


def bands(fig, spec) -> None:
    d = pl.read_parquet(D / "label_bands.parquet")
    totals = json.loads((D / "label_bands_totals.json").read_text())
    screens = [("Costanzo 2016 SGA (ε)", "genetic interaction ε", (-0.9, 0.5)),
               ("Ryan 2012 E-MAP (S)", "E-MAP score S", (-14, 8)),
               ("Dede 2020 (zdLFC)", "zdLFC", (-9, 5))]
    sub = spec.subgridspec(len(screens), 1, hspace=1.15)
    for i, (key, xl, (lo, hi)) in enumerate(screens):
        ax = fig.add_subplot(sub[i])
        x = d.filter(pl.col("screen") == key)
        bins = np.linspace(lo, hi, 90)
        for lab, c, z in ((None, AMB, 1), (0, NEG, 2), (1, POS, 3)):
            v = x.filter(pl.col("label").is_null() if lab is None else pl.col("label") == lab)["score"].to_numpy()
            h, _ = np.histogram(np.clip(v, lo, hi), bins)
            ax.fill_between(bins[:-1], np.maximum(h, 0.8), 0.8, step="post", color=c, lw=0, alpha=0.95, zorder=z)
        ax.set_yscale("log")
        ax.set_ylim(0.8, None)
        ax.set_xlim(lo, hi)
        ax.tick_params(axis="y", labelsize=5.8)
        ax.set_yticks([1, 1e2, 1e4, 1e6] if x.height > 1e5 else [1, 10, 100, 1000])
        ax.set_xlabel(xl, labelpad=1.5, fontsize=6.4)
        n = totals[key]
        ax.text(0.0, 1.06, key.split(" (")[0], transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
                fontweight="semibold", color=S.INK)
        ax.text(1.0, 1.06, f"{n / 1e6:.1f} M measured pairs" if n > 1e5 else f"{n:,} measured pairs",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=S.MUTED)
        if i == 0:
            S.panel(ax, "a", x=-0.12, y=1.3)
            ax.legend(handles=[Patch(color=POS, label="SL (positive)"), Patch(color=NEG, label="neutral (negative)"),
                               Patch(color=AMB, label="in between: unscored")],
                      loc="lower center", bbox_to_anchor=(0.5, 1.28), ncol=3, fontsize=5.8, handlelength=0.9,
                      handleheight=0.7, borderpad=0.1, columnspacing=1.0)
        if i == 1:
            ax.set_ylabel("measured pairs", fontsize=6.4)


def audit(ax) -> None:
    r = pl.read_csv(D / "replication.csv").with_columns(
        pl.col("source").map_elements(name, return_dtype=pl.String).alias("label"),
        pl.max_horizontal("cross_hi", "within_hi", "vs_included").alias("best"))
    def fill(row):
        if row["best"] is None:
            v = [float(x) for x in re.findall(r"0\.\d+", row["reason"] or "")]
            if v and "contradict" in (row["reason"] or ""):
                row["cross_lo"], row["cross_hi"], row["best"] = min(v), max(v), max(v)
        return row
    r = pl.DataFrame([fill(x) for x in r.iter_rows(named=True)], infer_schema_length=None)
    inc = r.filter(pl.col("included")).sort("best", descending=True)
    exc = r.filter(~pl.col("included")).with_columns(
        pl.col("reason").map_elements(reason, return_dtype=pl.String).alias("why")).sort(["why", "best"], descending=[False, True])
    y, rows = 0, []
    for block in (inc, exc):
        prev = None
        for row in block.iter_rows(named=True):
            if not row["included"] and prev is not None and row["why"] != prev:
                y += 0.6
            prev = row.get("why")
            rows.append((y, row)); y += 1
        y += 1.4
    ax.axvspan(0.28, 0.65, color="#fbf0ec", zorder=0)
    ax.axvline(0.65, color=S.CORAL, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.axvline(0.5, color=S.MUTED, lw=0.5, ls=(0, (1, 2)), zorder=1)
    prev_why = None
    for yy, row in rows:
        inc_ = row["included"]
        col = S.NAVY if inc_ else "#9aa5b1"
        for lo, hi, mk, off in ((row["cross_lo"], row["cross_hi"], "D", -0.16), (row["within_lo"], row["within_hi"], "o", 0.16)):
            if lo is None:
                continue
            ax.plot([lo, hi], [yy + off, yy + off], color=col, lw=1.1, alpha=0.6, solid_capstyle="round")
            ax.plot([lo, hi], [yy + off] * 2, mk, ms=3.0 if mk == "o" else 2.7, mfc=col if inc_ else "white", mec=col, mew=0.8)
        if row["vs_included"] is not None:
            ax.plot(row["vs_included"], yy - 0.16, "D", ms=2.7, mfc=S.DEV, mec=S.DEV)
        sp = SP.get(row["source"], "")
        ax.text(0.262, yy, row["label"], ha="right", va="center", fontsize=6.0, color=S.INK if inc_ else S.MUTED)
        ax.text(0.266, yy, SP_SHORT.get(sp, ""), ha="left", va="center", fontsize=5.0, color=S.MUTED, style="italic",
                transform=ax.transData, clip_on=False)
        if not inc_ and row["why"] != prev_why:
            ax.text(1.005, yy - 0.45, row["why"], ha="right", va="bottom", fontsize=5.6, color=S.CORAL, style="italic")
            prev_why = row["why"]
        if row["best"] is None and not inc_:
            ax.text(0.33, yy, "no independent re-measurement", fontsize=5.3, color=S.MUTED, va="center", style="italic")
    ax.set_xlim(0.28, 1.005)
    ax.set_ylim(y - 1.6, -1.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.4", "0.5", "", "0.65", "", "0.8", "0.9", "1.0"])
    ax.set_xlabel("label recovery by a re-measurement (AUROC)")
    ax.text(0.28, -1.05, f"included · {inc.height} sources", fontsize=6.6, fontweight="semibold", color=S.NAVY, va="bottom")
    ax.text(0.28, inc.height + 0.25, f"excluded · {exc.height} sources", fontsize=6.6, fontweight="semibold",
            color=S.MUTED, va="bottom")
    ax.text(0.655, y - 1.9, "admission bar", fontsize=5.6, color=S.CORAL, va="bottom")
    ax.legend(handles=[Line2D([], [], marker="D", ls="", ms=3, color=S.NAVY, label="cross-study"),
                       Line2D([], [], marker="D", ls="", ms=3, color=S.DEV, label="vs included studies"),
                       Line2D([], [], marker="o", ls="", ms=3.2, color=S.NAVY, label="own replicates / alleles")],
              loc="lower center", bbox_to_anchor=(0.42, 1.012), ncol=3, fontsize=5.8, handletextpad=0.1, borderpad=0.1,
              columnspacing=0.9)


def linkage(ax) -> None:
    g = pl.read_csv(D / "linkage.csv")
    edges = np.loadtxt(D / "linkage_edges.txt")
    ax.axvspan(10, 200, color="#f4f5f7", zorder=0)
    ax.axhline(1, color=S.MUTED, lw=0.6, ls=(0, (3, 3)))
    style_ = {"scer_emaps": (S.CORAL, "9 E-MAPs (S. cer.)", "o"), "costanzo2016": (S.NAVY, "Costanzo 2016 SGA", "s"),
"ryan2012": ("#8a63b8", "Ryan 2012 (S. pom.)", "D")}
    for src, (c, lab, mk) in style_.items():
        x = g.filter(pl.col("source") == src)
        base = x.filter(pl.col("bin") == -1)["rate"][0]
        x = x.filter((pl.col("bin") >= 0) & (pl.col("bin") < len(edges) - 1) & (pl.col("n") >= 250)).sort("bin")
        mid = np.sqrt(np.maximum(edges[x["bin"].to_numpy()], 12e3) * edges[x["bin"].to_numpy() + 1]) / 1e3
        ax.plot(mid, x["rate"] / base, marker=mk, ms=3.0, color=c, lw=1.1, label=lab, mec="white", mew=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(20, 1600); ax.set_ylim(0.4, 25)
    ax.set_xticks([25, 50, 100, 200, 500, 1000]); ax.set_xticklabels(["25", "50", "100", "200", "500", "1000"])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("distance between the two genes on one chromosome (kb)")
    ax.set_ylabel("SL rate ÷ rate on different\nchromosomes", fontsize=6.4)
    ax.text(28, 17, "linked: unscored", fontsize=6.0, color=S.CORAL, fontweight="semibold")
    ax.axvline(200, color=S.CORAL, lw=0.8)
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.6, labelspacing=0.25, borderpad=0.1)


def funnel(ax) -> None:
    f = json.loads((D / "funnel.json").read_text())
    total, scored = f["examples"], f["scored_examples"]
    conflict, linked = f["conflicting_pairs_unscored"], f["linked_pairs_unscored"]
    amb = total - scored - conflict - linked
    sp = {}
    for r in f["splits"]:
        sp[r["split"]] = sp.get(r["split"], 0) + r["scored"]
    rows = [("measured pairs", [("scored", scored, S.NAVY), ("ambiguous band", amb, AMB),
                                ("conflicting sources", conflict, S.CORAL), ("linked", linked, "#b8432a")]),
            ("scored pairs", [("train", sp["train"], S.TRAIN), ("dev", sp["dev"], S.DEV), ("test", sp["test"], S.TEST),
                              ("one gene held out", sp["dev_semi"] + sp["test_semi"], "#d9dee3"),
                              ("dev × test, dropped", sp["drop"], "#eef0f2")])]
    for i, (lab, parts) in enumerate(rows):
        tot = sum(v for _, v, _ in parts)
        x = 0
        for k, v, c in parts:
            ax.barh(i, v / tot, left=x, height=0.55, color=c, lw=0)
            if v / tot > 0.07:
                ax.text(x + v / tot / 2, i, f"{k}\n{v / 1e6:.2f} M", ha="center", va="center", fontsize=5.5,
                        color="white" if c in (S.NAVY, S.TRAIN, S.CORAL, S.DEV) else S.INK, linespacing=1.15)
            elif k in ("dev", "test"):
                ax.annotate(f"{k} {v / 1e6:.2f} M", (x + v / tot / 2, i + 0.27), (x + v / tot / 2 + (0.02 if k == "test" else -0.02), i + 0.62),
                            ha="left" if k == "test" else "right", va="center", fontsize=5.5, color=c,
                            fontweight="semibold", arrowprops=dict(arrowstyle="-", color=c, lw=0.5))
            x += v / tot
        ax.text(0.0, i - 0.36, f"{lab} · {tot / 1e6:.1f} M", ha="left", va="bottom", fontsize=6.2, fontweight="semibold")


    ax.text(1.0, 0.36, f"incl. {conflict:,} conflicting · {linked:,} linked (unscored)", fontsize=5.5,
            color=S.MUTED, va="top", ha="right")
    ax.set_xlim(0, 1); ax.set_ylim(1.75, -0.7)
    ax.axis("off")


def main() -> None:
    S.use()
    fig = plt.figure(figsize=(S.WIDE, 6.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.98], height_ratios=[2.25, 1.75, 0.95], wspace=0.5, hspace=0.62,
                          left=0.085, right=0.985, top=0.955, bottom=0.03)
    bands(fig, gs[0, 0])
    axl = fig.add_subplot(gs[1, 0]); linkage(axl); S.panel(axl, "c", x=-0.12)
    axf = fig.add_subplot(gs[2, 0]); funnel(axf); S.panel(axf, "d", x=-0.12, y=1.05)
    axa = fig.add_subplot(gs[:, 1]); audit(axa); S.panel(axa, "b", x=-0.32, y=1.0)
    S.save(fig, "fig2_labels", dpi=320)


if __name__ == "__main__":
    main()
