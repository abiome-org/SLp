"""Plot frozen saved reports; never reopen molecular outcome snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]/"results/slp11-transition"
    paths = [root/name/"report.json" for name in (
        "human-normalized-fusion-response32-ensemble731-733-v1",
        "human-normalized-fusion-response32-ensemble731-733-molecular-confirmation-v1")]
    reports = [json.loads(path.read_text()) for path in paths]
    contexts = list(reports[0]["results"])
    colors = ["#227f87", "#c15c3c"]
    labels = ["Development", "Reserved-gene confirmation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    fig.subplots_adjust(left=.08, right=.97, bottom=.27, top=.73, wspace=.33)
    fig.suptitle("SLp-1.1: the gain survives, the advancement rule fails", x=.08, ha="left", y=.96,
                 fontsize=16, fontweight="bold")
    fig.text(.08,.87,"Frozen three-seed human CRISPRi ensemble • 7,226 RNA readouts\n"
             "Same-source held intervention genes; no model or calibration refit after confirmation", fontsize=10, color="#46565c")
    for split, report in enumerate(reports):
        x = np.arange(2)+(split-.5)*.17
        for i, context in enumerate(contexts):
            item = report["results"][context]
            nll = item["bootstrap"]["ridge"]["deltaNllBaselineMinusWorld"]
            point = nll["estimate"]
            axes[0].errorbar(x[i], point,
                yerr=[[point-nll["ci95Low"]],[nll["ci95High"]-point]], fmt="o",
                color=colors[split], capsize=4, markersize=7,
                label=labels[split] if i==0 else None)
            ensemble = item["pointMetrics"]["ensemble"]["geneMacroAdjustedPearson"]
            ridge = item["pointMetrics"]["ridge"]["geneMacroAdjustedPearson"]
            axes[1].plot([x[i],x[i]],[ridge,ensemble],color=colors[split],linewidth=2)
            axes[1].scatter(x[i],ensemble,color=colors[split],s=55,marker="o")
            axes[1].scatter(x[i],ridge,facecolors="white",edgecolors=colors[split],s=45,marker="D")
    axes[0].axhline(.02, color="#776340",linestyle="--",linewidth=1.2)
    axes[0].text(1.33,.0208,"Fixed point\nthreshold",fontsize=8,color="#776340",ha="right")
    axes[0].axhline(0,color="#c5cbcc",linewidth=1)
    axes[0].set(ylabel="Likelihood gain over ridge (nats / target)",ylim=(-.004,.055),title="Development advantage shrinks")
    axes[1].set(ylabel="Perturbation-specific profile Pearson",ylim=(.17,.295),title="Specific signal exceeds ridge")
    axes[1].text(.03,.97,"● Ensemble    ◇ Matched feature ridge",transform=axes[1].transAxes,va="top",fontsize=9)
    for ax in axes:
        ax.set_xticks([0,1],["K562","RPE1"])
        ax.set_xlim(-.4,1.4)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
        ax.title.set_fontsize(11)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="upper left",bbox_to_anchor=(.075,.81),ncol=2,frameon=False,fontsize=9)
    fig.text(.08,.11,"Bars: 95% intervals from 1,000 intervention-gene bootstrap resamples.\n"
             "Confirmation gains: 0.01465 K562 / 0.01034 RPE1; both below the fixed 0.02-nat rule.\n"
             "Development is adaptive. This holdout is now retired from selection. No SOTA or SL-performance claim.",
             fontsize=9,color="#46565c",linespacing=1.5)
    output = root/"figures"/"human-confirmation-v1"
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix(".png"),dpi=180,facecolor="white")
    fig.savefig(output.with_suffix(".pdf"),facecolor="white")
    manifest = {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output.with_suffix(".json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
