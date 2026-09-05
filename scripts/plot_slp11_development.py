"""Plot saved development evidence; does not load molecular outcomes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]/"results/slp11-transition"
    runs = [json.loads((root/f"human-normalized-fusion-response32-exposure-seed{seed}-v1/report.json").read_text())
            for seed in (731, 732, 733)]
    audit = json.loads((root/"human-normalized-candidate-audit-v1/report.json").read_text())
    contexts = list(runs[0]["results"])
    colors = ["#106b78", "#55a2a6", "#a5d3c8"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 6.4))
    fig.subplots_adjust(top=.72, bottom=.27, left=.08, right=.97, wspace=.32)
    fig.suptitle("SLp-1.1: human molecular development", x=.08, ha="left", y=.97,
                 fontsize=19, fontweight="bold")
    fig.text(.08, .85, "Control-normalized Replogle CRISPRi | held intervention genes\n"
             "Protein + functional features, measured basal state, response-query decoder", fontsize=10, color="#44525a")
    for i, (run, color) in enumerate(zip(runs, colors)):
        x = np.arange(2)+(i-1)*.12
        axes[0].scatter(x, [run["results"][c]["world_delta_vs_ridge"] for c in contexts],
                        color=color, s=65, zorder=4, label=f"Seed {731+i}")
        axes[1].scatter(x, [run["results"][c]["world"]["gene_macro_profile_centroid_adjusted_pearson_mean"] for c in contexts],
                        color=color, s=65, zorder=4)
    for j, context in enumerate(contexts):
        boot = audit["contexts"][context]["bootstrap"]
        nll = boot["comparisons"]["ridge"]["deltaNllBaselineMinusWorld"]
        corr = boot["worldAdjustedPearson"]
        for ax, result in zip(axes, (nll, corr)):
            point = result["estimate"]
            ax.errorbar(j-.12, point, yerr=[[point-result["ci95Low"]], [result["ci95High"]-point]],
                        color=colors[0], capsize=4, linewidth=1.2, zorder=3)
        axes[1].scatter(j+.28, boot["ridgeAdjustedPearson"]["estimate"], color="#303940", marker="D", s=45,
                        label="Feature ridge" if j == 0 else None, zorder=4)
    axes[0].axhline(0, color="#8a979b", linewidth=1)
    axes[0].axhline(.02, color="#b87637", linestyle="--", linewidth=1.3, label="Development threshold")
    axes[0].set(ylabel="NLL improvement over ridge (nats / target)", ylim=(-.004, .038), title="Probability estimates improve")
    axes[1].set(ylabel="Centroid-adjusted profile Pearson", ylim=(.16, .33), title="Specific-response advantage is uncertain")
    for ax in axes:
        ax.set_xticks([0, 1], ["K562", "RPE1"])
        ax.set_xlim(-.4, 1.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.title.set_fontsize(11)
    handles, labels = axes[0].get_legend_handles_labels()
    other_handles, other_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles+other_handles, labels+other_labels, loc="upper left",
               bbox_to_anchor=(.075, .82), ncol=5, fontsize=8, frameon=False)
    fig.text(.08, .055, "95% intervals: 1,000 gene-bootstrap resamples for seed 731; dots: three training seeds.\n"
             "Adaptive development evidence. Original protected holdouts and SL benchmarks remain unopened.\n"
             "RPE1 does not meet the fixed 0.02-nat development rule. These results do not establish SOTA.",
             fontsize=9, color="#44525a", linespacing=1.6)
    output = root/"figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output/"human-development-v2.png", dpi=180, facecolor="white")
    fig.savefig(output/"human-development-v2.pdf", facecolor="white")
    plt.close(fig)
    print(output/"human-development-v2.png")


if __name__ == "__main__":
    main()
