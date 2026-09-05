"""Plot saved HepG2 diagnostic reports without reading molecular outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    names = ["zero_control", "equal_source_fitting_centroid",
             "control_nearest_source_physical_ridge", "equal_source_average_physical_ridge",
             "same_gene_source_response_mean", "world"]
    labels = ["Zero control", "Average source response", "Nearest-source ridge",
              "Average source ridge", "Same-gene source response", "SLp world"]
    colors = ["#bec8cc"] * 5 + ["#227f87"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), sharey=True)
    fig.subplots_adjust(left=.24, right=.97, top=.81, bottom=.25, hspace=.43, wspace=.24)
    fig.suptitle("SLp-1.1: frozen HepG2 transfer test", x=.055, ha="left", y=.97,
                 fontsize=18, fontweight="bold")
    pending = "bootstrapStatus" in report
    fig.text(.055, .90, "Human CRISPRi • 6,789 measured RNA queries • predictions frozen before outcomes\n"
             "Joint transfer across cell context, study and control population", fontsize=10, color="#46565c")
    for row, (stratum, title) in enumerate([
        ("seen", "1,665 genes with source training responses"),
        ("unseen", "725 genes absent from source training")
    ]):
        items = report["primary"][stratum]
        for col, (metric, xlabel) in enumerate([
            ("primaryGeneAveragedProfileMse", "Mean squared error  ↓"),
            ("primaryIndependentlyCenteredGeneMacroProfilePearson", "Perturbation-specific profile Pearson  ↑")
        ]):
            ax = axes[row, col]
            for index, name in enumerate(names):
                value = items[name][metric]
                if value is None:
                    ax.text(.007, index, "Undefined (constant forecast)", va="center", fontsize=8, color="#738087")
                    continue
                ax.barh(index, value, height=.62, color=colors[index])
                ax.text(value + (.0006 if col == 0 else .003), index, f"{value:.4f}" if col == 0 else f"{value:.3f}",
                        va="center", fontsize=8)
                boot = items[name].get("bootstrap", {})
                key = ("geneAveragedProfileMsePercentiles025_50_975" if col == 0 else
                       "independentlyCenteredGeneMacroProfilePearsonPercentiles025_50_975")
                bounds = boot.get(key)
                if bounds is not None:
                    ax.plot([bounds[0], bounds[2]], [index, index], color="#263e45", linewidth=1.2)
            ax.set_yticks(np.arange(len(names)), labels)
            ax.set_ylim(len(names) - .4, -.6)
            ax.set_xlim(0, .084 if col == 0 else .35)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.grid(axis="x", alpha=.13)
            ax.set_axisbelow(True)
        axes[row, 0].set_title(title, loc="left", fontsize=11, pad=10)
    fig.text(.055, .095, "Fixed test: fail in both groups. Seen genes lose correlation to same-gene transfer;\n"
             "unseen genes have 1.04% higher MSE than average-source ridge. Equal weight per gene.\n" +
             ("Point estimates shown; 1,000-draw gene bootstrap intervals pending." if pending else
              "Intervals: 1,000 gene-block bootstrap resamples; descriptive, without changing the decision."),
             fontsize=9, color="#46565c", linespacing=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = args.output.with_suffix("." + extension)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180, facecolor="white")
    args.output.with_suffix(".json").write_text(json.dumps({
        "report": str(args.report.resolve()),
        "reportSha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "bootstrapPending": pending,
    }, indent=2) + "\n")
    print(args.output.with_suffix(".png").resolve())


if __name__ == "__main__":
    main()
