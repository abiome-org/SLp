"""Render the frozen paired-endpoint development comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/slp11-transition/frangieh-paired-state-vs-static-scoring-v1/report.json"
EXPECTED = "d0c577e093198e9060a582cc5852b0db61246daa5772ae0c1e8451addc584b90"


def main():
    if hashlib.sha256(REPORT.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("frozen report changed")
    report = json.loads(REPORT.read_text())
    labels, points, intervals, world_r, ridge_r = [], [], [], [], []
    for context in ("Co-culture", "Control", "IFNγ"):
        for head in ("rna", "protein"):
            item = report["contexts"][context]["heads"][head]
            labels.append(context + " / " + ("RNA" if head == "rna" else "protein"))
            points.append(100 * item["gates"]["comparisons"]["physical1156"]["fractional_raw_mse_improvement"])
            ci = item["paired_gene_mse_bootstrap"]["physical1156"]
            intervals.append([100 * ci["fractional_raw_mse_improvement_ci025"],
                              100 * ci["fractional_raw_mse_improvement_ci975"]])
            world_r.append(item["world"]["query_centroid_adjusted_profile_pearson"])
            value = item["baselines"]["physical1156"]["query_centroid_adjusted_profile_pearson"]
            ridge_r.append(np.nan if value is None else value)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), sharey=True,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
        y = np.arange(len(labels))
        points, intervals = np.asarray(points), np.asarray(intervals)
        axes[0].hlines(y, intervals[:, 0], intervals[:, 1], color="#476780", linewidth=2)
        axes[0].scatter(points, y, color="#173b53", s=40, zorder=3)
        axes[0].axvline(0, color="#8b9399", linewidth=1)
        axes[0].axvline(1, color="#a26420", linestyle="--", linewidth=1)
        axes[0].set_yticks(y, labels)
        axes[0].set_xlabel("MSE improvement over physical ridge (%)\npositive values favor the model")
        axes[0].set_title("Prediction error · paired 95% bootstrap intervals", loc="left", fontsize=10)
        axes[1].scatter(world_r, y - .11, color="#173b53", marker="o", label="Paired state model", s=40)
        axes[1].scatter(ridge_r, y + .11, color="#c18b43", marker="s", label="Physical ridge", s=35)
        axes[1].axvline(0, color="#8b9399", linewidth=1)
        axes[1].axvline(.10, color="#a26420", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Intervention-specific profile correlation\nseparate prediction/truth centroids removed")
        axes[1].set_title("Landscape correlation · point estimates", loc="left", fontsize=10)
        axes[1].set_xlim(-.08, .15)
        fig.legend(*axes[1].get_legend_handles_labels(), loc="upper right",
                   bbox_to_anchor=(.985, .90), ncol=2, frameon=False, fontsize=9)
        for ax in axes:
            ax.set_ylim(5.65, -.7)
            ax.grid(axis="y", color="#e6e9ec", linewidth=.7)
        fig.suptitle("Paired RNA/protein pilot: 0 of 6 strata pass", x=.02, ha="left", fontsize=16, weight="bold")
        fig.text(.02, .91, "Frangieh 2021 · unseen intervention genes within three measured environments", color="#52616b")
        fig.text(.02, .025, "Adaptive development; 43 validation genes per environment; one training seed; 1,000 paired gene bootstrap draws.\n"
                 "Dashed lines: minimum +1% MSE and r = 0.10. Missing ridge points are undefined mean-limit correlations.",
                 fontsize=9, color="#52616b")
        fig.subplots_adjust(left=.19, right=.98, top=.80, bottom=.23, wspace=.14)
        output = ROOT / "results/slp11-transition/figures"
        output.mkdir(exist_ok=True)
        for extension in ("png", "pdf"):
            path = output / f"frangieh-paired-pilot-v2.{extension}"
            if path.exists():
                raise FileExistsError(path)
            fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        (output / "frangieh-paired-pilot-v2.json").write_text(json.dumps({
            "report_sha256": EXPECTED, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "labels": labels, "mse_improvement_percent": points.tolist(),
            "mse_improvement_95ci": intervals.tolist()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
