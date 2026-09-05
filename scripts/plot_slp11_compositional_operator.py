"""Render the fixed molecular composition experiment after scoring is complete."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    names = ["additive", "mean_residual", "weighted_additive", "state_ridge", "endpoint_ensemble", "observed_operator_ensemble", "autonomous_ensemble"]
    labels = ["Observed additive singles", "+ Mean residual", "Weighted additive", "+ State ridge", "+ Endpoint attention", "+ Observed-state operator", "Autonomous rollout (secondary)"]
    best = report["bestBaseline"]
    denominator = report["metrics"][best]["mse"]
    values = [report["metrics"][x]["mse"] / denominator for x in names]
    colors = ["#a8b4bf"] * 4 + ["#547b9a", "#16857b", "#936bad"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.7, 1]})
    y = np.arange(len(names))
    axes[0].barh(y, values, color=colors, height=0.64)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].axvline(1, color="#34424d", linestyle="--", linewidth=1)
    axes[0].axvline(0.95, color="#16857b", linestyle=":", linewidth=1)
    axes[0].set_xlabel("MSE / best fixed baseline MSE (lower is better)")
    axes[0].set_title("Held-combination molecular response")
    for i, value in enumerate(values):
        axes[0].text(value + 0.015, i, f"{value:.3f}", va="center", fontsize=9)
    axes[0].set_xlim(0, max(values) * 1.15)
    selected = ["state_ridge", "endpoint_ensemble", "observed_operator_ensemble", "autonomous_ensemble"]
    for i, name in enumerate(selected):
        points = [report["folds"][str(f)][name]["centeredNonadditivePearson"] for f in range(3)]
        axes[1].scatter([i - .09, i, i + .09], points, color=colors[i + 3], s=35, alpha=0.7)
        axes[1].scatter(i, report["metrics"][name]["centeredNonadditivePearson"], color=colors[i + 3], marker="D", s=80, edgecolor="white")
    axes[1].set_xticks(range(4), ["State\nridge", "Endpoint\nattention", "Observed-state\noperator", "Autonomous\n(secondary)"], fontsize=9)
    axes[1].axhline(0, color="#8795a1", linewidth=1)
    axes[1].set_ylabel("Centered nonadditive Pearson (higher is better)")
    axes[1].set_title("Beyond additive and shared response")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SLp-1-inspired composition test | Norman K562 CRISPRa", fontsize=15, fontweight="bold")
    fig.text(.5, .01, "59 held combinations · 3 folds · 3 neural seeds · observed constituent singles available\nPoints: individual folds. Diamonds: pooled score. Known-gene interpolation; no SL or temporal-dynamics claim.", ha="center", fontsize=9, color="#435361")
    fig.tight_layout(rect=[0, .09, 1, .94])
    fig.savefig(args.run / "comparison.png", dpi=160, bbox_inches="tight")
    fig.savefig(args.run / "comparison.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
