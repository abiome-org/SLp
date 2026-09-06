#!/usr/bin/env python3
"""Plot the completed MuSL CV3 readout comparison from saved score reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1 = ROOT / "results/slp11-transition/joint-world-compositional-musl-readout-v1/scores.json"
DEFAULT_V2 = ROOT / "results/slp11-transition/joint-world-compositional-musl-static-readout-v2/scores.json"
DEFAULT_SELECTED = ROOT / "results/slp11-transition/joint-world-training-selected-musl-readout-v1/score.json"
DEFAULT_OUTPUT = ROOT / "results/slp11-transition/joint-world-sl-readout-figure-v1"


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows(report: dict, arm: str, metric: str) -> np.ndarray:
    values = [float(row[metric]) for row in report["rows"] if row["arm"] == arm]
    if len(values) != 10:
        raise ValueError(f"expected 10 rows for {arm}, found {len(values)}")
    return np.asarray(values)


def _selected_rows(report: dict, metric: str) -> np.ndarray:
    values = [float(row[metric]) for row in report["rows"]]
    if len(values) != 10:
        raise ValueError(f"expected 10 selected rows, found {len(values)}")
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1)
    parser.add_argument("--v2", type=Path, default=DEFAULT_V2)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    v1, v2, selected = _read(args.v1), _read(args.v2), _read(args.selected)
    arms = [
        ("Retained baseline", _rows(v1, "baseline", "auroc"), _rows(v1, "baseline", "ap")),
        ("Descriptor static", _rows(v2, "static_control", "auroc"), _rows(v2, "static_control", "ap")),
        ("World augmented", _rows(v2, "world_augmented", "auroc"), _rows(v2, "world_augmented", "ap")),
        ("Locked selected", _selected_rows(selected, "auroc"), _selected_rows(selected, "averagePrecision")),
    ]
    colors = ["#718096", "#2B6CB0", "#2F855A", "#C05621"]
    markers = ["o", "s", "D", "*"]
    x = np.arange(len(arms), dtype=float)
    offsets = np.linspace(-0.10, 0.10, 10)

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), sharey=True)
    for axis, metric_index, title in zip(axes, (1, 2), ("AUROC", "Average precision (AP)")):
        for i, (label, auroc, ap) in enumerate(arms):
            values = (auroc, ap)[metric_index - 1]
            axis.scatter(x[i] + offsets, values, s=31, color=colors[i], alpha=0.62,
                         marker=markers[i], edgecolor="white", linewidth=0.45, zorder=2)
            macro = float(values.mean())
            axis.scatter(x[i], macro, s=150 if i == 3 else 105, color=colors[i],
                         marker=markers[i], edgecolor="#1A202C", linewidth=1.2, zorder=4)
            axis.text(x[i], macro + 0.008, f"{macro:.3f}", ha="center", va="bottom",
                      fontsize=9, fontweight="bold", color=colors[i])
        axis.set_title(title, pad=12)
        axis.set_xticks(x, [a[0].replace(" ", "\n", 1) for a in arms])
        axis.set_xlim(-0.55, 3.55)
        axis.set_ylim(0.70, 0.89)
        axis.grid(axis="y", color="#CBD5E0", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Held-pair score")
    axes[0].axhline(0.7895, color="#6B46C1", linestyle="--", linewidth=1.4, zorder=1)
    axes[0].text(3.48, 0.792, "MuSL published aggregate 0.7895\n(reference; unmatched modalities)",
                 ha="right", va="bottom", fontsize=8.5, color="#553C9A")

    fig.suptitle("Frozen world features improve a fixed synthetic-lethality readout",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.035,
             "Retrospective MuSL CV3 evaluation · 2 seeds × 5 folds · points show fold variation, not confidence intervals\n"
             "All arms use prepared feature/model artifacts; locked selection was fixed before this plot.",
             ha="center", va="center", fontsize=9, color="#4A5568")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22, wspace=0.12)

    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"slp11-sl-readout-comparison.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
