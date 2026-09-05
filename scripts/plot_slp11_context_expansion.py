"""Plot frozen ensemble context-expansion effects and descriptive intervals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "results/slp11-transition/human-context-ensemble-uncertainty-v1/report.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != "e1b4fc2df27bdcac2f88f9656d5ed40572f46060b2f57fbdf694e6c9b72a143e":
        raise ValueError("frozen uncertainty report mismatch")
    data = json.loads(source.read_text())
    names = list(data["contexts"])
    destination = root / "results/slp11-transition/figures/context-expansion-v1"
    for suffix in (".png", ".pdf", ".json"):
        if destination.with_suffix(suffix).exists():
            raise FileExistsError(destination.with_suffix(suffix))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    for ax, comparison, title in zip(
        axes, ("source4VsSource3", "source4VsRidge"),
        ("Effect of adding HepG2 training data", "Four-context ensemble versus ridge"), strict=True,
    ):
        for row, name in enumerate(names):
            result = data["contexts"][name][comparison]
            value = result["mseImprovementPercent"]
            low, high = result["pairedGeneBootstrap95PercentileInterval"]
            color = "#156b8a" if low > 0 else "#aa4b38" if high < 0 else "#697482"
            ax.errorbar(value, row, xerr=np.array([[value-low], [high-value]]),
                        fmt="o", capsize=5, color=color, markersize=7, linewidth=2)
            ax.annotate(f"{value:+.2f}%", (value, row), xytext=(0, 12),
                        textcoords="offset points", ha="center", color=color, fontsize=10)
        ax.axvline(0, color="#7e8894", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=12, pad=17)
        ax.set_xlabel("MSE improvement (%) · positive is better", labelpad=12)
        ax.grid(axis="x", alpha=.15)
        ax.set_ylim(3.55, -.55)
        ax.set_yticks(range(4), [f"{name}\n(n={data['contexts'][name][comparison]['genes']} genes)" for name in names])
    figure.suptitle("More context data helps HepG2, but does not advance the joint model",
                    fontsize=15, x=.54, y=.97)
    figure.text(.02, .055, "Fixed average of seeds 731–733. Bars: 95% paired-gene bootstrap intervals (2,000 samples).", fontsize=9)
    figure.text(.02, .025, "Adaptive development; intervals condition on the fitted models and exclude training/selection uncertainty.", fontsize=9)
    figure.subplots_adjust(left=.19, right=.97, top=.80, bottom=.23, wspace=.13)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(destination.with_suffix(suffix), dpi=180, facecolor="white")
    plt.close(figure)
    destination.with_suffix(".json").write_text(json.dumps({"sourceSha256": digest,
        "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    print(destination.with_suffix(".png"))


if __name__ == "__main__":
    main()
