"""Plot measured development tradeoffs from three immutable reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/slp11-transition"
REPORTS = {
    "kernel": (
        "human-gwps-nystrom-rbf512-physical-seed731-v1/report.json",
        "ceff4ea924df07dd930b980929c9227a6719421673974fbc0c065b3deac1184e",
    ),
    "bp": (
        "human-gwps-bp-ridge-source3-seed731-v2/report.json",
        "8a3d1ba2265dc09bf6856c97c7a791775ef3282594beed269f708f353d895a0a",
    ),
    "graph": (
        "human-gwps-gene-state-response32-state16-seed731-v1/report.json",
        "5ded419bca65c9dff6b88d0c4c65897b8d18d4431c1278d7027236295f14cf7a",
    ),
    "bp_kernel": (
        "human-gwps-bp-nystrom-rbf512-seed731-v1/report.json",
        "d8259c864460a21f9a13718b2190aad926ca58dc01409c0fab1220a6fbbd276c",
    ),
}
CONTEXTS = (
    ("replogle-2022-k562-essential-day-6", "K562 essential", 305),
    ("replogle-2022-rpe1-essential-day-7", "RPE1 essential", 360),
    ("replogle-2022-k562-gwps-day-8", "K562 genome-wide", 1491),
)
STYLES = {
    "Global state v2": ("#235976", "o"),
    "RBF kernel": ("#ba7625", "s"),
    "Ridge + biological process": ("#367d58", "D"),
    "RBF + biological process": ("#7860a8", "P"),
    "Gene states + response descriptors": ("#a05275", "^"),
}


def main() -> None:
    reports = {}
    for name, (relative, expected) in REPORTS.items():
        raw = (RESULTS / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"report checksum changed: {relative}")
        reports[name] = json.loads(raw)
    records = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 5.3))
        for axis, (context, title, genes) in zip(axes, CONTEXTS, strict=True):
            scores = reports["kernel"]["contexts"][context]["scores"]
            baseline = scores["fullPhysicalRidge"]
            candidates = {
                "Global state v2": scores["minimalControlV2"],
                "RBF kernel": scores["nystromRbf"],
                "Ridge + biological process": reports["bp"]["contexts"][context]["arms"][
                    "physical1156_bp128_present1"]["scores"],
                "RBF + biological process": reports["bp_kernel"]["contexts"][context]["candidate"],
            }
            graph = reports["graph"]["contexts"][context]["independentlyCentered"]
            candidates["Gene states + response descriptors"] = {
                "geneProfileMse": graph["primaryGeneAveragedProfileMse"],
                "independentlyCenteredPearson": graph["primaryIndependentlyCenteredGeneMacroProfilePearson"],
            }
            for label, value in candidates.items():
                improvement = 100 * (1 - value["geneProfileMse"] / baseline["geneProfileMse"])
                correlation = value.get("independentlyCenteredPearson", value.get("independentlyQueryCenteredPearson"))
                delta_r = correlation - baseline["independentlyCenteredPearson"]
                color, marker = STYLES[label]
                axis.scatter(improvement, delta_r, color=color, marker=marker,
                             s=75, edgecolors="white", linewidth=.7, label=label, zorder=4)
                records.append({"context": context, "model": label,
                                "mse_improvement_percent": improvement,
                                "independent_pearson_delta": delta_r})
            axis.scatter([0], [0], marker="+", color="#26343d", s=100, zorder=5)
            axis.axvline(0, color="#89949b", linewidth=1)
            axis.axhline(0, color="#89949b", linewidth=1)
            axis.grid(color="#e6e9ec", linewidth=.6)
            axis.set_title(f"{title}\n{genes:,} validation genes", loc="left", fontsize=11)
            axis.set_xlabel("MSE improvement over physical ridge (%)")
            axis.margins(x=.20, y=.25)
            axis.text(.97, .97, "Better: up and right", transform=axis.transAxes,
                      ha="right", va="top", fontsize=8, color="#52616b")
        axes[0].set_ylabel("Change in perturbation-specific correlation")
        fig.suptitle("Model development: error and landscape accuracy tell different stories",
                     x=.025, ha="left", fontsize=15, weight="bold")
        fig.text(.025, .91, "Human CRISPRi · identical held-gene profiles · each point is a frozen development result",
                 color="#52616b", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.50, .065),
                   ncol=3, frameon=False, fontsize=9)
        fig.text(.025, .022, "Separate prediction/truth query centroids removed before correlation. "
                 "Point estimates; no confidence intervals or launch-readiness claim. + marks ridge.",
                 fontsize=9, color="#52616b")
        fig.subplots_adjust(left=.08, right=.985, top=.77, bottom=.28, wspace=.32)
        output = RESULTS / "figures"
        output.mkdir(exist_ok=True)
        for extension in ("png", "pdf"):
            destination = output / f"source-frontier-v1.{extension}"
            if destination.exists():
                raise FileExistsError(destination)
            fig.savefig(destination, dpi=180, facecolor="white")
        plt.close(fig)
        (output / "source-frontier-v1.json").write_text(json.dumps({
            "reports": REPORTS, "records": records,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
