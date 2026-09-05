"""Compare equal-guide and sampled-cell guide-side estimands on fitting genes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = ROOT / "scripts/audit_slp11_frangieh_guide_reproducibility.py"
GUIDE_SPEC = importlib.util.spec_from_file_location("frangieh_guides", GUIDE_PATH)
GUIDE = importlib.util.module_from_spec(GUIDE_SPEC)
assert GUIDE_SPEC.loader is not None
GUIDE_SPEC.loader.exec_module(GUIDE)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cell_weighted_guide_sides(
    action_ids: np.ndarray,
    guide_ids: np.ndarray,
    num_cells: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the frozen alternating assignment and weight each side by cells."""
    action_ids = np.asarray(action_ids, dtype=str)
    guide_ids = np.asarray(guide_ids, dtype=str)
    num_cells = np.asarray(num_cells, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    genes, side_a, side_b = [], [], []
    for gene in sorted(set(action_ids)):
        rows = np.flatnonzero(action_ids == gene)
        rows = rows[np.argsort(guide_ids[rows], kind="stable")]
        if len(rows) < 2:
            continue
        a, b = rows[::2], rows[1::2]
        if not len(b):
            continue
        genes.append(gene)
        side_a.append(np.average(targets[a], axis=0, weights=num_cells[a]))
        side_b.append(np.average(targets[b], axis=0, weights=num_cells[b]))
    return np.asarray(genes), np.asarray(side_a, dtype=np.float32), np.asarray(side_b, dtype=np.float32)


def run(data_path: Path, prior_report_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = json.loads(prior_report_path.read_text(encoding="utf-8"))
    report = {
        "schema": "slp.frangieh-guide-cell-weighting/v1",
        "parent_report_sha256": digest(prior_report_path),
        "hypothesis": "Sampled-cell weighting reduces guide-split MSE by at least 10 percent without query-centroid-adjusted profile-correlation regression in each context and assay head.",
        "estimand": "Mean sampled cell population within each deterministic guide side; guide groups are not independent biological replicates.",
        "contexts": {},
    }
    decisions = []
    with np.load(data_path, allow_pickle=False) as data:
        action = data["action_ids"].astype(str)
        context = data["context_ids"].astype(str)
        guide = data["source_target_guide_sets"].astype(str)
        cells = data["num_cells"].astype(np.int64)
        fitting = np.zeros(len(action), dtype=bool)
        fitting[data["split_train"]] = True
        for ctx in sorted(set(context)):
            rows = fitting & (context == ctx)
            context_report = {}
            for head, targets in (("rna", data["rna_targets"]), ("adt", data["protein_targets"])):
                genes, equal_a, equal_b, _ = GUIDE.alternating_guide_sides(
                    action[rows], guide[rows], cells[rows], targets[rows]
                )
                weighted_genes, weighted_a, weighted_b = cell_weighted_guide_sides(
                    action[rows], guide[rows], cells[rows], targets[rows]
                )
                if not np.array_equal(genes, weighted_genes):
                    raise ValueError("weighting changed gene eligibility")
                equal = GUIDE.comparison(equal_a, equal_b)
                weighted = GUIDE.comparison(weighted_a, weighted_b)
                parent_equal = prior["contexts"][ctx]["heads"][head]["all"]["matched"]
                if not np.isclose(equal["raw_mse"], parent_equal["raw_mse"], rtol=0, atol=1e-15):
                    raise ValueError("equal-guide parent metric drift")
                improvement = (equal["raw_mse"] - weighted["raw_mse"]) / equal["raw_mse"]
                r_pass = (
                    weighted["query_centroid_adjusted_profile_pearson"]
                    >= equal["query_centroid_adjusted_profile_pearson"] - 1e-12
                )
                passed = bool(improvement >= 0.10 and r_pass)
                decisions.append(passed)
                context_report[head] = {
                    "genes": len(genes),
                    "equal_guide": equal,
                    "sampled_cell_weighted": weighted,
                    "fractional_mse_improvement": float(improvement),
                    "mse_pass": bool(improvement >= 0.10),
                    "correlation_no_regression_pass": bool(r_pass),
                    "pass": passed,
                }
            report["contexts"][ctx] = context_report
    report["decision"] = {
        "advance_weighted_estimand": bool(all(decisions)),
        "passed_checks": int(sum(decisions)),
        "total_checks": len(decisions),
    }
    report["limitations"] = [
        "Fitting genes only; this diagnostic does not alter the frozen development corpus or prior baseline.",
        "Weighting changes the estimand to the sampled cell population and does not create independent replicates.",
        "This is a guide/cell-population reproducibility comparison, not a biological noise ceiling.",
    ]
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--prior-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.data) != args.data_sha256 or digest(args.prior_report) != args.prior_report_sha256:
        raise ValueError("input hash mismatch")
    print(json.dumps(run(args.data, args.prior_report, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
