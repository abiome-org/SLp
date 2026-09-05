"""Fitting-only guide-set reproducibility diagnostic for Frangieh profiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "modules/slp-1-1-world-transition-v1/frangieh_basal_ridge.py"
SPEC = importlib.util.spec_from_file_location("frangieh_basal_ridge", MODULE_PATH)
METRIC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(METRIC)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def alternating_guide_sides(
    action_ids: np.ndarray,
    guide_ids: np.ndarray,
    num_cells: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make deterministic equal-guide-weight A/B profiles for repeated genes."""
    action_ids = np.asarray(action_ids, dtype=str)
    guide_ids = np.asarray(guide_ids, dtype=str)
    num_cells = np.asarray(num_cells, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float64)
    genes, side_a, side_b, minimum_cells = [], [], [], []
    for gene in sorted(set(action_ids)):
        rows = np.flatnonzero(action_ids == gene)
        rows = rows[np.argsort(guide_ids[rows], kind="stable")]
        if len(rows) < 2:
            continue
        a, b = rows[::2], rows[1::2]
        if not len(b):
            continue
        genes.append(gene)
        side_a.append(np.mean(targets[a], axis=0))
        side_b.append(np.mean(targets[b], axis=0))
        minimum_cells.append(min(int(np.sum(num_cells[a])), int(np.sum(num_cells[b]))))
    return (
        np.asarray(genes),
        np.asarray(side_a, dtype=np.float32),
        np.asarray(side_b, dtype=np.float32),
        np.asarray(minimum_cells, dtype=np.int64),
    )


def deterministic_different_gene_order(genes: np.ndarray) -> np.ndarray:
    genes = np.asarray(genes, dtype=str)
    if len(genes) < 2:
        return np.empty(0, dtype=np.int64)
    order = np.asarray(
        sorted(
            range(len(genes)),
            key=lambda index: hashlib.sha256(
                f"slp11-frangieh-guide-shuffle-v1|731|9606|{genes[index]}".encode()
            ).digest(),
        )
    )
    permutation = np.empty(len(genes), dtype=np.int64)
    permutation[order] = np.roll(order, -1)
    if np.any(genes == genes[permutation]):
        raise AssertionError("different-gene shuffle contains a self-pair")
    return permutation


def comparison(left: np.ndarray, right: np.ndarray) -> dict:
    score, per_gene = METRIC.query_centroid_adjusted_profile_pearson(left, right)
    return {
        "query_centroid_adjusted_profile_pearson": score,
        "undefined_genes": int(np.sum(~np.isfinite(per_gene))),
        "raw_mse": float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2)),
        "genes": len(left),
    }


def run(data_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as data:
        action = data["action_ids"].astype(str)
        context = data["context_ids"].astype(str)
        guides = data["source_target_guide_sets"].astype(str)
        counts = data["num_cells"].astype(np.int64)
        train_rows = np.zeros(len(action), dtype=bool)
        train_rows[data["split_train"]] = True
        report = {
            "schema": "slp.frangieh-guide-reproducibility/v1",
            "scope": "fitting genes only",
            "interpretation": "Guide/cell-population reproducibility diagnostic; not biological-replicate agreement and not a noise ceiling.",
            "contexts": {},
        }
        saved = {}
        for ctx in sorted(set(context)):
            rows = train_rows & (context == ctx)
            bins = {
                "1-4": int(np.sum((counts[rows] >= 1) & (counts[rows] < 5))),
                "5-19": int(np.sum((counts[rows] >= 5) & (counts[rows] < 20))),
                "20-99": int(np.sum((counts[rows] >= 20) & (counts[rows] < 100))),
                ">=100": int(np.sum(counts[rows] >= 100)),
            }
            context_report = {
                "guide_pseudobulks": int(np.sum(rows)),
                "cell_count_bins": bins,
                "heads": {},
            }
            for head, target in (("rna", data["rna_targets"]), ("adt", data["protein_targets"])):
                genes, side_a, side_b, minimum_cells = alternating_guide_sides(
                    action[rows], guides[rows], counts[rows], target[rows]
                )
                head_report = {}
                for label, stratum in (
                    ("all", np.ones(len(genes), dtype=bool)),
                    ("minimum_side_cells_lt20", minimum_cells < 20),
                    ("minimum_side_cells_ge20", minimum_cells >= 20),
                ):
                    if not np.any(stratum):
                        head_report[label] = {"genes": 0, "matched": None, "shuffled_different_gene": None}
                        continue
                    selected = np.flatnonzero(stratum)
                    matched = comparison(side_a[selected], side_b[selected])
                    permutation = deterministic_different_gene_order(genes[selected])
                    shuffled = (
                        comparison(side_a[selected], side_b[selected][permutation])
                        if len(permutation)
                        else None
                    )
                    head_report[label] = {
                        "genes": len(selected),
                        "matched": matched,
                        "shuffled_different_gene": shuffled,
                    }
                context_report["heads"][head] = head_report
                key = ctx.replace("γ", "gamma").replace("-", "_")
                saved[f"{key}_{head}_action_ids"] = genes
                saved[f"{key}_{head}_side_a"] = side_a
                saved[f"{key}_{head}_side_b"] = side_b
                saved[f"{key}_{head}_minimum_side_cells"] = minimum_cells
            report["contexts"][ctx] = context_report
    pairs_path = output_dir / "guide-side-profiles.npz"
    np.savez_compressed(pairs_path, **saved)
    report["profiles"] = {"sha256": digest(pairs_path), "bytes": pairs_path.stat().st_size}
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.data) != args.data_sha256:
        raise ValueError("input hash mismatch")
    print(json.dumps(run(args.data, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
