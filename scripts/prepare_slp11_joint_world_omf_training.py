#!/usr/bin/env python3
"""Create a fold-specific joint-world training snapshot with held Norman outcomes absent."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/derived/slp11-joint-world-expanded-string-v1"
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-joint-world-expanded-omf-fold0-v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def filter_norman(arrays: dict[str, np.ndarray], fold: int) -> dict[str, np.ndarray]:
    n = len(arrays["targets"])
    singles = np.asarray(arrays["single_rows"], dtype=np.int64)
    combinations = np.asarray(arrays["combination_rows"], dtype=np.int64)
    combination_fold = np.asarray(arrays["combination_fold"], dtype=np.int64)
    train_combination_index = np.flatnonzero(combination_fold != fold)
    kept_old = np.concatenate((singles, combinations[train_combination_index]))
    if len(np.unique(kept_old)) != len(kept_old) or np.any(kept_old < 0) or np.any(kept_old >= n):
        raise ValueError("invalid Norman row routing")
    old_to_new = np.full(n, -1, dtype=np.int64)
    old_to_new[kept_old] = np.arange(len(kept_old), dtype=np.int64)
    parents_old = np.asarray(arrays["combination_single_rows"], dtype=np.int64)[train_combination_index]
    parents_new = old_to_new[parents_old]
    if np.any(parents_new < 0):
        raise ValueError("training combination parent was removed")

    per_row = {
        "action_feature_index", "action_features", "action_mask", "basal", "observed",
        "target_cell_count", "target_control_z_mean", "target_observed", "targets",
    }
    result = {
        key: np.asarray(value)[kept_old].copy() if key in per_row else np.asarray(value).copy()
        for key, value in arrays.items()
        if key not in {"action_ids", "action_offsets"}
    }
    offsets = np.asarray(arrays["action_offsets"], dtype=np.int64)
    flattened = np.asarray(arrays["action_ids"])
    pieces = [flattened[offsets[row]:offsets[row + 1]] for row in kept_old]
    result["action_ids"] = np.concatenate(pieces)
    result["action_offsets"] = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum([len(piece) for piece in pieces], dtype=np.int64),
    ))
    result["single_rows"] = np.arange(len(singles), dtype=np.int64)
    result["combination_rows"] = np.arange(len(singles), len(kept_old), dtype=np.int64)
    result["combination_single_rows"] = parents_new
    result["combination_fold"] = combination_fold[train_combination_index]
    result["combination_common_query_mask"] = np.asarray(
        arrays["combination_common_query_mask"]
    )[train_combination_index].copy()
    result["schema"] = np.asarray("slp.joint-world-crispra-compositions-training-fold/v1")
    return result


def write_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--norman-fold", type=int, choices=range(3), default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite training snapshot: {args.output}")
    args.output.mkdir(parents=True)
    outputs = {}
    for context in ("k562", "rpe1", "gwps", "hepg2"):
        source = args.source / f"{context}.npz"
        destination = args.output / source.name
        shutil.copyfile(source, destination)
        outputs[context] = {"path": source.name, "sha256": sha256(destination), "copiedUnchanged": True}
    source_norman = args.source / "norman.npz"
    with np.load(source_norman, allow_pickle=False) as archive:
        original = {name: np.asarray(archive[name]) for name in archive.files}
    filtered = filter_norman(original, args.norman_fold)
    norman_path = args.output / "norman.npz"
    write_npz_new(norman_path, filtered)
    outputs["norman"] = {
        "path": norman_path.name,
        "sha256": sha256(norman_path),
        "rows": len(filtered["targets"]),
        "singles": len(filtered["single_rows"]),
        "trainingCombinations": len(filtered["combination_rows"]),
        "heldFold": args.norman_fold,
        "heldCombinationOutcomesPhysicallyAbsent": True,
    }
    manifest = {
        "schema": "slp.joint-world-omf-training-snapshot/v1",
        "source": {"path": str(args.source), "manifestSha256": sha256(args.source / "manifest.json")},
        "normanFold": args.norman_fold,
        "role": "training",
        "developmentOutcomesIncluded": False,
        "protectedTestOutcomesIncluded": False,
        "outputs": outputs,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
