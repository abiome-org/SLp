#!/usr/bin/env python3
"""Assemble the immutable eight-context joint-world v2 fitting snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/derived/slp11-joint-world-expanded-string-v1"
MCF = ROOT / "data/derived/slp11-gse164996-cropseq-populations-v4"
BASE_SOURCES = ("k562", "rpe1", "norman", "gwps", "hepg2")
MCF_TRAIN = ("mcf10a_full_d0", "mcf10a_full_d6", "mcf10a_tgfb1_d6")
MCF_HOLDOUT = "mcf10a_minimal_d6"
WEIGHTS = {
    "k562": .20, "rpe1": .20, "norman": .12, "gwps": .20, "hepg2": .08,
    "mcf10a_full_d0": 1/15, "mcf10a_full_d6": 1/15, "mcf10a_tgfb1_d6": 1/15,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_pair_fold(ids, seed: int = 731) -> int:
    genes = sorted(str(value) for value in ids if str(value))
    if len(genes) != 2 or genes[0] == genes[1]:
        raise ValueError(f"fold requires two distinct stable gene IDs: {genes}")
    token = f"{seed}|{genes[0]}|{genes[1]}".encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 3


def load(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, destination)
        return "copy"


def add_shared_folds(payload: dict) -> dict:
    singles = np.asarray(payload["single_rows"], np.int64)
    combinations = np.asarray(payload["combination_rows"], np.int64)
    folds = np.asarray([canonical_pair_fold(payload["action_ids"][row])
                        for row in combinations], np.int64)
    payload["single_rows"] = singles
    payload["combination_rows"] = combinations
    payload["combination_fold"] = folds
    if payload["combination_single_rows"].shape != (len(combinations), 2):
        raise ValueError("combination parent rows must align one-to-one with combinations")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--mcf", type=Path, default=MCF)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/derived/slp11-joint-world-context-transfer-v2-training-r4")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output must be a new immutable directory")
    args.output.mkdir(parents=True)
    manifest = {
        "schema": "slp.joint-world-context-transfer-training/v2-r4",
        "trainingSources": list(BASE_SOURCES + MCF_TRAIN),
        "sourceWeights": WEIGHTS,
        "weightPolicy": "legacy five-context weights scaled to 0.8; three MCF10A contexts share 0.2 equally",
        "combinationFold": {"seed": 731, "folds": 3, "selectedHoldoutFold": 0,
                            "identity": "SHA256(seed|sorted stable gene IDs), first 8 bytes mod 3"},
        "contextTransferHoldout": {
            "source": MCF_HOLDOUT,
            "path": str((args.mcf / f"{MCF_HOLDOUT}.npz").resolve().relative_to(ROOT)),
            "sha256": digest(args.mcf / f"{MCF_HOLDOUT}.npz"),
            "outcomesIncludedInTrainingSnapshot": False,
        },
        "normalizer": "unchanged from five-context expanded-string-v1",
        "sources": {},
    }
    normalizer = None
    for source in BASE_SOURCES:
        source_path = args.base / f"{source}.npz"
        destination = args.output / source_path.name
        method = link_or_copy(source_path, destination)
        data = load(source_path)
        if normalizer is None:
            normalizer = (data["feature_mean"], data["feature_scale"])
        else:
            np.testing.assert_array_equal(data["feature_mean"], normalizer[0])
            np.testing.assert_array_equal(data["feature_scale"], normalizer[1])
        manifest["sources"][source] = {"sha256": digest(destination), "bytes": destination.stat().st_size,
                                         "materialization": method, "populations": len(data["targets"])}
    for source in MCF_TRAIN:
        source_path = args.mcf / f"{source}.npz"
        data = add_shared_folds(load(source_path))
        np.testing.assert_array_equal(data["feature_mean"], normalizer[0])
        np.testing.assert_array_equal(data["feature_scale"], normalizer[1])
        destination = args.output / source_path.name
        np.savez_compressed(destination, **data)
        counts = np.bincount(data["combination_fold"], minlength=3)
        manifest["sources"][source] = {
            "sha256": digest(destination), "bytes": destination.stat().st_size,
            "populations": len(data["targets"]), "singles": len(data["single_rows"]),
            "combinations": len(data["combination_rows"]),
            "combinationFoldCounts": counts.tolist(),
            "fold0CombinationOutcomesUsedForFitting": False,
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
