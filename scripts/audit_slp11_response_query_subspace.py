#!/usr/bin/env python3
"""Fitting-only audit of the response-query decoder's output subspace."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
ARTIFACT = ROOT / "results/slp11-transition/human-source3-bp-neural-mean-pair-seed731-v2-finalization-v1"
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
PINS = {
    "data": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "modelSource": "fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef",
    "maskedModel": "690ed2d627aac7e17d81fdc35064aaa45bef065110377d26c00c218ba7ca6d14",
    "bpModel": "f1e0acf79c5326d4553ee77f45ccaa0d02628042672413d7089a17991e5d99fc",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("slp11_subspace_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collapse_fitting_profiles(
    targets: np.ndarray,
    action_ids: np.ndarray,
    context_index: np.ndarray,
    split_train: np.ndarray,
    context: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = split_train[context_index[split_train] == context]
    genes = np.unique(action_ids[rows].astype(str))
    profiles = np.stack(
        [targets[rows[action_ids[rows] == gene]].mean(axis=0, dtype=np.float64) for gene in genes]
    ).astype(np.float32)
    return genes, profiles


def orthonormal_basis(values: np.ndarray) -> tuple[np.ndarray, int]:
    left, singular, _ = np.linalg.svd(
        np.asarray(values, dtype=np.float64), full_matrices=False
    )
    threshold = max(values.shape) * np.finfo(np.float64).eps * singular.max(initial=0.0)
    rank = int(np.count_nonzero(singular > threshold))
    return left[:, :rank], rank


def captured_fraction(values: np.ndarray, basis: np.ndarray) -> float:
    total = float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
    projected = np.asarray(values, dtype=np.float64) @ basis
    return float(np.square(projected).sum(dtype=np.float64) / total)


def learned_query_basis(arm: str) -> np.ndarray:
    arm_path = ARTIFACT / arm
    source = ARTIFACT / "source/control_transition_model.py"
    expected_model = PINS["maskedModel"] if arm == "masked-bp-control" else PINS["bpModel"]
    if sha256(source) != PINS["modelSource"] or sha256(arm_path / "model.safetensors") != expected_model:
        raise ValueError("frozen neural payload drift")
    with np.load(arm_path / "reference.npz", allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    module = load_module(source)
    model = module.MinimalControlTransition(
        module.Config(
            int(reference["feature_mean"].shape[0]),
            int(reference["query_feature_mean"].shape[0]),
            hidden_dim=int(reference["hidden_dim"]),
            state_dim=int(reference["state_dim"]),
            dropout=float(reference["dropout"]),
        )
    )
    model.load_state_dict(load_file(arm_path / "model.safetensors"))
    model.eval()
    queries = (reference["query_features"] - reference["query_feature_mean"]) / reference[
        "query_feature_std"
    ]
    with torch.no_grad():
        query_state = model.query_encoder(torch.as_tensor(queries)).numpy()
    # Linear(state) @ query_state.T == state @ (query_state @ weight).T.
    decoder = query_state @ model.mean_state.weight.detach().numpy()
    return reference["delta_amplitude"][:, None].astype(np.float64) * decoder


def run(seed: int = 731) -> dict[str, object]:
    if sha256(DATA) != PINS["data"]:
        raise ValueError("development snapshot drift")
    with np.load(DATA, allow_pickle=False) as archive:
        if archive["split_test"].size:
            raise ValueError("unexpected test rows")
        targets = archive["targets"].astype(np.float32)
        observed = archive["observed"]
        action_ids = archive["action_ids"].astype(str)
        context_index = archive["context_index"]
        split_train = archive["split_train"]
        if not observed[split_train].all():
            raise ValueError("diagnostic requires a complete fitting panel")
    arm_bases = {arm: learned_query_basis(arm) for arm in ("masked-bp-control", "bp128-present")}
    result: dict[str, object] = {
        "schema": "slp.response-query-fitting-subspace-audit/v1",
        "dataSha256": PINS["data"],
        "quantitativeComputationRestrictedToTrainRows": True,
        "developmentArchiveMaterializationIncludesValidationRows": True,
        "validationRowsIndexedOrUsed": False,
        "rank": 128,
        "contexts": {},
    }
    with np.load(ARTIFACT / "bp128-present/reference.npz", allow_pickle=False) as archive:
        scales = archive["objective_query_scale"].astype(np.float64)
    with threadpool_limits(limits=2):
        for context, name in enumerate(CONTEXTS):
            genes, profiles = collapse_fitting_profiles(
                targets, action_ids, context_index, split_train, context
            )
            centered = profiles.astype(np.float64) - profiles.mean(axis=0, dtype=np.float64)
            standardized = centered / scales[context]
            total = float(np.square(standardized).sum(dtype=np.float64))
            _, singular, _ = randomized_svd(
                standardized,
                n_components=128,
                n_iter=5,
                random_state=seed,
                flip_sign=True,
            )
            oracle = float(np.square(singular).sum(dtype=np.float64) / total)
            learned = {}
            for arm, raw_basis in arm_bases.items():
                basis, rank = orthonormal_basis(raw_basis / scales[context, :, None])
                fraction = captured_fraction(standardized, basis)
                learned[arm] = {
                    "numericalRank": rank,
                    "capturedStandardizedLandscapeVariance": fraction,
                    "irreducibleRelativeSquaredErrorWithinSubspace": 1.0 - fraction,
                    "fractionOfRank128OracleVarianceCaptured": fraction / oracle,
                }
            result["contexts"][name] = {
                "fittingGenes": len(genes),
                "bestRank128CapturedStandardizedLandscapeVariance": oracle,
                "bestRank128IrreducibleRelativeSquaredError": 1.0 - oracle,
                "learnedQuerySubspaces": learned,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()
    print(json.dumps(run(args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
