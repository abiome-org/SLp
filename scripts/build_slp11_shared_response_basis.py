#!/usr/bin/env python3
"""Build a feasible shared rank-128 response basis from source3 fitting rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
ARM = ROOT / "results/slp11-transition/human-source3-bp-neural-mean-pair-seed731-v2-finalization-v1/bp128-present"
MODEL_SOURCE = ARM.parent / "source/control_transition_model.py"
OUTPUT = ROOT / "data/derived/slp11-human-response-basis/source3-shared-rank128-fitting-v1"
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
PINS = {
    "data": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "reference": "aea0c407fcc0e3199b2926eb9f639cc8fe4d8f3abd5eb3dd1055abc065c0dfef",
    "model": "f1e0acf79c5326d4553ee77f45ccaa0d02628042672413d7089a17991e5d99fc",
    "modelSource": "fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef",
}
RANK = 128
SEED = 731


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(entry) for entry in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def collapse_profiles(
    targets: np.ndarray, action_ids: np.ndarray, context_index: np.ndarray, context: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse already fitting-only rows equally within stable gene and context."""
    selected = np.flatnonzero(context_index == context)
    genes = np.unique(action_ids[selected].astype(str))
    profiles = np.stack(
        [targets[selected[action_ids[selected] == gene]].mean(0, dtype=np.float64) for gene in genes]
    ).astype(np.float32)
    return genes, profiles


def orthonormal_basis(values: np.ndarray) -> tuple[np.ndarray, int]:
    left, singular, _ = np.linalg.svd(
        np.asarray(values, dtype=np.float64), full_matrices=False
    )
    threshold = max(values.shape) * np.finfo(np.float64).eps * singular.max(initial=0.0)
    rank = int(np.count_nonzero(singular > threshold))
    return left[:, :rank], rank


def reconstruction(values: np.ndarray, raw_basis: np.ndarray) -> dict[str, float | int]:
    basis, rank = orthonormal_basis(raw_basis)
    values64 = np.asarray(values, dtype=np.float64)
    total = float(np.square(values64).sum(dtype=np.float64))
    captured = float(np.square(values64 @ basis).sum(dtype=np.float64))
    residual = total - captured
    return {
        "basisRank": rank,
        "capturedEnergyFraction": captured / total,
        "relativeSquaredError": residual / total,
        "meanSquaredError": residual / values64.size,
    }


def load_model_module(path: Path):
    spec = importlib.util.spec_from_file_location("slp11_shared_basis_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def actual_decoder_basis(reference: dict[str, np.ndarray]) -> np.ndarray:
    module = load_model_module(MODEL_SOURCE)
    model = module.MinimalControlTransition(
        module.Config(
            len(reference["feature_mean"]),
            len(reference["query_feature_mean"]),
            hidden_dim=int(reference["hidden_dim"]),
            state_dim=int(reference["state_dim"]),
            dropout=float(reference["dropout"]),
        )
    )
    model.load_state_dict(load_file(ARM / "model.safetensors"))
    model.eval()
    queries = (reference["query_features"] - reference["query_feature_mean"]) / reference[
        "query_feature_std"
    ]
    with torch.no_grad():
        encoded = model.query_encoder(torch.as_tensor(queries)).numpy()
    decoder = encoded @ model.mean_state.weight.detach().numpy()
    return reference["delta_amplitude"][:, None].astype(np.float64) * decoder


def run(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    paths = {"data": DATA, "reference": ARM / "reference.npz", "model": ARM / "model.safetensors", "modelSource": MODEL_SOURCE}
    for name, path in paths.items():
        if sha256(path) != PINS[name]:
            raise ValueError(f"input drift: {name}")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    shutil.copy2(Path(__file__), source / Path(__file__).name)
    protocol = {
        "schema": "slp.source3-shared-response-basis-protocol/v1",
        "status": "frozen-before-fitting-value-access",
        "construction": "Gene-collapse fitting records equally within each context; subtract frozen context control; divide each query by the same frozen delta_amplitude; keep context blocks uncentered; multiply every row in context c by 1/sqrt(n_fitting_genes_c); randomized SVD rank128 seed731,n_iter7.",
        "evaluation": "For each context, project raw deltas in its frozen fitting-SD standardized objective geometry; report uncentered objective reconstruction and separately gene-centered landscape energy. Compare the same geometry with the actual frozen BP neural decoder span.",
        "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": PINS[name]} for name, path in paths.items()},
        "sourceSha256": sha256(source / Path(__file__).name),
        "developmentArchiveMaterializationIncludesValidationRows": True,
        "quantitativeComputationRestrictedToTrainRows": True,
        "validationRowsIndexedOrUsed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "protocol.json", protocol)
    with np.load(DATA, allow_pickle=False) as archive:
        split_train = archive["split_train"]
        if archive["split_test"].size or not archive["observed"][split_train].all():
            raise ValueError("source3 fitting boundary drift")
        train_targets = archive["targets"][split_train].astype(np.float32)
        train_actions = archive["action_ids"][split_train].astype(str)
        train_context = archive["context_index"][split_train].astype(np.int64)
        train_record_ids = archive["record_ids"][split_train].astype(str)
        query_ids = archive["query_ids"].astype(str)
        context_ids = archive["context_ids"].astype(str)
    if tuple(context_ids) != CONTEXTS:
        raise ValueError("context identity drift")
    with np.load(ARM / "reference.npz", allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    if not np.array_equal(query_ids, reference["query_ids"]):
        raise ValueError("query identity drift")
    blocks = []
    gene_ids: dict[int, np.ndarray] = {}
    profiles_by_context: dict[int, np.ndarray] = {}
    for context in range(len(CONTEXTS)):
        genes, profiles = collapse_profiles(train_targets, train_actions, train_context, context)
        gene_ids[context] = genes
        profiles_by_context[context] = profiles
        normalized_delta = (profiles - reference["control_mean"][context]) / reference[
            "delta_amplitude"
        ]
        blocks.append((normalized_delta / np.sqrt(len(genes))).astype(np.float32))
    pooled = np.concatenate(blocks, axis=0)
    with threadpool_limits(limits=2):
        _, singular_values, components = randomized_svd(
            pooled,
            n_components=RANK,
            n_iter=7,
            random_state=SEED,
            flip_sign=True,
        )
    query_coordinates = (components.T * np.sqrt(RANK)).astype(np.float32)
    shared_raw_basis = reference["delta_amplitude"][:, None].astype(np.float64) * components.T
    actual_raw_basis = actual_decoder_basis(reference)
    contexts = {}
    for context, name in enumerate(CONTEXTS):
        delta = profiles_by_context[context].astype(np.float64) - reference["control_mean"][context]
        scale = reference["objective_query_scale"][context].astype(np.float64)
        uncentered = delta / scale
        centered = (delta - delta.mean(axis=0, dtype=np.float64)) / scale
        contexts[name] = {
            "fittingGenes": len(gene_ids[context]),
            "sharedBasis": {
                "uncenteredObjectiveReconstruction": reconstruction(
                    uncentered, shared_raw_basis / scale[:, None]
                ),
                "centeredLandscapeReconstruction": reconstruction(
                    centered, shared_raw_basis / scale[:, None]
                ),
            },
            "actualBpNeuralDecoder": {
                "uncenteredObjectiveReconstruction": reconstruction(
                    uncentered, actual_raw_basis / scale[:, None]
                ),
                "centeredLandscapeReconstruction": reconstruction(
                    centered, actual_raw_basis / scale[:, None]
                ),
            },
        }
    np.savez_compressed(
        output / "basis.npz",
        components=components.astype(np.float32),
        query_coordinates=query_coordinates,
        singular_values=singular_values.astype(np.float64),
        query_ids=query_ids,
        context_ids=context_ids,
        context0_fitting_action_ids=gene_ids[0],
        context1_fitting_action_ids=gene_ids[1],
        context2_fitting_action_ids=gene_ids[2],
        fitting_record_ids=train_record_ids,
        delta_amplitude=reference["delta_amplitude"],
        objective_query_scale=reference["objective_query_scale"],
        control_mean=reference["control_mean"],
    )
    report = {
        "schema": "slp.source3-shared-response-basis-result/v1",
        "protocolSha256": sha256(output / "protocol.json"),
        "basisSha256": sha256(output / "basis.npz"),
        "basis": {
            "rank": RANK,
            "queries": len(query_ids),
            "pooledRows": len(pooled),
            "queryCoordinates": "components.T*sqrt(128), so the existing /sqrt(state_dim) decoder has unit orthonormal component scale",
            "singularValuesSha256": hashlib.sha256(singular_values.tobytes()).hexdigest(),
        },
        "contexts": contexts,
        "developmentArchiveMaterializationIncludesValidationRows": True,
        "quantitativeComputationRestrictedToTrainRows": True,
        "validationRowsIndexedOrUsed": False,
        "testAccessed": False,
        "benchmarkAccessed": False,
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
