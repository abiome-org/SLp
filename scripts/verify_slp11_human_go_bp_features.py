#!/usr/bin/env python3
"""Reconstruct and verify the frozen source-three-fit human GO BP feature pack."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits


def load_builder(path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("slp11_go_bp_frozen_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen BP builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    builder_path = artifact / "source" / "build_slp11_human_go_bp_features.py"
    builder = load_builder(builder_path)
    sources = manifest["sources"]
    if builder.sha256_file(Path(sources["goaHuman"]["path"])) != sources["goaHuman"]["sha256"]:
        raise RuntimeError("GO source hash mismatch")
    if builder.sha256_file(Path(sources["ensemblMapping"]["path"])) != sources["ensemblMapping"]["sha256"]:
        raise RuntimeError("mapping source hash mismatch")
    data_path = Path(sources["developmentIdentityOnly"]["path"])
    if builder.sha256_file(data_path) != sources["developmentIdentityOnly"]["sha256"]:
        raise RuntimeError("development identity source hash mismatch")
    output_ids = builder.read_roster(artifact / "entity-ids.txt")
    fit_ids = builder.read_roster(artifact / "fit-entity-ids.txt")
    terms = tuple((artifact / "fit-term-ids.txt").read_text("ascii").splitlines())
    with np.load(data_path, allow_pickle=False) as data:
        expected_fit = tuple(sorted(set(data["action_ids"][data["split_train"]].astype(str))))
    if fit_ids != expected_fit:
        raise RuntimeError("fitting identity reconstruction mismatch")
    universe = tuple(sorted(set(output_ids) | set(fit_ids)))
    mapping_payload = Path(sources["ensemblMapping"]["path"]).read_bytes()
    go_payload = Path(sources["goaHuman"]["path"]).read_bytes()
    mapping, _ = builder.parse_mapping(mapping_payload, frozenset(universe))
    terms_by_gene, _ = builder.parse_bp_gaf(go_payload, mapping, universe)
    fit_matrix, output_matrix, reconstructed_terms, _ = builder.matrices_from_fit_terms(
        terms_by_gene, fit_ids, output_ids
    )
    if reconstructed_terms != terms:
        raise RuntimeError("term vocabulary reconstruction mismatch")
    with np.load(artifact / "human-go-bp-source3-fit-svd128-basis.npz", allow_pickle=False) as basis:
        components = basis["components"]
        if not np.array_equal(basis["term_id"].astype(str), np.asarray(terms)):
            raise RuntimeError("saved basis term identity mismatch")
    with np.load(artifact / "human-go-bp-source3-fit-svd128-features.npz", allow_pickle=False) as features:
        if not np.array_equal(features["entity_id"].astype(str), np.asarray(output_ids)):
            raise RuntimeError("feature entity identity mismatch")
        if np.any(features["entity_taxon"] != 9606):
            raise RuntimeError("feature taxonomy mismatch")
        projected = (output_matrix @ components.T).astype(np.float32)
        projection_error = float(np.max(np.abs(projected - features["feature_values"])))
        if projection_error > 2e-6:
            raise RuntimeError(f"saved projection mismatch: {projection_error}")
        if not np.array_equal(features["annotation_present"], (np.diff(output_matrix.indptr) > 0).astype(np.uint8)):
            raise RuntimeError("annotation-present flag mismatch")
    with threadpool_limits(limits=2):
        svd = TruncatedSVD(
            n_components=128,
            algorithm="randomized",
            n_iter=7,
            n_oversamples=10,
            power_iteration_normalizer="auto",
            random_state=731,
        ).fit(fit_matrix)
    component_error = float(np.max(np.abs(svd.components_.astype(np.float32) - components)))
    if component_error != 0.0:
        raise RuntimeError(f"SVD component determinism mismatch: {component_error}")
    verifier_copy = artifact / "source" / Path(__file__).name
    if not verifier_copy.exists():
        shutil.copyfile(Path(__file__), verifier_copy)
    report = {
        "schema": "slp.human-go-bp-source3-fit-svd-verification/v1",
        "status": "pass",
        "checks": {
            "pinnedSourceHashes": True,
            "fitRosterReconstructedFromSplitTrainIdentityOnly": True,
            "termVocabularyReconstructed": True,
            "svdComponentsBitwiseDeterministic": True,
            "projectionReconstructed": True,
            "entityIdentityAndTaxonomyExact": True,
            "annotationPresenceExact": True,
            "pickleRequired": False,
        },
        "maximumProjectionAbsoluteError": projection_error,
        "maximumComponentAbsoluteError": component_error,
        "featuresSha256": builder.sha256_file(artifact / "human-go-bp-source3-fit-svd128-features.npz"),
        "basisSha256": builder.sha256_file(artifact / "human-go-bp-source3-fit-svd128-basis.npz"),
        "builderSha256": builder.sha256_file(builder_path),
        "verifierSha256": builder.sha256_file(verifier_copy),
    }
    (artifact / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
