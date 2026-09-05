#!/usr/bin/env python3
"""Package minimal-control-v2 and make a target-free HepG2 BRCA1 forecast."""

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

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-control-transition-v2"
CHECKPOINT_SHA256 = "429791272736c59ae77cca72ccd5a6b51f60736c2213493fa6d924f215611d2d"
REFERENCE_SHA256 = "882e030021eeb3cfc427bf545e37b53f8836d0626624ca952472990e3d5bae5e"
DEVELOPMENT_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
STATIC_SHA256 = "a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b"
HEPG2_SHA256 = "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27"
PREDICTIONS_SHA256 = "8fb447ff43b8bf328b4a2c2e2cf8a3fafa6e33dab97a93685fd580df797b0a88"
BRCA1 = "ENSG00000012048"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def import_runtime(path: Path):
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "portable_minimal_control_runtime", path / "inference.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load packaged inference runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = Path(args.run)
    checkpoint = run_dir / "model.safetensors"
    reference_path = run_dir / "reference.npz"
    predictions_path = run_dir / "development-predictions.npz"
    inputs = {
        checkpoint: CHECKPOINT_SHA256,
        reference_path: REFERENCE_SHA256,
        predictions_path: PREDICTIONS_SHA256,
        Path(args.development): DEVELOPMENT_SHA256,
        Path(args.static_features): STATIC_SHA256,
        Path(args.hepg2_control): HEPG2_SHA256,
    }
    for path, expected in inputs.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    runtime_dir = output / "runtime"
    runtime_dir.mkdir()
    for source, destination in (
        (MODULE / "transition_model.py", runtime_dir / "transition_model.py"),
        (MODULE / "inference.py", runtime_dir / "inference.py"),
        (checkpoint, runtime_dir / "model.safetensors"),
        (run_dir / "model-config.json", runtime_dir / "model-config.json"),
    ):
        shutil.copyfile(source, destination)

    reference = load_npz(reference_path)
    development = load_npz(Path(args.development))
    hepg2 = load_npz(Path(args.hepg2_control))
    static = load_npz(Path(args.static_features))
    query_ids = reference["query_ids"]
    if (
        not np.array_equal(query_ids, development["query_ids"])
        or not np.array_equal(query_ids, hepg2["query_ids"])
        or int(hepg2["perturbed_expression_rows_read"]) != 0
        or hepg2["context_basal_expression"].shape != (1, len(query_ids))
        or hepg2["context_basal_observed"].shape != (1, len(query_ids))
        or str(hepg2["context_value_space"].item())
        != str(reference["context_value_space"].item())
    ):
        raise ValueError("control-only HepG2 or ordered query identity contract mismatch")
    common_panel = development["context_basal_observed"].all(0)
    if (
        int(common_panel.sum()) != 6789
        or not np.array_equal(hepg2["context_basal_observed"][0], common_panel)
        or not common_panel[reference["context_query_indices"]].all()
    ):
        raise ValueError("fixed common-control panel contract mismatch")
    np.savez_compressed(
        runtime_dir / "runtime-reference.npz",
        feature_mean=reference["feature_mean"],
        feature_std=reference["feature_std"],
        query_feature_mean=reference["query_feature_mean"],
        query_feature_std=reference["query_feature_std"],
        query_features=reference["query_features"],
        delta_amplitude=reference["delta_amplitude"],
        delta_amplitude_formula=reference["delta_amplitude_formula"],
        query_ids=query_ids,
        context_query_indices=reference["context_query_indices"],
        context_panel_mask=common_panel,
        context_value_space=reference["context_value_space"],
    )

    keys = list(zip(static["entity_taxon"].tolist(), static["entity_id"].tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError("static feature composite keys are not unique")
    lookup = {key: index for index, key in enumerate(keys)}
    if (9606, BRCA1) not in lookup:
        raise ValueError("BRCA1 exact Ensembl composite key absent from static pack")
    brca1_features = static["feature_values"][lookup[(9606, BRCA1)]].astype(np.float32)
    if brca1_features.shape != (577,) or not np.isfinite(brca1_features).all():
        raise ValueError("BRCA1 static feature row is invalid")
    np.savez_compressed(
        output / "action-features.npz",
        entity_taxon=np.asarray([9606], dtype=np.int64),
        entity_id=np.asarray([BRCA1]),
        feature_values=brca1_features[None, :],
        protein_present=np.asarray([brca1_features[320] > 0], dtype=np.bool_),
        go_projection_nonzero=np.asarray(
            [np.count_nonzero(brca1_features[321:]) > 0], dtype=np.bool_
        ),
        source_artifact_sha256=np.asarray(STATIC_SHA256),
    )
    shutil.copyfile(Path(args.hepg2_control), output / "hepg2-control-context.npz")

    runtime_module = import_runtime(runtime_dir)
    runtime = runtime_module.PortableMinimalControl(runtime_dir, device="cpu")
    zero_control = np.zeros(len(query_ids), dtype=np.float32)
    forecast = runtime.predict(
        brca1_features[None, :],
        hepg2["context_basal_expression"][0],
        hepg2["context_basal_observed"][0],
        zero_control,
        query_ids=query_ids,
    )
    if forecast["uncertainty_calibrated"] or "scale" in forecast:
        raise RuntimeError("target-free forecast unexpectedly claimed uncertainty")

    # Independent source-model reload verifies the packaged normalization path.
    from safetensors.torch import load_file

    with (runtime_dir / "model-config.json").open(encoding="utf-8") as stream:
        model = runtime_module.MinimalControlTransition(
            runtime_module.Config(**json.load(stream))
        ).eval()
    model.load_state_dict(load_file(runtime_dir / "model.safetensors", device="cpu"))
    frozen = load_npz(predictions_path)
    chosen = []
    for context_index in range(3):
        chosen.append(int(np.flatnonzero(frozen["context_index"] == context_index)[0]))
    chosen = np.asarray(chosen, dtype=np.int64)
    known_action_ids = frozen["action_ids"][chosen]
    known_actions = np.stack(
        [static["feature_values"][lookup[(9606, str(item))]] for item in known_action_ids]
    ).astype(np.float32)
    known_context_indices = frozen["context_index"][chosen]
    known = runtime.predict(
        known_actions,
        development["context_basal_expression"][known_context_indices],
        development["context_basal_observed"][known_context_indices],
        development["basal_control"][known_context_indices],
        query_ids=query_ids,
    )
    normalized_actions = (
        (known_actions - reference["feature_mean"]) / reference["feature_std"]
    ).astype(np.float32)
    normalized_queries = (
        (reference["query_features"] - reference["query_feature_mean"])
        / reference["query_feature_std"]
    ).astype(np.float32)
    with torch.no_grad():
        direct = model(
            torch.from_numpy(normalized_actions),
            torch.from_numpy(normalized_queries),
            torch.from_numpy(development["basal_control"][known_context_indices]),
            torch.from_numpy(reference["delta_amplitude"]),
            torch.ones((3, len(query_ids)), dtype=torch.float32),
            torch.from_numpy(normalized_queries[reference["context_query_indices"]]),
            torch.from_numpy(reference["context_values"][known_context_indices]),
            torch.from_numpy(reference["context_mask"][known_context_indices]),
        )
    direct_mean = direct["mean"].numpy()
    if not np.array_equal(known["mean"], direct_mean):
        raise RuntimeError("portable preprocessing differs from independent source reload")
    frozen_mean = frozen["mean"][chosen]
    frozen_max_abs = float(np.max(np.abs(known["mean"] - frozen_mean)))
    if not np.allclose(known["mean"], frozen_mean, rtol=1e-5, atol=2e-6):
        raise RuntimeError("portable known-context means differ from frozen GPU forecasts")

    empty = runtime.predict(
        np.empty((1, 0, 577), dtype=np.float32),
        hepg2["context_basal_expression"][0],
        hepg2["context_basal_observed"][0],
        zero_control,
        query_ids=query_ids,
        action_mask=np.empty((1, 0), dtype=np.bool_),
    )
    empty_identity = bool(
        np.array_equal(empty["mean"], zero_control[None, :])
        and np.count_nonzero(empty["delta"]) == 0
        and np.count_nonzero(empty["intervention_delta"]) == 0
        and np.array_equal(empty["state"], empty["basal_state"])
    )
    if not empty_identity:
        raise RuntimeError("packaged empty-action identity failed")
    chunks = []
    for indices in np.array_split(np.arange(len(query_ids), dtype=np.int64), 5):
        chunks.append(
            runtime.predict(
                brca1_features[None, :],
                hepg2["context_basal_expression"][0],
                hepg2["context_basal_observed"][0],
                zero_control,
                query_ids=query_ids,
                query_indices=indices,
            )["mean"]
        )
    chunk_max_abs = float(
        np.max(np.abs(np.concatenate(chunks, axis=1) - forecast["mean"]))
    )
    if chunk_max_abs > 2e-6:
        raise RuntimeError("packaged query chunk invariance failed")

    supported = hepg2["context_basal_observed"][0].astype(np.bool_)
    np.savez_compressed(
        output / "hepg2-brca1-forecast.npz",
        schema=np.asarray("slp.target-free-control-anchored-forecast/v1"),
        status=np.asarray("experimental-unvalidated-nonadvanced-checkpoint"),
        action_entity_taxon=np.asarray(9606, dtype=np.int64),
        action_entity_id=np.asarray(BRCA1),
        context_id=hepg2["context_ids"][0],
        context_value_space=hepg2["context_value_space"],
        query_ids=query_ids,
        query_supported=supported,
        supplied_control_mean=zero_control,
        molecular_mean=forecast["mean"][0].astype(np.float32),
        molecular_delta=forecast["delta"][0].astype(np.float32),
        state=forecast["state"][0].astype(np.float32),
        basal_state=forecast["basal_state"][0].astype(np.float32),
        intervention_delta=forecast["intervention_delta"][0].astype(np.float32),
        uncertainty_calibrated=np.asarray(False, dtype=np.bool_),
    )
    artifact_names = (
        "runtime/transition_model.py",
        "runtime/inference.py",
        "runtime/model.safetensors",
        "runtime/model-config.json",
        "runtime/runtime-reference.npz",
        "action-features.npz",
        "hepg2-control-context.npz",
        "hepg2-brca1-forecast.npz",
    )
    artifacts = {
        name: {
            "bytes": (output / name).stat().st_size,
            "sha256": sha256_file(output / name),
        }
        for name in artifact_names
    }
    manifest: dict[str, object] = {
        "schema": "slp.minimal-control-portable-target-free-demo/v1",
        "status": "experimental-unvalidated-nonadvanced-checkpoint",
        "checkpointAdvancementPassed": False,
        "organism": {"ncbiTaxon": 9606, "species": "Homo sapiens"},
        "action": {"entityId": BRCA1, "namespace": "Ensembl-gene"},
        "context": {
            "id": str(hepg2["context_ids"][0]),
            "sourceSha256": HEPG2_SHA256,
            "perturbedExpressionRowsRead": 0,
            "controlValueSpace": str(hepg2["context_value_space"].item()),
            "controlTokensPresent": int(supported.sum()),
        },
        "forecast": {
            "queries": len(query_ids),
            "supportedQueries": int(supported.sum()),
            "unsupportedQueries": int((~supported).sum()),
            "unsupportedQueryIds": query_ids[~supported].tolist(),
            "controlMean": "caller-supplied all-zero standardized control baseline",
            "uncertaintyCalibrated": False,
            "targetOutcomesRead": False,
            "interpretation": (
                "Functional target-free inference demonstration only; no HepG2 "
                "perturbed measurements were loaded or used for validation."
            ),
        },
        "decoderAmplitude": {
            "contextIndexed": False,
            "shape": list(reference["delta_amplitude"].shape),
            "provenance": str(reference["delta_amplitude_formula"].item()),
            "sourceContextsOnly": True,
        },
        "verification": {
            "knownContextRecords": len(chosen),
            "knownContextSourceReloadBitExact": True,
            "knownContextFrozenGpuMaxAbsError": frozen_max_abs,
            "emptyActionIdentity": empty_identity,
            "queryChunkMaxAbsError": chunk_max_abs,
            "orderedQueryIdsExact": True,
            "fixedContextMaskExact": True,
        },
        "inputs": {str(path): expected for path, expected in inputs.items()},
        "artifacts": artifacts,
    }
    write_json(output / "manifest.json", manifest)
    manifest["artifacts"]["manifest.json"] = {
        "bytes": (output / "manifest.json").stat().st_size,
        "sha256": sha256_file(output / "manifest.json"),
    }
    # The manifest hash cannot self-reference; report it on stdout and to caller.
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", required=True)
    result.add_argument("--development", required=True)
    result.add_argument("--static-features", required=True)
    result.add_argument("--hepg2-control", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    manifest = run(args)
    print(
        json.dumps(
            {
                "event": "portable-target-free-forecast-complete",
                "output": args.output,
                "manifestSha256": manifest["artifacts"]["manifest.json"]["sha256"],
                "supportedQueries": manifest["forecast"]["supportedQueries"],
                "unsupportedQueries": manifest["forecast"]["unsupportedQueries"],
                "verification": manifest["verification"],
            }
        )
    )


if __name__ == "__main__":
    main()
