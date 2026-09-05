#!/usr/bin/env python3
"""Generate target-free HepG2 forecasts from the failed physical/state128 model."""

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
INFERENCE_SOURCE = ROOT / "modules/slp-1-1-control-transition-v2/inference.py"
EXPECTED = {
    "checkpoint": "b1e55f2bcc8a29b6b2467a92ebedfdc1cc80ff8c343a6ab36916d638b9c48cf3",
    "reference": "a9f3fd2679b5a52e20dddddd427d8664b2c226f2db91bdae1e44a63e66568562",
    "candidate_report": "49333ade99f04d96e9d4c4ccc2fc01c002170b38f02d10f88fdc8559d274203d",
    "candidate_predictions": "501384b600c5f90fbe6ea22918777288f048091e71377ce8963cda6bd105039e",
    "model_source": "fdb4555bd0f7c0a0786539da67048f6985f4ec2f36ef7aa45bd22c7c6bfbb2ef",
    "features": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "control": "382626401ee38e8d5084ac9f86ffc44bd10408826fb85a94ede8eb908cdf5b27",
    "roster": "89e1819f22568fb9d35b31e84c338e27ea1f13c18c9de18fda266f83ff0e78e0",
    "fixed_context": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "inference_source": "da120d2dd8655d6cf90c684e5dbaa6a6aedd42bfefc1090f8bab121de6cd0d1b",
}


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


def validate_roster(roster: dict[str, np.ndarray]) -> None:
    required = {
        "population_ids",
        "source_construct_ids",
        "action_ids",
        "query_ids",
        "fitting_gene_seen",
        "source_query_measured",
        "context_control_query_observed",
    }
    if required - roster.keys():
        raise ValueError("forecast roster fields missing")
    if (
        len(roster["population_ids"]) != 2544
        or roster["action_ids"].shape != (2544,)
        or roster["source_construct_ids"].shape != (2544,)
        or len(set(roster["population_ids"].tolist())) != 2544
        or roster["query_ids"].shape != (7036,)
        or len(set(roster["query_ids"].tolist())) != 7036
        or roster["fitting_gene_seen"].shape != (2544,)
        or roster["source_query_measured"].shape != (7036,)
        or roster["context_control_query_observed"].shape != (7036,)
    ):
        raise ValueError("forecast roster shape or identity contract mismatch")
    if any(
        array.dtype != np.bool_
        for array in (
            roster["fitting_gene_seen"],
            roster["source_query_measured"],
            roster["context_control_query_observed"],
        )
    ):
        raise ValueError("forecast roster masks must be Boolean")


def selected_npz(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load only named arrays; compressed outcome arrays remain unopened."""
    with np.load(path, allow_pickle=False) as archive:
        missing = set(names) - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        return {name: archive[name] for name in names}


def import_runtime(path: Path):
    sys.path.insert(0, str(path))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "hepg2_physical_state128_runtime", path / "inference.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load portable inference runtime")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def run(args: argparse.Namespace) -> dict[str, object]:
    candidate = Path(args.candidate) / "model"
    paths = {
        "checkpoint": candidate / "model.safetensors",
        "reference": candidate / "reference.npz",
        "candidate_report": candidate / "report.json",
        "candidate_predictions": candidate / "development-predictions.npz",
        "model_source": candidate / "source/control_transition_model.py",
        "features": Path(args.features),
        "control": Path(args.control),
        "roster": Path(args.roster),
        "fixed_context": Path(args.fixed_context),
        "inference_source": INFERENCE_SOURCE,
    }
    for label, path in paths.items():
        actual = sha256_file(path)
        if actual != EXPECTED[label]:
            raise ValueError(f"{label} SHA-256 mismatch: {actual}")
    with paths["candidate_report"].open(encoding="utf-8") as stream:
        candidate_report = json.load(stream)
    if candidate_report.get("advancement", {}).get("passed") is not False:
        raise ValueError("candidate must retain failed development status")

    roster = selected_npz(
        paths["roster"],
        (
            "schema",
            "population_ids",
            "source_construct_ids",
            "source_transcript_labels",
            "action_ids",
            "fitting_gene_seen",
            "query_ids",
            "source_query_measured",
            "context_control_query_observed",
        ),
    )
    validate_roster(roster)
    control = selected_npz(
        paths["control"],
        (
            "context_ids",
            "query_ids",
            "context_basal_expression",
            "context_basal_observed",
            "context_value_space",
            "perturbed_expression_rows_read",
        ),
    )
    if (
        int(control["perturbed_expression_rows_read"]) != 0
        or control["context_basal_expression"].shape != (1, 7036)
        or control["context_basal_observed"].shape != (1, 7036)
        or not np.array_equal(control["query_ids"], roster["query_ids"])
        or not np.array_equal(
            control["context_basal_observed"][0],
            roster["context_control_query_observed"],
        )
    ):
        raise ValueError("HepG2 control-only descriptor contract mismatch")
    fixed = selected_npz(
        paths["fixed_context"],
        (
            "query_ids",
            "basal_control",
            "context_basal_expression",
            "context_basal_observed",
        ),
    )
    if not np.array_equal(fixed["query_ids"], roster["query_ids"]):
        raise ValueError("training-control query order differs from forecast roster")
    common_panel = fixed["context_basal_observed"].all(0)
    if (
        int(common_panel.sum()) != 6789
        or not np.array_equal(common_panel, control["context_basal_observed"][0])
    ):
        raise ValueError("fixed control panel differs between source and HepG2")

    reference = selected_npz(
        paths["reference"],
        (
            "feature_mean",
            "feature_std",
            "query_feature_mean",
            "query_feature_std",
            "query_features",
            "delta_amplitude",
            "delta_amplitude_formula",
            "query_ids",
            "context_query_indices",
            "context_values",
            "context_mask",
            "context_value_space",
        ),
    )
    if (
        not np.array_equal(reference["query_ids"], roster["query_ids"])
        or reference["feature_mean"].shape != (1156,)
        or reference["query_features"].shape != (7036, 1188)
        or not common_panel[reference["context_query_indices"]].all()
        or str(reference["context_value_space"].item())
        != str(control["context_value_space"].item())
    ):
        raise ValueError("candidate reference identity or dimension mismatch")
    features = selected_npz(
        paths["features"], ("feature_values", "entity_taxon", "entity_id")
    )
    keys = list(zip(features["entity_taxon"].tolist(), features["entity_id"].tolist()))
    if len(keys) != len(set(keys)) or features["feature_values"].shape != (10231, 1156):
        raise ValueError("physical static feature identity or shape mismatch")
    lookup = {key: index for index, key in enumerate(keys)}
    missing_actions = [
        str(item) for item in roster["action_ids"] if (9606, str(item)) not in lookup
    ]
    if missing_actions:
        raise ValueError(f"physical features missing {len(missing_actions)} roster actions")
    action_features = np.stack(
        [features["feature_values"][lookup[(9606, str(item))]] for item in roster["action_ids"]]
    ).astype(np.float32)
    if not np.isfinite(action_features).all():
        raise ValueError("roster action features contain nonfinite values")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    runtime_dir = output / "runtime"
    source_dir = output / "source"
    runtime_dir.mkdir()
    source_dir.mkdir()
    for source, destination in (
        (paths["model_source"], runtime_dir / "transition_model.py"),
        (INFERENCE_SOURCE, runtime_dir / "inference.py"),
        (paths["checkpoint"], runtime_dir / "model.safetensors"),
        (candidate / "model-config.json", runtime_dir / "model-config.json"),
        (Path(__file__), source_dir / Path(__file__).name),
    ):
        shutil.copyfile(source, destination)
    shutil.copyfile(paths["roster"], output / "forecast-roster.npz")
    np.savez_compressed(
        runtime_dir / "runtime-reference.npz",
        feature_mean=reference["feature_mean"],
        feature_std=reference["feature_std"],
        query_feature_mean=reference["query_feature_mean"],
        query_feature_std=reference["query_feature_std"],
        query_features=reference["query_features"],
        delta_amplitude=reference["delta_amplitude"],
        delta_amplitude_formula=reference["delta_amplitude_formula"],
        query_ids=reference["query_ids"],
        context_query_indices=reference["context_query_indices"],
        context_panel_mask=common_panel,
        context_value_space=reference["context_value_space"],
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable; no fallback permitted")
    runtime_module = import_runtime(runtime_dir)
    runtime = runtime_module.PortableMinimalControl(runtime_dir, device=args.device)
    query_ids = roster["query_ids"]
    zero_control = np.zeros(7036, dtype=np.float32)

    # Source reload uses only frozen predictions and source control descriptors.
    predictions = selected_npz(
        paths["candidate_predictions"],
        ("mean", "action_ids", "context_index"),
    )
    chosen = np.asarray(
        [int(np.flatnonzero(predictions["context_index"] == index)[0]) for index in range(3)],
        dtype=np.int64,
    )
    verify_actions = np.stack(
        [
            features["feature_values"][lookup[(9606, str(item))]]
            for item in predictions["action_ids"][chosen]
        ]
    ).astype(np.float32)
    verify_contexts = predictions["context_index"][chosen]
    portable_known = runtime.predict(
        verify_actions,
        fixed["context_basal_expression"][verify_contexts],
        fixed["context_basal_observed"][verify_contexts],
        fixed["basal_control"][verify_contexts],
        query_ids=query_ids,
    )
    from safetensors.torch import load_file

    with (runtime_dir / "model-config.json").open(encoding="utf-8") as stream:
        direct_model = runtime_module.MinimalControlTransition(
            runtime_module.Config(**json.load(stream))
        ).to(args.device)
    direct_model.load_state_dict(
        load_file(runtime_dir / "model.safetensors", device=args.device)
    )
    direct_model.eval()

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, device=args.device)

    normalized_actions = (
        (verify_actions - reference["feature_mean"]) / reference["feature_std"]
    ).astype(np.float32)
    normalized_queries = (
        (reference["query_features"] - reference["query_feature_mean"])
        / reference["query_feature_std"]
    ).astype(np.float32)
    with torch.no_grad():
        direct_known = direct_model(
            tensor(normalized_actions),
            tensor(normalized_queries),
            tensor(fixed["basal_control"][verify_contexts]),
            tensor(reference["delta_amplitude"]),
            torch.ones((3, 7036), device=args.device),
            tensor(normalized_queries[reference["context_query_indices"]]),
            tensor(reference["context_values"][verify_contexts]),
            tensor(reference["context_mask"][verify_contexts]),
        )["mean"].cpu().numpy()
    source_reload_max_abs = float(np.max(np.abs(portable_known["mean"] - direct_known)))
    if source_reload_max_abs > 2e-6:
        raise RuntimeError("portable known-context means differ from direct source reload")
    frozen_gpu_max_abs = float(
        np.max(np.abs(portable_known["mean"] - predictions["mean"][chosen]))
    )
    if not np.allclose(
        portable_known["mean"], predictions["mean"][chosen], rtol=1e-5, atol=2e-6
    ):
        raise RuntimeError("portable known-context means differ from frozen forecasts")

    empty = runtime.predict(
        np.empty((1, 0, 1156), dtype=np.float32),
        control["context_basal_expression"][0],
        control["context_basal_observed"][0],
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
        raise RuntimeError("empty-action control identity failed")
    one = runtime.predict(
        action_features[:1],
        control["context_basal_expression"][0],
        control["context_basal_observed"][0],
        zero_control,
        query_ids=query_ids,
    )
    chunks = []
    for indices in np.array_split(np.arange(7036, dtype=np.int64), 5):
        chunks.append(
            runtime.predict(
                action_features[:1],
                control["context_basal_expression"][0],
                control["context_basal_observed"][0],
                zero_control,
                query_ids=query_ids,
                query_indices=indices,
            )["mean"]
        )
    chunk_max_abs = float(np.max(np.abs(np.concatenate(chunks, axis=1) - one["mean"])))
    if chunk_max_abs > 2e-6:
        raise RuntimeError("query chunk invariance failed")

    forecast_path = output / "world-forecast.npy"
    forecast = np.lib.format.open_memmap(
        forecast_path, mode="w+", dtype=np.float32, shape=(2544, 7036)
    )
    for start in range(0, 2544, args.batch_size):
        stop = min(start + args.batch_size, 2544)
        result = runtime.predict(
            action_features[start:stop],
            control["context_basal_expression"][0],
            control["context_basal_observed"][0],
            zero_control,
            query_ids=query_ids,
        )
        if result["uncertainty_calibrated"] or "scale" in result:
            raise RuntimeError("target-free runtime emitted an uncertainty claim")
        forecast[start:stop] = result["mean"]
    forecast.flush()
    del forecast
    stored = np.load(forecast_path, mmap_mode="r", allow_pickle=False)
    if stored.shape != (2544, 7036) or stored.dtype != np.float32 or not np.isfinite(stored).all():
        raise RuntimeError("stored world forecast contract failed")

    artifact_paths = {
        "worldForecast": forecast_path,
        "forecastRoster": output / "forecast-roster.npz",
        "runtimeModel": runtime_dir / "transition_model.py",
        "runtimeInference": runtime_dir / "inference.py",
        "runtimeCheckpoint": runtime_dir / "model.safetensors",
        "runtimeConfig": runtime_dir / "model-config.json",
        "runtimeReference": runtime_dir / "runtime-reference.npz",
        "sourceGenerator": source_dir / Path(__file__).name,
    }
    artifacts = {
        label: {
            "path": str(path.relative_to(output)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for label, path in artifact_paths.items()
    }
    manifest: dict[str, object] = {
        "schema": "slp.hepg2-minimal-control-world-forecasts/v1",
        "status": "diagnostic-only-failed-development-candidate-target-free",
        "candidateAdvancementPassed": False,
        "candidate": {
            "architecture": "minimal-control-v2-physical1156-state128-response32-seed731",
            "developmentReportSha256": EXPECTED["candidate_report"],
            "checkpointSha256": EXPECTED["checkpoint"],
        },
        "identity": {
            "populationRecords": 2544,
            "uniqueActionGenes": len(set(roster["action_ids"].tolist())),
            "queries": 7036,
            "order": "exact copied baseline forecast-roster population and query axes",
            "rosterSha256": EXPECTED["roster"],
            "ncbiTaxon": 9606,
            "actionNamespace": "Ensembl-gene",
        },
        "context": {
            "id": str(control["context_ids"][0]),
            "descriptorSha256": EXPECTED["control"],
            "observedControlTokens": int(control["context_basal_observed"].sum()),
            "valueSpace": str(control["context_value_space"].item()),
            "suppliedControlMean": "all-zero standardized control molecular baseline",
        },
        "forecast": {
            "shape": [2544, 7036],
            "dtype": "float32",
            "uncertaintyCalibrated": False,
            "hepg2PerturbedExpressionRowsRead": 0,
            "hepg2OutcomesRead": False,
            "interpretation": "target-free diagnostic forecast; not validation or advancement",
        },
        "decoderAmplitude": {
            "contextIndexed": False,
            "shape": list(reference["delta_amplitude"].shape),
            "provenance": str(reference["delta_amplitude_formula"].item()),
        },
        "verification": {
            "sourceReloadKnownContexts": 3,
            "sourceReloadMaxAbsError": source_reload_max_abs,
            "frozenGpuMaxAbsError": frozen_gpu_max_abs,
            "emptyActionIdentity": empty_identity,
            "queryChunkMaxAbsError": chunk_max_abs,
            "orderedRosterExact": True,
            "fixedControlPanelExact": True,
            "allActionFeaturesPresent": True,
        },
        "inputs": {
            label: {"path": str(paths[label]), "sha256": EXPECTED[label]}
            for label in paths
        },
        "artifacts": artifacts,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate", required=True)
    result.add_argument("--features", required=True)
    result.add_argument("--control", required=True)
    result.add_argument("--roster", required=True)
    result.add_argument("--fixed-context", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--batch-size", type=int, default=256)
    return result


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    manifest = Path(args.output) / "manifest.json"
    print(
        json.dumps(
            {
                "event": "hepg2-world-forecasts-complete",
                "output": args.output,
                "manifestSha256": sha256_file(manifest),
                "forecastSha256": report["artifacts"]["worldForecast"]["sha256"],
                "verification": report["verification"],
            }
        )
    )


if __name__ == "__main__":
    main()
