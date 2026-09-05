#!/usr/bin/env python3
"""Fitting-only route ablation for one frozen gene-state checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def route_ablation_states(
    encoded: dict[str, torch.Tensor], route: str,
) -> dict[str, torch.Tensor]:
    """Replace one learned intervention route by its exact basal/null state."""
    result = dict(encoded)
    if route == "full":
        return result
    if route == "local_off":
        result["local_delta"] = torch.zeros_like(encoded["local_delta"])
        result["local_state"] = encoded["basal_node_state"]
        return result
    if route == "global_off":
        result["global_delta"] = torch.zeros_like(encoded["global_delta"])
        result["global_state"] = encoded["global_basal_state"]
        return result
    raise ValueError("route must be full, local_off, or global_off")


def gaussian_nll(
    prediction: np.ndarray, target: np.ndarray, observed: np.ndarray, scale: np.ndarray,
) -> float:
    safe_target = np.where(observed, target, 0.0)
    residual = np.where(observed, (prediction - safe_target) / scale, 0.0)
    entries = np.where(
        observed,
        0.5 * residual**2 + np.log(scale) + 0.5 * np.log(2 * np.pi),
        0.0,
    )
    return float(np.mean(entries.sum(1) / observed.sum(1)))


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA executor unavailable; no fallback")
    if args.output.exists():
        raise FileExistsError("immutable diagnostic output exists")
    for path, expected in (
        (args.data, args.data_sha256), (args.graph, args.graph_sha256),
        (args.model, args.model_sha256), (args.reference, args.reference_sha256),
        (args.core, args.core_sha256), (args.runner, args.runner_sha256),
        (args.config, args.config_sha256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"digest mismatch: {path}")
    core = load_source(args.core, "gene_state_route_audit_core")
    runner = load_source(args.runner, "gene_state_route_audit_runner")
    with (
        np.load(args.data, allow_pickle=False) as source,
        np.load(args.graph, allow_pickle=False) as graph_source,
        np.load(args.reference, allow_pickle=False) as reference_source,
    ):
        data = {key: source[key] for key in source.files}
        graph = {key: graph_source[key] for key in graph_source.files}
        reference = {key: reference_source[key] for key in reference_source.files}
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["static_features"] != 610:
        raise ValueError("diagnostic requires the frozen response32 checkpoint")
    train = np.asarray(data["split_train"], dtype=np.int64)
    selected = np.concatenate([
        train[data["context_index"][train] == context][:128]
        for context in range(len(data["context_ids"]))
    ])
    if len(selected) != 128 * len(data["context_ids"]) or not np.all(
        np.isin(selected, train)
    ):
        raise ValueError("each context requires 128 fitting rows")
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "selected-fitting-rows.npz",
        source_row_index=selected,
        record_ids=data["record_ids"][selected],
        action_ids=data["action_ids"][selected],
        context_index=data["context_index"][selected],
    )

    started = time.monotonic()
    device = torch.device("cuda")
    model = core.GeneStateCore(core.Config(**config)).to(device)
    model.load_state_dict(load_file(str(args.model)))
    model.eval()
    features = torch.as_tensor(
        runner.graph_node_features(graph, "response32"), device=device,
    )
    adjacency = runner.adjacency_from_arrays(graph).to(device)
    query = torch.as_tensor(reference["query_node_index"], device=device)
    basal = torch.as_tensor(reference["basal_values"], device=device)
    basal_observed = torch.as_tensor(reference["basal_observed"], device=device)
    amplitude = torch.as_tensor(reference["delta_amplitude"], device=device)
    action_index = torch.as_tensor(graph["action_node_index"], device=device)
    predictions = {name: [] for name in ("full", "local_off", "global_off")}
    for start in range(0, len(selected), 64):
        rows = selected[start:start + 64]
        row_tensor = torch.as_tensor(rows, device=device)
        context = torch.as_tensor(data["context_index"][rows], device=device)
        strength = runner.dense_action_strength(action_index[row_tensor], len(features))
        encoded = model.encode(
            features, basal[context], basal_observed[context], strength, adjacency,
        )
        control = torch.as_tensor(data["basal_control"][data["context_index"][rows]], device=device)
        for route, values in predictions.items():
            values.append(
                model.observe(
                    route_ablation_states(encoded, route), query, control, amplitude,
                )["mean"].cpu().numpy(),
            )
    predictions = {key: np.concatenate(value) for key, value in predictions.items()}
    scales = runner.fixed_exposure_scales(
        reference["mean_biological_variance"], reference["mean_sampling_variance"],
        data["context_index"][selected], data["num_cells_filtered"][selected],
    )
    target = data["targets"][selected]
    observed = data["observed"][selected]
    metrics = {}
    for context, name in enumerate(data["context_ids"].astype(str)):
        keep = data["context_index"][selected] == context
        full = predictions["full"][keep]
        metrics[name] = {
            route: {
                "fittingRowMeanGaussianNll": gaussian_nll(
                    prediction[keep], target[keep], observed[keep], scales[keep],
                ),
                "rmsPredictionChangeFromFull": float(np.sqrt(np.mean((prediction[keep] - full) ** 2))),
            }
            for route, prediction in predictions.items()
        }
    report = {
        "schema": "slp.source3-gene-state-fitting-route-ablation/v1",
        "scope": "observational route ablation within one fixed model; not checkpoint selection or biological mechanism evidence",
        "selection": "first 128 source-order fitting rows per context; validation rows excluded",
        "queries": len(data["query_ids"]),
        "rows": len(selected),
        "metrics": metrics,
        "seconds": time.monotonic() - started,
        "inputs": {
            str(path): sha256(path)
            for path in (
                args.data, args.graph, args.model, args.reference,
                args.core, args.runner, args.config,
            )
        },
        "selectedRowsSha256": sha256(args.output / "selected-fitting-rows.npz"),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"reportSha256": sha256(args.output / "report.json"), **metrics}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "graph", "model", "reference", "core", "runner"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
        parser.add_argument(f"--{name.replace('_', '-')}-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
