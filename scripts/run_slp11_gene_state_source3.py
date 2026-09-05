#!/usr/bin/env python3
"""Profile or train the fixed three-context gene-state development pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "modules/slp-1-1-gene-state-v1/gene_state.py"
BASELINES = ROOT / "modules/slp-1-1-world-transition-v1/transition_baselines.py"
SCORING = ROOT / "modules/slp-1-1-world-transition-v1/context_transfer_scoring.py"
DATA_SHA256 = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
V2_REFERENCE_SHA256 = "a9f3fd2679b5a52e20dddddd427d8664b2c226f2db91bdae1e44a63e66568562"
V2_EXPOSURE_SHA256 = "9cf5f4a5352dccaa7cb3d6c84e2123b16b190220a1ef9e03c933a887be6c81dd"
SEED = 731
SETTINGS = {
    "epochs": 32,
    "patience_evaluations": 10,
    "evaluate_every": 1,
    "learning_rate": 0.001,
    "weight_decay": 0.1,
    "max_seconds": 1800,
    "gradient_clip": 1.0,
    "scale_floor": 0.05,
}


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


def write_json(path: Path, value: object) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(entry) for entry in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def basal_node_inputs(
    context_values: np.ndarray,
    context_observed: np.ndarray,
    query_node_index: np.ndarray,
    nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map and normalize control-only expression onto the fixed node universe."""
    values = np.asarray(context_values, dtype=np.float64)
    observed = np.asarray(context_observed)
    query_node_index = np.asarray(query_node_index, dtype=np.int64)
    if (
        values.ndim != 2
        or observed.shape != values.shape
        or observed.dtype != np.bool_
        or query_node_index.shape != (values.shape[1],)
        or np.any(query_node_index < 0)
        or np.any(query_node_index >= nodes)
        or len(set(query_node_index.tolist())) != len(query_node_index)
    ):
        raise ValueError("control/query node axes do not align")
    safe = np.where(observed, values, 0.0)
    if not np.isfinite(safe).all() or np.any(observed.sum(axis=1) == 0):
        raise ValueError("observed controls must be finite and nonempty")
    mean = np.asarray([values[c, observed[c]].mean() for c in range(len(values))])
    scale = np.asarray([values[c, observed[c]].std() for c in range(len(values))])
    scale = np.maximum(scale, 1e-5)
    normalized = np.where(observed, (values - mean[:, None]) / scale[:, None], 0.0)
    node_values = np.zeros((len(values), nodes), dtype=np.float32)
    node_observed = np.zeros((len(values), nodes), dtype=np.bool_)
    node_values[:, query_node_index] = normalized.astype(np.float32)
    node_observed[:, query_node_index] = observed
    return node_values, node_observed, np.stack((mean, scale), axis=1)


def fixed_exposure_scales(
    biological_variance: np.ndarray,
    sampling_variance: np.ndarray,
    context_index: np.ndarray,
    num_cells: np.ndarray,
    floor: float = 0.05,
) -> np.ndarray:
    biological = np.asarray(biological_variance, dtype=np.float64)
    sampling = np.asarray(sampling_variance, dtype=np.float64)
    context = np.asarray(context_index, dtype=np.int64)
    count = np.asarray(num_cells, dtype=np.float64)
    if (
        biological.shape != sampling.shape
        or biological.ndim != 2
        or context.shape != count.shape
        or np.any(context < 0)
        or np.any(context >= len(biological))
        or not np.isfinite(biological).all()
        or not np.isfinite(sampling).all()
        or np.any(biological < 0)
        or np.any(sampling < 0)
        or not np.isfinite(count).all()
        or np.any(count <= 0)
        or not np.isfinite(floor)
        or floor <= 0
    ):
        raise ValueError("invalid frozen exposure components")
    variance = biological[context] + sampling[context] / count[:, None]
    return np.maximum(np.sqrt(variance), floor).astype(np.float32)


def dense_action_strength(action_node_index: torch.Tensor, nodes: int) -> torch.Tensor:
    """Map external intervention identities to unit presence on fixed nodes."""
    if action_node_index.ndim != 1 or action_node_index.dtype != torch.long:
        raise ValueError("action node index must be int64 [B]")
    if (action_node_index < 0).any() or (action_node_index >= nodes).any():
        raise ValueError("action node index outside graph")
    strength = torch.zeros(len(action_node_index), nodes, device=action_node_index.device)
    strength[torch.arange(len(action_node_index), device=action_node_index.device), action_node_index] = 1.0
    return strength


def gaussian_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if (
        prediction.shape != target.shape
        or observed.shape != target.shape
        or observed.dtype != torch.bool
        or scale.shape != target.shape
        or not torch.isfinite(prediction).all()
        or not torch.isfinite(torch.where(observed, target, 0.0)).all()
        or not torch.isfinite(scale).all()
        or not (scale > 0).all()
        or not (observed.sum(1) > 0).all()
    ):
        raise ValueError("invalid fixed-scale Gaussian loss inputs")
    residual = torch.where(observed, (prediction - torch.where(observed, target, 0.0)) / scale, 0.0)
    per_entry = 0.5 * residual.square() + torch.log(scale) + 0.5 * math.log(2 * math.pi)
    return (torch.where(observed, per_entry, 0.0).sum(1) / observed.sum(1)).mean()


def selection_gene_macro_nll(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
    action_ids: np.ndarray,
    context_index: np.ndarray,
) -> float:
    """Equal-context, equal-gene Gaussian NLL without computing other metrics."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(observed)
    scale = np.asarray(scale, dtype=np.float64)
    action_ids = np.asarray(action_ids, dtype=str)
    context_index = np.asarray(context_index, dtype=np.int64)
    if (
        target.shape != prediction.shape
        or observed.shape != prediction.shape
        or observed.dtype != np.bool_
        or scale.shape != prediction.shape
        or action_ids.shape != (len(prediction),)
        or context_index.shape != (len(prediction),)
        or np.any(observed.sum(1) == 0)
        or not np.isfinite(prediction[observed]).all()
        or not np.isfinite(target[observed]).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("invalid validation selection arrays")
    residual = np.where(observed, (prediction - target) / scale, 0.0)
    entries = np.where(
        observed,
        0.5 * residual**2 + np.log(scale) + 0.5 * np.log(2 * np.pi),
        0.0,
    )
    record_nll = entries.sum(1) / observed.sum(1)
    contexts = []
    for context in sorted(set(context_index.tolist())):
        selected = context_index == context
        genes = [
            float(np.mean(record_nll[selected & (action_ids == gene)]))
            for gene in sorted(set(action_ids[selected].tolist()))
        ]
        contexts.append(float(np.mean(genes)))
    return float(np.mean(contexts))


def adjacency_from_arrays(graph: dict[str, np.ndarray]) -> torch.Tensor:
    nodes = len(graph["node_ids"])
    indptr = np.asarray(graph["adjacency_indptr"], dtype=np.int64)
    indices = np.asarray(graph["adjacency_indices"], dtype=np.int64)
    weights = np.asarray(graph["adjacency_weights"], dtype=np.float32)
    if (
        indptr.shape != (nodes + 1,)
        or indptr[0] != 0
        or indptr[-1] != len(indices)
        or weights.shape != indices.shape
        or np.any(np.diff(indptr) < 0)
        or np.any(indices < 0)
        or np.any(indices >= nodes)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("invalid frozen CSR graph arrays")
    rows = np.repeat(np.arange(nodes), np.diff(indptr))
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(
            torch.from_numpy(np.stack((rows, indices))),
            torch.from_numpy(weights),
            (nodes, nodes),
        ).coalesce()


def graph_node_features(
    graph: dict[str, np.ndarray], feature_kind: str,
) -> np.ndarray:
    """Select and verify one frozen node-feature contract."""
    nodes = len(graph["node_ids"])
    static = np.asarray(graph["static_features"])
    static_observed = np.asarray(graph["static_feature_observed"])
    if (
        static.shape != (nodes, 577)
        or static_observed.shape != (nodes,)
        or static_observed.dtype != np.bool_
        or not np.isfinite(static).all()
        or np.count_nonzero(static[~static_observed])
    ):
        raise ValueError("graph static-feature missingness contract mismatch")
    if feature_kind == "static577":
        return static
    if feature_kind != "response32":
        raise ValueError("unknown node feature kind")
    combined = np.asarray(graph.get("node_features"))
    response_observed = np.asarray(graph.get("response_query_feature_observed"))
    if (
        combined.shape != (nodes, 610)
        or response_observed.shape != (nodes,)
        or response_observed.dtype != np.bool_
        or not np.isfinite(combined).all()
        or not np.array_equal(combined[:, :577], static)
        or not np.array_equal(combined[:, 609], response_observed.astype(combined.dtype))
        or np.count_nonzero(combined[~response_observed, 577:])
        or not np.all(response_observed[np.asarray(graph["query_node_index"], dtype=np.int64)])
    ):
        raise ValueError("graph response32 node-feature contract mismatch")
    return combined


def profile_actual_graph_cuda(
    core, graph: dict[str, np.ndarray], batch: int, feature_kind: str, repeats: int = 3,
):
    """Profile the frozen graph shape with synthetic values and no outcome access."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no profile fallback")
    if batch not in (32, 64) or repeats <= 0:
        raise ValueError("profile batch must be 32 or 64 and repeats positive")
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    nodes = len(graph["node_ids"])
    query_count = len(graph["query_node_index"])
    features = graph_node_features(graph, feature_kind)
    model = core.GeneStateCore(
        core.Config(features.shape[1], state=16, transition_hidden=64, decoder_hidden=32),
    ).to(device)
    static = torch.as_tensor(features, device=device)
    basal = torch.randn(batch, nodes, device=device)
    observed = torch.ones(batch, nodes, dtype=torch.bool, device=device)
    action_index = torch.arange(batch, device=device) % nodes
    strength = dense_action_strength(action_index, nodes)
    adjacency = adjacency_from_arrays(graph).to(device)
    query = torch.as_tensor(graph["query_node_index"], device=device)
    control = torch.zeros(batch, query_count, device=device)
    amplitude = torch.ones(query_count, device=device)
    torch.cuda.reset_peak_memory_stats()
    training_times, forward_times = [], []
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        encoded = model.encode(static, basal, observed, strength, adjacency)
        prediction = model.observe(encoded, query, control, amplitude)["mean"]
        prediction.square().mean().backward()
        torch.cuda.synchronize()
        training_times.append(time.perf_counter() - started)
    model.eval()
    with torch.no_grad():
        for _ in range(repeats):
            started = time.perf_counter()
            encoded = model.encode(static, basal, observed, strength, adjacency)
            model.observe(encoded, query, control, amplitude)["mean"]
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - started)
    train_seconds = float(np.mean(training_times))
    forward_seconds = float(np.mean(forward_times))
    return {
        "nodes": nodes,
        "edges": len(graph["adjacency_weights"]),
        "batch": batch,
        "staticFeatures": features.shape[1],
        "queries": query_count,
        "repeats": repeats,
        "meanForwardBackwardSeconds": train_seconds,
        "meanForwardSeconds": forward_seconds,
        "trainingExamplesPerSecond": batch / train_seconds,
        "forwardExamplesPerSecond": batch / forward_seconds,
        "peakAllocatedBytes": int(torch.cuda.max_memory_allocated()),
        "peakReservedBytes": int(torch.cuda.max_memory_reserved()),
    }


def validate_profile_choice(
    selected: dict[str, object],
    batch32: dict[str, object],
    fitting_rows: int,
    validation_rows: int,
) -> dict[str, float]:
    """Apply the pre-outcome B64 choice and fixed 32-epoch runtime rules."""
    required = (
        "batch", "nodes", "queries", "edges", "staticFeatures", "peakReservedBytes",
        "trainingExamplesPerSecond", "meanForwardBackwardSeconds", "meanForwardSeconds",
    )
    if any(name not in selected for name in required) or any(name not in batch32 for name in required):
        raise ValueError("profile report lacks required actual-shape measurements")
    if (
        int(selected["batch"]) != 64
        or int(batch32["batch"]) != 32
        or any(
            int(selected[name]) != int(batch32[name])
            for name in ("nodes", "queries", "edges", "staticFeatures")
        )
    ):
        raise ValueError("profiles must compare B64 and B32 on one exact graph shape")
    throughput_ratio = float(selected["trainingExamplesPerSecond"]) / float(
        batch32["trainingExamplesPerSecond"],
    )
    reserved_gib = float(selected["peakReservedBytes"]) / 2**30
    if reserved_gib > 9.0 or throughput_ratio < 1.15:
        raise ValueError("B64 fails the frozen operational batch-selection rule")
    train_per_epoch = math.ceil(fitting_rows / 64) * float(selected["meanForwardBackwardSeconds"])
    validation_per_epoch = math.ceil(validation_rows / 64) * float(selected["meanForwardSeconds"])
    schedule = SETTINGS["epochs"] * (train_per_epoch + validation_per_epoch)
    # One final selected-checkpoint forecast, one target-free reload forecast,
    # and 30 seconds for checkpoint/source reconstruction.
    projected_total = schedule + 2 * validation_per_epoch + 30.0
    if projected_total > SETTINGS["max_seconds"]:
        raise ValueError("fixed 32-epoch schedule is not projected inside 1800 seconds")
    return {
        "selectedBatch": 64,
        "reservedGiB": reserved_gib,
        "throughputRatioVsB32": throughput_ratio,
        "projectedTrainAndValidationSeconds": schedule,
        "projectedTotalSeconds": projected_total,
    }


def load_inputs(args, core):
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("source-three development digest mismatch")
    if sha256(args.graph) != args.graph_sha256:
        raise ValueError("gene-state graph adapter digest mismatch")
    if sha256(args.v2_reference) != V2_REFERENCE_SHA256:
        raise ValueError("frozen v2 reference digest mismatch")
    if sha256(args.v2_exposure) != V2_EXPOSURE_SHA256:
        raise ValueError("frozen v2 exposure digest mismatch")
    with (
        np.load(args.data, allow_pickle=False) as source,
        np.load(args.graph, allow_pickle=False) as graph_source,
        np.load(args.v2_reference, allow_pickle=False) as reference_source,
        np.load(args.v2_exposure, allow_pickle=False) as exposure_source,
    ):
        data = {key: source[key] for key in source.files}
        graph = {key: graph_source[key] for key in graph_source.files}
        reference = {key: reference_source[key] for key in reference_source.files}
        exposure = {key: exposure_source[key] for key in exposure_source.files}
    if len(data["split_test"]):
        raise ValueError("development input must contain no test rows")
    if not np.array_equal(reference["query_ids"], data["query_ids"]):
        raise ValueError("v2 reference query roster mismatch")
    if graph["action_node_index"].shape != (len(data["action_ids"]),):
        raise ValueError("graph action mapping shape mismatch")
    if graph["query_node_index"].shape != (len(data["query_ids"]),):
        raise ValueError("graph query mapping shape mismatch")
    node_ids = graph["node_ids"].astype(str)
    if (
        not np.array_equal(node_ids[graph["action_node_index"]], data["action_ids"].astype(str))
        or not np.array_equal(node_ids[graph["query_node_index"]], data["query_ids"].astype(str))
    ):
        raise ValueError("graph stable-identity mappings disagree with source rosters")
    train, validation = data["split_train"], data["split_validation"]
    if set(data["action_ids"][train]) & set(data["action_ids"][validation]):
        raise ValueError("development intervention split overlap")
    node_features = graph_node_features(graph, args.node_feature_kind)
    if (
        data["targets"].shape != data["observed"].shape
        or not data["observed"].all()
        or not np.isfinite(data["targets"]).all()
        or not np.all(data["context_basal_observed"].sum(1) == 6789)
    ):
        raise ValueError("fixed complete-query development contract mismatch")
    nodes = len(graph["node_ids"])
    basal_values, basal_observed, basal_stats = basal_node_inputs(
        data["context_basal_expression"],
        data["context_basal_observed"],
        graph["query_node_index"],
        nodes,
    )
    scales = fixed_exposure_scales(
        exposure["mean_biological_variance"],
        exposure["mean_sampling_variance"],
        data["context_index"],
        data["num_cells_filtered"],
        SETTINGS["scale_floor"],
    )
    adjacency = adjacency_from_arrays(graph)
    return (
        data, graph, reference, exposure, basal_values, basal_observed,
        basal_stats, scales, adjacency, node_features,
    )


def gene_metrics(baselines, prediction, scale, data, rows, reference):
    keys = [(9606, str(data["action_ids"][row])) for row in rows]
    groups: dict[tuple[int, str], list[int]] = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    reports = [
        baselines.evaluate(
            prediction[group], data["targets"][rows[group]], data["observed"][rows[group]],
            reference, scale[group], value_space=str(data["target_value_space"]),
        )
        for group in (np.asarray(value) for value in groups.values())
    ]
    report = baselines.evaluate(
        prediction, data["targets"][rows], data["observed"][rows], reference, scale,
        value_space=str(data["target_value_space"]),
    )
    for metric in ("nll", "mse", "profile_pearson_mean", "profile_centroid_adjusted_pearson_mean"):
        values = [
            item[metric] for item in reports
            if item[metric] is not None and np.isfinite(item[metric])
        ]
        report["gene_macro_" + metric] = float(np.mean(values)) if values else None
    report["intervention_genes"] = len(groups)
    return report


@torch.no_grad()
def reload_target_free_forecast(
    core_path: Path,
    config: dict[str, object],
    checkpoint_path: Path,
    graph_path: Path,
    reference_path: Path,
    roster_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reload a forecast using graph/reference/identity artifacts and no outcomes."""
    frozen = load_source(core_path, "gene_state_source3_frozen_reload")
    with np.load(graph_path, allow_pickle=False) as source:
        graph = {key: source[key] for key in source.files}
    with np.load(reference_path, allow_pickle=False) as source:
        reference = {key: source[key] for key in source.files}
    with np.load(roster_path, allow_pickle=False) as source:
        action_ids = source["action_ids"].astype(str)
        context = source["context_index"].astype(np.int64)
    if not np.array_equal(reference["node_ids"], graph["node_ids"]):
        raise ValueError("reference and graph node rosters disagree")
    node_lookup = {gene: row for row, gene in enumerate(graph["node_ids"].astype(str))}
    if any(gene not in node_lookup for gene in action_ids):
        raise ValueError("forecast action absent from frozen node roster")
    action_index = np.asarray([node_lookup[gene] for gene in action_ids], dtype=np.int64)
    device = torch.device("cuda")
    model = frozen.GeneStateCore(frozen.Config(**config)).to(device)
    model.load_state_dict(load_file(str(checkpoint_path)))
    model.eval()
    feature_kind = {577: "static577", 610: "response32"}.get(int(config["static_features"]))
    if feature_kind is None:
        raise ValueError("unsupported frozen node-feature width")
    static = torch.as_tensor(graph_node_features(graph, feature_kind), device=device)
    query_index = torch.as_tensor(reference["query_node_index"], device=device)
    basal = torch.as_tensor(reference["basal_values"], device=device)
    basal_observed = torch.as_tensor(reference["basal_observed"], device=device)
    control = torch.as_tensor(reference["control_mean"], device=device)
    amplitude = torch.as_tensor(reference["delta_amplitude"], device=device)
    adjacency = adjacency_from_arrays(graph).to(device)
    predictions = []
    for start in range(0, len(action_ids), batch_size):
        stop = min(start + batch_size, len(action_ids))
        local_context = torch.as_tensor(context[start:stop], device=device)
        local_action = torch.as_tensor(action_index[start:stop], device=device)
        strength = dense_action_strength(local_action, len(static))
        encoded = model.encode(
            static, basal[local_context], basal_observed[local_context], strength, adjacency,
        )
        predictions.append(
            model.observe(encoded, query_index, control[local_context], amplitude)["mean"].cpu().numpy(),
        )
    context_rows = torch.arange(len(control), device=device)
    empty_strength = torch.zeros(len(control), len(static), device=device)
    empty_encoded = model.encode(static, basal, basal_observed, empty_strength, adjacency)
    empty = model.observe(empty_encoded, query_index, control, amplitude)
    identity = {
        "meanBitExact": torch.equal(empty["mean"], control),
        "deltaNonzero": int(torch.count_nonzero(empty["delta"])),
        "globalDeltaNonzero": int(torch.count_nonzero(empty_encoded["global_delta"])),
        "localDeltaNonzero": int(torch.count_nonzero(empty_encoded["local_delta"])),
        "contexts": len(context_rows),
    }
    return np.concatenate(predictions), identity


def run(args) -> None:
    core = load_source(CORE, "gene_state_source3_core")
    if args.mode == "profile":
        if args.output.exists():
            raise FileExistsError("immutable profile output exists")
        args.output.mkdir(parents=True)
        if args.graph is not None:
            if args.graph_sha256 is None or sha256(args.graph) != args.graph_sha256:
                raise ValueError("actual graph profile digest mismatch")
            with np.load(args.graph, allow_pickle=False) as source:
                graph = {key: source[key] for key in source.files}
            dimensions = {
                "nodes": len(graph["node_ids"]), "batch": args.batch_size,
                "staticFeatures": graph_node_features(graph, args.node_feature_kind).shape[1],
                "queries": len(graph["query_node_index"]),
                "edges": len(graph["adjacency_weights"]),
            }
            profile_kind = "actual frozen graph/static shape with synthetic basal/action values"
        else:
            graph = None
            dimensions = {"nodes": 24000, "batch": 32, "staticFeatures": 577,
                          "queries": args.profile_query_count}
            profile_kind = "fully synthetic ring graph and values"
        protocol = {
            "schema": "slp.gene-state-synthetic-cuda-profile/v1",
            "source": {"path": str(CORE), "sha256": sha256(CORE)},
            "dimensions": dimensions,
            "profileKind": profile_kind,
            "access": "synthetic tensors only; no molecular outcomes",
        }
        write_json(args.output / "protocol.json", protocol)
        result = (
            profile_actual_graph_cuda(core, graph, args.batch_size, args.node_feature_kind)
            if graph is not None
            else core.profile_synthetic_cuda(query_count=args.profile_query_count)
        )
        write_json(args.output / "profile.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA executor unavailable; no fallback")
    if args.output.exists():
        raise FileExistsError("immutable training output exists")
    baselines = load_source(BASELINES, "gene_state_source3_baselines")
    scoring = load_source(SCORING, "gene_state_source3_scoring")
    (
    data, graph, reference, exposure, basal_values, basal_observed,
        basal_stats, scales, adjacency, node_features,
    ) = load_inputs(args, core)
    if sha256(args.v2_summary) != args.v2_summary_sha256 or sha256(
        args.landscape_report,
    ) != args.landscape_report_sha256:
        raise ValueError("pinned comparator report digest mismatch")
    if sha256(args.base577_report) != args.base577_report_sha256:
        raise ValueError("pinned base577 gene-state report digest mismatch")
    prior = json.loads(args.v2_summary.read_text(encoding="utf-8"))
    landscape = json.loads(args.landscape_report.read_text(encoding="utf-8"))
    base577 = json.loads(args.base577_report.read_text(encoding="utf-8"))
    if (
        sha256(args.selected_profile) != args.selected_profile_sha256
        or sha256(args.batch32_profile) != args.batch32_profile_sha256
    ):
        raise ValueError("actual-graph profile digest mismatch")
    selected_profile = json.loads(args.selected_profile.read_text(encoding="utf-8"))
    batch32_profile = json.loads(args.batch32_profile.read_text(encoding="utf-8"))
    profile_choice = validate_profile_choice(
        selected_profile, batch32_profile, len(data["split_train"]), len(data["split_validation"]),
    )
    if args.batch_size != int(profile_choice["selectedBatch"]):
        raise ValueError("runtime batch disagrees with frozen profile choice")
    config = core.Config(
        static_features=node_features.shape[1], state=16,
        transition_hidden=64, decoder_hidden=32,
    )
    args.output.mkdir(parents=True)
    source_dir = args.output / "source"
    source_dir.mkdir()
    for path in (CORE, BASELINES, SCORING, Path(__file__)):
        shutil.copyfile(path, source_dir / path.name)
    protocol = {
        "schema": "slp.source3-gene-state-development-pilot/v2",
        "hypothesis": (
            "Supplying frozen fitting-derived response32 descriptors to the shared gene-state "
            "encoder improves unseen-intervention molecular landscapes enough to pass the "
            "unchanged mean, ridge, and minimal-control-v2 gates."
            if args.node_feature_kind == "response32"
            else "An explicit two-hop per-gene state plus a global route improves unseen-intervention molecular forecasts across all three source contexts."
        ),
        "advancement": "Every context: gene-macro NLL improves at least .02 nats over fixed mean and full physical1156 ridge; training-centroid-adjusted r is at least .10; NLL and adjusted r do not regress from v2; independently prediction/truth-centered gene-profile r is at least .10 and does not regress from either v2 or full physical1156 ridge.",
        "interpretation": (
            "Controlled response-descriptor gene-state pilot paired to the base577 pilot. The "
            "response32 descriptors are fitting-derived rather than static priors and enter the "
            "shared node encoder, so they affect node, action, and query representations together."
            if args.node_feature_kind == "response32"
            else "Joint gene-state architecture test. Any change cannot be attributed solely to graph propagation because global state width/backbone also differ from v2."
        ),
        "config": asdict(config),
        "settings": {**SETTINGS, "batchSize": args.batch_size},
        "runtimeRule": (
            "At most 32 epochs with validation every epoch and patience10; hard 1800-second "
            "total CUDA-phase cap. If observed timing exhausts the cap, retain only the last "
            "fully evaluated best checkpoint, mark the schedule incomplete, skip unsafe final "
            "GPU verification when necessary, and make advancement impossible."
        ),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in {
                "development": args.data,
                "graph": args.graph,
                "v2Reference": args.v2_reference,
                "v2Exposure": args.v2_exposure,
                "v2Summary": args.v2_summary,
                "landscapeReport": args.landscape_report,
                "base577GeneStateReport": args.base577_report,
                "selectedB64Profile": args.selected_profile,
                "comparisonB32Profile": args.batch32_profile,
            }.items()
        },
        "profileChoice": profile_choice,
        "pairedComparison": (
            "The frozen base577 gene-state result is reported descriptively only and does not "
            "change the advancement rule."
        ),
        "normalization": {
            "nodeFeatures": (
                "exact base577 plus frozen v2 fitting-derived normalized response32 and explicit availability flag"
                if args.node_feature_kind == "response32"
                else "adapter fitting-action mean/SD over feature-covered genes; missing static nodes exact zero"
            ),
            "basal": "per-context mean/SD over 6789 observed fixed-panel control values only",
            "amplitude": str(reference["delta_amplitude_formula"]),
            "uncertainty": "frozen v2 mean-OOF biological variance plus core-control sampling variance / record cell count",
        },
        "accessibleModalities": (
            "base ESM2-8M321 plus frozen GO256; frozen v2 fitting-derived response32 descriptors and availability; fixed STRING experiment-confidence>=700 graph; source controls; source fitting outcomes"
            if args.node_feature_kind == "response32"
            else "base ESM2-8M321 plus frozen GO256; fixed STRING experiment-confidence>=700 graph; source controls; source fitting outcomes"
        ),
        "limitations": (
            "Action strength one is intervention presence, not measured knockdown, efficacy or dose. Response32 is quantitative source-fitting evidence, not a static prior, and affects node/action/query encoding together. No learned IDs, test outcomes, identified dynamics, causal-edge claim or decoder-only attribution."
            if args.node_feature_kind == "response32"
            else "Action strength one is intervention presence, not measured knockdown, efficacy or dose. No response-query basis, learned IDs, test outcomes, identified dynamics, causal-edge claim or isolated graph ablation."
        ),
        "sourceHashes": {path.name: sha256(path) for path in source_dir.iterdir()},
    }
    write_json(args.output / "protocol.json", protocol)

    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    static = torch.as_tensor(node_features, device=device)
    query_index = torch.as_tensor(graph["query_node_index"], device=device)
    action_index = torch.as_tensor(graph["action_node_index"], device=device)
    context_index = torch.as_tensor(data["context_index"], device=device)
    basal_tensor = torch.as_tensor(basal_values, device=device)
    basal_mask = torch.as_tensor(basal_observed, device=device)
    adjacency = adjacency.to(device)
    target = torch.as_tensor(data["targets"], device=device)
    observed = torch.as_tensor(data["observed"], device=device)
    control = torch.as_tensor(data["basal_control"], device=device)
    amplitude = torch.as_tensor(reference["delta_amplitude"], device=device)
    scale = torch.as_tensor(scales, device=device)
    train = np.asarray(data["split_train"], dtype=np.int64)
    validation = np.asarray(data["split_validation"], dtype=np.int64)
    model = core.GeneStateCore(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])
    rng = np.random.default_rng(SEED)

    def forward(rows: np.ndarray):
        row_tensor = torch.as_tensor(rows, device=device)
        contexts = context_index[row_tensor]
        strength = dense_action_strength(action_index[row_tensor], len(static))
        encoded = model.encode(
            static, basal_tensor[contexts], basal_mask[contexts], strength, adjacency,
        )
        result = model.observe(encoded, query_index, control[contexts], amplitude)
        return result, row_tensor

    def predict(rows: np.ndarray) -> np.ndarray:
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(rows), args.batch_size):
                result, _ = forward(rows[start:start + args.batch_size])
                parts.append(result["mean"].cpu().numpy())
        return np.concatenate(parts)

    context_names = [str(item) for item in data["context_ids"]]

    def evaluate_validation(prediction):
        reports, independent = {}, {}
        for context, name in enumerate(context_names):
            positions = np.flatnonzero(data["context_index"][validation] == context)
            rows = validation[positions]
            reports[name] = gene_metrics(
                baselines, prediction[positions], scales[rows], data, rows,
                reference["evaluation_perturbation_centroid"][context],
            )
            profiles = scoring.collapse_gene_profiles(
                prediction[positions], data["targets"][rows], data["observed"][rows],
                data["action_ids"][rows], data["record_ids"][rows],
            )
            independent[name] = scoring.score_gene_profiles(
                profiles, reference["evaluation_perturbation_centroid"][context],
            )
        return reports, independent

    best_score, best_epoch, best_state, best_prediction, stale = float("inf"), 0, None, None, 0
    incomplete_schedule = False
    history = []
    for epoch in range(1, SETTINGS["epochs"] + 1):
        if time.monotonic() - started >= SETTINGS["max_seconds"]:
            incomplete_schedule = True
            break
        model.train()
        losses = []
        order = rng.permutation(train)
        for start in range(0, len(train), args.batch_size):
            if time.monotonic() - started >= SETTINGS["max_seconds"]:
                incomplete_schedule = True
                break
            selection = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            result, row_tensor = forward(selection)
            loss = gaussian_loss(
                result["mean"], target[row_tensor], observed[row_tensor], scale[row_tensor],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_clip"])
            optimizer.step()
            losses.append((float(loss.detach()), len(selection)))
        if incomplete_schedule:
            break
        if epoch % SETTINGS["evaluate_every"]:
            continue
        prediction = predict(validation)
        score = selection_gene_macro_nll(
            prediction,
            data["targets"][validation],
            data["observed"][validation],
            scales[validation],
            data["action_ids"][validation],
            data["context_index"][validation],
        )
        entry = {
            "epoch": epoch,
            "score": score,
            "trainRowMeanNll": sum(value * count for value, count in losses)
            / sum(count for _, count in losses),
            "seconds": time.monotonic() - started,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if score < best_score - 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_prediction = prediction.copy()
        else:
            stale += 1
        if stale >= SETTINGS["patience_evaluations"]:
            break
        if time.monotonic() - started >= SETTINGS["max_seconds"]:
            incomplete_schedule = True
            break
    if best_state is None or best_prediction is None:
        raise RuntimeError("no complete checkpoint inside time cap")
    model.load_state_dict(best_state)
    prediction = best_prediction
    reports, independent = evaluate_validation(prediction)
    decisions = {}
    for name in context_names:
        old = prior["results"][name]
        old_landscape = landscape["results"][name]
        current = reports[name]
        current_independent = independent[name]["primaryIndependentlyCenteredGeneMacroProfilePearson"]
        def at_least(value, threshold):
            return value is not None and np.isfinite(value) and value >= threshold

        def at_most(value, threshold):
            return value is not None and np.isfinite(value) and value <= threshold

        checks = {
            "nllGainVsMeanAtLeast002": old["mean"]["gene_macro_nll"] - current["gene_macro_nll"] >= 0.02,
            "nllGainVsRidgeAtLeast002": old["ridge"]["gene_macro_nll"] - current["gene_macro_nll"] >= 0.02,
            "adjustedRAtLeast010": at_least(
                current["gene_macro_profile_centroid_adjusted_pearson_mean"], 0.10,
            ),
            "nllNoRegressionVsV2": at_most(
                current["gene_macro_nll"], old["world"]["gene_macro_nll"],
            ),
            "adjustedRNoRegressionVsV2": at_least(
                current["gene_macro_profile_centroid_adjusted_pearson_mean"],
                old["world"]["gene_macro_profile_centroid_adjusted_pearson_mean"],
            ),
            "independentRAtLeast010": at_least(current_independent, 0.10),
            "independentRNoRegressionVsRidge": at_least(
                current_independent,
                old_landscape["full_physical_ridge"][
                    "primaryIndependentlyCenteredGeneMacroProfilePearson"
                ],
            ),
            "independentRNoRegressionVsV2": at_least(
                current_independent,
                old_landscape["minimal_control_v2"][
                    "primaryIndependentlyCenteredGeneMacroProfilePearson"
                ],
            ),
        }
        decisions[name] = {"checks": checks, "passed": all(checks.values())}

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_file(best_state, str(args.output / "model.safetensors"))
    write_json(args.output / "model-config.json", asdict(config))
    np.savez_compressed(
        args.output / "reference.npz",
        context_ids=data["context_ids"],
        node_ids=graph["node_ids"],
        query_ids=data["query_ids"],
        query_node_index=graph["query_node_index"],
        basal_values=basal_values,
        basal_observed=basal_observed,
        basal_normalization=basal_stats,
        control_mean=data["basal_control"],
        delta_amplitude=reference["delta_amplitude"],
        mean_biological_variance=exposure["mean_biological_variance"],
        mean_sampling_variance=exposure["mean_sampling_variance"],
    )
    np.savez_compressed(
        args.output / "development-predictions.npz",
        mean=prediction,
        record_ids=data["record_ids"][validation],
        action_ids=data["action_ids"][validation],
        context_index=data["context_index"][validation],
    )
    if sha256(args.graph) != args.graph_sha256:
        raise RuntimeError("graph adapter changed before reload verification")
    projected_reload = math.ceil(len(validation) / args.batch_size) * float(
        selected_profile["meanForwardSeconds"],
    ) + 5.0
    verification_incomplete = time.monotonic() - started + projected_reload > SETTINGS["max_seconds"]
    if verification_incomplete:
        reload_drift = None
        identity = None
    else:
        repeated, identity = reload_target_free_forecast(
            source_dir / CORE.name,
            asdict(config),
            args.output / "model.safetensors",
            args.graph,
            args.output / "reference.npz",
            args.output / "development-predictions.npz",
            args.batch_size,
        )
        reload_drift = float(np.max(np.abs(repeated - prediction)))
        if reload_drift > 1e-6 or not identity["meanBitExact"] or any(
            identity[key] for key in ("deltaNonzero", "globalDeltaNonzero", "localDeltaNonzero")
        ):
            raise RuntimeError("target-free source reload or empty-action identity failed")
    evidence_complete = not incomplete_schedule and not verification_incomplete
    if not evidence_complete:
        for decision in decisions.values():
            decision["checks"]["fixedScheduleAndVerificationComplete"] = False
            decision["passed"] = False
    report = {
        "schema": "slp.source3-gene-state-development-result/v2",
        "bestEpoch": best_epoch,
        "completedSchedule": not incomplete_schedule,
        "verificationComplete": not verification_incomplete,
        "trainingSeconds": time.monotonic() - started,
        "parameters": parameter_count,
        "sourceReloadMaxAbsError": reload_drift,
        "emptyActionIdentity": identity,
        "contexts": {name: {"world": reports[name], "independentlyCentered": independent[name]} for name in context_names},
        "comparators": {
            name: {
                "mean": prior["results"][name]["mean"],
                "fullPhysical1156Ridge": prior["results"][name]["ridge"],
                "minimalControlV2": prior["results"][name]["world"],
                "base577GeneState": base577["contexts"][name],
                "independentlyCenteredFullPhysical1156Ridge": landscape["results"][name]["full_physical_ridge"],
                "independentlyCenteredMinimalControlV2": landscape["results"][name]["minimal_control_v2"],
            }
            for name in context_names
        },
        "advancement": {"contexts": decisions, "passed": all(item["passed"] for item in decisions.values())},
        "history": history,
        "interpretation": protocol["interpretation"],
        "artifacts": {
            name: sha256(args.output / name)
            for name in (
                "protocol.json", "model-config.json", "model.safetensors",
                "reference.npz", "development-predictions.npz",
            )
        },
    }
    write_json(args.output / "report.json", report)
    print(json.dumps({"reportSha256": sha256(args.output / "report.json")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("profile", "train"), required=True)
    parser.add_argument("--profile-query-count", type=int, default=7036)
    parser.add_argument("--batch-size", type=int, choices=(32, 64), default=32)
    parser.add_argument(
        "--node-feature-kind", choices=("static577", "response32"), default="static577",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--graph-sha256")
    parser.add_argument("--v2-reference", type=Path)
    parser.add_argument("--v2-exposure", type=Path)
    parser.add_argument("--v2-summary", type=Path)
    parser.add_argument("--v2-summary-sha256")
    parser.add_argument("--landscape-report", type=Path)
    parser.add_argument("--landscape-report-sha256")
    parser.add_argument("--base577-report", type=Path)
    parser.add_argument("--base577-report-sha256")
    parser.add_argument("--selected-profile", type=Path)
    parser.add_argument("--selected-profile-sha256")
    parser.add_argument("--batch32-profile", type=Path)
    parser.add_argument("--batch32-profile-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "train" and any(
        getattr(args, name) is None
        for name in (
            "data", "graph", "graph_sha256", "v2_reference", "v2_exposure",
            "v2_summary", "v2_summary_sha256", "landscape_report", "landscape_report_sha256",
            "base577_report", "base577_report_sha256",
            "selected_profile", "selected_profile_sha256", "batch32_profile",
            "batch32_profile_sha256",
        )
    ):
        parser.error("train mode requires all data, graph, frozen-reference and comparator inputs")
    run(args)


if __name__ == "__main__":
    main()
