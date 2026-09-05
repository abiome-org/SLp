#!/usr/bin/env python3
"""Freeze target-free K562-prior forecasts under the RPE1 control context."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "results/slp11-transition/k562-essential-count-latent-state-seed731-v1"
FREEZE = MODEL_DIR / "FROZEN-BEFORE-DEVELOPMENT-V2.json"
ADAPTER_DIR = ROOT / "modules/slp-1-1-count-prior-context-adapter-v1"
RPE_STATIC = ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz"
RPE_ROSTER = ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz"
RPE_ROUTING = ROOT / "data/derived/slp11-human-rpe1-essential-singlecell-metadata-v1/cell-routing-metadata.npz"
RPE_CONTROL = ROOT / "data/derived/slp11-human-rpe1-essential-count-control/reconstruction-train-nt-gem-v1/gem-control-reference.npz"
K_ROSTER = ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz"
OUTPUT = ROOT / "results/slp11-transition/k562-count-prior-rpe1-unfitted-context-forecasts-v2"
PINS = {
    MODEL_DIR / "model.safetensors": "c7cc6a369f8b63d936c535f7cc59439fec38033202d4b98616b02270df74f3f8",
    MODEL_DIR / "reference.npz": "8020753e9e2597b08cb94c5351772be05986b286f61e0f7a26be26fbfabae4f6",
    MODEL_DIR / "protocol.json": "a85d2ab7cb83760a818614f20ab28d2936c3604c4f9236293c18b355391b89e7",
    MODEL_DIR / "artifact-manifest.json": "7f0151d7af61782613407cad22de111df997a12f60e4723cc4c8faaeeb0e24b5",
    FREEZE: "e2f875e42675d54f3690eecf43c71cb1c1cecd762112cffd8545932a13555113",
    MODEL_DIR / "source/count_latent_state.py": "75df347a82151074c0ce6f4c732106e70ed17126aff07d017294894421d30bac",
    RPE_STATIC: "621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816",
    RPE_ROSTER: "b9e1b169c2be4ac756e94f465009dc5bef80d06bc0652950c3cf6916d26d1e56",
    RPE_ROUTING: "10f3d313a5671122bde10a9bd586e3a2808d6f9b554f737ddcbbc28becc5e2f2",
    RPE_CONTROL: "c0c2eab217d00f9555b6ab5725cd2c49f56b1ecdf34b7af47f303eee9d1b8e20",
    K_ROSTER: "f2ee702a0714ca7f11f4fd2aa96f4c1825617c0e4f2bcdac42135cd0ba938d7b",
}
QUERY_COUNT = 8749
ACTION_COUNT = 1666
CONTEXTS = 56
SECONDS = 600.0


class ContextForecastError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compresslevel=9)
    return output.getvalue()


def metadata_gem_counts(
    routing: dict[str, np.ndarray], fitting_ids: np.ndarray, gem_ids: np.ndarray
) -> np.ndarray:
    """Count reconstruction-training fitting cells without opening expression."""
    actions = routing["action_ids"].astype(str)
    role = routing["intervention_role"].astype(str)
    reconstruction = routing["reconstruction_role"].astype(str)
    unresolved = np.asarray(routing["unresolved_action"], dtype=np.bool_)
    control = np.asarray(routing["is_control"], dtype=np.bool_)
    gems = np.asarray(routing["gem_group"], dtype=np.int64)
    selected = (role == "train") & (reconstruction == "train") & (~unresolved) & (~control)
    genes = np.asarray(fitting_ids).astype(str)
    groups = np.asarray(gem_ids, dtype=np.int64)
    if genes.shape != (ACTION_COUNT,) or groups.shape != (CONTEXTS,):
        raise ContextForecastError("fitting action or GEM roster size drift")
    gene_index = {gene: row for row, gene in enumerate(genes)}
    gem_index = {int(gem): row for row, gem in enumerate(groups)}
    result = np.zeros((len(genes), len(groups)), dtype=np.int64)
    for gene, gem in zip(actions[selected], gems[selected], strict=True):
        if gene not in gene_index or int(gem) not in gem_index:
            raise ContextForecastError("selected metadata row is outside frozen rosters")
        result[gene_index[gene], gem_index[int(gem)]] += 1
    if np.any(result.sum(1) <= 0) or int(result.sum()) != 142601:
        raise ContextForecastError("RPE1 fitting metadata support drift")
    return result


def chunk_consistency(left: np.ndarray, right: np.ndarray) -> dict[str, float | bool]:
    """Evaluate the frozen tolerance for equivalent query chunkings."""
    first, second = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ContextForecastError("chunk comparison arrays must be finite and aligned")
    difference = np.abs(first - second)
    maximum_absolute = float(difference.max(initial=0))
    maximum_relative = float(
        np.max(difference / np.maximum(1.0, np.abs(second)), initial=0)
    )
    maximum_log = float(
        np.max(np.abs(np.log1p(first) - np.log1p(second)), initial=0)
    )
    return {
        "queryChunkExact": bool(np.array_equal(first, second)),
        "queryChunkMaximumAbsoluteCp10kDifference": maximum_absolute,
        "queryChunkMaximumRelativeCp10kDifferenceWithUnitFloor": maximum_relative,
        "queryChunkMaximumAbsoluteLog1pDifference": maximum_log,
        "queryChunkWithinTolerance": maximum_relative <= 1e-6 and maximum_log <= 1e-6,
    }


def protocol() -> dict[str, object]:
    return {
        "schema": "slp.k562-count-prior-rpe1-unfitted-context-forecast-protocol/v1",
        "hypothesis": "The frozen K562 count-state prior may retain descriptive action-conditioned structure when supplied an RPE1 control-only context, without RPE1 perturbation fitting.",
        "decisionRule": "Descriptive forecast freeze only. Later scoring must report all 1,666 genes and the prespecified 1,443 K562-seen/223 RPE1-only action strata. No result selects or changes an architecture or checkpoint.",
        "model": "Original frozen K562 count-latent prior checkpoint; prior marginal mean only, no posterior encoder, cell counts, library sizes, or uncertainty claim.",
        "externalContext": "RPE1 reconstruction-training NT smoothed CP10k rates for 56 GEM groups and raw static577 query descriptors on the exact 8,749 source query axis.",
        "featureTransform": "Use the original K562 reference.npz float32 feature_mean/feature_scale/feature_clip for both RPE1 query and action raw static577 features. The separately fitted RPE1 float64 normalizer is intentionally not used.",
        "gemWeights": "Per fitting action, reconstruction-training RPE1 cell counts by GEM derived from routing metadata only, normalized to sum one.",
        "numericalVerification": "Empty-action context mixture must be bit-exact. Equivalent query chunkings may differ only within maximum relative CP10k 1e-6 with unit floor and maximum absolute ln1p(CP10k) 1e-6.",
        "strata": {"k562SeenAction": 1443, "rpe1OnlyAction": 223},
        "laterComparators": {
            "rpe1AnchoredRidgeModel": {"path": "results/slp11-transition/rpe1-essential-count-anchored-static-ridge-seed731-v1/model.npz", "sha256": "bd144e36b5618c6225828501492edfa5449cef07442041c1d1cc20645b1473bc"},
            "qualification": "RPE1 ridge and mean are fitted in the target context; they do not have the same training-data access as this unfitted-context prior.",
        },
        "inputs": {str(path.relative_to(ROOT)): expected for path, expected in PINS.items()},
        "source": {
            "runnerSha256": sha256_file(Path(__file__).resolve()),
            "adapterSha256": sha256_file(ADAPTER_DIR / "inference.py"),
            "coreSha256": sha256_file(ADAPTER_DIR / "count_latent_state.py"),
        },
        "limits": {"cpuThreads": 2, "wallSeconds": SECONDS},
        "accessBoundary": {"routingMetadata": True, "controlMomentsDerivedReference": True, "staticFeatures": True, "rpeFittingPerturbationMoments": False, "rpeReconstructionHeld": False, "rpeDevelopment": False, "test": False},
    }


def prepare(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError("immutable context forecast output already exists")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    shutil.copy2(Path(__file__).resolve(), source / "runner.py")
    shutil.copy2(ADAPTER_DIR / "inference.py", source / "inference.py")
    shutil.copy2(ADAPTER_DIR / "count_latent_state.py", source / "count_latent_state.py")
    value = protocol()
    (output / "protocol.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "slp.k562-count-prior-rpe1-unfitted-context-prepared/v2",
        "protocolSha256": sha256_file(output / "protocol.json"),
        "rpeFittingPerturbationMomentsRead": False,
        "rpeDevelopmentRead": False,
        "testRead": False,
    }
    (output / "PREPARED.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def forecast(output: Path) -> dict[str, object]:
    if not (output / "PREPARED.json").exists() or (output / "forecasts-before-rpe-fitting-outcomes.npz").exists():
        raise ContextForecastError("prepare once before freezing forecasts")
    if json.loads((output / "protocol.json").read_text()) != protocol():
        raise ContextForecastError("frozen protocol or source changed")
    for path, expected in PINS.items():
        if sha256_file(path) != expected:
            raise ContextForecastError(f"frozen input hash mismatch: {path}")
    source = output / "source"
    for name, live in (("runner.py", Path(__file__).resolve()), ("inference.py", ADAPTER_DIR / "inference.py"), ("count_latent_state.py", ADAPTER_DIR / "count_latent_state.py")):
        if sha256_file(source / name) != sha256_file(live):
            raise ContextForecastError(f"prepared source drift: {name}")
    started = time.perf_counter()
    static, roster, routing, control, k_roster = map(
        load_npz, (RPE_STATIC, RPE_ROSTER, RPE_ROUTING, RPE_CONTROL, K_ROSTER)
    )
    query_ids = roster["query_ids"].astype(str)
    fitting_ids = roster["fitting_action_ids"].astype(str)
    if query_ids.shape != (QUERY_COUNT,) or fitting_ids.shape != (ACTION_COUNT,):
        raise ContextForecastError("RPE1 query/action roster drift")
    if not np.array_equal(query_ids, control["query_ids"].astype(str)):
        raise ContextForecastError("RPE1 static/control query axes differ")
    gem_count = metadata_gem_counts(routing, fitting_ids, control["gem_group"])
    gem_weight = gem_count.astype(np.float64) / gem_count.sum(1, keepdims=True)
    k_seen = np.isin(fitting_ids, k_roster["fitting_action_ids"].astype(str))
    if int(k_seen.sum()) != 1443 or int((~k_seen).sum()) != 223:
        raise ContextForecastError("prespecified action strata drift")
    raw_query = static["feature_values"][roster["query_entity_index"]]
    raw_action = static["feature_values"][roster["fitting_action_entity_index"]]
    adapter = load_module(source / "inference.py", "rpe_external_count_prior")
    predictor = adapter.ContextPriorPredictor(
        MODEL_DIR, freeze_receipt=FREEZE, device="cpu"
    )
    cp_parts, log_parts = [], []
    support = []
    with threadpool_limits(2), torch.no_grad():
        for left in range(0, ACTION_COUNT, 32):
            if time.perf_counter() - started > SECONDS:
                raise TimeoutError("external-context forecast exceeded 600 seconds")
            stop = min(left + 32, ACTION_COUNT)
            result = predictor.predict(
                raw_action[left:stop], raw_query, query_ids,
                control["basal_rate"], control["basal_mask"], gem_weight[left:stop],
                chunk_size=512,
            )
            cp_parts.append(result["mean_cp10k"])
            log_parts.append(result["mean_log1p_cp10k"])
            support.append(result["query_supported"])
    cp10k = np.concatenate(cp_parts)
    log1p = np.concatenate(log_parts)
    supported = np.concatenate(support)
    empty = predictor.predict(
        np.zeros((1, 577), np.float32), raw_query, query_ids,
        control["basal_rate"], control["basal_mask"], gem_weight[:1],
        action_mask=np.zeros((1, 1), np.bool_), chunk_size=257,
    )
    empty_expected = gem_weight[:1] @ control["basal_rate"].astype(np.float64)
    chunk_a = predictor.predict(
        raw_action[:2], raw_query, query_ids, control["basal_rate"],
        control["basal_mask"], gem_weight[:2], chunk_size=257,
    )["mean_cp10k"]
    chunk_b = cp10k[:2]
    verification = {
        "emptyContextIdentityExact": bool(np.array_equal(empty["mean_cp10k"], empty_expected)),
        "emptyContextIdentityMaximumAbsoluteDifference": float(np.max(np.abs(empty["mean_cp10k"] - empty_expected))),
        "allQueriesSupported": bool(supported.all()),
        **chunk_consistency(chunk_a, chunk_b),
    }
    if not all((verification["emptyContextIdentityExact"], verification["queryChunkWithinTolerance"], verification["allQueriesSupported"])):
        raise ContextForecastError(f"external context contract verification failed: {verification}")
    arrays = {
        "schema": np.asarray("slp.k562-count-prior-rpe1-unfitted-context-forecasts/v2"),
        "entity_taxon": np.full(ACTION_COUNT, 9606, dtype=np.int64),
        "action_ids": fitting_ids,
        "query_ids": query_ids,
        "gem_group": np.asarray(control["gem_group"], dtype=np.int16),
        "gem_cell_count": gem_count,
        "gem_weights": gem_weight,
        "is_k562_seen_action": k_seen,
        "action_static_all_zero": np.all(raw_action == 0, axis=1),
        "query_static_all_zero": np.all(raw_query == 0, axis=1),
        "query_supported": supported,
        "mean_cp10k": cp10k,
        "mean_log1p_cp10k": log1p,
        "source_model_sha256": np.asarray(PINS[MODEL_DIR / "model.safetensors"]),
        "source_reference_sha256": np.asarray(PINS[MODEL_DIR / "reference.npz"]),
        "rpe_control_sha256": np.asarray(PINS[RPE_CONTROL]),
        "rpe_static_sha256": np.asarray(PINS[RPE_STATIC]),
    }
    artifact_path = output / "forecasts-before-rpe-fitting-outcomes.npz"
    artifact_path.write_bytes(deterministic_npz(arrays))
    report = {
        "schema": "slp.k562-count-prior-rpe1-unfitted-context-forecast-freeze/v2",
        "protocolSha256": sha256_file(output / "protocol.json"),
        "forecastSha256": sha256_file(artifact_path),
        "modelSha256": predictor.model_sha256,
        "sourceReferenceSha256": predictor.reference_sha256,
        "actions": ACTION_COUNT, "queries": QUERY_COUNT, "contexts": CONTEXTS,
        "strata": {"k562SeenActions": int(k_seen.sum()), "rpe1OnlyActions": int((~k_seen).sum())},
        "staticMissing": {"actions": int(np.all(raw_action == 0, axis=1).sum()), "queries": int(np.all(raw_query == 0, axis=1).sum())},
        "priorTotalCp10kQuantiles": np.quantile(cp10k.sum(1), [0, .25, .5, .75, 1]).tolist(),
        "verification": verification,
        "elapsedSeconds": time.perf_counter() - started,
        "rpeFittingPerturbationMomentsRead": False,
        "rpeReconstructionHeldRead": False,
        "rpeDevelopmentRead": False,
        "testRead": False,
    }
    report_path = output / "FORECASTS-FROZEN-BEFORE-RPE-FITTING-OUTCOMES.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--forecast", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.prepare == args.forecast:
        raise SystemExit("select exactly one of --prepare or --forecast")
    value = prepare(args.output_dir) if args.prepare else forecast(args.output_dir)
    print(json.dumps(value, indent=2, sort_keys=True))
