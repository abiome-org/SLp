"""Fitting-only audit of static-query decoding for supervised rank-32 responses."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
from safetensors.torch import save_file
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/slp11-transition/human-essential-count-query-decoder-capacity-fitting-v1"
CORE = ROOT / "modules/slp-1-1-count-static-ridge-v1/count_static_ridge.py"
RANK_CORE = ROOT / "modules/slp-1-1-reduced-rank-response-v1/response_model.py"
DECODER_CORE = ROOT / "modules/slp-1-1-query-decoder-capacity-v1/query_decoder.py"
SEED = 731
ALPHA = 1000.0
RANK = 32
UPDATES = 2000
BATCH = 1024

CONTEXTS = {
    "k562": {
        "moments": ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1/fitting-action-moments.npz",
        "static": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz",
        "baseline": ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1/model.npz",
    },
    "rpe1": {
        "moments": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/fitting-action-moments.npz",
        "static": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz",
        "baseline": ROOT / "results/slp11-transition/rpe1-essential-count-anchored-static-ridge-seed731-v1/model.npz",
    },
}

PINS = {
    CORE: "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
    RANK_CORE: "da5989fef73891a2fe79a5802858bc583721e3c7e8f9c72e5fde971df3eb92bb",
    CONTEXTS["k562"]["moments"]: "a1f44a15a42c5b56e4ce897fde6ebba97298fc296105c6c870ee0e740331694e",
    CONTEXTS["k562"]["static"]: "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659",
    CONTEXTS["k562"]["baseline"]: "dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3",
    CONTEXTS["rpe1"]["moments"]: "d15def86aead06b0bc75ab63c77513735ec7c57d65012bff72f3947bc654895c",
    CONTEXTS["rpe1"]["static"]: "621e1e9f0dffc740ef42382b1b2898f629edd5037e8a02d411e8d30e815ed816",
    CONTEXTS["rpe1"]["baseline"]: "bd144e36b5618c6225828501492edfa5449cef07442041c1d1cc20645b1473bc",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def prepare(output: Path) -> dict:
    for path, expected in PINS.items():
        if sha256(path) != expected:
            raise ValueError(f"pinned source changed: {path}")
    protocol = {
        "schema": "slp.human-essential-count-query-decoder-capacity-fitting-protocol/v1",
        "hypothesis": "A fixed static-feature query decoder preserves the supervised rank-32 teacher's full-native-query held-fitting-gene MSE within one percent in both K562 and RPE1.",
        "advancementRule": "For each source, aggregate three-fold decoder OOF MSE must be <=1.01 times the exact rank-32 teacher OOF MSE. Both sources must pass.",
        "population": "All fitting intervention genes only: 1,443 K562 and 1,666 RPE1; three seed731 global gene folds; no development, reconstruction-held, test, or benchmark values.",
        "teacher": "Within each fold, exact alpha1000 regularized rank-32 response factors are fitted only on the other two fitting-gene folds. T=(U32.T@whitened_rhs).T [native queries,32], A=(normalized action-design_mean)@(V/sqrt(D+1000)@U32).",
        "queryDecoder": {
            "input": "raw static577 query features transformed by the fold's fitting-action feature mean and scale; no learned query ID",
            "target": "T columns divided by their native-query RMS without centering; RMS uses training-fold-derived T only",
            "architecture": "577->256 GELU->32 plus bias-free linear 577->32 residual; both output maps zero initialized; dropout0",
            "optimizer": "AdamW lr0.001 weight_decay0.01, exactly2000 updates, uniform query batches1024 with replacement, seed731; no early stopping or sweep",
        },
        "metrics": "Descriptor standardized/raw MSE over native queries; held-fitting-gene full-query raw residual MSE for decoded teacher, exact rank32 teacher and full alpha1000 ridge.",
        "interpretationBoundary": "This diagnoses whether static query features can reproduce a fitting-supervised native-panel output basis. It does not establish prediction of unmeasured queries or a launchable model.",
        "snapshots": {
            source: {
                kind: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": PINS[path]}
                for kind, path in paths.items()
            }
            for source, paths in CONTEXTS.items()
        },
        "sources": {
            "ridgeCore": {"path": str(CORE.relative_to(ROOT)), "sha256": PINS[CORE]},
            "exactRankCore": {"path": str(RANK_CORE.relative_to(ROOT)), "sha256": PINS[RANK_CORE]},
            "queryDecoderCore": {"path": str(DECODER_CORE.relative_to(ROOT)), "sha256": sha256(DECODER_CORE)},
            "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
        },
        "compute": {"device": "local RTX 4070", "profileBeforeRun": True, "maximumSeconds": 600, "maximumMemoryBytes": 8 << 30},
        "developmentOpened": False,
        "testOpened": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    write_new(path, canonical_json(protocol))
    return protocol


def load_source(source: str, core) -> dict[str, np.ndarray]:
    paths = CONTEXTS[source]
    moments = load_npz(paths["moments"])
    static = load_npz(paths["static"])
    baseline = load_npz(paths["baseline"])
    genes = moments["action_ids"].astype(str)
    queries = moments["query_ids"].astype(str)
    if not np.array_equal(queries, baseline["query_ids"].astype(str)):
        raise ValueError(f"{source} query/baseline order mismatch")
    entity = static["entity_id"].astype(str)
    lookup = {value: row for row, value in enumerate(entity)}
    if len(lookup) != len(entity) or any(item not in lookup for item in (*genes, *queries)):
        raise ValueError(f"{source} static identity coverage mismatch")
    features = np.asarray(static["feature_values"], dtype=np.float32)
    action = features[[lookup[item] for item in genes]]
    query = features[[lookup[item] for item in queries]]
    target = core.response_from_cp10k_moments(moments["cp10k_sum"], moments["cell_count"])
    target -= core.control_anchor(baseline["basal_rate"], moments["gem_cell_count"])
    folds = np.asarray([core.global_gene_fold(item, SEED) for item in genes], dtype=np.int8)
    return {"genes": genes, "queries": queries, "action": action, "query": query, "target": target, "folds": folds}


def make_fold(source_data, fold: int, core, rank_core, decoder_core):
    fitting = source_data["folds"] != fold
    held = ~fitting
    teacher = rank_core.fit(source_data["action"][fitting], source_data["target"][fitting], rank=RANK, alpha=ALPHA)
    state = core.fit_state(source_data["action"][fitting], source_data["target"][fitting])
    factors = decoder_core.rank32_factors(state, alpha=ALPHA, rank=RANK)
    np.testing.assert_allclose(teacher.state_projection, factors["state_projection"], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(teacher.query_loading, factors["query_loading"], atol=2e-12, rtol=2e-12)
    query_input = (source_data["query"].astype(np.float64) - teacher.feature_mean) / teacher.feature_scale
    target_loading = teacher.query_loading.T
    scale = decoder_core.rms_scale(target_loading)
    return {
        "fitting": fitting,
        "held": held,
        "teacher": teacher,
        "state": state,
        "query_input": query_input.astype(np.float32),
        "target_loading": target_loading,
        "loading_scale": scale,
    }


def train_decoder(inputs: np.ndarray, targets: np.ndarray, decoder_core, device: torch.device, updates: int, *, profile: bool = False):
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = decoder_core.QueryDecoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    x = torch.from_numpy(np.asarray(inputs, np.float32)).to(device)
    y = torch.from_numpy(np.asarray(targets, np.float32)).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    started = time.perf_counter()
    last = None
    for _ in range(updates):
        index = torch.randint(len(x), (BATCH,), generator=generator, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[index])
        loss = torch.mean(torch.square(prediction - y[index]))
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        prediction = model(x).cpu().numpy()
    return model, prediction, {"updates": updates, "seconds": seconds, "lastBatchLoss": last, "profile": profile}


def run(output: Path) -> dict:
    protocol = prepare(output)
    if (output / "report.json").exists():
        raise FileExistsError("diagnostic already completed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    core = load_module(CORE, "query_decoder_capacity_ridge")
    rank_core = load_module(RANK_CORE, "query_decoder_capacity_rank")
    decoder_core = load_module(DECODER_CORE, "query_decoder_capacity_core")
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        source_data = {source: load_source(source, core) for source in CONTEXTS}
        profile_fold = make_fold(source_data["k562"], 0, core, rank_core, decoder_core)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    _, _, profile = train_decoder(
        profile_fold["query_input"],
        profile_fold["target_loading"] / profile_fold["loading_scale"],
        decoder_core,
        device,
        100,
        profile=True,
    )
    projected_training_seconds = profile["seconds"] * (len(CONTEXTS) * 3 * UPDATES / 100)
    projected_total_seconds = projected_training_seconds + (time.perf_counter() - started) * 6
    profile.update({
        "projectedTrainingSeconds": projected_training_seconds,
        "projectedTotalSeconds": projected_total_seconds,
        "peakCudaAllocatedBytes": int(torch.cuda.max_memory_allocated()),
        "parameters": decoder_core.expected_parameter_count(),
    })
    write_new(output / "profile.json", canonical_json(profile))
    print(json.dumps({"stage": "profile", **profile}), flush=True)
    if projected_total_seconds > 600 or profile["peakCudaAllocatedBytes"] > (8 << 30):
        raise RuntimeError("actual-shape profile exceeds frozen compute cap")

    source_reports = {}
    output_arrays: dict[str, np.ndarray] = {"schema": np.asarray("slp.human-essential-count-query-decoder-capacity-fitting/v1")}
    artifact_hashes = {}
    for source, data in source_data.items():
        per_gene = {name: np.empty(len(data["genes"]), dtype=np.float64) for name in ("decoder", "teacher", "full_ridge")}
        fold_reports = []
        for fold in range(3):
            if time.perf_counter() - started > 600:
                raise TimeoutError("frozen 600-second cap reached")
            with threadpool_limits(limits=2):
                item = make_fold(data, fold, core, rank_core, decoder_core)
            standardized_target = item["target_loading"] / item["loading_scale"]
            model, standardized_prediction, training = train_decoder(
                item["query_input"], standardized_target, decoder_core, device, UPDATES
            )
            predicted_loading = standardized_prediction.astype(np.float64) * item["loading_scale"]
            held_features = data["action"][item["held"]]
            teacher_prediction = item["teacher"].predict(held_features)
            action = decoder_core.action_state(core, item["state"], held_features, item["teacher"].state_projection)
            decoded_prediction = decoder_core.reconstruct(item["teacher"].intercept, action, predicted_loading)
            full_prediction = core.predict_residual(item["state"], held_features, "1000")
            exact = decoder_core.reconstruct(item["teacher"].intercept, action, item["target_loading"])
            np.testing.assert_allclose(exact, teacher_prediction, atol=2e-12, rtol=2e-12)
            for name, prediction in (("decoder", decoded_prediction), ("teacher", teacher_prediction), ("full_ridge", full_prediction)):
                per_gene[name][item["held"]] = np.mean(np.square(prediction - data["target"][item["held"]]), axis=1)
            descriptor_standardized = float(np.mean(np.square(standardized_prediction - standardized_target)))
            descriptor_raw = float(np.mean(np.square(predicted_loading - item["target_loading"])))
            fold_path = output / "folds" / f"{source}-fold{fold}.npz"
            write_new(fold_path, deterministic_npz({
                "schema": np.asarray("slp.human-essential-count-query-decoder-capacity-fold/v1"),
                "source_id": np.asarray(source),
                "fold": np.asarray(fold, dtype=np.int8),
                "query_ids": data["queries"],
                "held_gene_ids": data["genes"][item["held"]],
                "loading_rms": item["loading_scale"],
                "teacher_query_loading": item["target_loading"],
                "decoded_query_loading": predicted_loading,
                "teacher_action_state": action,
                "target_mean": item["teacher"].intercept,
            }))
            weight_path = output / "folds" / f"{source}-fold{fold}.safetensors"
            weight_path.parent.mkdir(parents=True, exist_ok=True)
            if weight_path.exists():
                raise FileExistsError(weight_path)
            save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(weight_path))
            artifact_hashes[str(fold_path.relative_to(output)).replace("\\", "/")] = sha256(fold_path)
            artifact_hashes[str(weight_path.relative_to(output)).replace("\\", "/")] = sha256(weight_path)
            report = {
                "fold": fold,
                "fittingGenes": int(item["fitting"].sum()),
                "heldGenes": int(item["held"].sum()),
                "descriptorStandardizedMse": descriptor_standardized,
                "descriptorRawMse": descriptor_raw,
                "forecastMse": {name: float(values[item["held"]].mean()) for name, values in per_gene.items()},
                "training": training,
                "foldArtifact": {"path": str(fold_path.relative_to(output)).replace("\\", "/"), "sha256": sha256(fold_path)},
                "weights": {"path": str(weight_path.relative_to(output)).replace("\\", "/"), "sha256": sha256(weight_path)},
            }
            fold_reports.append(report)
            print(json.dumps({"stage": "fold", "source": source, **report}), flush=True)
        mse = {name: float(values.mean()) for name, values in per_gene.items()}
        source_reports[source] = {
            "genes": len(data["genes"]),
            "queries": len(data["queries"]),
            "mse": mse,
            "decoderWithinOnePercentOfTeacher": mse["decoder"] <= 1.01 * mse["teacher"],
            "folds": fold_reports,
        }
        output_arrays[f"{source}_gene_ids"] = data["genes"]
        output_arrays[f"{source}_fold"] = data["folds"]
        for name, values in per_gene.items():
            output_arrays[f"{source}_{name}_mse"] = values
    scores_path = output / "per-gene-mse.npz"
    write_new(scores_path, deterministic_npz(output_arrays))
    shutil.copyfile(Path(__file__).resolve(), output / "executed-source.py")
    report = {
        "schema": "slp.human-essential-count-query-decoder-capacity-fitting-report/v1",
        "status": "complete",
        "protocol": {"path": "protocol.json", "sha256": sha256(output / "protocol.json")},
        "profile": {"path": "profile.json", "sha256": sha256(output / "profile.json")},
        "sources": source_reports,
        "advancement": {
            "passes": all(item["decoderWithinOnePercentOfTeacher"] for item in source_reports.values()),
            "rule": protocol["advancementRule"],
        },
        "perGeneMse": {"path": scores_path.name, "sha256": sha256(scores_path)},
        "artifacts": artifact_hashes,
        "runtime": {"seconds": time.perf_counter() - started, "maximumSeconds": 600, "peakCudaAllocatedBytes": int(torch.cuda.max_memory_allocated())},
        "developmentOpened": False,
        "testOpened": False,
        "interpretationBoundary": protocol["interpretationBoundary"],
    }
    write_new(output / "report.json", canonical_json(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output.resolve()) if args.command == "prepare" else run(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
