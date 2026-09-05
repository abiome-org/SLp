"""Build fixed-anchor control-cell coexpression fingerprints for K562 and RPE1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from numpy.lib import format as npformat
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "modules/slp-1-1-control-coexpression-v1/control_coexpression.py"
RESULT_DIR = ROOT / "results/slp11-transition/human-control-coexpression-reliability-v1"
DATA_DIR = ROOT / "data/derived/slp11-human-control-coexpression/static577-gaussian64-leave-self-out-v1"
REFERENCE = ROOT / "results/slp11-transition/k562-essential-count-latent-state-seed731-v1/reference.npz"
CONTEXTS = {
    "k562": {
        "context_id": "replogle-2022-k562-essential-day-6",
        "manifest": ROOT / "data/derived/slp11-human-k562-essential-raw-cells-v2/manifest.json",
        "static": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz",
        "roster": ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz",
        "expected_queries": 8563,
        "expected_controls": 9609,
        "expected_gems": 48,
    },
    "rpe1": {
        "context_id": "replogle-2022-rpe1-essential-day-7",
        "manifest": ROOT / "data/derived/slp11-human-rpe1-essential-raw-cells-v1/manifest.json",
        "static": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/rpe1-essential-count-static577.npz",
        "roster": ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz",
        "expected_queries": 8749,
        "expected_controls": 10350,
        "expected_gems": 56,
    },
}
DIMENSIONS = 64
SEED = 731
GATE_MEDIAN_COSINE = 0.5
GATE_DEFINED_FRACTION = 0.9


def load_module():
    spec = importlib.util.spec_from_file_location("slp11_control_coexpression_runtime", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CC = load_module()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            npformat.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_identity_inputs():
    with np.load(REFERENCE, allow_pickle=False) as ref:
        reference = {key: ref[key].copy() for key in ref.files}
    loaded = {}
    for name, cfg in CONTEXTS.items():
        manifest = json.loads(cfg["manifest"].read_text(encoding="utf-8"))
        with np.load(cfg["static"], allow_pickle=False) as archive:
            static = {key: archive[key].copy() for key in archive.files}
        with np.load(cfg["roster"], allow_pickle=False) as archive:
            roster = {key: archive[key].copy() for key in archive.files}
        query_ids = roster["query_ids"].astype(str)
        if len(query_ids) != cfg["expected_queries"] or len(set(query_ids.tolist())) != len(query_ids):
            raise ValueError(f"{name} query identity contract failed")
        if not np.array_equal(static["entity_id"][roster["query_entity_index"]], query_ids):
            raise ValueError(f"{name} static/query index mismatch")
        control = [item for item in manifest["shards"] if item["role"] == "control"]
        if sum(int(item["rows"]) for item in control) != cfg["expected_controls"]:
            raise ValueError(f"{name} control row contract failed")
        loaded[name] = {"manifest": manifest, "static": static, "roster": roster, "control": control}
    if not np.array_equal(reference["query_ids"].astype(str), loaded["k562"]["roster"]["query_ids"].astype(str)):
        raise ValueError("K562 count reference query order mismatch")
    return reference, loaded


def build_anchor(reference, loaded):
    k_ids = loaded["k562"]["roster"]["query_ids"].astype(str)
    r_ids = loaded["rpe1"]["roster"]["query_ids"].astype(str)
    common_ids = np.asarray(sorted(set(k_ids.tolist()).intersection(r_ids.tolist())), dtype="U15")
    if len(common_ids) != 7226:
        raise ValueError(f"expected 7,226 common queries, got {len(common_ids)}")
    rows = {}
    raw = {}
    for name in CONTEXTS:
        query_ids = loaded[name]["roster"]["query_ids"].astype(str)
        lookup = {gene: i for i, gene in enumerate(query_ids)}
        rows[name] = np.asarray([lookup[gene] for gene in common_ids], dtype=np.int64)
        entity_rows = loaded[name]["roster"]["query_entity_index"][rows[name]]
        raw[name] = loaded[name]["static"]["feature_values"][entity_rows]
    if not np.array_equal(raw["k562"], raw["rpe1"]):
        different = int(np.any(raw["k562"] != raw["rpe1"], axis=1).sum())
        raise ValueError(f"common raw static coordinates disagree for {different} genes")
    normalized = CC.normalize_static_float32(
        raw["k562"], reference["feature_mean"], reference["feature_scale"], float(reference["feature_clip"])
    )
    weights, gaussian = CC.fixed_anchor_weights(normalized, DIMENSIONS, SEED)
    weights_repeat, gaussian_repeat = CC.fixed_anchor_weights(normalized, DIMENSIONS, SEED)
    if not np.array_equal(weights, weights_repeat) or not np.array_equal(gaussian, gaussian_repeat):
        raise ValueError("fixed anchor construction is not deterministic")
    return common_ids, rows, normalized, weights, gaussian


def prepare() -> None:
    reference, loaded = load_identity_inputs()
    common_ids, rows, normalized, weights, gaussian = build_anchor(reference, loaded)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "slp.human-control-coexpression-reliability-protocol/v2",
        "status": "frozen-before-control-count-access",
        "hypothesis": "Control-cell expression distributions yield stable gene fingerprints beyond the supplied control mean and static protein/GO coordinates.",
        "advancementRule": {
            "allContextsMustPass": True,
            "definedQueryFractionMinimum": GATE_DEFINED_FRACTION,
            "medianDefinedQuerySplitHalfCosineMinimum": GATE_MEDIAN_COSINE,
            "failureAction": "stop feature branch; retain diagnostic report but do not emit candidate feature packs",
        },
        "accessibleModalities": [
            "reconstruction-training non-targeting raw counts",
            "cell barcode",
            "GEM group",
            "full-native-panel library size",
            "static ESM2-8M/protein-presence/shared-MF-CC-GO features",
        ],
        "excluded": ["targeting cells", "reconstruction-held cells", "development-validation cells", "test cells", "perturbation outcomes"],
        "identity": {
            "taxonomy": 9606,
            "commonAnchorQueries": int(len(common_ids)),
            "commonAnchorOrder": "ascending exact unversioned ENSG",
            "contextNativeQueryCounts": {name: int(len(loaded[name]["roster"]["query_ids"])) for name in CONTEXTS},
            "commonQueryLfSha256": hashlib.sha256(("\n".join(common_ids.tolist()) + "\n").encode("ascii")).hexdigest(),
        },
        "anchor": {
            "staticTransform": "original K562 count-reference float32: nan_to_num(raw,0); clip((raw-feature_mean)/feature_scale,+/-feature_clip)",
            "gaussian": "numpy.random.default_rng(731).standard_normal((577,64))/sqrt(577), float64",
            "postprocess": "center every projected column over the 7,226 common queries, then L2-normalize each column",
            "sameWeightsBothContexts": True,
            "nativeOnlyQueryWeight": 0.0,
            "dimensions": DIMENSIONS,
            "seed": SEED,
        },
        "measurement": {
            "cellValue": "ln1p(10000 * raw count / sum raw counts over every native source query column)",
            "libraryCovariate": "natural-log full-native-panel raw library size, centered within GEM",
            "regression": "per GEM and per query OLS on intercept plus centered log library, NT controls only",
            "fingerprint": "Pearson(residual query, residual anchor score with that query's W contribution removed), pooled after GEM-specific residualization",
            "degenerate": "zero coordinate with explicit coordinate-present=false",
        },
        "splitHalf": {
            "assignment": "first8bigendian(SHA256('slp11-control-coexpression-v1|731|'+barcode)) % 2",
            "regression": "independently refit within each GEM and half",
            "score": "per-native-query cosine between the two 64D leave-self-out fingerprints; zero-norm pair is undefined",
        },
        "inputs": {
            "reference": {"path": str(REFERENCE.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(REFERENCE)},
            **{
                name: {
                    key: {"path": str(cfg[key].relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(cfg[key])}
                    for key in ("manifest", "static", "roster")
                }
                for name, cfg in CONTEXTS.items()
            },
        },
        "implementation": {
            "module": str(MODULE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "moduleSha256": sha256(MODULE_PATH),
            "runner": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "runnerSha256": sha256(Path(__file__).resolve()),
            "numpy": np.__version__,
            "executionAmendment": "v1 stopped before raw-count access because role arrays were vector-valued; v2 validates that every row has the frozen control/train role without changing any scientific setting",
        },
        "targetOutput": str(DATA_DIR.relative_to(ROOT)).replace("\\", "/"),
    }
    atomic_write(RESULT_DIR / "protocol.json", canonical_json(protocol))
    anchor_arrays = {
        "common_query_ids": common_ids,
        "common_query_rows_k562": rows["k562"],
        "common_query_rows_rpe1": rows["rpe1"],
        "normalized_static_features": normalized,
        "anchor_weights": weights,
        "gaussian_projection": gaussian,
    }
    atomic_write(RESULT_DIR / "frozen-anchor.npz", deterministic_npz(anchor_arrays))
    print(json.dumps({"protocol": str(RESULT_DIR / "protocol.json"), "protocolSha256": sha256(RESULT_DIR / "protocol.json"), "anchorSha256": sha256(RESULT_DIR / "frozen-anchor.npz")}, indent=2))


def load_dense_shard(path: Path, expected_sha: str, expected_queries: int):
    if sha256(path) != expected_sha:
        raise ValueError(f"shard hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        required = {"raw_data", "raw_indices", "raw_indptr", "raw_shape", "cell_ids", "gem_group", "library_size", "is_control", "intervention_role", "reconstruction_role"}
        if not required.issubset(z.files):
            raise ValueError(f"shard schema missing keys: {path}")
        if (
            not np.all(z["is_control"])
            or not np.all(z["intervention_role"].astype(str) == "control")
            or not np.all(z["reconstruction_role"].astype(str) == "train")
        ):
            raise ValueError(f"inadmissible shard role: {path}")
        shape = tuple(int(v) for v in z["raw_shape"])
        if shape[1] != expected_queries:
            raise ValueError("shard query width mismatch")
        matrix = sparse.csr_matrix((z["raw_data"], z["raw_indices"], z["raw_indptr"]), shape=shape)
        library = z["library_size"].astype(np.int64)
        if np.any(library <= 0) or not np.array_equal(np.asarray(matrix.sum(axis=1)).reshape(-1), library):
            raise ValueError("library sizes do not equal full native-panel row sums")
        dense = matrix.toarray().astype(np.float64)
        dense *= 10000.0 / library[:, None]
        np.log1p(dense, out=dense)
        return dense, np.log(library.astype(np.float64)), z["gem_group"].astype(np.int64), z["cell_ids"].astype(str)


def native_anchor_matrix(query_ids: np.ndarray, common_ids: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    common_lookup = {gene: i for i, gene in enumerate(common_ids.astype(str))}
    native = np.zeros((len(query_ids), weights.shape[1]), dtype=np.float64)
    common_index = []
    common_weight_rows = []
    for i, gene in enumerate(query_ids.astype(str)):
        if gene in common_lookup:
            row = common_lookup[gene]
            native[i] = weights[row]
            common_index.append(i)
            common_weight_rows.append(row)
    if len(common_index) != len(common_ids):
        raise ValueError("native/common anchor mapping is incomplete")
    order = np.argsort(common_weight_rows)
    return native, np.asarray(common_index, dtype=np.int64)[order]


def run_context(name: str, loaded, common_ids: np.ndarray, weights: np.ndarray):
    cfg = CONTEXTS[name]
    query_ids = loaded["roster"]["query_ids"].astype(str)
    native_weights, common_index = native_anchor_matrix(query_ids, common_ids, weights)
    groups = cfg["expected_gems"]
    first = [CC.empty_first_pass(groups, len(query_ids)) for _ in range(3)]
    shards = loaded["control"]
    started = time.perf_counter()
    for number, shard in enumerate(shards, 1):
        path = cfg["manifest"].parent / shard["path"]
        x, log_library, gem, barcode = load_dense_shard(path, shard["sha256"], len(query_ids))
        group = gem - 1
        if group.min() < 0 or group.max() >= groups:
            raise ValueError(f"{name} GEM IDs outside 1..{groups}")
        half = np.fromiter((CC.barcode_half(item, SEED) for item in barcode), dtype=np.int8, count=len(barcode))
        CC.update_first_pass(first[0], x, log_library, group)
        for h in (0, 1):
            take = half == h
            CC.update_first_pass(first[h + 1], x[take], log_library[take], group[take])
        print(f"{name} first pass {number}/{len(shards)}", flush=True)
    parameters = [CC.regression_parameters(item) for item in first]
    moments = [CC.empty_second_pass(len(query_ids), DIMENSIONS) for _ in range(3)]
    for number, shard in enumerate(shards, 1):
        path = cfg["manifest"].parent / shard["path"]
        x, log_library, gem, barcode = load_dense_shard(path, shard["sha256"], len(query_ids))
        group = gem - 1
        half = np.fromiter((CC.barcode_half(item, SEED) for item in barcode), dtype=np.int8, count=len(barcode))
        residual = CC.residualize(x, log_library, group, *parameters[0])
        CC.update_second_pass(moments[0], residual, common_index, weights)
        del residual
        for h in (0, 1):
            take = half == h
            residual = CC.residualize(x[take], log_library[take], group[take], *parameters[h + 1])
            CC.update_second_pass(moments[h + 1], residual, common_index, weights)
            del residual
        print(f"{name} second pass {number}/{len(shards)}", flush=True)
    derived = [CC.fingerprints_from_moments(item["cross"], item["var_x"], item["var_z"], native_weights) for item in moments]
    repeated = CC.fingerprints_from_moments(moments[0]["cross"], moments[0]["var_x"], moments[0]["var_z"], native_weights)
    if not all(np.array_equal(a, b) for a, b in zip(derived[0], repeated)):
        raise ValueError("fingerprint reconstruction is not exactly deterministic")
    cosine, cosine_defined = CC.split_half_cosine(derived[1][0], derived[2][0])
    defined_values = cosine[cosine_defined]
    quantiles = np.quantile(defined_values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]) if len(defined_values) else np.full(7, np.nan)
    summary = {
        "contextId": cfg["context_id"],
        "nativeQueries": int(len(query_ids)),
        "commonAnchorQueries": int(len(common_ids)),
        "controlCells": int(first[0]["count"].sum()),
        "gemGroups": int(groups),
        "gemCellCountMinimum": int(first[0]["count"].min()),
        "gemCellCountMaximum": int(first[0]["count"].max()),
        "halfGemCellCountMinimum": int(min(first[1]["count"].min(), first[2]["count"].min())),
        "coordinatePresentFraction": float(derived[0][1].mean()),
        "queriesWithAnyCoordinate": int(derived[0][1].any(axis=1).sum()),
        "splitHalfDefinedQueries": int(cosine_defined.sum()),
        "splitHalfUndefinedQueries": int((~cosine_defined).sum()),
        "splitHalfDefinedFraction": float(cosine_defined.mean()),
        "splitHalfCosineQuantiles": dict(zip(["minimum", "q05", "q25", "median", "q75", "q95", "maximum"], [float(v) for v in quantiles])),
        "elapsedSeconds": float(time.perf_counter() - started),
    }
    summary["passesReliabilityGate"] = bool(
        summary["splitHalfDefinedFraction"] >= GATE_DEFINED_FRACTION
        and summary["splitHalfCosineQuantiles"]["median"] >= GATE_MEDIAN_COSINE
    )
    action_ids = loaded["roster"]["action_ids"].astype(str)
    query_lookup = {gene: i for i, gene in enumerate(query_ids)}
    action_query_index = np.asarray([query_lookup.get(gene, -1) for gene in action_ids], dtype=np.int64)
    action_present = action_query_index >= 0
    action_features = np.zeros((len(action_ids), DIMENSIONS), dtype=np.float32)
    action_coordinate_present = np.zeros((len(action_ids), DIMENSIONS), dtype=bool)
    action_features[action_present] = derived[0][0][action_query_index[action_present]].astype(np.float32)
    action_coordinate_present[action_present] = derived[0][1][action_query_index[action_present]]
    arrays = {
        "query_ids": query_ids.astype("U15"),
        "query_features": derived[0][0].astype(np.float32),
        "query_coordinate_present": derived[0][1],
        "query_feature_present": derived[0][1].any(axis=1),
        "split_half_cosine": cosine.astype(np.float32),
        "split_half_defined": cosine_defined,
        "action_ids": action_ids.astype("U15"),
        "action_features": action_features,
        "action_coordinate_present": action_coordinate_present,
        "action_query_present": action_present,
        "action_query_index": action_query_index,
        "entity_taxon": np.asarray(9606, dtype=np.int64),
    }
    diagnostics = {
        "query_ids": query_ids.astype("U15"),
        "split_half_cosine": cosine.astype(np.float32),
        "split_half_defined": cosine_defined,
        "full_coordinate_present": derived[0][1],
        "half0_coordinate_present": derived[1][1],
        "half1_coordinate_present": derived[2][1],
        "full_var_x": moments[0]["var_x"],
        "full_var_z": moments[0]["var_z"],
    }
    return arrays, diagnostics, summary


def run() -> None:
    protocol_path = RESULT_DIR / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError("run --prepare before opening control counts")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen-before-control-count-access":
        raise ValueError("protocol is not a pre-access freeze")
    if protocol["implementation"]["runnerSha256"] != sha256(Path(__file__).resolve()):
        raise ValueError("runner changed after protocol freeze")
    if protocol["implementation"]["moduleSha256"] != sha256(MODULE_PATH):
        raise ValueError("module changed after protocol freeze")
    reference, loaded_all = load_identity_inputs()
    common_ids, rows, normalized, weights, gaussian = build_anchor(reference, loaded_all)
    with np.load(RESULT_DIR / "frozen-anchor.npz", allow_pickle=False) as anchor:
        if not np.array_equal(anchor["common_query_ids"], common_ids) or not np.array_equal(anchor["anchor_weights"], weights):
            raise ValueError("frozen anchor replay mismatch")
    products = {}
    diagnostic_hashes = {}
    summaries = {}
    for name in CONTEXTS:
        arrays, diagnostics, summary = run_context(name, loaded_all[name], common_ids, weights)
        products[name] = arrays
        summaries[name] = summary
        diagnostic_path = RESULT_DIR / f"{name}-split-half-diagnostics.npz"
        atomic_write(diagnostic_path, deterministic_npz(diagnostics))
        diagnostic_hashes[name] = sha256(diagnostic_path)
    passed = all(item["passesReliabilityGate"] for item in summaries.values())
    output_hashes = {}
    if passed:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name, arrays in products.items():
            path = DATA_DIR / f"{name}-control-coexpression64.npz"
            payload = deterministic_npz(arrays)
            atomic_write(path, payload)
            first_hash = sha256(path)
            if payload != deterministic_npz(arrays):
                raise ValueError("deterministic NPZ byte replay failed")
            output_hashes[name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": first_hash}
        anchor_path = DATA_DIR / "common-static-anchor64.npz"
        anchor_payload = deterministic_npz({
            "common_query_ids": common_ids,
            "normalized_static_features": normalized,
            "anchor_weights": weights,
            "gaussian_projection": gaussian,
            "entity_taxon": np.asarray(9606, dtype=np.int64),
        })
        atomic_write(anchor_path, anchor_payload)
        output_hashes["anchor"] = {"path": str(anchor_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(anchor_path)}
    report = {
        "schema": "slp.human-control-coexpression-reliability-report/v1",
        "status": "candidate-feature-packs-emitted" if passed else "reliability-gate-failed-feature-branch-stopped",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "frozenAnchorSha256": sha256(RESULT_DIR / "frozen-anchor.npz"),
        "contexts": summaries,
        "gatePassedAllContexts": passed,
        "diagnosticSha256": diagnostic_hashes,
        "outputs": output_hashes,
        "limitations": [
            "Fingerprints describe covariance among non-targeting control cells and contain no perturbation effect estimates.",
            "Split-half agreement can include stable GEM, technical, cell-state, and biological covariance; it does not establish causal regulation.",
            "Action features are exact RNA-query identity lookups; actions absent from the native RNA panel remain explicit zero/missing rows.",
            "The fixed random anchor preserves a 64-dimensional sketch rather than the complete control covariance matrix.",
        ],
    }
    atomic_write(RESULT_DIR / "report.json", canonical_json(report))
    if passed:
        manifest = {
            "schema": "slp.human-control-coexpression-static-features/v1",
            "status": "complete-after-prespecified-control-only-reliability-gate",
            "report": {"path": str((RESULT_DIR / "report.json").relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(RESULT_DIR / "report.json")},
            "outputs": output_hashes,
            "featureColumns": "64 leave-self-out Pearson correlations to the shared fixed static-derived anchor",
            "coverage": summaries,
        }
        atomic_write(DATA_DIR / "manifest.json", canonical_json(manifest))
    print(json.dumps({"passed": passed, "report": str(RESULT_DIR / "report.json"), "reportSha256": sha256(RESULT_DIR / "report.json"), "outputs": output_hashes}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("choose exactly one of --prepare or --run")
    prepare() if args.prepare else run()


if __name__ == "__main__":
    main()
