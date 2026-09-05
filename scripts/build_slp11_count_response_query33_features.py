"""Build source-native fitting-derived response-query33 feature packs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "modules/slp-1-1-count-response-query-features-v1/response_query_features.py"
OUTPUT = ROOT / "data/derived/slp11-human-count-response-query33/rank32-alpha1000-full-fitting-v3"
MODELS = {
    "k562": {
        "path": ROOT / "results/slp11-transition/human-essential-count-response-rank32-seed731-v1/model-k562.npz",
        "sha256": "6267584a4a69dc30899b18d0c9660e0c73d2b8383a1e4911571295a1ea57ae44",
        "context_id": "replogle-2022-k562-essential-day-6",
    },
    "rpe1": {
        "path": ROOT / "results/slp11-transition/human-essential-count-response-rank32-seed731-v1/model-rpe1.npz",
        "sha256": "ff864e96d02fb81b64baadc36c164de61a01d9e7d31a2609f78b64d48107be70",
        "context_id": "replogle-2022-rpe1-essential-day-7",
    },
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


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build(output: Path) -> dict:
    core = load_module(CORE, "slp11_count_response_query33_build")
    receipts = {}
    for source, entry in MODELS.items():
        model_path = entry["path"]
        if sha256(model_path) != entry["sha256"]:
            raise ValueError(f"rank32 model changed: {source}")
        with np.load(model_path, allow_pickle=False) as model:
            if str(model["schema"]) != "slp.reduced-rank-response-model/v1" or str(model["source_id"]) != source:
                raise ValueError(f"unexpected rank32 model identity: {source}")
            query_ids = model["query_ids"].astype(str)
            loading = np.asarray(model["query_loading"], dtype=np.float64)
            intercept = np.asarray(model["intercept"], dtype=np.float64)
            rank = int(model["rank"])
            alpha = float(model["alpha"])
        if rank != 32 or alpha != 1000.0:
            raise ValueError("response model parameterization mismatch")
        raw, rms, normalized = core.response_query33(loading, intercept)
        arrays = {
            "schema": np.asarray("slp.human-count-response-query33/v1"),
            "source_id": np.asarray(source),
            "context_id": np.asarray(entry["context_id"]),
            "entity_taxon": np.full(len(query_ids), 9606, dtype=np.int64),
            "query_ids": query_ids,
            "raw_response_query33": raw,
            "response_query33_rms": rms,
            "normalized_response_query33": normalized.astype(np.float32),
            "rank": np.asarray(rank, dtype=np.int64),
            "alpha": np.asarray(alpha, dtype=np.float64),
            "rank_model_sha256": np.asarray(entry["sha256"]),
            "query_ids_lf_sha256": np.asarray(core.lf_roster_sha256(query_ids)),
            "fitting_outcome_derived": np.asarray(True),
            "development_outcomes_accessed": np.asarray(False),
            "test_outcomes_accessed": np.asarray(False),
            "normalization_formula": np.asarray("raw33 / sqrt(mean_query(raw33^2)); no centering; exact-zero RMS replaced by1"),
        }
        core.validate_pack(arrays)
        path = output / f"response-query33-{source}.npz"
        write_new(path, deterministic_npz(arrays))
        receipts[source] = {
            "path": path.name,
            "sha256": sha256(path),
            "queries": len(query_ids),
            "queryIdsLfSha256": core.lf_roster_sha256(query_ids),
            "rankModel": {"path": str(model_path.relative_to(ROOT)).replace("\\", "/"), "sha256": entry["sha256"]},
            "contextId": entry["context_id"],
            "rawShape": list(raw.shape),
            "maximumNormalizedColumnRmsError": float(np.max(np.abs(np.sqrt(np.mean(np.square(normalized), axis=0)) - 1.0))),
        }
    manifest = {
        "schema": "slp.human-count-response-query33-manifest/v1",
        "status": "complete",
        "packs": receipts,
        "featureContract": {
            "modes": {
                "static-zero33": "query static577 plus33 zeros; action static577 plus33 zeros",
                "response33": "query static577 plus source-native normalized response33; action static577 plus33 zeros",
            },
            "width": 610,
            "panelAdapter": {"path": str(CORE.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(CORE)},
            "panelDataApi": {"path": "modules/slp-1-1-count-panel-data-v1/panel_data.py", "sha256": "a8f1ee3537041d20e1dda330c20ec0f73b3265ac63024eb3114ec1161d072c66"},
        },
        "provenance": "The 33 descriptors are derived from source-specific full-fitting molecular outcomes through frozen rank32 models. They are not static biology and cannot be treated as new-query or cross-context coordinates.",
        "alignment": "No Procrustes or other cross-source basis alignment is applied. K562 and RPE1 coordinates remain native and context dependent.",
        "outcomeAccess": {"fitting": True, "development": False, "test": False},
        "implementation": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(Path(__file__).resolve())},
    }
    manifest_path = output / "manifest.json"
    write_new(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2, sort_keys=True))
