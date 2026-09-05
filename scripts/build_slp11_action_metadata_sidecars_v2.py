"""Build current three-context action metadata and explicit assay-head sidecars."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1_BUILDER_PATH = ROOT / "scripts/build_slp11_action_metadata_sidecars.py"
V2_CONTRACT_PATH = ROOT / "modules/slp-1-1-world-transition-v1/action_observation_metadata_v2.py"
sys.path.insert(0, str(V2_CONTRACT_PATH.parent))
CURRENT_PATH = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
CURRENT_SHA = "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c"
OUTPUT = ROOT / "data/derived/slp11-action-observation-metadata-v2"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decorate(
    arrays: dict[str, np.ndarray], *, transform: str, head: str,
) -> dict[str, np.ndarray]:
    result = dict(arrays)
    result["schema"] = np.asarray("slp.action-observation-metadata/v2")
    result["numeric_transform"] = np.asarray(transform)
    result["assay_head_id"] = np.asarray(head)
    result["assay_head_routing"] = np.asarray(
        "mechanical-source-plus-target-value-space-routing; no outcome-based selection"
    )
    result["assay_head_separation_required"] = np.asarray(True, dtype=np.bool_)
    result["mode_source_confounded"] = np.asarray(True, dtype=np.bool_)
    return result


def current_replogle(v1) -> dict[str, np.ndarray]:
    path = v1._verify(CURRENT_PATH, CURRENT_SHA)
    with np.load(path, allow_pickle=False) as source:
        records = source["record_ids"]
        contexts = source["context_ids"]
        context_index = source["context_index"]
        actions = source["action_ids"]
        query_ids = source["query_ids"]
        value_space = str(source["target_value_space"].item())
    expected_contexts = [
        "replogle-2022-k562-essential-day-6",
        "replogle-2022-rpe1-essential-day-7",
        "replogle-2022-k562-gwps-day-8",
    ]
    if contexts.tolist() != expected_contexts or len(records) != 13_058 or len(query_ids) != 7_036:
        raise RuntimeError("current three-context development contract drift")
    days = np.asarray([6.0, 7.0, 8.0], dtype=np.float32)[context_index]
    n = len(records)
    arrays = v1._common(
        source_id="figshare-plus:20029387/replogle-2022-essential-plus-gwps",
        source_sha=CURRENT_SHA, records=records, contexts=contexts,
        context_index=context_index, offsets=np.arange(n + 1), actions=actions,
        taxon=9606, mode="crispri-repression", exposure=days,
        exposure_present=np.ones(n, dtype=np.bool_),
        constructs=v1._string([item.split("|", 1)[1] for item in records]),
        construct_present=np.ones(n, dtype=np.bool_), replicates=v1._string([""] * n),
        replicate_present=np.zeros(n, dtype=np.bool_), observation_units=v1._string([""] * n),
        observation_unit_present=np.zeros(n, dtype=np.bool_), query_ids=query_ids,
        query_taxon=9606, query_modality="single-cell-derived-pseudobulk-RNA",
        value_space=value_space,
        normalization_group="replogle-author-per-gemgroup-core-control-zscore-v1",
        dose_status="unknown-not-deposited-as-model-input",
        efficacy_status="post-intervention-expression-is-an-outcome-and-forbidden-as-forecast-input",
        construct_role="exact-source-population-label-audit-only-not-learnable-identity",
    )
    return _decorate(
        arrays,
        transform=(
            "author-normalized: per-cell UMI scaling to experiment median core-control UMI; "
            "per-gemgroup per-gene core-control z-score; arithmetic mean across cells in "
            "each perturbation population; no second normalization or log transform"
        ),
        head="rna-replogle-author-gemgroup-control-zscore-v1",
    )


def build() -> dict[str, object]:
    if OUTPUT.exists():
        raise RuntimeError(f"immutable output already exists: {OUTPUT}")
    v1 = _module("slp_action_metadata_builder_v1", V1_BUILDER_PATH)
    contract = _module("action_observation_metadata_v2", V2_CONTRACT_PATH)
    bundles = {
        "replogle-current-three-context-development-metadata-v2.npz": current_replogle(v1),
        "norman-development-metadata-v2.npz": _decorate(
            v1.norman_arrays(),
            transform=(
                "per eligible cell: full 33694-source-gene UMI denominator, CP10K, "
                "log2(1+x); arithmetic mean within exact guide construct; per-query "
                "centering and scaling by eligible core-control-cell mean and population SD"
            ),
            head="rna-norman-percell-full-library-log2cp10k-control-zscore-v2",
        ),
        "yeast-proteome-fitting-metadata-v2.npz": _decorate(
            v1.yeast_arrays(),
            transform=(
                "log2 of each observed positive upstream batch-corrected MaxLFQ relative "
                "protein intensity; no pseudocount, centering or imputation"
            ),
            head="proteome-yeast-log2-batch-corrected-maxlfq-v1",
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, arrays in bundles.items():
        contract.validate_arrays(arrays)
        path = OUTPUT / name
        v1._write_npz(path, arrays)
        sizes = np.diff(arrays["action_offsets"])
        artifacts[name] = {
            "sha256": _hash(path), "bytes": path.stat().st_size,
            "records": len(arrays["record_ids"]), "actions": len(arrays["action_ids"]),
            "uniqueActionGenes": len(set(arrays["action_ids"].tolist())),
            "singleRecords": int(np.count_nonzero(sizes == 1)),
            "doubleRecords": int(np.count_nonzero(sizes == 2)),
            "queries": len(arrays["query_ids"]), "contexts": arrays["context_ids"].tolist(),
            "mode": sorted(set(arrays["action_mode"].tolist())),
            "targetValueSpace": str(arrays["target_value_space"].item()),
            "numericTransform": str(arrays["numeric_transform"].item()),
            "assayHeadId": str(arrays["assay_head_id"].item()),
            "dosePresent": int(arrays["action_dose_present"].sum()),
            "efficacyPresent": int(arrays["action_efficacy_present"].sum()),
        }
    manifest = {
        "schema": "slp.action-observation-metadata-sidecars/v2",
        "status": "derived-metadata-only-not-omf-admitted",
        "supersedesForCurrentTraining": "slp11-action-observation-metadata-v1 Replogle two-context sidecar",
        "historicalV1Preserved": True,
        "sources": {
            "currentThreeContextDevelopment": {"path": str(CURRENT_PATH.relative_to(ROOT)), "sha256": CURRENT_SHA},
            "normanDevelopment": {"path": str(v1.NORMAN_PATH.relative_to(ROOT)), "sha256": v1.NORMAN_SHA},
            "yeastFittingCorpus": {"path": str(v1.YEAST_PATH.relative_to(ROOT)), "sha256": v1.YEAST_SHA},
        },
        "sourceCode": {
            "contract": {"path": str(V2_CONTRACT_PATH.relative_to(ROOT)), "sha256": _hash(V2_CONTRACT_PATH)},
            "builder": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _hash(Path(__file__).resolve())},
            "v1MetadataExtraction": {"path": str(V1_BUILDER_PATH.relative_to(ROOT)), "sha256": _hash(V1_BUILDER_PATH)},
            "normanLiteralTransform": {"path": "modules/slp-1-1-world-transition-v1/norman_data_v2.py", "sha256": _hash(ROOT / "modules/slp-1-1-world-transition-v1/norman_data_v2.py")},
        },
        "artifacts": artifacts,
        "access": {
            "developmentMetadataOnly": True, "quantitativeOutcomeAnalysis": False,
            "testOnlyFilesAccessed": False, "hepg2JurkatPerturbedAccessed": False,
            "syntheticLethalityAccessed": False,
        },
        "mechanicalHeadSeparation": {
            "basis": "source plus target value-space metadata fixed without reading target values",
            "heads": sorted({item["assayHeadId"] for item in artifacts.values()}),
            "normalizationEquivalent": False,
            "modeSourceConfounded": True,
            "timeSourceConfounded": True,
            "dose": "unknown in every source; explicit masks false; no mode-derived surrogate",
            "efficacy": "post-intervention outcome/QC and forbidden as a forecast input; explicit masks false",
            "guideConstructIds": "audit-only provenance; forbidden as learnable identity shortcuts",
        },
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "manifestSha256": _hash(manifest_path), **manifest}


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
