"""Build metadata-only action/observation sidecars for existing development corpora."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "modules/slp-1-1-world-transition-v1/action_observation_metadata.py"
REPL_PATH = ROOT / "data/derived/slp11-human/replogle-k562-rpe1-author-normalized-development-v2.npz"
REPL_SHA = "88de5164fca4e2504ac5b459ab4226c161eb586dd04700d5784da4bb53048659"
NORMAN_PATH = ROOT / "data/derived/slp11-norman-author-normalized-v2/norman-2019-author-normalized-development-v2.npz"
NORMAN_SHA = "ab81e7ed07d7f111b3dfc964cece28a2db7de0dcf5975f6ff1a3bc2db0be683e"
YEAST_PATH = ROOT / ".omf/runs/01a06e28-2f3f-7ccb-b71d-8b7654fc26ca/stages/compose/proteome-corpus-compose-v1/corpus-v1-2.tar"
YEAST_SHA = "0a5322c46e15e8a15d17000e8993c0ad642fcc70bc8fff00cbba8fb2905708bf"
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-action-observation-metadata-v1"


def _load_module():
    spec = importlib.util.spec_from_file_location("action_observation_metadata", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load metadata contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or path.is_symlink() or _hash(resolved) != expected:
        raise RuntimeError(f"source snapshot drift: {path}")
    return resolved


def _string(values: list[str] | tuple[str, ...] | np.ndarray) -> np.ndarray:
    items = [str(item) for item in values]
    return np.asarray(items, dtype=f"<U{max(1, max(map(len, items), default=1))}")


def _scalar(value: str) -> np.ndarray:
    return np.asarray(value, dtype=f"<U{len(value)}")


def _common(
    *, source_id: str, source_sha: str, records: np.ndarray, contexts: np.ndarray,
    context_index: np.ndarray, offsets: np.ndarray, actions: np.ndarray, taxon: int,
    mode: str, exposure: np.ndarray, exposure_present: np.ndarray,
    constructs: np.ndarray, construct_present: np.ndarray, replicates: np.ndarray,
    replicate_present: np.ndarray, observation_units: np.ndarray,
    observation_unit_present: np.ndarray, query_ids: np.ndarray, query_taxon: int,
    query_modality: str, value_space: str, normalization_group: str,
    dose_status: str, efficacy_status: str, construct_role: str,
) -> dict[str, np.ndarray]:
    count = len(actions)
    return {
        "schema": _scalar("slp.action-observation-metadata/v1"),
        "source_id": _scalar(source_id), "source_sha256": _scalar(source_sha),
        "record_ids": _string(records), "context_ids": _string(contexts),
        "context_index": np.asarray(context_index, dtype=np.int64),
        "action_offsets": np.asarray(offsets, dtype=np.int64),
        "action_taxon": np.full(count, taxon, dtype=np.int64), "action_ids": _string(actions),
        "action_mode": _string([mode] * count),
        "action_mode_present": np.ones(count, dtype=np.bool_),
        "action_dose": np.zeros(count, dtype=np.float32),
        "action_dose_present": np.zeros(count, dtype=np.bool_),
        "action_efficacy": np.zeros(count, dtype=np.float32),
        "action_efficacy_present": np.zeros(count, dtype=np.bool_),
        "exposure_days": np.asarray(exposure, dtype=np.float32),
        "exposure_days_present": np.asarray(exposure_present, dtype=np.bool_),
        "construct_ids": _string(constructs),
        "construct_present": np.asarray(construct_present, dtype=np.bool_),
        "replicate_ids": _string(replicates),
        "replicate_present": np.asarray(replicate_present, dtype=np.bool_),
        "observation_unit_ids": _string(observation_units),
        "observation_unit_present": np.asarray(observation_unit_present, dtype=np.bool_),
        "query_taxon": np.full(len(query_ids), query_taxon, dtype=np.int64),
        "query_ids": _string(query_ids), "query_modality": _scalar(query_modality),
        "target_value_space": _scalar(value_space),
        "normalization_group": _scalar(normalization_group),
        "dose_status": _scalar(dose_status), "efficacy_status": _scalar(efficacy_status),
        "construct_role": _scalar(construct_role),
        "time_source_confounded": np.asarray(True, dtype=np.bool_),
        "guide_ids_model_input_allowed": np.asarray(False, dtype=np.bool_),
    }


def replogle_arrays(path: Path = REPL_PATH) -> dict[str, np.ndarray]:
    with np.load(_verify(path, REPL_SHA), allow_pickle=False) as source:
        records = source["record_ids"]
        contexts = source["context_ids"]
        context_index = source["context_index"]
        actions = source["action_ids"]
        query_ids = source["query_ids"]
        value_space = str(source["target_value_space"].item())
    constructs = _string([item.split("|", 1)[1] for item in records])
    days = np.asarray([6.0, 7.0], dtype=np.float32)[context_index]
    n = len(records)
    return _common(
        source_id="figshare-plus:20029387/replogle-2022-essential", source_sha=REPL_SHA,
        records=records, contexts=contexts, context_index=context_index,
        offsets=np.arange(n + 1), actions=actions, taxon=9606, mode="crispri-repression",
        exposure=days, exposure_present=np.ones(n, dtype=np.bool_), constructs=constructs,
        construct_present=np.ones(n, dtype=np.bool_), replicates=_string([""] * n),
        replicate_present=np.zeros(n, dtype=np.bool_), observation_units=_string([""] * n),
        observation_unit_present=np.zeros(n, dtype=np.bool_), query_ids=query_ids,
        query_taxon=9606, query_modality="single-cell-derived-pseudobulk-RNA",
        value_space=value_space, normalization_group="replogle-author-per-gemgroup-core-control-zscore-v1",
        dose_status="unknown-not-deposited-as-model-input",
        efficacy_status="post-intervention-expression-is-an-outcome-and-forbidden-as-forecast-input",
        construct_role="exact-source-population-label-audit-only-not-learnable-identity",
    )


def norman_arrays(path: Path = NORMAN_PATH) -> dict[str, np.ndarray]:
    with np.load(_verify(path, NORMAN_SHA), allow_pickle=False) as source:
        records = source["record_ids"]
        contexts = source["context_ids"]
        context_index = source["context_index"]
        offsets = source["action_offsets"]
        actions = source["action_ids"]
        constructs = source["construct_ids"]
        query_ids = source["query_ids"]
        value_space = str(source["target_value_space"].item())
    n = len(records)
    return _common(
        source_id="GEO:GSE133344", source_sha=NORMAN_SHA, records=records,
        contexts=contexts, context_index=context_index, offsets=offsets, actions=actions,
        taxon=9606, mode="crispra-activation", exposure=np.full(n, 5.0, np.float32),
        exposure_present=np.ones(n, dtype=np.bool_), constructs=constructs,
        construct_present=np.ones(n, dtype=np.bool_), replicates=_string([""] * n),
        replicate_present=np.zeros(n, dtype=np.bool_), observation_units=_string([""] * n),
        observation_unit_present=np.zeros(n, dtype=np.bool_), query_ids=query_ids,
        query_taxon=9606, query_modality="single-cell-derived-pseudobulk-RNA",
        value_space=value_space, normalization_group="norman-slp-percell-full-library-core-control-zscore-v2",
        dose_status="unknown-not-deposited-as-model-input",
        efficacy_status="post-intervention-expression-is-an-outcome-and-forbidden-as-forecast-input",
        construct_role="exact-author-guide-identity-audit-only-not-learnable-identity",
    )


def _read_npz_member(tar: tarfile.TarFile, name: str) -> dict[str, np.ndarray]:
    stream = tar.extractfile(name)
    if stream is None:
        raise RuntimeError(f"missing archive member: {name}")
    with np.load(io.BytesIO(stream.read()), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def yeast_arrays(path: Path = YEAST_PATH) -> dict[str, np.ndarray]:
    _verify(path, YEAST_SHA)
    with tarfile.open(path, "r:") as archive:
        names = {item.name for item in archive.getmembers() if item.isfile()}
        manifest_stream = archive.extractfile("composite-corpus/corpus.json")
        if manifest_stream is None:
            raise RuntimeError("yeast corpus manifest missing")
        manifest = json.load(manifest_stream)
        entities = _read_npz_member(archive, "composite-corpus/entities.npz")
        queries = _read_npz_member(archive, "composite-corpus/queries.npz")
        shard_names = [f"composite-corpus/{item['path']}" for item in manifest["shards"]]
        if not set(shard_names).issubset(names):
            raise RuntimeError("yeast corpus shard missing")
        keys = ("record_id", "replicate_id", "observation_unit_id", "action_entity_index")
        shard_arrays = [_read_npz_member(archive, name) for name in shard_names]
        merged = {key: np.concatenate([item[key] for item in shard_arrays], axis=0) for key in keys}
    records = merged["record_id"]
    n = len(records)
    action_index = merged["action_entity_index"][:, 0]
    actions = entities["entity_id"][action_index]
    query_index = queries["query_entity_index"]
    query_ids = entities["entity_id"][query_index]
    context_id = "slp-context:mendeley-w8jtmnszd9-v2-prototrophic-sm"
    return _common(
        source_id="mendeley:w8jtmnszd9.2", source_sha=YEAST_SHA, records=records,
        contexts=_string([context_id]), context_index=np.zeros(n, dtype=np.int64),
        offsets=np.arange(n + 1), actions=actions, taxon=4932, mode="gene-deletion",
        exposure=np.zeros(n, dtype=np.float32), exposure_present=np.zeros(n, dtype=np.bool_),
        constructs=_string([""] * n), construct_present=np.zeros(n, dtype=np.bool_),
        replicates=merged["replicate_id"], replicate_present=np.ones(n, dtype=np.bool_),
        observation_units=merged["observation_unit_id"],
        observation_unit_present=np.ones(n, dtype=np.bool_), query_ids=query_ids,
        query_taxon=4932, query_modality="quantitative-proteome-relative-intensity",
        value_space=manifest["normalization"]["valueSpace"],
        normalization_group="yeast-log2-batch-corrected-maxlfq-relative-intensity-v1",
        dose_status="not-applicable-to-constitutive-deletion; deletion-completeness-not-quantified-as-input",
        efficacy_status="separate-knockout-protein-detection-is-downstream-QC-and-forbidden-as-forecast-input",
        construct_role="no-delivery-construct-in-canonical-corpus; exact-sample-replicate-provenance-retained",
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    path.write_bytes(buffer.getvalue())


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    if output.exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    contract = _load_module()
    bundles = {
        "replogle-development-action-observation-metadata-v1.npz": replogle_arrays(),
        "norman-development-action-observation-metadata-v1.npz": norman_arrays(),
        "yeast-proteome-fitting-action-observation-metadata-v1.npz": yeast_arrays(),
    }
    output.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, object] = {}
    for name, arrays in bundles.items():
        contract.validate_arrays(arrays)
        path = output / name
        _write_npz(path, arrays)
        artifacts[name] = {
            "sha256": _hash(path), "bytes": path.stat().st_size,
            "records": len(arrays["record_ids"]), "actions": len(arrays["action_ids"]),
            "queries": len(arrays["query_ids"]), "mode": sorted(set(arrays["action_mode"].tolist())),
            "exposureDaysPresent": int(arrays["exposure_days_present"].sum()),
            "dosePresent": int(arrays["action_dose_present"].sum()),
            "efficacyPresent": int(arrays["action_efficacy_present"].sum()),
        }
    manifest = {
        "schema": "slp.action-observation-metadata-sidecars/v1",
        "status": "derived-metadata-only-not-omf-admitted",
        "contractSource": {"path": str(MODULE_PATH.relative_to(ROOT)), "sha256": _hash(MODULE_PATH)},
        "inputs": {
            "replogleDevelopment": {"path": str(REPL_PATH.relative_to(ROOT)), "sha256": REPL_SHA},
            "normanDevelopment": {"path": str(NORMAN_PATH.relative_to(ROOT)), "sha256": NORMAN_SHA},
            "yeastFittingCorpus": {"path": str(YEAST_PATH.relative_to(ROOT)), "sha256": YEAST_SHA},
        },
        "artifacts": artifacts,
        "inputAccess": {
            "human": "development train-plus-validation arrays only; test-only files unopened",
            "yeast": "fitting-only composed corpus metadata arrays",
            "hepg2Jurkat": "no perturbed outcomes accessed",
            "syntheticLethality": "not accessed",
        },
        "modelInputPolicy": {
            "allowed": ["stable taxon-qualified action gene set", "verified intervention mode", "verified exposure days"],
            "auditOnly": ["guide/construct ID", "replicate ID", "observation-unit ID"],
            "forbidden": ["post-intervention efficacy", "target-gene response as an input", "mode-derived fabricated strength"],
        },
        "jointModelAssessment": {
            "directionConditioningPossible": True,
            "reason": "all three corpora have verified discrete intervention mode and stable target-gene sets",
            "strengthConditioningPossible": False,
            "timeConditioningScope": "verified for Replogle and Norman only; yeast exposure day is absent",
            "normalizationMatched": False,
            "normalizationGroups": [
                "Replogle author per-gemgroup core-control z-score pseudobulk RNA",
                "Norman SLp per-cell full-library core-control z-score pseudobulk RNA",
                "yeast log2 batch-corrected MaxLFQ relative-intensity proteome",
            ],
            "confounding": "mode, source, assay, normalization, time and mostly context are not independently crossed",
            "recommendedNextStep": "add a learned mode token and assay/value-space-specific decoder heads; retain source-stratified fitting and within-source held-gene evaluation before testing cross-mode transfer",
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"manifest": str(manifest_path), "manifestSha256": _hash(manifest_path), **manifest}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
