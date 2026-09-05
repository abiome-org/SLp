"""Build a metadata-only routing sidecar for Replogle RPE1 essential cells."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/sources/replogle-2022-rpe1-essential-singlecell-v1"
SOURCE = SOURCE_DIR / "rpe1_raw_singlecell_01.h5ad"
OUTPUT = ROOT / "data/derived/slp11-human-rpe1-essential-singlecell-metadata-v1"
REPORT = ROOT / "results/slp11-transition/rpe1-essential-singlecell-metadata-audit-v1"
SOURCE_BYTES = 8_700_873_216
SOURCE_MD5 = "6a2a9d0d2bf4ec147f4d1104043b268c"
SOURCE_SHA256 = "9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1"
ENSG = re.compile(r"^ENSG[0-9]+$")
TERMINAL = re.compile(r"_([^_]+)$")
CONTROL = "non-targeting"


def action_role(action: str, seed: int = 731) -> str:
    if action == "":
        return "control"
    if ENSG.fullmatch(action) is None:
        return "unresolved-excluded"
    bucket = int.from_bytes(hashlib.sha256(f"slp11-development-v1|{seed}|9606|{action}".encode()).digest()[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test-excluded"


def reconstruction_role(barcode: str, intervention_role: str, seed: int = 731) -> str:
    if intervention_role not in {"train", "control"}:
        return "none"
    bucket = int.from_bytes(hashlib.sha256(f"slp11-rpe1-essential-cell-reconstruction-v1|{seed}|{barcode}".encode()).digest()[:8], "big") % 100
    return "train" if bucket < 90 else "validation"


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def categorical(handle: h5py.File, group: str, name: str) -> np.ndarray:
    codes = np.asarray(handle[f"{group}/{name}"][:], np.int64)
    categories = _decode(handle[f"{group}/__categories/{name}"][:])
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError(f"{group}/{name} has invalid categorical code")
    return categories[codes]


def _strings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=str)
    width = max(1, *(len(x) for x in values))
    return values.astype(f"<U{width}")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _quantiles(values: list[int]) -> dict[str, float]:
    result = np.quantile(values, [0, 0.1, 0.5, 0.9, 1])
    return dict(zip(("minimum", "p10", "median", "p90", "maximum"), map(float, result), strict=True))


def run() -> dict[str, object]:
    receipt = json.loads((SOURCE_DIR / "complete.json").read_text())
    if SOURCE.stat().st_size != SOURCE_BYTES or receipt.get("md5") != SOURCE_MD5 or receipt.get("sha256") != SOURCE_SHA256:
        raise ValueError("verified source receipt drift")
    with h5py.File(SOURCE, "r") as handle:
        if set(handle) != {"X", "obs", "var"}:
            raise ValueError("unexpected H5AD root")
        matrix = handle["X"]
        if matrix.shape != (247_914, 8_749) or matrix.dtype != np.float32:
            raise ValueError("raw single-cell matrix contract drift")
        barcodes = _decode(handle["obs/cell_barcode"][:])
        gem_group = np.asarray(handle["obs/gem_group"][:], np.int16)
        gene = categorical(handle, "obs", "gene")
        gene_id = categorical(handle, "obs", "gene_id")
        transcript = categorical(handle, "obs", "transcript")
        population = categorical(handle, "obs", "gene_transcript")
        guide_pair = categorical(handle, "obs", "sgID_AB")
        query_ids = _decode(handle["var/gene_id"][:])
        query_names = categorical(handle, "var", "gene_name")
        in_matrix = np.asarray(handle["var/in_matrix"][:], bool)
        umi_count = np.asarray(handle["obs/UMI_count"][:], np.float32)
        core_adjusted = np.asarray(handle["obs/core_adjusted_UMI_count"][:], np.float32)
        core_scale = np.asarray(handle["obs/core_scale_factor"][:], np.float32)
        z_umi = np.asarray(handle["obs/z_gemgroup_UMI"][:], np.float32)
        mito = np.asarray(handle["obs/mitopercent"][:], np.float32)
        structure = {"shape": list(matrix.shape), "dtype": str(matrix.dtype), "chunks": matrix.chunks, "compression": matrix.compression, "offset": int(matrix.id.get_offset()), "storageBytes": int(matrix.id.get_storage_size())}
    if len(set(barcodes)) != len(barcodes) or len(set(query_ids)) != len(query_ids) or any(ENSG.fullmatch(x) is None for x in query_ids) or not np.all(in_matrix):
        raise ValueError("cell or fully measured query identities invalid")
    controls = gene_id == CONTROL
    unresolved = gene_id == "nan"
    target = ~(controls | unresolved)
    control_agreement = controls & (gene == CONTROL) & (transcript == CONTROL) & np.char.endswith(population, "_non-targeting_non-targeting_non-targeting")
    control_guide = np.asarray([all(part.startswith("non-targeting_") for part in value.split("|")) for value in guide_pair])
    if not np.array_equal(controls, control_agreement) or not np.array_equal(controls, control_guide):
        raise ValueError("control metadata fields disagree")
    if any(ENSG.fullmatch(x) is None for x in gene_id[target]):
        raise ValueError("resolved action ID is not stable ENSG")
    if any(len(value.split("|")) != 2 for value in guide_pair):
        raise ValueError("guide-pair cardinality drift")
    terminal = np.asarray([TERMINAL.search(value).group(1) if TERMINAL.search(value) else "" for value in population])
    if not np.array_equal(terminal[target], gene_id[target]) or not np.all(terminal[unresolved] == "nan"):
        raise ValueError("population terminal action disagrees with gene_id")
    actions = np.where(controls, "", gene_id)
    roles = np.asarray([action_role(str(action)) for action in actions])
    reconstruction = np.asarray([reconstruction_role(str(barcode), str(role)) for barcode, role in zip(barcodes, roles, strict=True)])
    if not np.all(reconstruction[unresolved] == "none"):
        raise AssertionError("unresolved action entered reconstruction domain")
    groups = Counter(zip(population.tolist(), gem_group.tolist(), strict=True))
    reconstruction_groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for pop, gem, role in zip(population, gem_group, reconstruction, strict=True):
        if role != "none":
            reconstruction_groups[(str(pop), int(gem))].add(str(role))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    metadata_path = OUTPUT / "cell-routing-metadata.npz"
    np.savez_compressed(
        metadata_path,
        schema=np.asarray("slp.replogle-rpe1-essential-cell-routing/v1"), source_sha256=np.asarray(SOURCE_SHA256),
        source_row_index=np.arange(len(barcodes), dtype=np.int64), cell_ids=_strings(barcodes),
        context_id=np.asarray("replogle-2022-rpe1-essential-day-7"), entity_taxon=np.full(len(barcodes), 9606, np.int64),
        action_ids=_strings(actions), intervention_role=_strings(roles), reconstruction_role=_strings(reconstruction),
        is_control=controls, unresolved_action=unresolved, gene_symbols=_strings(gene), gene_transcript=_strings(population),
        transcript_labels=_strings(transcript), guide_pair_ids=_strings(guide_pair), gem_group=gem_group,
        umi_count=umi_count, core_adjusted_umi_count=core_adjusted, core_scale_factor=core_scale,
        z_gemgroup_umi=z_umi, mitochondrial_fraction=mito, query_ids=_strings(query_ids),
        query_taxon=np.full(len(query_ids), 9606, np.int64), query_names=_strings(query_names), query_in_matrix=in_matrix,
        matrix_value_space=np.asarray("raw nonnegative UMI counts stored float32"),
        library_size_definition=np.asarray("sum raw X across exact ordered 8749 source query columns; obs/UMI_count comparison is diagnostic because it may precede retained-gene filtering"),
    )
    role_counts = Counter(roles.tolist())
    role_genes = {role: len(set(actions[roles == role].tolist()) - {"", "nan"}) for role in sorted(role_counts)}
    control_by_gem = Counter(gem_group[controls].tolist())
    unresolved_detail = {symbol: int(np.sum(unresolved & (gene == symbol))) for symbol in sorted(set(gene[unresolved].tolist()))}
    report = {
        "schema": "slp.replogle-rpe1-essential-singlecell-metadata-audit/v1",
        "source": {"path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "bytes": SOURCE_BYTES, "md5": SOURCE_MD5, "sha256": SOURCE_SHA256, "figshareFileId": 35775606, "url": "https://ndownloader.figshare.com/files/35775606", "rights": "rights/figshare-replogle-2022-rpe1-essential-singlecell-cc-by-4.0.yaml"},
        "matrixStructure": structure, "cells": len(barcodes), "queries": len(query_ids),
        "queryRosterOrderedSha256": hashlib.sha256(("\n".join(query_ids) + "\n").encode()).hexdigest(),
        "queryInMatrix": int(in_matrix.sum()), "gemGroups": len(set(gem_group.tolist())),
        "interventionRoleCells": dict(sorted(role_counts.items())), "interventionRoleGenes": role_genes,
        "unresolvedActions": {"cells": int(unresolved.sum()), "sourceGeneId": "nan", "geneSymbols": unresolved_detail, "policy": "quarantine metadata; no quantitative X access"},
        "controls": {"cells": int(controls.sum()), "guidePairsAllVerifiedNonTargeting": True, "gemGroups": len(control_by_gem), "cellsPerGemGroup": _quantiles(list(control_by_gem.values()))},
        "actionMetadata": {"uniqueResolvedStableGenes": len(set(actions[target].tolist())), "populationLabels": len(set(population.tolist())), "stableGuidePairs": len(set(guide_pair.tolist())), "guidesPerCell": 2, "populationActionExactAgreement": True, "guidePairRetainedAsProvenanceNotModelIdentity": True},
        "populationGroups": {"definition": "exact gene_transcript x gem_group", "groups": len(groups), "cellsPerGroup": _quantiles(list(groups.values())), "groupsWithAtLeast2Cells": sum(n >= 2 for n in groups.values()), "groupsWithAtLeast10Cells": sum(n >= 10 for n in groups.values())},
        "reconstructionSplit": {"domain": "globally fitting stable actions plus verified controls only", "hash": "sha256(slp11-rpe1-essential-cell-reconstruction-v1|731|cell_barcode), first8bytes mod100 <90 train", "counts": dict(sorted(Counter(reconstruction.tolist()).items())), "exactPopulationGemGroupsWithBothRoles": sum(value == {"train", "validation"} for value in reconstruction_groups.values()), "interpretation": "held cells within fitting interventions/controls; not held-intervention evidence"},
        "plannedNormalization": {"rawDenominator": "each allowlisted cell exact sum over all8749 ordered source X columns", "rawDenominatorRoster": "query_ids in cell-routing-metadata.npz", "obsUmiCountUse": "diagnostic only", "controls": "verified non-targeting guide pairs across56 GEM groups"},
        "metadataSidecar": {"path": str(metadata_path.relative_to(ROOT)).replace("\\", "/"), "bytes": metadata_path.stat().st_size, "sha256": _sha(metadata_path)},
        "access": {"XValuesRead": 0, "validationOutcomeValuesRead": 0, "testOutcomeValuesRead": 0, "unresolvedOutcomeValuesRead": 0, "metadataRowsRead": len(barcodes)},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    REPORT.mkdir(parents=True, exist_ok=False)
    (REPORT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    value = run()
    print(json.dumps({"metadata": value["metadataSidecar"], "roles": value["interventionRoleCells"], "reportSha256": _sha(REPORT / "report.json")}))
