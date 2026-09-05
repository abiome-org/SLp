"""Build a metadata-only routing sidecar for Replogle K562 essential cells."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/sources/replogle-2022-k562-essential-singlecell-v1"
SOURCE = SOURCE_DIR / "K562_essential_raw_singlecell_01.h5ad"
OUTPUT = ROOT / "data/derived/slp11-human-k562-essential-singlecell-metadata-v1"
REPORT = ROOT / "results/slp11-transition/k562-essential-singlecell-metadata-audit-v1"
SOURCE_BYTES = 10_661_879_995
SOURCE_MD5 = "4f1122ce1c7f13299a68df6459a266d3"
SOURCE_SHA256 = "3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee"
ENSG = re.compile(r"^ENSG[0-9]+$")
TERMINAL_ENSG = re.compile(r"_(ENSG[0-9]+)$")
CONTROL = "non-targeting"


def action_role(action: str, seed: int = 731) -> str:
    if action == "":
        return "control"
    bucket = int.from_bytes(hashlib.sha256(f"slp11-development-v1|{seed}|9606|{action}".encode()).digest()[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test-excluded"


def reconstruction_role(barcode: str, intervention_role: str, seed: int = 731) -> str:
    if intervention_role not in {"train", "control"}:
        return "none"
    bucket = int.from_bytes(hashlib.sha256(f"slp11-k562-essential-cell-reconstruction-v1|{seed}|{barcode}".encode()).digest()[:8], "big") % 100
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
    width = max(1, *(len(str(x)) for x in values))
    return np.asarray(values, dtype=f"<U{width}")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _quantiles(values: list[int]) -> dict[str, float]:
    q = np.quantile(values, [0, .1, .5, .9, 1])
    return dict(zip(("minimum", "p10", "median", "p90", "maximum"), map(float, q), strict=True))


def run() -> dict[str, object]:
    receipt = json.loads((SOURCE_DIR / "complete.json").read_text())
    if SOURCE.stat().st_size != SOURCE_BYTES or receipt.get("md5") != SOURCE_MD5 or receipt.get("sha256") != SOURCE_SHA256:
        raise ValueError("verified source receipt drift")
    with h5py.File(SOURCE, "r") as handle:
        if set(handle) != {"X", "obs", "var"}:
            raise ValueError("unexpected H5AD root")
        matrix = handle["X"]
        if matrix.shape != (310_385, 8_563) or matrix.dtype != np.float32:
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
        structure = {"shape": list(matrix.shape), "dtype": str(matrix.dtype), "chunks": matrix.chunks, "compression": matrix.compression, "storageBytes": int(matrix.id.get_storage_size())}
    if len(set(barcodes)) != len(barcodes) or len(set(query_ids)) != len(query_ids) or any(ENSG.fullmatch(x) is None for x in query_ids):
        raise ValueError("cell or query stable identities invalid")
    controls = gene_id == CONTROL
    control_agreement = controls & (gene == CONTROL) & (transcript == CONTROL) & np.char.endswith(population, "_non-targeting_non-targeting_non-targeting")
    control_guide = np.asarray([all(part.startswith("non-targeting_") for part in value.split("|")) for value in guide_pair])
    if not np.array_equal(controls, control_agreement) or not np.array_equal(controls, control_guide):
        raise ValueError("control metadata fields disagree")
    actions = np.where(controls, "", gene_id)
    if any(ENSG.fullmatch(x) is None for x in actions[~controls]):
        raise ValueError("target action IDs are not stable ENSG")
    terminal = np.asarray([TERMINAL_ENSG.search(value).group(1) if TERMINAL_ENSG.search(value) else "" for value in population])
    if not np.array_equal(terminal[~controls], actions[~controls]):
        raise ValueError("population terminal ENSG disagrees with action")
    if any(len(value.split("|")) != 2 for value in guide_pair):
        raise ValueError("guide-pair cardinality drift")
    roles = np.asarray([action_role(str(action)) for action in actions])
    reconstruction = np.asarray([reconstruction_role(str(barcode), str(role)) for barcode, role in zip(barcodes, roles, strict=True)])
    groups = Counter(zip(population.tolist(), gem_group.tolist(), strict=True))
    reconstruction_groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for pop, gem, role in zip(population, gem_group, reconstruction, strict=True):
        if role != "none":
            reconstruction_groups[(str(pop), int(gem))].add(str(role))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    metadata_path = OUTPUT / "cell-routing-metadata.npz"
    np.savez_compressed(
        metadata_path,
        schema=np.asarray("slp.replogle-k562-essential-cell-routing/v1"),
        source_sha256=np.asarray(SOURCE_SHA256), source_row_index=np.arange(len(barcodes), dtype=np.int64),
        cell_ids=_strings(barcodes), context_id=np.asarray("replogle-2022-k562-essential-day-6"),
        entity_taxon=np.full(len(barcodes), 9606, np.int64), action_ids=_strings(actions),
        intervention_role=_strings(roles), reconstruction_role=_strings(reconstruction),
        is_control=controls, gene_symbols=_strings(gene), gene_transcript=_strings(population),
        transcript_labels=_strings(transcript), guide_pair_ids=_strings(guide_pair), gem_group=gem_group,
        umi_count=umi_count, core_adjusted_umi_count=core_adjusted, core_scale_factor=core_scale,
        z_gemgroup_umi=z_umi, mitochondrial_fraction=mito,
        query_ids=_strings(query_ids), query_taxon=np.full(len(query_ids), 9606, np.int64),
        query_names=_strings(query_names), query_in_matrix=in_matrix,
        matrix_value_space=np.asarray("raw nonnegative UMI counts stored float32"),
        library_size_definition=np.asarray("sum raw X across exact ordered 8563 source query columns; equality to obs/UMI_count requires later allowlisted X audit"),
    )
    role_counts = Counter(roles.tolist())
    role_genes = {role: len(set(actions[roles == role].tolist()) - {""}) for role in sorted(role_counts)}
    control_by_gem = Counter(gem_group[controls].tolist())
    report = {
        "schema": "slp.replogle-k562-essential-singlecell-metadata-audit/v1",
        "source": {"path": str(SOURCE.relative_to(ROOT)), "bytes": SOURCE_BYTES, "md5": SOURCE_MD5, "sha256": SOURCE_SHA256, "figshareFileId": 35773219, "url": "https://ndownloader.figshare.com/files/35773219", "rights": "rights/figshare-replogle-2022-k562-essential-singlecell-cc-by-4.0.yaml"},
        "matrixStructure": structure,
        "cells": len(barcodes), "queries": len(query_ids), "queryRosterOrderedSha256": hashlib.sha256(("\n".join(query_ids) + "\n").encode()).hexdigest(),
        "queryInMatrix": int(in_matrix.sum()), "gemGroups": len(set(gem_group.tolist())),
        "interventionRoleCells": dict(sorted(role_counts.items())), "interventionRoleGenes": role_genes,
        "controls": {"cells": int(controls.sum()), "guidePairsAllVerifiedNonTargeting": True, "gemGroups": len(control_by_gem), "cellsPerGemGroup": _quantiles(list(control_by_gem.values()))},
        "actionMetadata": {"uniqueStableGenes": len(set(actions.tolist()) - {""}), "populationLabels": len(set(population.tolist())), "stableGuidePairs": len(set(guide_pair.tolist())), "guidesPerCell": 2, "populationActionExactAgreement": True, "guidePairRetainedAsProvenanceNotModelIdentity": True},
        "populationGroups": {"definition": "exact gene_transcript x gem_group", "groups": len(groups), "cellsPerGroup": _quantiles(list(groups.values())), "groupsWithAtLeast2Cells": sum(n >= 2 for n in groups.values()), "groupsWithAtLeast10Cells": sum(n >= 10 for n in groups.values())},
        "reconstructionSplit": {"domain": "fitting-action plus verified control cells only", "hash": "sha256(slp11-k562-essential-cell-reconstruction-v1|731|cell_barcode), first8bytes mod100 <90 train", "counts": dict(sorted(Counter(reconstruction.tolist()).items())), "exactPopulationGemGroupsWithBothRoles": sum(value == {"train", "validation"} for value in reconstruction_groups.values()), "interpretation": "held cells from already fitting interventions/controls; reconstruction validation is not held-intervention evidence"},
        "normalizationPlan": {"rawDenominator": "each allowlisted cell's exact sum over all 8563 ordered source X columns", "rawDenominatorRoster": "query_ids in cell-routing-metadata.npz", "authorNormalizedAlternative": "per-cell UMI scaling to experiment median core-control UMI then per-gemgroup gene-wise core-control z score", "controls": "verified non-targeting guide pairs in every one of 48 gem groups", "noEfficacyPredictor": True},
        "shardPlan": {"accessOrder": "metadata role allowlist before every X row selection", "fitting": "train actions plus controls in sparse CSR shards; reconstruction roles retained", "developmentValidation": "separate validation-action shards", "excluded": "test-excluded action X rows never read", "maximumRowsPerShard": 2048, "noDenseWholeMatrix": True},
        "metadataSidecar": {"path": str(metadata_path.relative_to(ROOT)), "bytes": metadata_path.stat().st_size, "sha256": _sha(metadata_path)},
        "access": {"XValuesRead": 0, "validationOutcomeValuesRead": 0, "testOutcomeValuesRead": 0, "metadataRowsRead": len(barcodes)},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    REPORT.mkdir(parents=True, exist_ok=False)
    (REPORT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    value = run()
    print(json.dumps({"metadata": value["metadataSidecar"], "report": str((REPORT / 'report.json').relative_to(ROOT)), "reportSha256": _sha(REPORT / "report.json")}))
