"""Build a benchmark-agnostic multi-action DLD1 pack from GSE337988."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import warnings

import h5py
import numpy as np
import rdata


ROOT = Path(__file__).resolve().parents[2]
MOIS = ("0.1", "0.2", "0.5", "1.0", "3.0", "5.0")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def assignments(obj, guide: dict[str, str]):
    columns = obj.colData.listData
    cells = np.asarray(columns["cell"], str)
    bad = np.asarray(columns["is.bad"], bool)
    raw = np.asarray(columns["assignment"], str)
    members: list[tuple[str, ...]] = []
    kind = np.full(len(cells), -1, dtype="int8")
    for index, text in enumerate(raw):
        guide_ids = [] if text in ("", "NA") else text.split(",")
        targets = [guide.get(guide_id) for guide_id in guide_ids]
        if bad[index] or not guide_ids or any(target is None for target in targets):
            members.append(())
            continue
        if all(target == "NTC" for target in targets):
            kind[index] = 0
            members.append(())
            continue
        genes = tuple(sorted(set(targets)))
        if "NTC" in targets or len(genes) != len(targets) or len(genes) > 8:
            members.append(())
            continue
        kind[index] = len(genes)
        members.append(genes)
    return cells, members, kind


def source_rows(raw: Path, moi: str, guide: dict[str, str], action_id: dict[str, int]):
    rds = raw / f"GSE337988_pilot_processed_objects_MOI_{moi}_se.rds"
    h5 = raw / f"GSE337988_pilot_processed_objects_MOI_{moi}_assays.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rds(rds)
    cells, members, cardinality = assignments(obj, guide)
    symbols = np.char.upper(np.asarray(obj.rowRanges.elementMetadata.listData["symbol"], str))
    support: dict[tuple[str, ...], int] = {}
    for genes, card in zip(members, cardinality):
        if card > 0:
            support[genes] = support.get(genes, 0) + 1
    replicates = {genes: min(4, count // 4) for genes, count in support.items() if count >= 4}
    keys = sorted((genes, replicate) for genes, count in replicates.items() for replicate in range(count))
    key_id = {key: index for index, key in enumerate(keys)}
    group = np.full(len(cells), -1, dtype="int32")
    supported_rows: dict[tuple[str, ...], list[int]] = {genes: [] for genes in replicates}
    for index, genes in enumerate(members):
        if genes in supported_rows:
            supported_rows[genes].append(index)
    for genes, indices in supported_rows.items():
        ordered = sorted(indices, key=lambda index: cell_hash(cells[index]))
        for rank, index in enumerate(ordered):
            group[index] = key_id[(genes, rank % replicates[genes])]
    counts = np.bincount(group[group >= 0], minlength=len(keys)).astype("int32")
    sums = np.zeros((len(keys), len(symbols)), dtype="float32")
    control_sum = np.zeros(len(symbols), dtype="float64")
    control_cells = 0
    with h5py.File(h5) as handle:
        matrix = handle["assay001"]
        if matrix.shape != (len(cells), len(symbols)):
            raise ValueError(f"matrix axes disagree for MOI {moi}")
        for start in range(0, len(cells), 256):
            stop = min(len(cells), start + 256)
            state = np.asarray(matrix[start:stop], dtype="float32")
            library_size = state.sum(axis=1).clip(1)
            state *= (1e4 / library_size)[:, None]
            np.log1p(state, out=state)
            batch_cardinality = cardinality[start:stop]
            controls = batch_cardinality == 0
            if np.any(controls):
                control_sum += state[controls].sum(axis=0)
                control_cells += int(controls.sum())
            batch_group = group[start:stop]
            for group_id in np.unique(batch_group[batch_group >= 0]):
                sums[group_id] += state[batch_group == group_id].sum(axis=0)
    if not control_cells or np.any(counts == 0):
        raise ValueError(f"missing controls or empty pseudoreplicate for MOI {moi}")
    control = control_sum / control_cells
    target = sums / counts[:, None] - control
    actions = np.full((len(keys), 8), -1, dtype="int32")
    for row, (genes, _) in enumerate(keys):
        actions[row, : len(genes)] = [action_id[gene] for gene in genes]
    audit = {
        "moi": moi,
        "cells": len(cells),
        "control_cells": control_cells,
        "supported_sets": len(replicates),
        "pseudobulks": len(keys),
        "pseudobulks_by_cardinality": {
            str(card): int(((actions >= 0).sum(axis=1) == card).sum())
            for card in np.unique((actions >= 0).sum(axis=1))
        },
        "rds_sha256": sha256(rds),
        "h5_sha256": sha256(h5),
    }
    return actions, target.astype("float32"), counts, symbols, audit


def build(raw: Path, output: Path) -> dict[str, object]:
    raw = Path(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    guide_path = raw / "GSE337988_NGS6194_crispr_demux_index_map.csv.gz"
    guide = {
        row["index"]: row["gene_symbol"].upper()
        for row in csv.DictReader(gzip.open(guide_path, "rt"))
    }
    action_names = np.asarray(sorted(set(guide.values()) - {"NTC"}))
    action_id = {name: index for index, name in enumerate(action_names)}
    rows = [source_rows(raw, moi, guide, action_id) for moi in MOIS]
    features = rows[0][3]
    if any(not np.array_equal(features, row[3]) for row in rows[1:]):
        raise ValueError("expression feature axes differ across MOI")
    actions = np.concatenate([row[0] for row in rows])
    target = np.concatenate([row[1] for row in rows])
    condition = np.concatenate(
        [np.repeat(f"CRISPRi|120h|MOI={moi}", len(row[0])) for moi, row in zip(MOIS, rows)]
    )
    cell_count = np.concatenate([row[2] for row in rows])
    valid = actions >= 0
    action_modes = np.full(actions.shape, "", dtype="<U10")
    action_modes[valid] = "repression"
    action_doses = valid.astype("int8")
    pack = output / "gse337988_dld1_multi_action_pseudobulk_v1.npz"
    np.savez_compressed(
        pack,
        actions=actions,
        action_modes=action_modes,
        action_doses=action_doses,
        action_names=action_names,
        target=target,
        target_semantics=np.asarray("perturbation_delta"),
        target_feature_name=features,
        cardinality=valid.sum(axis=1).astype("int8"),
        source_id=np.asarray("GSE337988"),
        context_id=np.asarray("DLD1"),
        experimental_condition_id=condition,
        cell_count=cell_count,
    )
    source_manifest = raw / "source_manifest.json"
    audit = {
        "schema": "slp-data-release-audit-v1",
        "release_id": "data/perturbseq/gse337988-dld1-multi-action-pseudobulk-v1",
        "source": {
            "name": "Joint analysis of multiply perturbed cells improves statistical power and cost efficiency in Perturb-seq screens",
            "accession": "GSE337988",
            "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE337988",
            "source_manifest_sha256": sha256(source_manifest) if source_manifest.exists() else None,
        },
        "license": {
            "id": "NCBI-GEO-PUBLIC-DATA",
            "evidence": "NCBI states that it places no restrictions on use or distribution of GEO data, while noting that submitters may assert rights.",
            "policy": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
        },
        "transformations": "Strict source guide assignments were mapped to distinct non-targeting-free gene sets. Sets with at least four cells were split into one to four deterministic barcode-hash pseudoreplicates. Counts were normalized per cell as log1p counts per 10,000, averaged, and differenced from the MOI-matched non-targeting mean.",
        "schema_description": "NPZ with up to eight gene actions per row, per-action repression mode and dose, a 35-target vocabulary, full-gene perturbation-delta state, DLD1 context, six MOI conditions, and source cell counts.",
        "population": "DLD1 Zim3-dCas9 CRISPRi cells across six pilot multiplicities of infection",
        "endpoints": ["mean per-cell log1p(CP10K) expression change from MOI-matched non-targeting control"],
        "split_construction": "No train/test split is embedded. The hard generalization gate constructs deterministic folds downstream.",
        "exclusions": "Bad, unassigned, mixed non-targeting, repeated-target-guide, and exact sets supported by fewer than four cells were excluded. Raw RDS/H5 files are checksum-manifested locally but not duplicated in this cleaned release.",
        "rows": len(target),
        "unique_action_targets": len(action_names),
        "expression_features": len(features),
        "sources": [row[4] for row in rows],
        "sl_labels_used": False,
        "files": [{"path": pack.name, "bytes": pack.stat().st_size, "sha256": sha256(pack)}],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/gse337988_pilot")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/data/gse337988-dld1-multi-action-pseudobulk-v1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
