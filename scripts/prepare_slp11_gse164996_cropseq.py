#!/usr/bin/env python3
"""Prepare condition-separated MCF10A CRISPR-KO population profiles from GSE164996."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import h5py
from scipy.io import mmread

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sources/gse164996-combinatorial-cropseq-v1"
REFERENCE = ROOT / "data/derived/slp11-joint-world-expanded-string-v1/gwps.npz"
FOUR_CONTEXT = ROOT / "data/derived/slp11-human-four-context-v2/development.npz"
GTF = ROOT / "data/sources/replogle-perturbseq-gi-code/data_sharing/cellranger-GRCh38-1.2.0_only_genes.gtf"
STATIC = ROOT / "data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2/human-static-esm8m-shared-go-mf-cc-features.npz"
STRING = ROOT / "data/tooling/slim-5a7e9ade/data/gene_string_embeddings.v0.3.h5"
CONDITIONS = {
    1: ("mcf10a_full_d0", "full medium", 0),
    2: ("mcf10a_full_d6", "full medium", 6),
    3: ("mcf10a_tgfb1_d6", "TGF-beta1 supplemented medium", 6),
    4: ("mcf10a_minimal_d6", "minimal medium", 6),
}
CONTROLS = frozenset({"CTRL0001", "CTRL0002"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guide_symbol(guide: str) -> str:
    return guide.rsplit("_", 1)[0]


def classify_call(feature_call: str, held_symbols: set[str]):
    """Return (kind, guide tuple, symbol tuple), rejecting ambiguous captures."""
    guides = feature_call.split("|")
    if len(guides) != 2 or len(set(guides)) != 2:
        return None
    canonical = frozenset(guides)
    if canonical in (frozenset({"CTRL0001", "CTRL0002"}),
                     frozenset({"CTRL0001", "hRosa26_2"})):
        return "control", (), ()
    # hRosa26 is overridden as control by the authors only in its exact
    # hRosa26+CTRL1 construct; its pairing with a target is not a two-gene KO.
    if "hRosa26_2" in guides:
        return None
    targets = [guide for guide in guides if guide not in CONTROLS]
    symbols = [guide_symbol(guide) for guide in targets]
    if len(set(symbols)) != len(symbols) or any(symbol in held_symbols for symbol in symbols):
        return None
    if not targets:
        return None
    if len(targets) == 1:
        # Deposited calls do not expose vector slots. The author code labels a
        # target paired with either numbered control as an SKO.
        if not any(guide in CONTROLS for guide in guides):
            return None
        return "single", tuple(targets), tuple(symbols)
    return "double", tuple(sorted(targets)), tuple(sorted(symbols))


def stable_symbol_map(symbols: set[str]) -> dict[str, str]:
    pattern = re.compile(r'gene_id "([^"]+)".*gene_name "([^"]+)"')
    found: dict[str, set[str]] = {symbol: set() for symbol in symbols}
    with GTF.open(encoding="utf-8") as stream:
        for line in stream:
            match = pattern.search(line)
            if match and match.group(2) in found:
                found[match.group(2)].add(match.group(1).split(".")[0])
    bad = {key: sorted(value) for key, value in found.items() if len(value) != 1}
    if bad:
        raise ValueError(f"symbols do not map uniquely through pinned GTF: {bad}")
    return {key: next(iter(value)) for key, value in found.items()}


def read_axis(path: Path) -> list[str]:
    with gzip.open(path, "rt", newline="") as stream:
        return [line.rstrip("\n\r").split("\t")[0].split(".")[0] for line in stream]


def read_barcodes(path: Path) -> list[str]:
    with gzip.open(path, "rt", newline="") as stream:
        return [line.strip() for line in stream]


def selected_groups(path: Path, held_symbols: set[str], minimum_cells: int):
    calls = {}
    rejected = Counter()
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            parsed = classify_call(row["feature_call"], held_symbols)
            if parsed is None:
                rejected["ambiguous_or_excluded"] += 1
            else:
                calls[row["cell_barcode"]] = parsed
    counts = Counter(parsed for parsed in calls.values())
    retained = {key for key, count in counts.items() if count >= minimum_cells}
    return calls, counts, retained, rejected


def build_condition(index: int, reference: dict, held_ids: set[str], minimum_cells: int):
    name, medium, day = CONDITIONS[index]
    matrix_path = SOURCE / f"GSE164996_S{index}_filtered_matrix.mtx.gz"
    feature_path = SOURCE / f"GSE164996_S{index}_filtered_features.tsv.gz"
    barcode_path = SOURCE / f"GSE164996_S{index}_filtered_barcodes.tsv.gz"
    call_path = SOURCE / f"GSE164996_S{index}_protospacer_calls_per_cell.csv.gz"

    all_symbols = {guide_symbol(g) for g in (
        "NF1_1 PTEN_5 SMAD4_5 CASP8_5 CBFB_2 CDH1_5 RB1_3 TP53_3 NF2_2 "
        "TBX3_3 USP9X_3 TP53_4 TP53_5 NF2_1"
    ).split()}
    symbol_ids = stable_symbol_map(all_symbols)
    held_symbols = {symbol for symbol, stable in symbol_ids.items() if stable in held_ids}
    calls, raw_counts, retained, rejected = selected_groups(call_path, held_symbols, minimum_cells)
    control_keys = [key for key in retained if key[0] == "control"]
    if len(control_keys) != 1:
        # Both exact control constructs collapse to the same biological key.
        raise ValueError(f"{name}: expected pooled negative-control group")

    feature_ids = read_axis(feature_path)
    barcodes = read_barcodes(barcode_path)
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError(f"{name}: duplicate stable query IDs")
    reference_query_ids = reference["query_ids"].astype(str)
    feature_lookup = {gene: row for row, gene in enumerate(feature_ids)}
    query_mask = np.asarray([gene in feature_lookup for gene in reference_query_ids])
    query_ids = reference_query_ids[query_mask]
    if not len(query_ids):
        raise ValueError(f"{name}: no overlap with reference query axis")
    barcode_lookup = {barcode: col for col, barcode in enumerate(barcodes)}
    if len(barcode_lookup) != len(barcodes):
        raise ValueError(f"{name}: duplicate cell barcodes")

    matrix = mmread(matrix_path).tocsc().astype(np.float64)
    if matrix.shape != (len(feature_ids), len(barcodes)):
        raise ValueError(f"{name}: matrix axes disagree with metadata")
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    if np.any(totals <= 0):
        raise ValueError(f"{name}: zero-library cells")
    selected = matrix[[feature_lookup[gene] for gene in query_ids], :]
    selected = selected.multiply(10000.0 / totals)
    selected.data = np.log1p(selected.data)
    selected = selected.tocsc()

    def mean_for(key):
        columns = [barcode_lookup[barcode] for barcode, parsed in calls.items()
                   if parsed == key and barcode in barcode_lookup]
        if len(columns) != raw_counts[key]:
            raise ValueError(f"{name}: guide calls do not align exactly to filtered barcodes")
        return np.asarray(selected[:, columns].mean(axis=1)).ravel().astype(np.float32), len(columns)

    basal, control_cells = mean_for(control_keys[0])
    singles = {key[1][0]: key for key in retained if key[0] == "single"}
    rows = []
    for key in sorted(retained):
        kind, guides, symbols = key
        if kind == "control":
            continue
        if kind == "double" and not all(guide in singles for guide in guides):
            continue
        target, cells = mean_for(key)
        rows.append((kind, guides, symbols, target, cells))
    row_lookup = {guides: i for i, (_, guides, _, _, _) in enumerate(rows)}
    action_features = np.zeros((len(rows), 2, reference["query_features"].shape[1]), np.float32)
    action_mask = np.zeros((len(rows), 2), bool)
    action_ids = np.full((len(rows), 2), "", dtype="U15")
    parents = np.full((len(rows), 2), -1, np.int64)
    static = load_npz(STATIC)
    static_lookup = {gene: i for i, gene in enumerate(static["entity_id"].astype(str))}
    with h5py.File(STRING, "r") as string:
        for row_index, (kind, guides, symbols, _, _) in enumerate(rows):
            stable = [symbol_ids[symbol] for symbol in symbols]
            for position, (gene, symbol) in enumerate(zip(stable, symbols)):
                feature = np.zeros(642, np.float32)
                if gene in static_lookup:
                    feature[:577] = static["feature_values"][static_lookup[gene]]
                if symbol in string:
                    feature[577:641] = string[symbol][:]
                    feature[641] = 1.0
                if not np.any(feature):
                    raise ValueError(f"{name}: intervention {gene} has no admitted static features")
                action_ids[row_index, position] = gene
                action_features[row_index, position] = feature
                action_mask[row_index, position] = True
            if kind == "double":
                parents[row_index] = [row_lookup[singles[guide][1]] for guide in guides]
    targets = np.stack([row[3] for row in rows])
    observed = np.ones_like(targets, dtype=bool)
    action_guide_ids = np.full((len(rows), 2), "", dtype="U16")
    for row_index, row in enumerate(rows):
        action_guide_ids[row_index, :len(row[1])] = row[1]
    roster_ids = np.unique(action_ids[action_mask])
    roster_features = np.stack([
        action_features[np.argwhere(action_ids == gene)[0][0], np.argwhere(action_ids == gene)[0][1]]
        for gene in roster_ids
    ])
    payload = {
        "schema": np.asarray("slp.joint-world-cropseq-population/v4"),
        "source_id": np.asarray(name), "ncbi_taxon": np.asarray(9606),
        "cell_line": np.asarray("MCF10A"), "mode_id": np.asarray(2),
        "intervention_mode": np.asarray("CRISPR-Cas9 knockout"), "assay_id": np.asarray(4),
        "assay": np.asarray("mean per-cell log1p(CP10K)"),
        "target_units": np.asarray("mean per-cell log1p(CP10K)"),
        "medium": np.asarray(medium), "day": np.asarray(day),
        "query_ids": query_ids, "query_features": reference["query_features"][query_mask],
        "feature_mean": reference["feature_mean"], "feature_scale": reference["feature_scale"],
        "action_ids": action_ids, "action_guide_ids": action_guide_ids,
        "action_features": action_features, "action_mask": action_mask,
        "action_roster_ids": roster_ids, "action_roster_features": roster_features,
        "targets": targets, "basal": np.broadcast_to(basal, targets.shape).copy(),
        "observed": observed, "cell_counts": np.asarray([row[4] for row in rows]),
        "control_cell_count": np.asarray(control_cells),
        "control_context_values": basal / np.float32(np.log(2.0)),
        "control_context_observed": np.ones_like(basal, dtype=bool),
        "single_rows": np.asarray([i for i, row in enumerate(rows) if row[0] == "single"], np.int64),
        "combination_rows": np.asarray([i for i, row in enumerate(rows) if row[0] == "double"], np.int64),
        "combination_single_rows": parents[[i for i, row in enumerate(rows) if row[0] == "double"]],
        "held_gene_ids": np.asarray(sorted(held_ids)),
    }
    report = {
        "condition": name, "day": day, "medium": medium,
        "matrixCells": len(barcodes), "calledCells": len(calls), "controlCells": control_cells,
        "retainedCells": int(sum(row[4] for row in rows) + control_cells),
        "singleViews": int(len(payload["single_rows"])),
        "doubleViewsWithBothParents": int(len(payload["combination_rows"])),
        "interventionGenes": int(len(set(action_ids[action_mask]))),
        "queryGenes": int(len(query_ids)),
        "referenceQueriesAbsent": int((~query_mask).sum()), "minimumCellsPerView": minimum_cells,
        "heldSymbolsExcluded": sorted(held_symbols), "rejectedCalls": dict(rejected),
    }
    return payload, report


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def main():
    global SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/derived/slp11-gse164996-cropseq-populations-v4")
    parser.add_argument("--minimum-cells", type=int, default=10)
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    if args.output.exists():
        raise FileExistsError("output must be a new immutable directory")
    reference = load_npz(REFERENCE)
    four = load_npz(FOUR_CONTEXT)
    held_ids = set(four["action_ids"][four["split_role"] == "validation"].astype(str))
    if len(held_ids) != 1492:
        raise ValueError(f"expected 1,492 global validation genes, found {len(held_ids)}")
    args.output.mkdir(parents=True)
    manifest = {
        "schema": "slp.gse164996-cropseq-derived/v4", "accession": "GEO:GSE164996",
        "protectedTestOpened": False, "normalization": "per-cell library-size CP10K then log1p; population mean",
        "routing": "author exact-two-guide SKO/DKO rules; vector slots unavailable; all calls containing any four-context validation intervention gene excluded before aggregation",
        "globalValidationRoster": {"count": len(held_ids), "source": str(FOUR_CONTEXT.relative_to(ROOT)),
                                   "sourceSha256": sha256(FOUR_CONTEXT)},
        "sourceFiles": {}, "contexts": {},
    }
    for path in sorted(SOURCE.glob("GSE164996_*")):
        manifest["sourceFiles"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    for index in CONDITIONS:
        payload, report = build_condition(index, reference, held_ids, args.minimum_cells)
        path = args.output / f"{CONDITIONS[index][0]}.npz"
        np.savez_compressed(path, **payload)
        report.update({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        manifest["contexts"][CONDITIONS[index][0]] = report
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
