"""Fitting-only audit of Replogle source3 population endpoints and efficacy support."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
OUTPUT = ROOT / "results/slp11-transition/human-source3-fitting-endpoint-audit-v1"
SOURCES = (
    ROOT / "data/sources/human/K562_essential_normalized_bulk_01.h5ad",
    ROOT / "data/sources/human/rpe1_normalized_bulk_01.h5ad",
    ROOT / "data/sources/human/K562_gwps_normalized_bulk_01.h5ad",
)
PINS = {
    "development": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "K562_essential_normalized_bulk_01.h5ad": "c1ca6456c9c9f1aa2b02c496eb64d1dc3e6a852edbd744d682b8d2c95fd36829",
    "rpe1_normalized_bulk_01.h5ad": "a3c5bfd0f15d63938bc80c9b8874b9cd761e3a23caf5ffe7966bae4e887ec89d",
    "K562_gwps_normalized_bulk_01.h5ad": "37e48c474d8b5dead4151f96ea8f5fe7bbe6beb10eeea48685b740c3f74490a2",
}
SINGLECELL = (
    {
        "context": "replogle-2022-k562-essential-day-6",
        "raw": {"fileId": 35773219, "name": "K562_essential_raw_singlecell_01.h5ad", "bytes": 10_661_879_995, "md5": "4f1122ce1c7f13299a68df6459a266d3"},
        "normalized": {"fileId": 35773075, "name": "K562_essential_normalized_singlecell_01.h5ad", "bytes": 10_661_879_995, "md5": "f1e221fbf6eac774c21c4242ed440c3f"},
        "knownCells": 310_385,
        "guideMetadata": "Published inspection verifies obs.gene_transcript, obs.sgID_AB, obs.gene, gem_group and cell barcode in the raw artifact.",
    },
    {
        "context": "replogle-2022-rpe1-essential-day-7",
        "raw": {"fileId": 35775606, "name": "rpe1_raw_singlecell_01.h5ad", "bytes": 8_700_873_216, "md5": "6a2a9d0d2bf4ec147f4d1104043b268c"},
        "normalized": {"fileId": 35775554, "name": "rpe1_normalized_singlecell_01.h5ad", "bytes": 8_700_873_216, "md5": "2c36a053960f3fae157adacdbccd4485"},
        "knownCells": None,
        "guideMetadata": "Exact obs guide-field schema requires metadata inspection after acquisition; do not assume K562 field equality.",
    },
    {
        "context": "replogle-2022-k562-gwps-day-8",
        "raw": {"fileId": 35775507, "name": "K562_gwps_raw_singlecell_01.h5ad", "bytes": 65_830_941_948, "md5": "887e3e6a8c8df6eadf7a3030a53c9546"},
        "normalized": {"fileId": 35774440, "name": "K562_gwps_normalized_singlecell_01.h5ad", "bytes": 65_830_941_948, "md5": "6cd393e369506849ebf959175989d632"},
        "knownCells": None,
        "guideMetadata": "Exact obs guide-field schema requires metadata inspection after acquisition; do not infer stable guide IDs from bulk gene_transcript labels.",
    },
)
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
CONTROL_SUFFIX = "_non-targeting_non-targeting_non-targeting"
QC_FIELDS = (
    "num_cells_filtered", "num_cells_unfiltered", "UMI_count_unfiltered",
    "z_gemgroup_UMI", "mean_leverage_score", "std_leverage_score",
    "TE_ratio", "fold_expr", "control_expr", "pct_expr", "mitopercent",
    "energy_test_p_value", "mann_whitney_counts", "anderson_darling_counts",
    "cnv_score_z",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def stable_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = np.asarray(left, np.float64), np.asarray(right, np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3:
        return None
    x = x - x[0]
    y = y - y[0]
    x -= x.mean()
    y -= y.mean()
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    eps = 8 * np.finfo(np.float64).eps * math.sqrt(len(x))
    tx = eps * max(1.0, float(np.max(np.abs(left))))
    ty = eps * max(1.0, float(np.max(np.abs(right))))
    if nx <= tx or ny <= ty:
        return None
    return float(x @ y / (nx * ny))


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def stable_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = np.asarray(left, np.float64), np.asarray(right, np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    return stable_pearson(average_ranks(x[finite]), average_ranks(y[finite]))


def correlations(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    return {"pearson": stable_pearson(left, right), "spearman": stable_spearman(left, right)}


def centered_response_norm(targets: np.ndarray, observed: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, np.float64)
    observed = np.asarray(observed, bool)
    if targets.ndim != 2 or observed.shape != targets.shape:
        raise ValueError("target/mask shape mismatch")
    counts = observed.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("fitting context has an unsupported query")
    centroid = np.where(observed, targets, 0.0).sum(axis=0) / counts
    residual = np.where(observed, targets - centroid, 0.0)
    row_count = observed.sum(axis=1)
    if np.any(row_count < 2):
        raise ValueError("fitting record lacks two observed queries")
    row_mean = residual.sum(axis=1) / row_count
    residual = np.where(observed, residual - row_mean[:, None], 0.0)
    return np.sqrt(np.square(residual).sum(axis=1) / row_count)


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {k: None for k in ("minimum", "p10", "median", "p90", "maximum")}
    q = np.quantile(finite, [0, .1, .5, .9, 1])
    return dict(zip(("minimum", "p10", "median", "p90", "maximum"), map(float, q), strict=True))


def read_fitting_matrix(source: Path, source_rows: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    with h5py.File(source, "r") as handle:
        records = decode(handle["obs/gene_transcript"][:])
        genes = decode(handle["var/gene_id"][:])
        if len(set(records)) != len(records) or len(set(genes)) != len(genes):
            raise ValueError("source identities are not unique")
        lookup = {record: i for i, record in enumerate(records)}
        rows = np.asarray([lookup[str(record)] for record in source_rows], np.int64)
        if len(set(rows.tolist())) != len(rows):
            raise ValueError("fitting source rows are duplicated")
        query_lookup = {gene: i for i, gene in enumerate(genes)}
        columns = np.asarray([query_lookup.get(str(gene), -1) for gene in query_ids], np.int64)
        present = columns >= 0
        targets = np.zeros((len(rows), len(query_ids)), np.float32)
        observed = np.zeros_like(targets, bool)
        order = np.argsort(rows, kind="stable")
        for offset in range(0, len(rows), 256):
            local = order[offset:offset + 256]
            selected = rows[local]
            block = np.asarray(handle["X"][selected, :], np.float32)
            aligned = block[:, columns[present]]
            finite = np.isfinite(aligned)
            targets[np.ix_(local, present)] = np.where(finite, aligned, 0.0)
            observed[np.ix_(local, present)] = finite
        inverse_order = np.argsort(order, kind="stable")
        metadata = {
            name: np.asarray(handle[f"obs/{name}"][rows[order]], np.float64)[inverse_order]
            for name in QC_FIELDS
        }
        core = np.asarray(handle["obs/core_control"][:], bool)
        controls = np.asarray([r.endswith(CONTROL_SUFFIX) for r in records])
        source_info = {
            "sourceRows": len(records), "sourceQueries": len(genes),
            "sourceCoreControls": int(np.count_nonzero(core & controls)),
            "sourceAllControls": int(np.count_nonzero(controls)),
            "sourceQueriesOnFixedPanel": int(present.sum()),
            "sourceQueriesAbsentFromFixedPanel": int((~present).sum()),
            "obsFields": sorted(handle["obs"].keys()),
        }
    return targets, observed, metadata, source_info


def prepare(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    for path in (DATA, *SOURCES):
        expected = PINS["development" if path == DATA else path.name]
        if sha256(path) != expected:
            raise ValueError(f"input hash drift: {path}")
    protocol = {
        "schema": "slp.human-source3-fitting-endpoint-audit-protocol/v1",
        "label": "fitting-only development diagnostic",
        "hypothesis": "Population cell count and measured on-target CRISPRi reduction covary with perturbation-specific molecular response magnitude, indicating that equal population means mix intervention efficacy with measurement precision.",
        "fixedDiagnostic": {
            "population": "only split_train records, separately in each of three contexts; no rows filtered by quality or outcome",
            "onTargetReduction": "fixed context core-control mean in author-z target space minus the exact action-ENSG query value",
            "responseNorm": "RMS after subtracting the context fitting per-query centroid and then each record's observed-query mean",
            "correlations": "Pearson and Spearman for reduction vs log1p filtered cells, reduction vs response norm, and log1p filtered cells vs response norm",
            "descriptiveFlags": "absolute Spearman >=0.10 marks material association; efficacy-aware support additionally requires positive-reduction fraction >0.5 and reduction-vs-norm Spearman >=0.10",
        },
        "guidePolicy": "Report guide agreement only if stable source guide IDs exist. gene_transcript P1/P2 or transcript labels are population labels, not assumed guide identifiers.",
        "outcomeMetadataPolicy": "Source QC/efficacy columns are inventoried on fitting rows but are post-intervention measurements and forbidden forecast inputs.",
        "inputs": {"development": {"path": str(DATA.relative_to(ROOT)), "sha256": PINS["development"]}, "sources": [{"path": str(p.relative_to(ROOT)), "sha256": PINS[p.name]} for p in SOURCES]},
        "sourceCode": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
        "singleCellInventory": {
            "metadataSource": "official Figshare public API article 20029387 v1 queried 2026-09-05",
            "recordUrl": "https://plus.figshare.com/articles/dataset/20029387",
            "license": "CC BY 4.0",
            "files": [{**item, "raw": {**item["raw"], "url": f"https://ndownloader.figshare.com/files/{item['raw']['fileId']}"}, "normalized": {**item["normalized"], "url": f"https://ndownloader.figshare.com/files/{item['normalized']['fileId']}"}} for item in SINGLECELL],
            "downloaded": False,
        },
        "testOnlyAccess": False, "hepg2JurkatAccess": False, "syntheticLethalityAccess": False,
        "quantitativeSourceAccessAfterProtocolFreeze": True,
    }
    path = output / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return protocol


def audit(output: Path) -> dict[str, object]:
    protocol_path = output / "protocol.json"
    if not protocol_path.is_file() or (output / "report.json").exists():
        raise ValueError("frozen protocol absent or report already exists")
    protocol = json.loads(protocol_path.read_text())
    if protocol["sourceCode"]["sha256"] != sha256(Path(__file__).resolve()):
        raise ValueError("executing source differs from frozen protocol")
    started = time.perf_counter()
    with np.load(DATA, allow_pickle=False) as archive:
        train = np.asarray(archive["split_train"], np.int64)
        if len(archive["split_test"]):
            raise ValueError("development artifact unexpectedly includes test rows")
        actions = archive["action_ids"].astype(str)
        contexts = np.asarray(archive["context_index"], np.int64)
        records = archive["record_ids"].astype(str)
        query_ids = archive["query_ids"].astype(str)
        basal = np.asarray(archive["basal_control"], np.float64)
    reports: dict[str, object] = {}
    for context, (context_id, source) in enumerate(zip(CONTEXTS, SOURCES, strict=True)):
        selected = train[contexts[train] == context]
        source_records = np.asarray([record.split("|", 1)[1] for record in records[selected]])
        targets, observed, metadata, source_info = read_fitting_matrix(source, source_records, query_ids)
        response_norm = centered_response_norm(targets, observed)
        query_lookup = {gene: i for i, gene in enumerate(query_ids)}
        mapped = np.asarray([gene in query_lookup for gene in actions[selected]])
        columns = np.asarray([query_lookup.get(gene, 0) for gene in actions[selected]], np.int64)
        row = np.arange(len(selected))
        on_target_observed = mapped & observed[row, columns]
        reduction = basal[context, columns[on_target_observed]] - targets[row[on_target_observed], columns[on_target_observed]]
        cells = metadata["num_cells_filtered"]
        log_cells = np.log1p(cells)
        duplicate_genes = sum(np.count_nonzero(actions[selected] == gene) > 1 for gene in set(actions[selected]))
        qc = {}
        for name, values in metadata.items():
            qc[name] = {"fittingRows": len(values), "finiteRows": int(np.isfinite(values).sum()), "quantiles": quantiles(values)}
        reduction_norm = correlations(reduction, response_norm[on_target_observed])
        count_norm = correlations(log_cells, response_norm)
        reduction_count = correlations(reduction, log_cells[on_target_observed])
        reports[context_id] = {
            **source_info,
            "fittingRecords": len(selected), "fittingGenes": len(set(actions[selected])),
            "genesWithMultiplePopulationRecords": int(duplicate_genes),
            "stableGuideIdsAvailable": False,
            "guideAgreementComputed": False,
            "guideLimitation": "gene_transcript retains population/program labels such as P1, P2 or P1P2 but no stable sgRNA ID or sequence; duplicate records cannot be called two-guide replicates",
            "exactActionQuery": {"mappedRows": int(mapped.sum()), "unmappedRows": int((~mapped).sum()), "mappedButUnobservedRows": int(np.count_nonzero(mapped & ~observed[row, columns])), "observedRows": int(on_target_observed.sum())},
            "controlBaseline": {"space": "author per-gemgroup core-control z score", "rms": float(np.sqrt(np.mean(np.square(basal[context])))), "maximumAbsolute": float(np.max(np.abs(basal[context])))},
            "filteredCellCounts": quantiles(cells),
            "centeredResponseNorm": quantiles(response_norm),
            "onTargetReduction": {"quantiles": quantiles(reduction), "positiveFraction": float(np.mean(reduction > 0)), "median": float(np.median(reduction))},
            "correlations": {"onTargetReductionVsLog1pFilteredCells": reduction_count, "onTargetReductionVsCenteredResponseNorm": reduction_norm, "log1pFilteredCellsVsCenteredResponseNorm": count_norm},
            "descriptiveFlags": {
                "materialCellCountAssociation": bool(count_norm["spearman"] is not None and abs(count_norm["spearman"]) >= .10),
                "efficacyAwareModelingSupported": bool(float(np.mean(reduction > 0)) > .5 and reduction_norm["spearman"] is not None and reduction_norm["spearman"] >= .10),
            },
            "postInterventionMetadataInventory": qc,
        }
        del targets, observed
    result = {
        "schema": "slp.human-source3-fitting-endpoint-audit/v1",
        "label": "fitting-only development diagnostic",
        "protocolSha256": sha256(protocol_path),
        "hypothesis": protocol["hypothesis"],
        "contexts": reports,
        "endpointContract": {
            "observationUnit": "author perturbation-population pseudobulk mean; unequal contributing filtered cells",
            "targetSpace": "author per-cell UMI scaling to experiment median core-control UMI, per-gemgroup gene-wise core-control z score, then arithmetic population mean; no second normalization/log",
            "controls": "source core-control populations define zero-centered target space; separate raw core-control expression defines context state and is not used in this diagnostic",
            "constructProvenance": "population gene_transcript label only; stable guide IDs, gemgroup composition, and per-cell outcomes absent from bulk artifact",
            "efficacy": "exact on-target RNA response and source fold_expr/control_expr fields are post-intervention outcomes, diagnostic-only and unavailable for unseen-action forecasting",
            "cellCount": "num_cells_filtered is exposure/precision metadata, not a molecular-mean predictor",
        },
        "singleCellFeasibility": {
            "officialRecord": "https://plus.figshare.com/articles/dataset/20029387",
            "license": "CC BY 4.0",
            "rawFiles": [{"context": item["context"], **item["raw"], "url": f"https://ndownloader.figshare.com/files/{item['raw']['fileId']}", "knownCells": item["knownCells"], "guideMetadata": item["guideMetadata"]} for item in SINGLECELL],
            "normalizedFiles": [{"context": item["context"], **item["normalized"], "url": f"https://ndownloader.figshare.com/files/{item['normalized']['fileId']}"} for item in SINGLECELL],
            "rawTotalBytes": int(sum(item["raw"]["bytes"] for item in SINGLECELL)),
            "normalizedTotalBytes": int(sum(item["normalized"]["bytes"] for item in SINGLECELL)),
            "downloadStatus": "not downloaded",
            "boundedPreparation": "Acquire raw files one at a time with size/MD5/SHA checks; inspect obs/var before X; mechanically exclude global held/test genes from cell-row allowlist; stream sparse X into <=2048-cell pickle-free CSR shards. Never materialize the 65.8-GB GWPS object or dense cell-by-gene matrix in RAM.",
            "rtx4070Assessment": "A compact denoising/latent model over bounded sparse shards is feasible; full dense 7036-query reconstruction per cell and monolithic AnnData loading are not. Profile K562 essential first, then GWPS only after shard statistics and disk capacity pass.",
        },
        "access": {"fittingSourceXRowsRead": int(sum(v["fittingRecords"] for v in reports.values())), "validationSourceXRowsRead": 0, "testSourceXRowsRead": 0, "hepg2Jurkat": False, "syntheticLethality": False},
        "runtimeSeconds": time.perf_counter() - started,
        "limitations": [
            "Associations are descriptive and cannot distinguish sampling noise, selection, guide efficacy or biology.",
            "No stable guide identifiers are present, so same-gene guide agreement is not estimable from these bulk artifacts.",
            "The three contexts differ in cell line, day and screen design; cross-context endpoint distributions are not normalization-equivalent replicates.",
        ],
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = audit(args.output) if args.run else prepare(args.output)
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output / ("report.json" if args.run else "protocol.json")), "runtimeSeconds": result.get("runtimeSeconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
