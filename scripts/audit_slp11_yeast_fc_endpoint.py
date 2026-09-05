"""Describe fitting-only saturation in the frozen Nadal-Ribelles FC endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVELOPMENT = ROOT / "data/derived/slp11-yeast-atlas-response/nadal-ribelles-control-nacl-development-v1/development.npz"
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-yeast-atlas-response/nadal-ribelles-fc-endpoint-diagnostic-v1"
PREFIX_NAME = "seus_split-prefix-2mib.bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def context_statistics(targets, observed, rows, cell_counts) -> dict:
    values = targets[rows][observed[rows]]
    absolute = np.abs(values)
    mean_absolute = np.asarray([np.abs(targets[row][observed[row]]).mean() for row in rows])
    extreme_fraction = np.asarray([(np.abs(targets[row][observed[row]]) > 20).mean() for row in rows])
    log_cells = np.log1p(cell_counts[rows].astype(np.float64))
    median_negative, median_positive = [], []
    for row in rows:
        row_values = targets[row][observed[row]]
        negative = row_values[row_values < -20]
        positive = row_values[row_values > 20]
        median_negative.append(np.median(negative) if len(negative) else np.nan)
        median_positive.append(np.median(positive) if len(positive) else np.nan)

    def extreme_fit(y: np.ndarray) -> dict:
        finite = np.isfinite(y)
        x = np.log(cell_counts[rows].astype(np.float64))[finite]
        y = y[finite]
        if len(y) < 2:
            return {"records": int(len(y)), "intercept": None, "slope_per_ln_cell_count": None, "r_squared": None, "residual_sd": None}
        design = np.column_stack((np.ones(len(x)), x))
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = design @ coefficients
        total = np.sum((y - y.mean()) ** 2)
        return {
            "records": int(len(y)),
            "intercept": float(coefficients[0]),
            "slope_per_ln_cell_count": float(coefficients[1]),
            "r_squared": float(1 - np.sum((y - prediction) ** 2) / total),
            "residual_sd": float(np.std(y - prediction)),
        }

    rounded, rounded_counts = np.unique(np.round(values[absolute > 10], 6), return_counts=True)
    most_common = np.argsort(rounded_counts)[-10:][::-1]
    return {
        "records": int(len(rows)),
        "observed_values": int(len(values)),
        "value_minimum": float(values.min()),
        "value_maximum": float(values.max()),
        "absolute_quantiles": dict(zip(
            ["0", "0.25", "0.5", "0.75", "0.8", "0.9", "0.95", "0.99", "1"],
            np.quantile(absolute, [0, .25, .5, .75, .8, .9, .95, .99, 1]).tolist(),
            strict=True,
        )),
        "absolute_threshold_fractions": {
            f"gt_{threshold}": float(np.mean(absolute > threshold))
            for threshold in (1, 2, 5, 10, 20, 22, 23, 24, 25, 30)
        },
        "fraction_abs_gt10_and_le20": float(np.mean((absolute > 10) & (absolute <= 20))),
        "extreme_abs_gt20_sign_fraction": {
            "negative": float(np.mean(values[absolute > 20] < 0)),
            "positive": float(np.mean(values[absolute > 20] > 0)),
        },
        "record_mean_absolute_quantiles": dict(zip(
            ["0", "0.25", "0.5", "0.75", "1"],
            np.quantile(mean_absolute, [0, .25, .5, .75, 1]).tolist(), strict=True,
        )),
        "record_extreme_fraction_quantiles": dict(zip(
            ["0", "0.25", "0.5", "0.75", "1"],
            np.quantile(extreme_fraction, [0, .25, .5, .75, 1]).tolist(), strict=True,
        )),
        "cell_count_association": {
            "pearson_log1p_cells_vs_mean_absolute": correlation(log_cells, mean_absolute),
            "spearman_log1p_cells_vs_mean_absolute": rank_correlation(log_cells, mean_absolute),
            "pearson_log1p_cells_vs_fraction_abs_gt20": correlation(log_cells, extreme_fraction),
            "spearman_log1p_cells_vs_fraction_abs_gt20": rank_correlation(log_cells, extreme_fraction),
            "median_negative_extreme_fit": extreme_fit(np.asarray(median_negative)),
            "median_positive_extreme_fit": extreme_fit(np.asarray(median_positive)),
        },
        "most_frequent_extreme_values_rounded_6dp": [
            {"value": float(rounded[index]), "count": int(rounded_counts[index])}
            for index in most_common
        ],
    }


def prefix_profile(path: Path) -> dict:
    compressed = path.read_bytes()
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    decompressed = decompressor.decompress(compressed, 32 * 1024 * 1024)
    return {
        "path": path.name,
        "bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "decompressed_prefix_bytes": len(decompressed),
        "decompressed_prefix_sha256": hashlib.sha256(decompressed).hexdigest(),
        "gzip_stream_complete": bool(decompressor.eof),
        "r_serialization_header": decompressed[:7].decode("ascii"),
        "observed_initial_structure": ["saved object symbol seus", "named vector length 2", "first element S4", "early slot strings assays and counts"],
    }


def build(development_path: Path, output_dir: Path) -> dict:
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError("immutable report already exists")
    prefix_path = output_dir / PREFIX_NAME
    if not prefix_path.is_file() or prefix_path.stat().st_size != 2 * 1024 * 1024:
        raise ValueError("exact capped HTTP prefix is absent")
    with np.load(development_path, allow_pickle=False) as archive:
        targets = archive["targets"].astype(np.float32)
        observed = archive["observed"].astype(bool)
        training_rows = archive["split_train"].astype(np.int64)
        context_index = archive["context_index"].astype(np.int64)
        context_ids = archive["context_ids"].astype(str)
        cell_counts = archive["num_cells"].astype(np.int64)
        validation_count = len(archive["split_validation"])
    contexts = {}
    for index, name in enumerate(context_ids):
        rows = training_rows[context_index[training_rows] == index]
        contexts[name] = context_statistics(targets, observed, rows, cell_counts)
    zenodo_path = ROOT / "data/sources/nadal-ribelles-2025-yeast-metadata-v1/zenodo-record-14062629.json"
    readme_path = ROOT / "data/sources/nadal-ribelles-2025-yeast-metadata-v1/README.txt"
    figures_path = ROOT / "data/sources/nadal-ribelles-2025-yeast-metadata-v1/Figures_Rev.R"
    report = {
        "schema": "slp.nadal-ribelles-fc-endpoint-diagnostic/v1",
        "status": "fitting-only-endpoint-diagnostic-no-target-modification",
        "source_code": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(Path(__file__).resolve())},
        "input": {"path": str(development_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(development_path)},
        "scope": {
            "fitting_records_only": int(len(training_rows)),
            "validation_records_excluded_from_statistics": validation_count,
            "target_clipping_or_filtering": False,
            "model_fitting": False,
            "storage_note": "the compressed NPZ target member contains all development rows; only split_train indices enter every statistic",
        },
        "contexts": contexts,
        "interpretation": {
            "observed": "About one fifth of fitting values jump from ordinary magnitudes directly beyond abs=20, with almost no mass between 10 and 20. More than 94% of abs>20 values are negative. Per-record extreme prevalence and mean absolute response are strongly anticorrelated with genotype cell count; sign-specific extreme medians are nearly linear in ln(cell count).",
            "inference_not_proof": "The discontinuous magnitude regime and its cell-count dependence are consistent with a zero-expression or pseudocount floor in the unavailable upstream differential-expression calculation. The deposited summary-generating code only copies logfoldchanges from CSV files, so the precise estimator and cause cannot be established.",
            "decision": "Do not clip or reinterpret the frozen FC endpoint. Retain it as source-diagnostic evidence and require raw-count reconstruction before using this source for a world-model response objective.",
        },
        "raw_archive_profile": {
            **prefix_profile(prefix_path),
            "http": {
                "url": "https://zenodo.org/api/records/14062629/files/seus_split.RData/content",
                "request": "Range bytes=0-2097151; stream closed at 2 MiB",
                "status": 206,
                "content_range": "bytes 0-2097151/5907877873",
                "archive_bytes": 5907877873,
                "archive_md5": "65bb56efd8120f32f65c044de5f040aa",
                "range_requests_supported": True,
            },
            "author_evidence": {
                "README": {"path": str(readme_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(readme_path), "claim": "seus_split contains separately processed Control and NaCl Seurat objects"},
                "Figures_Rev.R": {"path": str(figures_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(figures_path), "claim": "author code calls GetAssayData(x, assay='RNA', slot='counts') on each seus object"},
                "Zenodo": {"path": str(zenodo_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(zenodo_path), "license": "CC-BY-4.0"},
            },
        },
        "raw_extractor_feasibility": {
            "feasible_in_principle": True,
            "current_parser_supported": False,
            "reason": "the existing bounded reader handles simple lists/data.frames but not S4 Seurat/Assay/dgCMatrix graphs; the prefix reaches an S4 object and a large counts payload immediately",
            "compression_constraint": "the archive is one gzip stream, so HTTP ranges cannot provide random access to serialized slots; every pass must sequentially decompress the archive",
            "bounded_plan": [
                "Expand the CC-BY-4.0 rights/source allowlist to seus_split.RData before full acquisition; the current summary rights record explicitly excludes full Seurat objects.",
                "Download and verify the exact 5,907,877,873-byte archive against MD5 65bb56efd8120f32f65c044de5f040aa.",
                "Extend the pinned XDR reader with S4 and generic skip handlers that preserve reference-table semantics while discarding numeric vectors in bounded chunks.",
                "Pass 1: sequentially skip assay payloads but retain RNA dgCMatrix dimensions/dimnames/column pointers and only required cell metadata; classify intervention genes with the frozen SGD mapping and exclude protected/development validation/test before the next value pass.",
                "Pass 2: sequentially revisit RNA counts and decode only dgCMatrix i/x ranges belonging to fitting intervention cells and source controls, emitting bounded sparse shards or direct fitting pseudobulk accumulators.",
                "Verify cell/column names, gene/row names, exact Control/NaCl separation, integer nonnegative counts, and aggregate cell counts against ptb_summary before defining any normalized endpoint.",
            ],
            "memory_projection": "Metadata, dgCMatrix column pointers, stable mappings, and bounded shard/aggregate buffers can remain below 6 GiB; full generic rdata conversion or dense matrices cannot.",
            "runtime_uncertainty": "A 2 MiB prefix cannot estimate full sequential decompression time or raw sparse nnz. Profile a complete first pass after acquisition and stop if two passes cannot meet a separately frozen compute bound.",
        },
        "access_boundary": "No full Seurat archive downloaded; no raw count value decoded; no validation record contributes to reported distributions.",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.development.resolve(), args.output_dir.resolve())
    print(json.dumps({"contexts": result["contexts"], "raw_archive_profile": result["raw_archive_profile"]}, indent=2))
