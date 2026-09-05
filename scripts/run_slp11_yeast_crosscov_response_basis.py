#!/usr/bin/env python3
"""Compare fold-local mean-PCA and split-half cross-covariance RNA bases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[1]
FOLD_SOURCE = ROOT / "modules" / "slp-1-1-yeast-static-baseline-v1" / "static_baseline.py"
FOLD_SOURCE_SHA256 = "88e51be7dfbb175844f6d2f6c884d482129f38b24af15b3d4528bff82088e57f"
INPUT_SHA256 = "dab8b4bbf21bd0a584e77f5fd69d82df41e366ad6d034e9ba7be62896b588689"
INPUT_SCHEMA = "slp.yeast-fitting-split-half-sufficient-statistics/v1"
SEED = 731
RANK = 32
FOLDS = 3
POSITIVE_EIGEN_RELATIVE_FLOOR = 1e-10
POSITIVE_EIGEN_ABSOLUTE_FLOOR = 1e-12


class BasisDiagnosticError(ValueError):
    """Raised when fitting-only response-basis inputs violate the contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compresslevel=9)
    return output.getvalue()


def load_fold_module():
    if sha256_file(FOLD_SOURCE) != FOLD_SOURCE_SHA256:
        raise BasisDiagnosticError("grouped-fold source hash mismatch")
    spec = importlib.util.spec_from_file_location("_slp11_yeast_response_basis_folds", FOLD_SOURCE)
    if spec is None or spec.loader is None:
        raise BasisDiagnosticError("cannot load grouped-fold source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def orient_columns(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def crosscov_operator(a_centered: np.ndarray, b_centered: np.ndarray) -> LinearOperator:
    if a_centered.shape != b_centered.shape or a_centered.ndim != 2:
        raise BasisDiagnosticError("centered split halves must have matching matrices")
    n, queries = a_centered.shape
    if n < 2 or queries < 2 or not np.isfinite(a_centered).all() or not np.isfinite(b_centered).all():
        raise BasisDiagnosticError("invalid centered split-half matrices")

    def matvec(vector: np.ndarray) -> np.ndarray:
        return (
            a_centered.T @ (b_centered @ vector)
            + b_centered.T @ (a_centered @ vector)
        ) / np.float64(2 * n)

    def matmat(matrix: np.ndarray) -> np.ndarray:
        return (
            a_centered.T @ (b_centered @ matrix)
            + b_centered.T @ (a_centered @ matrix)
        ) / np.float64(2 * n)

    return LinearOperator(
        (queries, queries), matvec=matvec, matmat=matmat, dtype=np.float64
    )


def fit_bases(
    a_fit: np.ndarray, b_fit: np.ndarray, rank: int = RANK, seed: int = SEED
) -> dict[str, np.ndarray]:
    if a_fit.shape != b_fit.shape or a_fit.ndim != 2 or a_fit.shape[0] <= rank:
        raise BasisDiagnosticError("fit matrices do not support the requested rank")
    mean_a = a_fit.mean(axis=0, dtype=np.float64)
    mean_b = b_fit.mean(axis=0, dtype=np.float64)
    ac = np.asarray(a_fit - mean_a, dtype=np.float64)
    bc = np.asarray(b_fit - mean_b, dtype=np.float64)
    mean_profiles = (ac + bc) * np.float64(0.5)
    _, singular_values, pca_vt = randomized_svd(
        mean_profiles,
        n_components=rank,
        n_iter=7,
        random_state=seed,
        flip_sign=True,
    )
    pca_basis = orient_columns(pca_vt.T)

    operator = crosscov_operator(ac, bc)
    v0 = np.random.default_rng(seed).standard_normal(ac.shape[1])
    eigenvalues, eigenvectors = eigsh(
        operator,
        k=rank,
        which="LA",
        v0=v0,
        tol=1e-6,
        maxiter=500,
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    floor = max(
        POSITIVE_EIGEN_ABSOLUTE_FLOOR,
        float(np.max(np.abs(eigenvalues))) * POSITIVE_EIGEN_RELATIVE_FLOOR,
    )
    positive = eigenvalues > floor
    cross_basis = orient_columns(eigenvectors[:, positive])
    if cross_basis.shape[1] < 1:
        raise BasisDiagnosticError("cross-covariance has no retained positive eigenvalue")
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "pca_basis": pca_basis,
        "pca_singular_values": singular_values.astype(np.float64),
        "cross_basis": cross_basis,
        "cross_eigenvalues": eigenvalues[positive],
        "cross_all_requested_eigenvalues": eigenvalues,
        "cross_positive_floor": np.asarray(floor, dtype=np.float64),
    }


def project(values: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - mean
    return (centered @ basis) @ basis.T


def row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(ac * ac, axis=1) * np.sum(bc * bc, axis=1))
    result = np.full(a.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = np.sum(ac[valid] * bc[valid], axis=1) / denominator[valid]
    return result


def metrics(
    a_centered: np.ndarray, b_centered: np.ndarray, full_trace: float
) -> dict[str, float | int]:
    correlations = row_pearson(a_centered, b_centered)
    trace = float(np.mean(np.sum(a_centered * b_centered, axis=1)))
    return {
        "geneMeanProjectedMse": float(
            np.mean(np.mean(np.square(a_centered - b_centered), axis=1))
        ),
        "geneMeanIndependentQueryCenteredPearson": float(np.nanmean(correlations)),
        "geneMedianIndependentQueryCenteredPearson": float(np.nanmedian(correlations)),
        "undefinedPearsonGenes": int(np.isnan(correlations).sum()),
        "heldCrossCovarianceTrace": trace,
        "fractionOfFullHeldCrossCovarianceTraceCaptured": (
            float(trace / full_trace) if full_trace != 0 else float("nan")
        ),
    }


def load_statistics(path: Path) -> dict[str, np.ndarray]:
    if sha256_file(path) != INPUT_SHA256:
        raise BasisDiagnosticError("split-half statistics hash mismatch")
    with np.load(path, allow_pickle=False) as source:
        required = {
            "schema",
            "contexts",
            "query_ids",
            "gene_ids",
            "half_sum",
            "half_num_cells",
        }
        if not required.issubset(source.files):
            raise BasisDiagnosticError("split-half statistics schema is incomplete")
        schema = str(source["schema"])
        contexts = source["contexts"].astype(str)
        query_ids = source["query_ids"].astype(str)
        gene_ids = source["gene_ids"].astype(str)
        half_sum = source["half_sum"].copy()
        half_num_cells = source["half_num_cells"].copy()
    if (
        schema != INPUT_SCHEMA
        or contexts.tolist() != ["Control", "NaCl"]
        or query_ids.shape != (6683,)
        or gene_ids.shape != (1516,)
        or half_sum.shape != (2, 2, 1516, 6683)
        or half_sum.dtype != np.float64
        or half_num_cells.shape != (2, 2, 1516)
        or half_num_cells.dtype != np.int64
        or query_ids.tolist() != sorted(set(query_ids.tolist()))
        or gene_ids.tolist() != sorted(set(gene_ids.tolist()))
    ):
        raise BasisDiagnosticError("split-half statistics contract mismatch")
    return {
        "contexts": contexts,
        "query_ids": query_ids,
        "gene_ids": gene_ids,
        "half_sum": half_sum,
        "half_num_cells": half_num_cells,
    }


def environment_profiles(
    statistics: dict[str, np.ndarray], context_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = statistics["half_num_cells"][context_index]
    observed = (counts[0] > 0) & (counts[1] > 0)
    sums = statistics["half_sum"][context_index][:, observed, :]
    selected_counts = counts[:, observed]
    a = sums[0] / selected_counts[0, :, None]
    b = sums[1] / selected_counts[1, :, None]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise BasisDiagnosticError("nonfinite observed split-half profile")
    return statistics["gene_ids"][observed], a, b


def protocol(input_path: Path) -> dict[str, object]:
    return {
        "schema": "slp.yeast-crosscov-response-basis-protocol/v1",
        "status": "frozen-before-fitting-only-basis-estimation",
        "hypothesis": "A rank-32 positive split-half cross-covariance basis improves held-fitting-gene A/B profile correlation by at least 0.02 over rank-32 PCA in both Control and NaCl.",
        "advancementRule": "cross-covariance gene-mean independent-query-centered Pearson minus PCA is at least 0.02 in each environment; no aggregate can conceal an environment failure",
        "comparators": ["raw full 6683-query profile", "rank-32 PCA of fitting-fold split-half means"],
        "crossCovariance": "C=(Ac.T@Bc+Bc.T@Ac)/(2*n), largest-algebraic 32 eigenpairs, retain eigenvalues above max(1e-12,maxabs*1e-10)",
        "centering": "separate A and B per-query means estimated only from the two fitting folds",
        "evaluation": "project held A/B centered profiles separately; equal-gene MSE, per-gene Pearson, and projected/full held cross-covariance trace",
        "rank": RANK,
        "seed": SEED,
        "folds": {
            "count": FOLDS,
            "domain": "slp11-yeast-static-baseline-inner-v1",
            "assignmentScope": "all 1516 stable fitting gene IDs before environment support restriction",
        },
        "snapshots": {
            "fittingSplitHalfStatistics": {
                "path": str(input_path).replace("\\", "/"),
                "sha256": INPUT_SHA256,
            },
            "groupedFoldSource": {
                "path": str(FOLD_SOURCE.relative_to(ROOT)).replace("\\", "/"),
                "sha256": FOLD_SOURCE_SHA256,
            },
        },
        "accessBoundary": {
            "savedFittingSufficientStatisticsOnly": True,
            "rawCountsRead": False,
            "developmentValidationRead": False,
            "benchmarkRead": False,
        },
        "limitations": [
            "Projection reproducibility is not intervention forecasting.",
            "The halves retain shared batch, clone, genotype-assignment, library-preparation and normalization effects.",
            "The saved sufficient statistics do not support fit-fold batch adjustment, so technical confounding remains.",
        ],
        "compute": {"cpuThreads": 2, "maximumSeconds": 600, "maximumMemoryGiB": 4},
    }


def run_profile(
    statistics: dict[str, np.ndarray], folds: np.ndarray, output: Path
) -> dict[str, object]:
    gene_ids, a, b = environment_profiles(statistics, 0)
    index = {gene: row for row, gene in enumerate(statistics["gene_ids"].tolist())}
    local_folds = np.asarray([folds[index[gene]] for gene in gene_ids])
    fit = local_folds != 0
    started = time.monotonic()
    fitted = fit_bases(a[fit], b[fit])
    elapsed = time.monotonic() - started
    report = {
        "schema": "slp.yeast-crosscov-response-basis-profile/v1",
        "environment": "Control",
        "fold": 0,
        "fitGenes": int(fit.sum()),
        "queries": int(a.shape[1]),
        "elapsedSeconds": elapsed,
        "projectedSixFitsSeconds": elapsed * 6,
        "crossPositiveRank": int(fitted["cross_basis"].shape[1]),
        "estimatedWorkingBytes": int(a[fit].nbytes * 4 + a.shape[1] * RANK * 8 * 3),
    }
    output.write_bytes(canonical_json(report))
    return report


def run_all(
    statistics: dict[str, np.ndarray], folds: np.ndarray, output_dir: Path
) -> dict[str, object]:
    started = time.monotonic()
    contexts: dict[str, object] = {}
    for context_index, context in enumerate(statistics["contexts"].tolist()):
        gene_ids, a, b = environment_profiles(statistics, context_index)
        global_index = {
            gene: row for row, gene in enumerate(statistics["gene_ids"].tolist())
        }
        local_folds = np.asarray([folds[global_index[gene]] for gene in gene_ids])
        fold_reports: list[dict[str, object]] = []
        aggregate: dict[str, list[np.ndarray]] = {
            "raw_a": [],
            "raw_b": [],
            "pca_a": [],
            "pca_b": [],
            "cross_a": [],
            "cross_b": [],
        }
        for fold in range(FOLDS):
            fit = local_folds != fold
            held = local_folds == fold
            fitted = fit_bases(a[fit], b[fit])
            raw_a = a[held] - fitted["mean_a"]
            raw_b = b[held] - fitted["mean_b"]
            pca_a = project(a[held], fitted["mean_a"], fitted["pca_basis"])
            pca_b = project(b[held], fitted["mean_b"], fitted["pca_basis"])
            cross_a = project(a[held], fitted["mean_a"], fitted["cross_basis"])
            cross_b = project(b[held], fitted["mean_b"], fitted["cross_basis"])
            full_trace = float(np.mean(np.sum(raw_a * raw_b, axis=1)))
            for name, values in (
                ("raw_a", raw_a),
                ("raw_b", raw_b),
                ("pca_a", pca_a),
                ("pca_b", pca_b),
                ("cross_a", cross_a),
                ("cross_b", cross_b),
            ):
                aggregate[name].append(values)
            fold_metrics = {
                "rawFullQuery": metrics(raw_a, raw_b, full_trace),
                "pcaRank32": metrics(pca_a, pca_b, full_trace),
                "positiveCrossCovariance": metrics(cross_a, cross_b, full_trace),
            }
            basis_arrays = {
                "query_ids": statistics["query_ids"],
                "fit_gene_ids": gene_ids[fit],
                "held_gene_ids": gene_ids[held],
                "mean_a": fitted["mean_a"],
                "mean_b": fitted["mean_b"],
                "pca_basis": fitted["pca_basis"],
                "pca_singular_values": fitted["pca_singular_values"],
                "cross_basis": fitted["cross_basis"],
                "cross_eigenvalues": fitted["cross_eigenvalues"],
                "cross_all_requested_eigenvalues": fitted[
                    "cross_all_requested_eigenvalues"
                ],
                "cross_positive_floor": fitted["cross_positive_floor"],
            }
            basis_name = f"{context.lower()}-fold-{fold}-bases.npz"
            basis_path = output_dir / basis_name
            basis_path.write_bytes(deterministic_npz(basis_arrays))
            fold_reports.append(
                {
                    "fold": fold,
                    "fitGenes": int(fit.sum()),
                    "heldGenes": int(held.sum()),
                    "crossPositiveRank": int(fitted["cross_basis"].shape[1]),
                    "metrics": fold_metrics,
                    "basis": {
                        "path": basis_name,
                        "sha256": sha256_file(basis_path),
                        "bytes": basis_path.stat().st_size,
                    },
                }
            )
        combined = {name: np.vstack(value) for name, value in aggregate.items()}
        combined_full_trace = float(
            np.mean(np.sum(combined["raw_a"] * combined["raw_b"], axis=1))
        )
        aggregate_metrics = {
            "rawFullQuery": metrics(
                combined["raw_a"], combined["raw_b"], combined_full_trace
            ),
            "pcaRank32": metrics(
                combined["pca_a"], combined["pca_b"], combined_full_trace
            ),
            "positiveCrossCovariance": metrics(
                combined["cross_a"], combined["cross_b"], combined_full_trace
            ),
        }
        delta = (
            aggregate_metrics["positiveCrossCovariance"][
                "geneMeanIndependentQueryCenteredPearson"
            ]
            - aggregate_metrics["pcaRank32"][
                "geneMeanIndependentQueryCenteredPearson"
            ]
        )
        contexts[context] = {
            "observedGenes": len(gene_ids),
            "folds": fold_reports,
            "aggregate": aggregate_metrics,
            "crossCovarianceMinusPcaMeanPearson": delta,
            "passesPointZeroTwoRule": bool(delta >= 0.02),
        }
    passed = all(
        context["passesPointZeroTwoRule"] for context in contexts.values()
    )
    return {
        "schema": "slp.yeast-crosscov-response-basis-report/v1",
        "contexts": contexts,
        "advancement": passed,
        "decision": "pass" if passed else "fail",
        "runtimeSeconds": time.monotonic() - started,
        "interpretation": "Fitting-only A/B projection reproducibility; not intervention forecasting or causal biology.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("profile", "run"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / "protocol.json"
    frozen_protocol = protocol(args.input)
    if protocol_path.exists():
        if protocol_path.read_bytes() != canonical_json(frozen_protocol):
            raise BasisDiagnosticError("existing frozen protocol differs")
    else:
        protocol_path.write_bytes(canonical_json(frozen_protocol))
    statistics = load_statistics(args.input)
    fold_module = load_fold_module()
    folds = fold_module.grouped_folds(
        statistics["gene_ids"], folds=FOLDS, seed=SEED
    )
    if args.mode == "profile":
        output = args.output_dir / "profile.json"
        if output.exists():
            raise BasisDiagnosticError("profile output already exists")
        result = run_profile(statistics, folds, output)
    else:
        output = args.output_dir / "report.json"
        if output.exists():
            raise BasisDiagnosticError("report output already exists")
        result = run_all(statistics, folds, args.output_dir)
        output.write_bytes(canonical_json(result))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
