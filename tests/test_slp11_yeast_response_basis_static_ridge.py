from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_slp11_yeast_response_basis_static_ridge.py"
SPEC = importlib.util.spec_from_file_location("_response_basis_static_ridge_test", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_projection_reconstruction_is_exact_for_in_span_residual() -> None:
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    residual = np.asarray([[2.0, -3.0, 0.0]])
    reference = np.asarray([[10.0, 20.0, 30.0]])
    scores, raw = MODULE.projection_reconstruction(residual, basis, reference)
    np.testing.assert_array_equal(scores, [[2.0, -3.0]])
    np.testing.assert_array_equal(raw, reference + residual)


def test_fold_basis_does_not_read_held_profiles() -> None:
    rng = np.random.default_rng(731)
    genes, queries = 60, 24
    gene_ids = np.asarray([f"SGD:G{i:04d}" for i in range(genes)])
    fold = np.arange(genes) % 3
    sums = rng.normal(size=(1, 2, genes, queries))
    counts = np.ones((1, 2, genes), dtype=np.int64)
    values = {"gene_ids": gene_ids, "half_sum": sums.copy(), "half_num_cells": counts}
    allowed = set(gene_ids[fold != 0])

    class Basis:
        @staticmethod
        def fit_bases(a, b, rank, seed):
            _, _, vt = np.linalg.svd((a + b) / 2, full_matrices=False)
            return {"mean_a": a.mean(0), "mean_b": b.mean(0), "pca_basis": vt[:rank].T, "cross_basis": vt[:rank].T, "cross_eigenvalues": np.ones(rank), "pca_singular_values": np.ones(rank)}

    first = MODULE.fit_basis(values, 0, allowed, Basis, rank=4)
    values["half_sum"][:, :, fold == 0] = 1e12
    second = MODULE.fit_basis(values, 0, allowed, Basis, rank=4)
    np.testing.assert_array_equal(first["basis_gene_ids"], second["basis_gene_ids"])
    np.testing.assert_allclose(first["pca_basis"], second["pca_basis"], atol=0, rtol=0)


def test_batch_reference_and_latent_offset_reconstruct_absolute_profile() -> None:
    basis = np.asarray([[1.0], [0.0]])
    residual = np.asarray([[4.0, 0.0]])
    reference = np.asarray([[7.0, 11.0]])
    latent, reconstructed = MODULE.projection_reconstruction(residual, basis, reference)
    np.testing.assert_array_equal(latent, [[4.0]])
    np.testing.assert_array_equal(reconstructed, [[11.0, 11.0]])
