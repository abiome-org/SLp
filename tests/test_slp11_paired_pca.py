import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).parents[1]
PATH = ROOT / "modules/slp-1-1-paired-pca-v1/paired_pca.py"
SPEC = importlib.util.spec_from_file_location("slp11_paired_pca_test", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def stats(rna, protein):
    return MOD.fit_stats(lambda: iter([(sparse.csr_matrix(rna), protein)]), floor=0.05)


def dense_balanced(rna, protein, fitted):
    return np.column_stack((
        (rna - fitted.rna_mean) / fitted.rna_sd * fitted.rna_weight,
        (protein - fitted.protein_mean) / fitted.protein_sd * fitted.protein_weight,
    ))


def test_sparse_affine_products_match_dense_matrix_products():
    rng = np.random.default_rng(2)
    rna = rng.poisson(1.2, size=(7, 5)).astype(float)
    protein = rng.normal(size=(7, 3))
    fitted = stats(rna, protein)
    balanced = dense_balanced(rna, protein, fitted)
    right = rng.normal(size=(8, 4))
    cell_right = rng.normal(size=(7, 4))
    np.testing.assert_allclose(
        MOD.affine_right(sparse.csr_matrix(rna), protein, right, fitted),
        balanced @ right,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        MOD.affine_transpose(sparse.csr_matrix(rna), protein, cell_right, fitted),
        balanced.T @ cell_right,
        atol=1e-12,
    )


def test_streaming_subspace_matches_direct_svd_for_low_rank_fixture():
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(240, 2))
    loading = rng.normal(size=(2, 8))
    dense = latent @ loading
    rna, protein = dense[:, :6], dense[:, 6:]
    fitted = stats(rna, protein)

    def batches():
        for start in range(0, len(rna), 37):
            yield sparse.csr_matrix(rna[start:start + 37]), protein[start:start + 37]

    model = MOD.fit_streaming_pca(
        batches, fitted, rank=2, oversample=2, passes=3, seed=731,
    )
    _, _, direct = np.linalg.svd(dense_balanced(rna, protein, fitted), full_matrices=False)
    expected = direct[:2].T @ direct[:2]
    actual = model.components @ model.components.T
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_linear_encoder_commutes_with_exact_guide_averaging():
    rng = np.random.default_rng(7)
    rna = rng.poisson(1.0, size=(11, 5)).astype(float)
    protein = rng.normal(size=(11, 2))
    fitted = stats(rna, protein)
    q, _ = np.linalg.qr(rng.normal(size=(7, 3)))
    model = MOD.StreamingPca(q, np.ones(3), fitted, 3, 1, 731)
    cell_average = model.encode(sparse.csr_matrix(rna), protein).mean(axis=0)
    guide_mean = model.encode(sparse.csr_matrix(rna.mean(axis=0)[None]), protein.mean(axis=0)[None])[0]
    np.testing.assert_allclose(cell_average, guide_mean, atol=1e-12)


def test_held_gene_exclusion_rejects_any_overlap():
    MOD.assert_held_genes_excluded(np.asarray(["ENSG1", "", "ENSG2"]), np.asarray(["ENSG3"]))
    with pytest.raises(MOD.PairedPcaError, match="held genes"):
        MOD.assert_held_genes_excluded(np.asarray(["ENSG1"]), np.asarray(["ENSG1"]))


def test_portable_forecast_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(9)
    fitted = MOD.PairedStats(
        np.zeros(3), np.ones(3), np.zeros(2), np.ones(2), 20,
    )
    components, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    pca = MOD.StreamingPca(components, np.asarray([4.0, 2.0]), fitted, 3, 1, 731)
    ridge = MOD.LatentRidge(
        np.zeros(4), np.ones(4), 10.0,
        rng.normal(size=(2, 4, 2)), rng.normal(size=(2, 2)), 10_000.0,
    )
    artifact = MOD.PcaForecastArtifact(
        pca, ridge, rng.normal(size=(2, 3)), rng.normal(size=(2, 2)),
        np.asarray(["a", "b"]),
    )
    features = rng.normal(size=(5, 4))
    context = np.asarray([0, 1, 1, 0, 1])
    expected = artifact.forecast(features, context)
    path = tmp_path / "artifact.npz"
    artifact.save(path)
    actual = MOD.PcaForecastArtifact.load(path).forecast(features, context)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_allclose(left, right)
