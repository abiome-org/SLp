from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "scripts/build_slp11_shared_response_basis.py"
    spec = importlib.util.spec_from_file_location("shared_basis_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gene_collapse_is_equal_record_and_context_restricted() -> None:
    builder = load_builder()
    targets = np.asarray([[1.0, 3.0], [3.0, 5.0], [8.0, 2.0], [9.0, 4.0]])
    actions = np.asarray(["g1", "g1", "g2", "g1"])
    contexts = np.asarray([0, 0, 0, 1])
    genes, profiles = builder.collapse_profiles(targets, actions, contexts, 0)
    assert genes.tolist() == ["g1", "g2"]
    np.testing.assert_allclose(profiles, [[2.0, 4.0], [8.0, 2.0]])


def test_reconstruction_reports_exact_span_and_residual() -> None:
    builder = load_builder()
    values = np.asarray([[2.0, 3.0], [4.0, 5.0]])
    captured = builder.reconstruction(values, np.eye(2))
    assert captured["capturedEnergyFraction"] == 1.0
    assert captured["relativeSquaredError"] == 0.0


def test_reconstruction_handles_reordered_rank_deficient_columns() -> None:
    builder = load_builder()
    basis = np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    captured = builder.reconstruction(np.asarray([[3.0, 0.0, 0.0]]), basis)
    assert captured["basisRank"] == 1
    assert captured["capturedEnergyFraction"] == 1.0
