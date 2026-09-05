from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_audit():
    path = ROOT / "scripts/audit_slp11_response_query_subspace.py"
    spec = importlib.util.spec_from_file_location("subspace_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_captured_fraction_distinguishes_aligned_and_orthogonal_landscapes() -> None:
    audit = load_audit()
    basis, rank = audit.orthonormal_basis(np.asarray([[1.0], [0.0], [0.0]]))
    assert rank == 1
    assert audit.captured_fraction(np.asarray([[2.0, 0.0, 0.0]]), basis) == 1.0
    assert audit.captured_fraction(np.asarray([[0.0, 3.0, 0.0]]), basis) == 0.0


def test_orthonormal_basis_handles_leading_dependent_columns() -> None:
    audit = load_audit()
    values = np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    basis, rank = audit.orthonormal_basis(values)
    assert rank == 1
    assert audit.captured_fraction(np.asarray([[4.0, 0.0, 0.0]]), basis) == 1.0
