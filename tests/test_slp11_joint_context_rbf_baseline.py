from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load(
    "slp11_joint_context_rbf_test",
    ROOT / "scripts/run_slp11_joint_context_rbf_baseline.py",
)
HELPER = load(
    "slp11_joint_context_helper_test",
    ROOT / "scripts/run_slp11_nystrom_rbf_baseline.py",
)
INFERENCE = load(
    "slp11_joint_context_inference_test",
    ROOT / "scripts/inference_slp11_joint_context_rbf.py",
)


def test_equal_context_gene_weights_have_global_mean_one_and_equal_mass() -> None:
    contexts = np.asarray([0, 0, 1, 1, 1, 2])
    weights = RUNNER.equal_context_gene_weights(contexts)
    assert float(weights.mean()) == pytest.approx(1.0)
    masses = [float(weights[contexts == context].sum()) for context in range(3)]
    assert masses == pytest.approx([2.0, 2.0, 2.0])


def test_weighted_ridge_global_mean_limit_uses_equal_context_gene_mean() -> None:
    contexts = np.asarray([0, 0, 1])
    weights = RUNNER.equal_context_gene_weights(contexts)
    features = np.asarray([[0.0], [2.0], [8.0]], dtype=np.float32)
    targets = np.asarray([[0.0], [2.0], [10.0]], dtype=np.float32)
    state = RUNNER.fit_weighted_ridge(features, targets, weights)
    prediction = RUNNER.predict_weighted_ridge(
        state, np.asarray([[100.0]], dtype=np.float32), "global-mean-limit"
    )
    assert prediction.item() == pytest.approx(5.5)


def test_context_kernel_and_kronecker_design_contract() -> None:
    generator = np.random.default_rng(4)
    controls = generator.normal(size=(3, 8)).astype(np.float32)
    context_basis, _, report = RUNNER.fit_context_kernel(controls)
    actions = generator.normal(size=(3, 512)).astype(np.float32)
    design = RUNNER.design_matrix(context_basis, actions)
    assert design.shape == (3, 1539)
    assert report["retainedEigenvalues"] == 3
    assert np.array_equal(design[:, :3], context_basis)


def test_dropped_action_eigenvectors_are_right_zero_padded() -> None:
    values = np.ones((2, 511), dtype=np.float32)
    padded = RUNNER.pad_action_basis(values)
    assert padded.shape == (2, 512)
    assert np.array_equal(padded[:, :511], values)
    assert np.array_equal(padded[:, 511], np.zeros(2, dtype=np.float32))


def test_global_held_gene_is_excluded_in_every_context() -> None:
    genes = np.asarray(["ENSG00000000001", "ENSG00000000001", "ENSG00000000002"])
    fold = HELPER.global_gene_fold("ENSG00000000001", seed=731)
    held = RUNNER.held_gene_mask(HELPER, genes, fold, 731)
    assert np.array_equal(held[:2], [True, True])
    assert held[2] == (
        HELPER.global_gene_fold("ENSG00000000002", seed=731) == fold
    )


def test_fold_cv_uses_context_specific_scale() -> None:
    prediction = np.asarray([[1.0], [10.0]], dtype=np.float32)
    truth = np.zeros_like(prediction)
    contexts = np.asarray([0, 1])
    score, by_context = RUNNER.fold_objective(
        prediction,
        truth,
        contexts,
        {0: np.asarray([1.0]), 1: np.asarray([10.0])},
    )
    assert by_context == pytest.approx({"0": 1.0, "1": 1.0})
    assert score == pytest.approx(1.0)


def test_empty_action_identity_is_bitwise_exact() -> None:
    control = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
    returned = INFERENCE.prediction_or_identity(control, None)
    assert np.array_equal(returned, control)
    assert returned is not control
