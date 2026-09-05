"""Focused tests for frozen neural-OOF development evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_slp11_neural_oof_calibration.py"
SPEC = importlib.util.spec_from_file_location("evaluate_slp11_neural_oof_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


def test_component_scales_preserve_sampling_and_apply_floor() -> None:
    biological = np.asarray([[0.0, 4.0], [1.0, 9.0]])
    sampling = np.asarray([[1.0, 0.0], [3.0, 7.0]])
    result = EVALUATION._component_scales(
        biological,
        sampling,
        np.asarray([100.0, 4.0]),
        np.asarray([0, 1]),
    )
    assert result.shape == (2, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0], [0.1, 2.0])
    np.testing.assert_allclose(result[1], [np.sqrt(1.75), np.sqrt(10.75)])


def test_point_metrics_do_not_change_adjusted_mean_with_scale() -> None:
    prediction = np.asarray([[1.0, -1.0, 0.5]])
    target = np.asarray([[0.5, -0.5, 1.0]])
    mask = np.ones_like(target, dtype=bool)
    genes = np.asarray(["ENSG1"])
    first = EVALUATION.audit.gene_summaries(
        prediction, target, mask, genes, np.zeros(3), np.ones_like(target)
    )
    second = EVALUATION.audit.gene_summaries(
        prediction, target, mask, genes, np.zeros(3), np.full_like(target, 2.0)
    )
    assert EVALUATION._point(first)["geneMacroAdjustedPearson"] == EVALUATION._point(
        second
    )["geneMacroAdjustedPearson"]
    assert EVALUATION._point(first)["geneMacroNll"] != EVALUATION._point(second)[
        "geneMacroNll"
    ]
