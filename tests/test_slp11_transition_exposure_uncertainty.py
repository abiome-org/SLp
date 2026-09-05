from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from exposure_uncertainty import (
    ExposureUncertaintyError,
    fit_exposure_uncertainty,
)


def _exact_residual_fixture():
    exposures = np.tile(np.asarray([5.0, 10.0, 20.0, 40.0]), 8)
    contexts = np.repeat([0, 1], exposures.size)
    counts = np.tile(exposures, 2)
    biological = np.asarray([[0.04, 0.09], [0.16, 0.25]])
    sampling = np.asarray([[0.8, 1.2], [1.6, 2.0]])
    sign = np.tile(np.asarray([-1.0, 1.0]), counts.size // 2)
    residuals = np.empty((counts.size, 2))
    for row, (context, count) in enumerate(zip(contexts, counts, strict=True)):
        residuals[row] = sign[row] * np.sqrt(
            biological[context] + sampling[context] / count
        )
    return residuals, np.ones_like(residuals, dtype=np.bool_), counts, contexts, biological, sampling


def test_joint_nnls_recovers_known_context_heteroscedasticity() -> None:
    residuals, observed, counts, contexts, biological, sampling = _exact_residual_fixture()
    model = fit_exposure_uncertainty(residuals, observed, counts, contexts)

    np.testing.assert_allclose(model.biological_variance_, biological, atol=1e-12)
    np.testing.assert_allclose(model.sampling_variance_, sampling, atol=1e-12)
    assert model.identifiability_warning is not None
    assert "does not identify" in model.identifiability_warning
    assert not model.sampling_from_controls_.any()


def test_core_controls_supply_sampling_slope() -> None:
    residuals, observed, counts, contexts, biological, sampling = _exact_residual_fixture()
    control_counts = np.tile(np.asarray([4.0, 4.0, 8.0, 8.0, 16.0, 16.0]), 2)
    control_contexts = np.repeat([0, 1], 6)
    control_background = np.asarray([[0.02, 0.03], [0.05, 0.07]])
    signs = np.tile(np.asarray([-1.0, 1.0]), 6)
    controls = np.empty((12, 2))
    for row, (context, count) in enumerate(
        zip(control_contexts, control_counts, strict=True)
    ):
        controls[row] = signs[row] * np.sqrt(
            control_background[context] + sampling[context] / count
        )

    model = fit_exposure_uncertainty(
        residuals,
        observed,
        counts,
        contexts,
        control_targets=controls,
        control_observed=np.ones_like(controls, dtype=np.bool_),
        control_num_cells=control_counts,
        control_context_index=control_contexts,
    )

    np.testing.assert_allclose(model.sampling_variance_, sampling, atol=1e-12)
    np.testing.assert_allclose(model.biological_variance_, biological, atol=1e-12)
    assert model.sampling_from_controls_.all()
    assert model.identifiability_warning is None


def test_scales_are_finite_at_small_exposure_and_respect_floor() -> None:
    residuals, observed, counts, contexts, _, _ = _exact_residual_fixture()
    model = fit_exposure_uncertainty(
        residuals, observed, counts, contexts, scale_floor=0.2
    )

    scales = model.scales(np.asarray([1.0, 2.0]), np.asarray([0, 1]))
    assert scales.shape == (2, residuals.shape[1])
    assert np.isfinite(scales).all()
    assert np.all(scales >= 0.2)


def test_count_is_likelihood_only_and_cannot_enter_mean_prediction() -> None:
    residuals, observed, counts, contexts, _, _ = _exact_residual_fixture()
    model = fit_exposure_uncertainty(residuals, observed, counts, contexts)

    assert not hasattr(model, "predict")
    assert set(inspect.signature(model.scales).parameters) == {
        "num_cells",
        "context_index",
    }
    with pytest.raises(TypeError):
        model.scales(counts, contexts, action_features=np.ones((counts.size, 3)))


def test_partial_control_contract_and_nonpositive_counts_are_rejected() -> None:
    residuals, observed, counts, contexts, _, _ = _exact_residual_fixture()
    with pytest.raises(ExposureUncertaintyError, match="all four control"):
        fit_exposure_uncertainty(
            residuals,
            observed,
            counts,
            contexts,
            control_targets=residuals,
        )
    counts[0] = 0.0
    with pytest.raises(ExposureUncertaintyError, match="positive cell counts"):
        fit_exposure_uncertainty(residuals, observed, counts, contexts)
