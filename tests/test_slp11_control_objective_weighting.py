from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-control-transition-v2"
sys.path.insert(0, str(MODULE))

from objective_weighting import (
    EQUAL_CONTEXT_GENE_V1,
    UNIFORM_ROW_V1,
    training_row_weights,
)

SPEC = importlib.util.spec_from_file_location("weighted_v2_model", MODULE / "transition_model.py")
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)

LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "weighted_launcher", ROOT / "scripts/run_slp11_minimal_control_common_context.py"
)
assert LAUNCHER_SPEC is not None and LAUNCHER_SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(LAUNCHER_SPEC)
sys.modules[LAUNCHER_SPEC.name] = LAUNCHER
LAUNCHER_SPEC.loader.exec_module(LAUNCHER)


def test_equal_context_gene_weights_give_equal_mass() -> None:
    contexts = np.asarray([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    actions = np.asarray(["A", "A", "B", "C", "C", "C", "D"])
    weights = training_row_weights(
        contexts, actions, objective=EQUAL_CONTEXT_GENE_V1
    )
    assert weights.mean() == pytest.approx(1.0, abs=1e-15)
    for context in (0, 1):
        assert weights[contexts == context].sum() == pytest.approx(3.5)
        local_actions = actions[contexts == context]
        local_weights = weights[contexts == context]
        gene_masses = [local_weights[local_actions == gene].sum() for gene in set(local_actions)]
        assert gene_masses == pytest.approx([1.75, 1.75])


def test_uniform_objective_is_exact_ones() -> None:
    weights = training_row_weights(
        np.asarray([0, 1, 1]),
        np.asarray(["A", "B", "B"]),
        objective=UNIFORM_ROW_V1,
    )
    np.testing.assert_array_equal(weights, np.ones(3, dtype=np.float64))


def test_weighted_gaussian_loss_all_ones_preserves_legacy_and_masks_targets() -> None:
    prediction = {
        "mean": torch.tensor([[0.0, 2.0], [1.0, -1.0]], dtype=torch.float32),
        "scale": torch.tensor([[1.0, 0.5], [2.0, 1.5]], dtype=torch.float32),
    }
    observed = torch.tensor([[True, False], [True, True]])
    target = torch.tensor([[1.0, float("nan")], [2.0, 0.0]], dtype=torch.float32)
    legacy = MODEL.gaussian_loss(prediction, target, observed)
    weighted = MODEL.gaussian_loss(
        prediction, target, observed, row_weight=torch.ones(2)
    )
    assert torch.equal(weighted, legacy)
    changed_masked = target.clone()
    changed_masked[0, 1] = 1e30
    assert torch.equal(
        weighted,
        MODEL.gaussian_loss(
            prediction, changed_masked, observed, row_weight=torch.ones(2)
        ),
    )


def test_weighted_gaussian_loss_does_not_renormalize_batch_weights() -> None:
    prediction = {
        "mean": torch.zeros((2, 1), dtype=torch.float32),
        "scale": torch.ones((2, 1), dtype=torch.float32),
    }
    target = torch.tensor([[1.0], [2.0]])
    observed = torch.ones((2, 1), dtype=torch.bool)
    per_record = torch.stack(
        [
            MODEL.gaussian_loss(
                {"mean": prediction["mean"][i : i + 1], "scale": prediction["scale"][i : i + 1]},
                target[i : i + 1],
                observed[i : i + 1],
            )
            for i in range(2)
        ]
    )
    weights = torch.tensor([0.5, 2.0])
    actual = MODEL.gaussian_loss(prediction, target, observed, row_weight=weights)
    assert torch.equal(actual, (per_record * weights).mean())
    assert not torch.isclose(actual, (per_record * weights).sum() / weights.sum())


def test_launcher_model_source_hook_requires_exact_hash() -> None:
    path = MODULE / "transition_model.py"
    digest = LAUNCHER.sha256_file(path)
    loaded, resolved, actual = LAUNCHER.load_transition_model(path, digest)
    assert resolved == path.resolve()
    assert actual == digest
    assert loaded.Config is not None and loaded.MinimalControlTransition is not None
    with pytest.raises(ValueError, match="SHA-256"):
        LAUNCHER.load_transition_model(path, "0" * 64)
