from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from slp12_r2_specificity import original_coordinates, query_mean, wrong_identity


def test_wrong_identity_preserves_targets_context_and_action_semantics():
    batch = {
        "action_mask": torch.ones(3, 1, dtype=torch.bool),
        "action_sequence": torch.arange(12).reshape(3, 1, 4),
        "action_annotation": torch.arange(6).reshape(3, 1, 2),
        "action_known": torch.ones(3, 1, 2, dtype=torch.bool),
        "action_method": torch.tensor([[1], [2], [3]]),
        "target": torch.tensor([[4.0], [5.0], [6.0]]),
        "context": torch.randn(3, 4),
    }
    changed = wrong_identity(batch)
    assert torch.equal(changed["action_sequence"][0], batch["action_sequence"][2])
    for key in ("action_method", "target", "context", "action_mask"):
        assert changed[key] is batch[key]
    assert torch.equal(batch["action_sequence"], torch.arange(12).reshape(3, 1, 4))


def test_query_mean_uses_only_admitted_observed_fitting_coordinates():
    view = SimpleNamespace(
        families=[{"indices": np.array([0, 2])}],
        query=np.array([14, 18]),
        targets=np.array([[1.0, 99.0], [9000.0, 9000.0], [3.0, 8.0]]),
        observed=np.array([[True, False], [True, True], [True, True]]),
    )
    assert query_mean(view) == {14: 2.0, 18: 8.0}


def test_centered_diagnostic_restores_original_metric_coordinates():
    view = SimpleNamespace(
        query=np.array([7, 3]),
        query_offsets=np.array([[10.0, 20.0]]),
        scales=np.array([[0.0, 2.0]]),
        native_scales=np.array([[12.0, 4.0]]),
    )
    gain, shift = original_coordinates(view, 0, np.array([3, 7]))
    # Centered predictions [1, -2] mean native measurements [22, 6].
    np.testing.assert_allclose(np.array([1.0, -2.0]) * gain + shift, [2.5, -1.5])
    raw = SimpleNamespace()
    gain, shift = original_coordinates(raw, 0, np.array([3, 7]))
    np.testing.assert_array_equal(gain, [1.0, 1.0])
    np.testing.assert_array_equal(shift, [0.0, 0.0])
