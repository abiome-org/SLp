"""Portable runtime contracts for minimal control transition v2."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-control-transition-v2"
sys.path.insert(0, str(MODULE))
SPEC = importlib.util.spec_from_file_location("minimal_control_inference_test", MODULE / "inference.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNTIME
SPEC.loader.exec_module(RUNTIME)


def test_import_ignores_wrong_cached_generic_transition_module() -> None:
    wrong = type(sys)("transition_model")
    wrong.Config = object
    wrong.TransitionWorld = nn.Identity
    previous = sys.modules.get("transition_model")
    sys.modules["transition_model"] = wrong
    try:
        unique_spec = importlib.util.spec_from_file_location(
            "minimal_control_inference_collision_regression", MODULE / "inference.py"
        )
        unique_runtime = importlib.util.module_from_spec(unique_spec)
        sys.modules[unique_spec.name] = unique_runtime
        unique_spec.loader.exec_module(unique_runtime)
        assert unique_runtime.MinimalControlTransition.__name__ == "MinimalControlTransition"
        assert unique_runtime.Config.__name__ == "Config"
    finally:
        if previous is None:
            sys.modules.pop("transition_model", None)
        else:
            sys.modules["transition_model"] = previous


def package(tmp_path: Path) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(731)
    config = RUNTIME.Config(5, 7, hidden_dim=16, state_dim=8, dropout=0.0)
    model = RUNTIME.MinimalControlTransition(config).eval()
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    (tmp_path / "model-config.json").write_text(
        json.dumps(
            {
                "action_feature_dim": 5,
                "query_feature_dim": 7,
                "hidden_dim": 16,
                "state_dim": 8,
                "dropout": 0.0,
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(731)
    query_ids = np.asarray([f"ENSG{i:011d}" for i in range(11)])
    panel = np.asarray([1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.bool_)
    np.savez_compressed(
        tmp_path / "runtime-reference.npz",
        feature_mean=np.zeros(5, dtype=np.float32),
        feature_std=np.ones(5, dtype=np.float32),
        query_feature_mean=np.zeros(7, dtype=np.float32),
        query_feature_std=np.ones(7, dtype=np.float32),
        query_features=rng.normal(size=(11, 7)).astype(np.float32),
        delta_amplitude=np.ones(11, dtype=np.float32),
        query_ids=query_ids,
        context_query_indices=np.asarray([0, 2, 4, 7], dtype=np.int64),
        context_panel_mask=panel,
        context_value_space=np.asarray("synthetic-fixed-control"),
    )
    context = rng.normal(size=11).astype(np.float32)
    context[~panel] = np.nan
    actions = rng.normal(size=(2, 5)).astype(np.float32)
    return RUNTIME.PortableMinimalControl(tmp_path), query_ids, context, actions


def test_target_free_prediction_and_query_chunks(tmp_path: Path) -> None:
    runtime, query_ids, context, actions = package(tmp_path)
    mask = runtime.context_panel_mask
    control = np.zeros(len(query_ids), dtype=np.float32)
    full = runtime.predict(
        actions,
        context,
        mask,
        control,
        query_ids=query_ids,
    )
    assert full["mean"].shape == (2, len(query_ids))
    assert full["state"].shape == (2, 8)
    assert full["uncertainty_calibrated"] is False
    assert "scale" not in full
    pieces = []
    for indices in (np.arange(4), np.arange(4, 8), np.arange(8, 11)):
        part = runtime.predict(
            actions,
            context,
            mask,
            control,
            query_ids=query_ids,
            query_indices=indices,
        )
        pieces.append(part["mean"])
        np.testing.assert_allclose(part["state"], full["state"], rtol=0, atol=0)
    np.testing.assert_allclose(np.concatenate(pieces, axis=1), full["mean"], rtol=1e-6, atol=1e-7)


def test_empty_action_identity_and_explicit_measurement_scale(tmp_path: Path) -> None:
    runtime, query_ids, context, _ = package(tmp_path)
    control = np.arange(len(query_ids), dtype=np.float32)[None, :]
    result = runtime.predict(
        np.empty((1, 0, 5), dtype=np.float32),
        context,
        runtime.context_panel_mask,
        control,
        query_ids=query_ids,
        action_mask=np.empty((1, 0), dtype=np.bool_),
        measurement_scale=np.full(len(query_ids), 0.75, dtype=np.float32),
    )
    assert np.array_equal(result["mean"], control)
    assert np.count_nonzero(result["delta"]) == 0
    assert np.count_nonzero(result["intervention_delta"]) == 0
    assert np.array_equal(result["state"], result["basal_state"])
    assert result["uncertainty_calibrated"] is True
    assert np.all(result["scale"] == np.float32(0.75))


def test_identity_and_fixed_panel_are_strict(tmp_path: Path) -> None:
    runtime, query_ids, context, actions = package(tmp_path)
    wrong_ids = query_ids.copy()
    wrong_ids[[0, 1]] = wrong_ids[[1, 0]]
    try:
        runtime.predict(
            actions,
            context,
            runtime.context_panel_mask,
            np.zeros(len(query_ids), dtype=np.float32),
            query_ids=wrong_ids,
        )
    except ValueError as error:
        assert "ordered roster" in str(error)
    else:
        raise AssertionError("misordered query identities were accepted")
    wrong_mask = runtime.context_panel_mask.copy()
    wrong_mask[0] = False
    try:
        runtime.predict(
            actions,
            context,
            wrong_mask,
            np.zeros(len(query_ids), dtype=np.float32),
            query_ids=query_ids,
        )
    except ValueError as error:
        assert "fixed-panel mask" in str(error)
    else:
        raise AssertionError("context panel drift was accepted")
