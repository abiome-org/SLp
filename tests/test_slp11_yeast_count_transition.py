from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_slp11_yeast_count_transition.py"
EXECUTOR = ROOT / "scripts/execute_slp11_yeast_count_transition.py"
MODULE_ROOT = ROOT / "modules/slp-1-1-yeast-count-transition-v1"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load(SCRIPT, "slp11_yeast_count_transition_runner")
executor = load(EXECUTOR, "slp11_yeast_count_transition_executor")
core = load(MODULE_ROOT / "control_transition_model.py", "slp11_yeast_count_core")
inference = load(MODULE_ROOT / "inference.py", "slp11_yeast_count_inference")


def test_stored_npz_member_memmap_reads_exact_uncompressed_array(tmp_path):
    path = tmp_path / "values.npz"
    expected = np.arange(35, dtype=np.float64).reshape(5, 7)
    np.savez(path, sum=expected, other=np.full((200, 200), -99.0))
    mapped = runner.stored_npz_memmap(path, "sum")
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(mapped[[1, 4]], expected[[1, 4]])
    compressed = tmp_path / "compressed.npz"
    np.savez_compressed(compressed, sum=expected)
    with pytest.raises(ValueError, match="compressed"):
        runner.stored_npz_memmap(compressed, "sum")


def test_row_weights_equalize_context_and_gene_but_follow_cells_within_gene():
    contexts = np.asarray([0, 0, 0, 0, 1, 1])
    actions = np.asarray(["A", "A", "B", "B", "C", "D"])
    cells = np.asarray([1, 3, 2, 2, 5, 7])
    weights = runner.row_weights(contexts, actions, cells)
    assert weights.mean() == pytest.approx(1)
    assert weights[1] / weights[0] == pytest.approx(3)
    for context in (0, 1):
        assert weights[contexts == context].sum() == pytest.approx(3)
    assert weights[(contexts == 0) & (actions == "A")].sum() == pytest.approx(1.5)
    assert weights[(contexts == 0) & (actions == "B")].sum() == pytest.approx(1.5)


def test_frozen_core_supports_exact_empty_identity_at_requested_dimensions():
    model = core.MinimalControlTransition(
        core.Config(577, 577, hidden_dim=128, state_dim=128, dropout=0.2)
    ).eval()
    query = torch.zeros((5, 577))
    control = torch.randn((2, 5))
    with torch.no_grad():
        result = model(
            torch.empty((2, 0, 577)),
            query,
            control,
            torch.ones(5),
            torch.ones((2, 5)),
            query[:3],
            torch.zeros((2, 3)),
            torch.ones((2, 3), dtype=torch.bool),
            action_mask=torch.empty((2, 0), dtype=torch.bool),
        )
    assert torch.equal(result["mean"], control)
    assert torch.count_nonzero(result["delta"]) == 0
    assert torch.count_nonzero(result["intervention_delta"]) == 0


def test_portable_predictor_uses_batch_control_and_no_exposure_input(tmp_path):
    artifact = tmp_path / "artifact"
    source = artifact / "source"
    source.mkdir(parents=True)
    shutil.copy2(MODULE_ROOT / "control_transition_model.py", source)
    model = core.MinimalControlTransition(
        core.Config(577, 577, hidden_dim=128, state_dim=128, dropout=0.2)
    )
    save_file(model.state_dict(), str(artifact / "model.safetensors"))
    rng = np.random.default_rng(731)
    control = rng.normal(size=(2, 4)).astype(np.float32)
    np.savez(
        artifact / "reference.npz",
        batch_ids=np.asarray(["B1", "B2"]),
        batch_context_index=np.asarray([0, 1]),
        control_mean=control,
        feature_mean=np.zeros(577, np.float32),
        feature_std=np.ones(577, np.float32),
        query_features_normalized=rng.normal(size=(4, 577)).astype(np.float32),
        basal_query_indices=np.asarray([0, 1]),
        basal_values_normalized=np.zeros((2, 2), np.float32),
        basal_mask=np.ones((2, 2), bool),
        delta_amplitude=np.ones(4, np.float32),
        objective_query_scale=np.ones((2, 4), np.float32),
    )
    predictor = inference.Predictor(artifact)
    empty = predictor.predict_empty(np.asarray([1, 0]))
    assert np.array_equal(empty, control[[1, 0]])
    prediction = predictor.predict(np.zeros((2, 577)), np.asarray([0, 1]))
    assert prediction.shape == (2, 4)
    assert np.isfinite(prediction).all()


def test_corrected_centering_marks_constant_prediction_undefined():
    truth = np.asarray([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [0.0, 3.0, 1.0]])
    constant = np.tile(np.asarray([8.0, -2.0, 5.0]), (3, 1))
    result = executor.metrics(truth, constant)
    assert result["independentlyQueryCenteredPearson"] is None
    assert result["independentlyQueryCenteredUndefinedGenes"] == 3
