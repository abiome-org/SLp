from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modules/slp-1-1-scouter-adapted-baseline-v1/scouter_model.py"
SPEC = importlib.util.spec_from_file_location("slp11_scouter_adapted_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)

RUNNER_SOURCE = ROOT / "scripts/run_slp11_scouter_adapted_baseline.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("slp11_scouter_runner_test", RUNNER_SOURCE)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


def small_model() -> nn.Module:
    return MODEL.ScouterAdaptedBaseline(
        MODEL.Config(
            query_dim=7,
            action_feature_dim=5,
            control_hidden=(11, 9),
            control_state_dim=3,
            generator_hidden=(13,),
        )
    ).eval()


def test_author_architecture_defaults() -> None:
    model = MODEL.ScouterAdaptedBaseline(MODEL.Config(7036, 1156))
    control_linears = [item for item in model.control_encoder if isinstance(item, nn.Linear)]
    generator_linears = [item for item in model.generator if isinstance(item, nn.Linear)]
    assert [(item.in_features, item.out_features) for item in control_linears] == [
        (7036, 2048),
        (2048, 512),
        (512, 64),
    ]
    assert [(item.in_features, item.out_features) for item in generator_linears] == [
        (1220, 2048),
        (2048, 7036),
    ]
    assert sum(isinstance(item, nn.BatchNorm1d) for item in model.modules()) == 3
    assert not any(isinstance(item, nn.Dropout) for item in model.modules())


def test_action_mask_order_and_nonfinite_padding_are_inert() -> None:
    torch.manual_seed(1)
    model = small_model()
    actions = torch.randn(2, 3, 5)
    controls = torch.randn(2, 7)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    padded = actions.clone()
    padded[~mask] = torch.nan
    first = model(padded, controls, mask)
    second = model(padded[:, [1, 0, 2]], controls, mask[:, [1, 0, 2]])
    assert torch.equal(first, second)


def test_direct_action_features_affect_output_and_receive_no_embedding_lookup() -> None:
    model = small_model()
    assert not any(isinstance(item, nn.Embedding) for item in model.modules())
    actions = torch.randn(3, 1, 5, requires_grad=True)
    prediction = model(actions, torch.randn(3, 7))
    prediction.square().mean().backward()
    assert actions.grad is not None
    assert torch.count_nonzero(actions.grad) > 0


def test_uniform_row_gaussian_loss_ignores_masked_nan() -> None:
    prediction = torch.tensor([[0.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = torch.tensor([[1.0, float("nan")], [1.0, 2.0]])
    observed = torch.tensor([[True, False], [True, True]])
    scale = torch.tensor([[1.0, float("nan")], [2.0, 2.0]])
    loss = MODEL.gaussian_loss(prediction, target, observed, scale)
    expected_first = 0.5 * (torch.log(torch.tensor(2.0 * torch.pi)) + 1.0)
    expected_second = 0.5 * (torch.log(torch.tensor(8.0 * torch.pi)) + 1.0)
    assert float(loss.detach()) == pytest.approx(
        float((expected_first + expected_second) / 2.0)
    )
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_invalid_shapes_fail_closed() -> None:
    model = small_model()
    with pytest.raises(ValueError):
        model(torch.zeros(2, 5), torch.zeros(2, 7))
    with pytest.raises(ValueError):
        MODEL.gaussian_loss(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.ones(1, 2),
        )


def test_context_validation_weights_are_gene_macro() -> None:
    weights = RUNNER.gene_macro_weights(["a", "a", "b"])
    assert weights.tolist() == pytest.approx([0.25, 0.25, 0.5])
    assert float(weights.sum()) == pytest.approx(1.0)


def test_batching_avoids_batchnorm_singleton() -> None:
    chunks = RUNNER.batches(torch.arange(513).numpy(), 256)
    assert [len(item) for item in chunks] == [256, 257]
    assert torch.equal(torch.as_tensor(chunks[0]), torch.arange(256))


def test_independent_centering_removes_common_profile() -> None:
    target = torch.tensor([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [3.0, 7.0, 9.0]]).numpy()
    prediction = target + torch.tensor([[10.0, -3.0, 2.0]]).numpy()
    report = RUNNER.independently_centered_gene_correlation(
        prediction,
        target,
        torch.ones(3, 3, dtype=torch.bool).numpy(),
        ["a", "b", "c"],
    )
    assert report["correlation"] == pytest.approx(1.0)


def test_basal_normalization_uses_common_panel_and_zeros_missing() -> None:
    values = torch.tensor([[1.0, 2.0, 10.0], [4.0, 8.0, 20.0]]).numpy()
    observed = torch.tensor([[True, True, False], [True, True, False]]).numpy()
    normalized, means, standard_deviations = RUNNER.normalize_basal_controls(
        values, observed, 2
    )
    assert normalized[:, 2].tolist() == [0.0, 0.0]
    assert normalized[:, :2].mean(axis=1).tolist() == pytest.approx([0.0, 0.0])
    assert means[:, 0].tolist() == pytest.approx([1.5, 6.0])
    assert standard_deviations[:, 0].tolist() == pytest.approx([0.5, 2.0])
