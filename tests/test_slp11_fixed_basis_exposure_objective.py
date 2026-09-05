from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exposure_never_enters_fixed_model_mean() -> None:
    core = load(
        ROOT / "modules/slp-1-1-fixed-query-transition-v1/transition_model.py",
        "fixed_exposure_contract_core",
    )
    model = core.FixedQueryTransition(
        core.Config(3, 4, state_dim=2, hidden_dim=5, dropout=0.0)
    ).eval()
    parameters = inspect.signature(model.forward).parameters
    assert "num_cells" not in parameters
    args = (
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.1, -0.2]]),
        torch.ones(2),
        torch.ones((1, 2)),
        torch.ones((1, 1, 4)),
        torch.zeros((1, 1)),
        torch.ones((1, 1), dtype=torch.bool),
    )
    first = model(*args)["mean"]
    second = model(*args)["mean"]
    assert torch.equal(first, second)


def test_portable_predictor_has_no_exposure_argument() -> None:
    inference = load(
        ROOT / "modules/slp-1-1-fixed-query-exposure-v1/inference.py",
        "fixed_exposure_inference_contract",
    )
    parameters = inspect.signature(inference.Predictor.predict).parameters
    assert tuple(parameters) == ("self", "raw_action_features", "context_index")


def test_precision_rows_uses_count_only_in_loss_precision() -> None:
    runner = load(
        ROOT / "scripts/run_slp11_source3_fixed_basis_exposure_objective.py",
        "fixed_exposure_runner_contract",
    )
    biological = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    sampling = torch.tensor([[4.0, 8.0]], dtype=torch.float64)
    contexts = np.asarray([0, 0])
    counts = np.asarray([1.0, 4.0])
    precision = runner.precision_rows(
        np.asarray([0, 1]), contexts, counts, biological, sampling, 0.5
    )
    assert torch.allclose(
        precision, torch.tensor([[0.1, 0.05], [0.25, 0.125]], dtype=torch.float64)
    )


def test_batch_sequence_is_deterministic() -> None:
    objective = load(
        ROOT / "results/slp11-transition/"
        "human-source3-bp-neural-mean-pair-seed731-v2-finalization-v1/"
        "source/mean_objective.py",
        "fixed_exposure_batch_contract",
    )
    first = list(
        objective.deterministic_shuffled_batches(
            np.arange(12), batch_size=4, steps=7, seed=731
        )
    )
    second = list(
        objective.deterministic_shuffled_batches(
            np.arange(12), batch_size=4, steps=7, seed=731
        )
    )
    assert all(
        np.array_equal(left, right) for left, right in zip(first, second, strict=True)
    )
