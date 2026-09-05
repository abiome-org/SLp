from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


def test_extended_action_initialization_preserves_old_parameters() -> None:
    runner = load(ROOT / "scripts/run_slp11_source3_bp_neural_mean_pair.py", "bp_pair_test")
    model_module = load(runner.MODEL, "bp_pair_model_test")
    new, old = runner.initialize_extended(model_module)
    for name, source in old.state_dict().items():
        target = new.state_dict()[name]
        if name == "action_encoder.0.weight":
            assert torch.equal(target[:, :1156], source)
            assert torch.count_nonzero(target[:, 1156:]) == 0
        else:
            assert torch.equal(target, source)


def test_both_arms_have_equal_width_and_control_tail_is_zero() -> None:
    runner = load(ROOT / "scripts/run_slp11_source3_bp_neural_mean_pair.py", "bp_pair_data_test")
    _, actions, references, audit = runner.load_data()

    assert actions["masked-bp-control"].shape[1] == 1285
    assert actions["bp128-present"].shape == actions["masked-bp-control"].shape
    assert not actions["masked-bp-control"][:, 1156:].any()
    assert audit["fittingGenes"] == 6866
    assert audit["baseNormalizerBitExact"]
    assert all(reference["feature_mean"].shape == (1285,) for reference in references.values())


def test_finalizer_selects_two_fitting_rows_per_context() -> None:
    finalizer = load(
        ROOT / "scripts/finalize_slp11_source3_bp_neural_mean_pair.py",
        "bp_pair_finalizer_test",
    )
    split = torch.tensor([0, 1, 2, 3, 4, 5, 6]).numpy()
    contexts = torch.tensor([0, 2, 0, 1, 1, 2, 2]).numpy()
    selected = finalizer.select_probe_rows(split, contexts)
    assert selected.tolist() == [0, 2, 3, 4, 1, 5]
    assert contexts[selected].tolist() == [0, 0, 1, 1, 2, 2]
