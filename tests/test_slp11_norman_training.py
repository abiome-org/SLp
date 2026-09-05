from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load():
    module_dir = Path(__file__).parents[1] / "modules" / "slp-1-1-world-transition-v1"
    sys.path.insert(0, str(module_dir))
    path = module_dir / "train_norman.py"
    spec = importlib.util.spec_from_file_location("slp11_train_norman", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = _load()


def test_action_sets_unpack_and_sum_with_masks() -> None:
    records = TRAIN.unpack_actions(
        np.asarray(["A", "B", "C"]), np.asarray([0, 1, 3], dtype=np.int64)
    )
    lookup = {
        "A": np.ones(TRAIN.FEATURE_DIM, dtype=np.float32),
        "B": np.full(TRAIN.FEATURE_DIM, 2, dtype=np.float32),
        "C": np.full(TRAIN.FEATURE_DIM, 3, dtype=np.float32),
    }
    values, mask = TRAIN.action_feature_tensor(records, lookup)
    assert records == (("A",), ("B", "C"))
    assert mask.tolist() == [[True, False], [True, True]]
    assert np.array_equal((values * mask[..., None]).sum(1)[:, 0], [1, 5])


def test_validation_strata_counts_held_constituents() -> None:
    candidates = [f"ENSG{index:011d}" for index in range(1, 5000)]
    by_role: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for item in candidates:
        by_role[TRAIN.constituent_split(item)].append(item)
    records = (
        (by_role["validation"][0],),
        (by_role["train"][0], by_role["validation"][1]),
        (by_role["validation"][2], by_role["validation"][3]),
    )
    strata = TRAIN.validation_strata(records)
    assert strata["single"].tolist() == [0]
    assert strata["double"].tolist() == [1, 2]
    assert strata["oneHeldConstituent"].tolist() == [0, 1]
    assert strata["twoHeldConstituents"].tolist() == [2]


def test_ridge_and_oof_predictions_are_deterministic() -> None:
    rng = np.random.default_rng(731)
    x = rng.normal(size=(20, 4))
    y = rng.normal(size=(20, 3))
    ids = [f"record-{index}" for index in range(20)]
    first = TRAIN.oof_predictions(x, y, ids, alpha=100.0)
    second = TRAIN.oof_predictions(x, y, ids, alpha=100.0)
    assert first.shape == y.shape
    np.testing.assert_array_equal(first, second)


def test_evaluation_reports_canonical_action_set_aggregation() -> None:
    prediction = np.asarray([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]])
    target = prediction.copy()
    observed = np.ones_like(prediction, dtype=np.bool_)
    reference = np.zeros(2)
    scale = np.ones(2)
    validation = next(
        f"ENSG{index:011d}"
        for index in range(1, 5000)
        if TRAIN.constituent_split(f"ENSG{index:011d}") == "validation"
    )
    report = TRAIN.evaluate(
        prediction,
        target,
        observed,
        reference,
        scale,
        ((validation,), (validation,), (validation,)),
    )
    assert report["all"]["records"] == 3
    assert report["canonicalActionSet"]["records"] == 1
