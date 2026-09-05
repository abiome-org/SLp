import importlib.util
import sys
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "modules/slp-1-1-joint-world-v3/response_model.py"
SPEC = importlib.util.spec_from_file_location("joint_world_response_v3", PATH)
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


def test_small_panel_shrinkage_groups_duplicate_action_descriptors(tmp_path):
    # Two guide views per action share a descriptor. Their outcomes must remain
    # together when estimating how much feature-dependent signal generalizes.
    features = np.repeat(np.eye(4, dtype=np.float32), 2, axis=0)
    targets = np.repeat(np.arange(4, dtype=np.float64), 2)[:, None]
    model = MODEL.fit(features, targets, rank=3, alpha=1.0)
    assert 0.0 <= model.shrinkage <= 1.0

    path = tmp_path / "prior.npz"
    MODEL.save(path, model, query_ids=["q"], source_id="fixture")
    restored = MODEL.load(path)
    assert restored.shrinkage == model.shrinkage
    assert np.array_equal(restored.predict(features), model.predict(features))


def test_large_panel_retains_unshrunk_reduced_rank_behavior():
    rng = np.random.default_rng(731)
    features = rng.normal(size=(65, 5)).astype(np.float32)
    targets = rng.normal(size=(65, 7))
    model = MODEL.fit(features, targets, rank=3, alpha=10.0)
    assert model.shrinkage == 1.0


def test_template_saturation_uses_only_explicit_fitting_combinations():
    template = np.array([2.0, -1.0])
    mask = np.array([[True, False], [True, True], [True, True]])
    per_action = np.broadcast_to(template, (3, 2, 2)).copy()
    basal = np.zeros((3, 2))
    # Row 1 has one template in its target, selecting full subtraction from
    # the two-template raw sum. Row 2 is deliberately contradictory held data.
    targets = np.array([[2.0, -1.0], [2.0, -1.0], [200.0, -100.0]])
    saturation = MODEL.calibrate_template_saturation(
        per_action, targets, basal, mask, np.array([0, 1]), template)
    assert saturation == 1.0


def test_apply_prior_increment_handles_starting_state_and_empty_actions():
    template = np.array([2.0, -1.0])
    per_action = np.broadcast_to(template, (3, 2, 2)).copy()
    mask = np.array([[True, True], [True, False], [False, False]])
    result = MODEL.apply_prior_increment(
        per_action, template, mask, np.array([True, False, True]), 1.0)
    assert np.array_equal(result[0], template)  # basal start retains one template
    assert np.array_equal(result[1], np.zeros(2))  # parent start receives increment only
    assert np.array_equal(result[2], np.zeros(2))  # empty action is exact zero
