import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-joint-world-v4"


def load_sources():
    old_path = list(sys.path)
    sys.path.insert(0, str(MODULE))
    try:
        for name in ("world_model", "response_model"):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location("joint_world_v4_inference", MODULE / "inference.py")
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)
        return inference, sys.modules["world_model"]
    finally:
        sys.path[:] = old_path


def fixture(tmp_path):
    inference, world = load_sources()
    cfg = world.Config(feature_dim=5, width=16, state_slots=2, heads=4)
    model = world.SharedWorldModel(cfg)
    torch.nn.init.normal_(model.transition_output.weight, std=.02)
    root = tmp_path / "model"
    for name in ("checkpoints", "adapters", "priors"):
        (root / name).mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), root / "checkpoints/step-000000.safetensors")
    settings = {"config": cfg.__dict__, "contexts": {"k562": {
        "mode": 0, "assay": 0, "response_scale": 2.0, "template_saturation": 0.25}}}
    (root / "config.json").write_text(json.dumps(settings))
    np.savez(root / "normalizer.npz", feature_mean=np.zeros(5), feature_scale=np.ones(5))
    qf = np.arange(20, dtype=np.float32).reshape(4, 5) / 20
    qids = np.array(["q0", "q1", "q2", "q3"])
    np.savez(root / "adapters/k562.npz", query_ids=qids, query_features=qf,
             observed_query_mask=np.array([True, True, False, True]),
             observation_indices=np.array([0, 1, 3]))
    np.savez(root / "priors/k562.npz", schema=np.asarray("slp.reduced-rank-response-model/v1"),
             source_id=np.asarray("k562"), rank=np.asarray(2), alpha=np.asarray(1.), query_ids=qids,
             feature_mean=np.zeros(5), feature_scale=np.ones(5), design_mean=np.zeros(5),
             state_projection=np.eye(5)[:, :2], query_loading=np.arange(8).reshape(2, 4) / 20,
             intercept=np.array([.1, .2, .3, .4]))
    return inference.JointWorldBundle(root, "step-000000.safetensors"), qids


def test_latent_rollout_empty_single_and_query_contract(tmp_path):
    bundle, qids = fixture(tmp_path)
    actions = np.arange(30, dtype=np.float64).reshape(3, 2, 5) / 30
    mask = np.array([[False, False], [True, False], [True, True]])
    basal = np.arange(12, dtype=np.float64).reshape(3, 4) / 7
    rollout = bundle.predict_latent_rollout("k562", actions, mask, basal, batch_size=2, query_chunk=2)
    assert rollout.shape == basal.shape and rollout.dtype == np.float64
    assert np.isfinite(rollout).all()
    assert np.array_equal(rollout[0], basal[0])
    ordinary = bundle.predict("k562", actions[1:2], mask[1:2], basal[1:2], query_chunk=2)
    np.testing.assert_allclose(rollout[1], ordinary[0], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(bundle.query_ids("k562"), qids)
    np.testing.assert_array_equal(bundle.supported_query_mask("k562"), [True, True, False, True])
