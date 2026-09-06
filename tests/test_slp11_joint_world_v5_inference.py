import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-joint-world-v5"


def fixture(tmp_path):
    old = list(sys.path); sys.path.insert(0, str(MODULE))
    try:
        for name in ("world_model", "response_model"):
            sys.modules.pop(name, None)
        ws = importlib.util.spec_from_file_location("world_model", MODULE / "world_model.py")
        world = importlib.util.module_from_spec(ws); sys.modules["world_model"] = world; ws.loader.exec_module(world)
        cfg = world.Config(feature_dim=5, width=16, state_slots=2, heads=4)
        model = world.SharedWorldModel(cfg)
        root = tmp_path / "model"
        for name in ("checkpoints", "adapters", "priors"): (root / name).mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), root / "checkpoints/step-000000.safetensors")
        (root / "config.json").write_text(json.dumps({"config": cfg.__dict__, "contexts": {"k562": {
            "mode": 0, "assay": 0, "response_scale": 2., "template_saturation": .25}}}))
        np.savez(root / "normalizer.npz", feature_mean=np.zeros(5), feature_scale=np.ones(5))
        qids = np.array(["q0", "q1", "q2", "q3"]); qf = np.arange(20).reshape(4, 5).astype(np.float32) / 20
        np.savez(root / "adapters/k562.npz", query_ids=qids, query_features=qf,
                 observed_query_mask=np.array([1, 1, 0, 1], bool), observation_indices=np.array([0, 1, 3]))
        np.savez(root / "priors/k562.npz", schema=np.asarray("slp.reduced-rank-response-model/v1"), source_id=np.asarray("k562"),
                 rank=np.asarray(2), alpha=np.asarray(1.), query_ids=qids, feature_mean=np.zeros(5), feature_scale=np.ones(5),
                 design_mean=np.zeros(5), state_projection=np.eye(5)[:, :2], query_loading=np.arange(8).reshape(2, 4) / 20,
                 intercept=np.array([.1, .2, .3, .4]))
        spec = importlib.util.spec_from_file_location("joint_world_v5_inference", MODULE / "inference.py")
        inference = importlib.util.module_from_spec(spec); spec.loader.exec_module(inference)
        return inference.JointWorldBundle(root, "step-000000.safetensors"), inference
    finally: sys.path[:] = old


def test_transported_rollout_empty_single_order_and_prefix_anchors(tmp_path):
    bundle, inference = fixture(tmp_path)
    torch.manual_seed(9)
    with torch.no_grad():
        bundle.model.transition_output.weight.normal_(0, .03)
        if bundle.model.transition_output.bias is not None: bundle.model.transition_output.bias.normal_(0, .01)
    actions = np.arange(30, dtype=np.float64).reshape(3, 2, 5) / 30
    mask = np.array([[False, False], [True, False], [True, True]])
    basal = np.arange(12, dtype=np.float64).reshape(3, 4) / 7
    rollout = bundle.predict_latent_rollout("k562", actions, mask, basal, batch_size=3, query_chunk=2)
    assert np.array_equal(rollout[0], basal[0])
    direct = bundle.predict("k562", actions, mask, basal, batch_size=3, query_chunk=2)
    np.testing.assert_allclose(rollout[1], direct[1], rtol=0, atol=1e-7)
    reverse = bundle.predict_latent_rollout("k562", actions[2:3, ::-1], mask[2:3, ::-1], basal[2:3], query_chunk=2)
    assert np.isfinite(rollout).all()
    assert np.max(np.abs(rollout[2] - reverse[0])) > 1e-7

    calls = []
    original = bundle.model.transport_action
    def recording(state, prior, next_prior, *args):
        calls.append((prior.detach().clone(), next_prior.detach().clone()))
        return original(state, prior, next_prior, *args)
    bundle.model.transport_action = recording
    bundle.predict_latent_rollout("k562", actions[2:3], mask[2:3], basal[2:3])
    assert len(calls) == 2
    torch.testing.assert_close(calls[1][0], calls[0][1])
    assert not torch.equal(calls[0][0], calls[0][1])
    assert not torch.equal(calls[1][0], calls[1][1])
