import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch

DIR = Path(__file__).resolve().parents[1] / "modules/slp-1-1-joint-world-v3"
sys.path.insert(0, str(DIR))
SPEC = importlib.util.spec_from_file_location("joint_v3_train", DIR / "train.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def test_generated_parent_uses_no_parent_outcome_and_is_detached():
    model = TRAIN.SharedWorldModel(TRAIN.Config(feature_dim=4, width=8, state_slots=2, heads=2, control_context=True))
    basal = torch.arange(15, dtype=torch.float32).reshape(3,5) / 10
    action_mask = torch.tensor([[True,False]]*3)
    priors = torch.zeros((3,2,5)); priors[:,0] = .3
    data = {"basal":basal, "mode":0, "assay":0, "response_scale":.5,
        "query_features":torch.randn(5,4), "action_features":torch.randn(3,2,4),
        "action_mask":action_mask, "prior_per_action":priors,
        "control_context_values":basal, "control_context_mask":torch.ones_like(basal,dtype=torch.bool)}
    # No targets key exists: generating an intermediate may not consume its measured endpoint.
    prediction = TRAIN.predict_parent_queries(model,data,[0,2],np.array([0,2]),np.array([2,4]))
    torch.testing.assert_close(prediction,basal[[0,2]][:,[0,2,4]]+.3)
    assert not prediction.requires_grad


def test_direction_loss_ignores_shared_template_and_handles_no_signal():
    target = torch.tensor([[1.,3.,2.],[2.,1.,4.],[4.,2.,1.]])
    shared = torch.tensor([[10.,-4.,8.]])
    loss = TRAIN.centered_direction_loss(target+shared,target)
    torch.testing.assert_close(loss,torch.tensor(0.),atol=1e-6,rtol=0.)
    prediction = torch.zeros_like(target,requires_grad=True)
    flat = TRAIN.centered_direction_loss(prediction,torch.ones_like(target))
    flat.backward()
    assert flat.item()==0 and torch.isfinite(prediction.grad).all()
