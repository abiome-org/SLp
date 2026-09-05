import importlib.util
from pathlib import Path
import sys

import torch


PATH=Path(__file__).parents[1]/'modules/slp-1-1-joint-world-v3/world_model.py'
SPEC=importlib.util.spec_from_file_location('slp11_joint_world_v3',PATH)
WORLD=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=WORLD;SPEC.loader.exec_module(WORLD)


def model(control_context=True):
    torch.manual_seed(731)
    return WORLD.SharedWorldModel(WORLD.Config(
        feature_dim=7,width=16,state_slots=4,heads=4,
        control_context=control_context,mode_count=3,assay_count=5))


def encoded(m):
    features=torch.randn(6,7);observed=torch.randn(2,6);basal=torch.randn(2,6)
    mask=torch.tensor([[True,True,False,True,True,False],[True,False,True,True,False,True]])
    modes=torch.tensor([0,2]);assays=torch.tensor([1,4])
    context=torch.randn(2,6)*4;context_mask=mask.clone()
    state=m.encode(observed,basal,features,mask,modes,assays,context,context_mask)
    return state,(features,observed,basal,mask,modes,assays,context,context_mask)


def test_query_set_and_action_set_permutations_are_invariant():
    m=model();state,args=encoded(m);features,observed,basal,mask,modes,assays,context,context_mask=args
    p=torch.tensor([4,0,5,2,1,3])
    permuted=m.encode(observed[:,p],basal[:,p],features[p],mask[:,p],modes,assays,context[:,p],context_mask[:,p])
    torch.testing.assert_close(permuted,state,rtol=1e-5,atol=1e-6)
    actions=torch.randn(2,3,7);action_mask=torch.tensor([[True,True,False],[True,True,True]])
    # Make the initialized zero delta observable without changing the architecture.
    torch.nn.init.normal_(m.transition_output.weight,std=.02)
    order=torch.tensor([2,0,1])
    left=m.transition(state,actions,action_mask,modes,assays)
    right=m.transition(state,actions[:,order],action_mask[:,order],modes,assays)
    torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)


def test_empty_action_is_exact_identity_even_with_nonfinite_padding():
    m=model(False);state=torch.randn(2,4,16);actions=torch.full((2,3,7),float('nan'))
    mask=torch.zeros(2,3,dtype=torch.bool);modes=torch.tensor([0,1]);assays=torch.tensor([0,1])
    changed=m.transition(state,actions,mask,modes,assays)
    assert torch.equal(changed,state)


def test_fixed_scale_action_sum_makes_cardinality_detectable():
    m=model(False);state=torch.randn(1,4,16);one=torch.randn(1,1,7)
    torch.nn.init.normal_(m.transition_output.weight,std=.05)
    modes=torch.tensor([0]);assays=torch.tensor([0])
    one_state=m.transition(state,one,torch.ones(1,1,dtype=torch.bool),modes,assays)
    two_state=m.transition(state,one.expand(-1,2,-1),torch.ones(1,2,dtype=torch.bool),modes,assays)
    assert not torch.allclose(one_state,two_state,rtol=1e-5,atol=1e-6)


def test_zero_initialized_transition_has_finite_first_step_gradient():
    m=model(False);state=torch.randn(2,4,16);actions=torch.randn(2,2,7)
    mask=torch.ones(2,2,dtype=torch.bool);modes=torch.tensor([0,1]);assays=torch.tensor([0,1])
    changed=m.transition(state,actions,mask,modes,assays)
    torch.testing.assert_close(changed,state,rtol=0,atol=0)
    changed.square().mean().backward()
    gradient=m.transition_output.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
