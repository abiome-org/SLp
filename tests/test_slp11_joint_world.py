import importlib.util, sys
from pathlib import Path
import json, subprocess
import numpy as np
import torch
from safetensors.torch import save_file

PATH=Path(__file__).resolve().parents[1]/'modules/slp-1-1-joint-world-v1/world_model.py'
spec=importlib.util.spec_from_file_location('slp11_joint_world',PATH)
world=importlib.util.module_from_spec(spec); sys.modules[spec.name]=world; spec.loader.exec_module(world)

def fixture():
 torch.manual_seed(731); model=world.SharedWorldModel(world.Config(feature_dim=7,width=16,state_slots=3,heads=4))
 b,q,a=2,9,3; observed=torch.randn(b,q); basal=torch.randn(b,q); queries=torch.randn(q,7)
 mask=torch.ones(b,q,dtype=torch.bool); actions=torch.randn(b,a,7); action_mask=torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
 modes=torch.tensor([0,1]); assays=torch.tensor([0,1])
 return model,(observed,basal,queries,mask,actions,action_mask,modes,assays)

def test_set_symmetries_and_empty_action_identity():
 model,x=fixture(); observed,basal,queries,mask,actions,action_mask,modes,assays=x; model.eval()
 state=model.encode(observed,basal,queries,mask,modes,assays)
 permutation=torch.tensor([1,0,2])
 left=model.transition(state,actions,action_mask,modes,assays)
 right=model.transition(state,actions[:,permutation],action_mask[:,permutation],modes,assays)
 torch.testing.assert_close(left,right,rtol=1e-6,atol=1e-6)
 empty=torch.zeros_like(action_mask)
 assert torch.equal(model.transition(state,actions,empty,modes,assays),state)
 assert torch.equal(model(observed,basal,queries,mask,actions,empty,queries,modes,assays),torch.zeros((2,9)))

def test_query_permutation_and_masked_encoder():
 model,x=fixture(); observed,basal,queries,mask,_,_,modes,assays=x; model.eval(); order=torch.randperm(len(queries))
 state=model.encode(observed,basal,queries,mask,modes,assays)
 permuted=model.encode(observed[:,order],basal[:,order],queries[order],mask[:,order],modes,assays)
 torch.testing.assert_close(state,permuted,rtol=1e-5,atol=1e-6)
 decoded=model.decode(state,queries,assays); decoded_permuted=model.decode(state,queries[order],assays)
 torch.testing.assert_close(decoded[:,order],decoded_permuted,rtol=1e-5,atol=1e-6)

def test_observation_binding_distinguishes_values_assigned_to_different_genes():
 model,x=fixture(); observed,basal,queries,mask,_,_,modes,assays=x; model.eval()
 original=model.encode(observed,basal,queries,mask,modes,assays)
 reassigned=observed.clone(); reassigned[:,[0,1]]=reassigned[:,[1,0]]
 changed=model.encode(reassigned,basal,queries,mask,modes,assays)
 assert not torch.allclose(original,changed)

def test_transition_has_initial_gradient_and_learns_observed_state_dependence():
 model,x=fixture(); observed,basal,queries,mask,actions,action_mask,modes,assays=x
 optimizer=torch.optim.SGD(model.parameters(),lr=.1)
 delta=model(observed,basal,queries,mask,actions,action_mask,queries,modes,assays)
 loss=(delta-1).square().mean(); optimizer.zero_grad(); loss.backward()
 assert model.transition_output.weight.grad is not None
 assert model.transition_output.weight.grad.abs().sum()>0
 optimizer.step()
 first=model(observed,basal,queries,mask,actions,action_mask,queries,modes,assays)
 second=model(observed+torch.linspace(0,1,len(queries)),basal,queries,mask,actions,action_mask,queries,modes,assays)
 assert not torch.allclose(first,second)

def test_all_missing_observation_is_rejected():
 model,x=fixture(); observed,basal,queries,mask,_,_,modes,assays=x; mask[0]=False
 try: model.encode(observed,basal,queries,mask,modes,assays)
 except ValueError as error: assert 'at least one' in str(error)
 else: raise AssertionError('all-missing observation accepted')

def test_control_context_is_distinct_from_zero_centered_target_baseline():
 model,x=fixture(); observed,basal,queries,mask,_,_,modes,assays=x
 model=world.SharedWorldModel(world.Config(feature_dim=7,width=16,state_slots=3,heads=4,control_context=True))
 model.eval(); zeros=torch.zeros_like(observed)
 context=torch.linspace(0,4,observed.shape[1]).expand_as(observed)
 missing=torch.zeros_like(mask)
 first=model.encode(zeros,zeros,queries,mask,modes,assays,context,mask)
 second=model.encode(zeros,zeros,queries,mask,modes,assays,context.flip(1),mask)
 assert not torch.allclose(first,second)
 absent=model.encode(zeros,zeros,queries,mask,modes,assays,context,missing)
 masked_nan=model.encode(zeros,zeros,queries,mask,modes,assays,torch.full_like(context,float('nan')),missing)
 torch.testing.assert_close(absent,masked_nan)
 order=torch.randperm(len(queries))
 permuted=model.encode(zeros[:,order],zeros[:,order],queries[order],mask[:,order],modes,assays,context[:,order],mask[:,order])
 torch.testing.assert_close(first,permuted,rtol=1e-5,atol=1e-6)

def test_portable_bundle_step_zero_matches_observed_plus_prior(tmp_path):
 cfg=world.Config(feature_dim=5,width=16,state_slots=2,heads=4)
 model=world.SharedWorldModel(cfg); root=tmp_path/'model'; (root/'checkpoints').mkdir(parents=True); (root/'adapters').mkdir(); (root/'priors').mkdir()
 save_file(model.state_dict(),root/'checkpoints/step-000000.safetensors')
 (root/'config.json').write_text(json.dumps({'config':cfg.__dict__,'contexts':{'k562':{'mode':0,'assay':0,'response_scale':2.}},'observation_queries':3,'seed':731}))
 np.savez(root/'normalizer.npz',feature_mean=np.zeros(5,np.float32),feature_scale=np.ones(5,np.float32))
 qf=np.arange(20,dtype=np.float32).reshape(4,5)/20; qids=np.array(['q0','q1','q2','q3'])
 np.savez(root/'adapters/k562.npz',query_ids=qids,query_features=qf,observed_query_mask=np.ones(4,bool),observation_indices=np.array([0,1,2]))
 fm=np.zeros(5); fs=np.ones(5); dm=np.zeros(5); projection=np.eye(5)[:,:2]; loading=np.arange(8,dtype=float).reshape(2,4)/20; intercept=np.zeros(4)
 np.savez(root/'priors/k562.npz',schema=np.asarray('slp.reduced-rank-response-model/v1'),source_id=np.asarray('k562'),rank=np.asarray(2),alpha=np.asarray(1.),query_ids=qids,feature_mean=fm,feature_scale=fs,design_mean=dm,state_projection=projection,query_loading=loading,intercept=intercept)
 actions=np.ones((2,1,5),np.float64); mask=np.array([[True],[False]]); basal=np.array([[3.1,3.2,3.3,3.4],[np.pi,np.e,1/3,1/7]],np.float64); request=tmp_path/'request.npz'; output=tmp_path/'out.npz'; np.savez(request,actions=actions,action_mask=mask,basal=basal)
 subprocess.run([sys.executable,str(PATH.with_name('inference.py')),'--model',str(root),'--checkpoint','step-000000.safetensors','--context','k562','--input',str(request),'--output',str(output)],check=True)
 with np.load(output,allow_pickle=False) as result:
  expected=basal.astype(float); expected[0]+=((actions[0,0].astype(float)-fm)/fs-dm)@projection@loading
  np.testing.assert_allclose(result['predictions'],expected,rtol=0,atol=1e-6); np.testing.assert_array_equal(result['query_ids'],qids)
  assert result['predictions'].dtype==np.float64
  assert np.array_equal(result['predictions'][1],basal[1])
