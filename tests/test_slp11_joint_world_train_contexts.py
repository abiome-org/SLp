import importlib.util,sys
from pathlib import Path
import numpy as np

MODULE=Path(__file__).resolve().parents[1]/'modules/slp-1-1-joint-world-v1';sys.path.insert(0,str(MODULE));spec=importlib.util.spec_from_file_location('joint_train',MODULE/'train.py');train=importlib.util.module_from_spec(spec);sys.modules[spec.name]=train;spec.loader.exec_module(train)

def test_expanded_single_action_context_contract(tmp_path):
 rng=np.random.default_rng(3); n,q,f=8,5,4; path=tmp_path/'gwps.npz'; output=tmp_path/'out';(output/'priors').mkdir(parents=True);(output/'adapters').mkdir()
 context=np.arange(q,dtype=np.float32); context_mask=np.ones(q,bool)
 np.savez(path,targets=rng.normal(size=(n,q)).astype(np.float32),basal=np.zeros((n,q),np.float32),observed=np.ones((n,q),bool),action_features=rng.normal(size=(n,1,f)).astype(np.float32),action_mask=np.ones((n,1),bool),query_features=rng.normal(size=(q,f)).astype(np.float32),query_ids=np.array([f'q{i}' for i in range(q)]),feature_mean=np.zeros(f,np.float32),feature_scale=np.ones(f,np.float32),mode_id=np.asarray(0),assay_id=np.asarray(2),control_context_values=context,control_context_observed=context_mask)
 data=train.prepare_context(path,output,0,np.random.default_rng(731),3,2)
 assert data['prior_per_action'].shape==(n,1,q);assert data['mode']==0 and data['assay']==2
 np.testing.assert_array_equal(data['control_context_values'][0],context)
 with np.load(output/'adapters/gwps.npz',allow_pickle=False) as adapter:np.testing.assert_array_equal(adapter['control_context_values'],context)
