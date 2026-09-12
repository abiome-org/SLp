"""Isolated checks for holdout routing and masked quantitative training loss."""
from pathlib import Path
import subprocess
import sys

MODULE = Path(__file__).resolve().parents[1] / 'modules/slp-1-2'


def execute(source):
    subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, ' + repr(str(MODULE)) + ')\n' + source],
                   check=True, capture_output=True, text=True, timeout=45)


def test_population_holdout_cannot_change_fitted_scale_or_supply_mixed_pairs():
    execute('''
import copy,numpy as np
from data import Population,partition_source
s=object.__new__(Population)
s.ids=np.array([['a',''],['b',''],['held',''],['a','held'],['held','held2']])
s.target=np.array([[1.],[3.],[1000.],[10000.],[100000.]],np.float32)
s.basal=np.zeros_like(s.target);s.observed=np.ones_like(s.target,dtype=bool)
v=copy.deepcopy(s)
assert partition_source(s,{'held','held2'},False)
assert partition_source(v,{'held','held2'},True)
assert set(s.ids[:,0])=={'a','b'}
assert len(v.ids)==2 and np.isin(v.ids,['held','held2','']).all()
assert np.isclose(s.scale,np.sqrt(5)) and s.scale==v.scale
assert (s.parent_rows==-1).all() and (v.parent_rows==-1).all()
''')


def test_unmeasured_nan_targets_do_not_poison_loss_or_gradients():
    execute('''
import torch
from data import batch_template,to_device
from model import WorldModel,Config
from train import objective
torch.set_num_threads(2)
m=WorldModel(Config(width=32,heads=4,layers=2)).train()
b=batch_template(2,2,1,3)
b['query_mask'][:,-1]=False;b['target'][:,-1]=float('nan')
b['query_features'][:,-1]=float('nan');b['query_anchor'][:,-1]=float('nan')
b=to_device(b,'cpu')
for flow in (False,True):
    m.zero_grad(set_to_none=True)
    loss,_=objective(m,b,flow=flow,noise_scale=1.)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
''')


def test_control_supervision_masks_actions_without_revealing_cell_targets():
    execute('''
import numpy as np
from data import batch_template,apply_controls
b=batch_template(2,2,2,3);b['task']='molecular'
b['action_mask'][:]=True;b['target'][:]=9;b['query_anchor'][:]=5
b['control_target']=np.full((2,3),2.,np.float32)
b['control_anchor']=np.full((2,3),1.,np.float32)
b['observation_ids'][:]=[1,2];b['query_ids'][:]=[2,3,4]
b['observation_mask'][:]=True
selected=np.array([True,False])
apply_controls(b,selected)
assert not b['action_mask'][0].any() and b['action_mask'][1].all()
assert (b['target'][0]==2).all() and (b['query_anchor'][0]==1).all()
assert (b['target'][1]==9).all() and (b['query_anchor'][1]==5).all()
assert 'control_target' not in b and 'control_anchor' not in b
assert b['observation_mask'][0].tolist()==[True,False]
assert b['observation_mask'][1].all()
f=batch_template(2,1,2,1);f['task']='yeast_fitness';f['target'][:]=-.5
f['action_mask'][:]=True;apply_controls(f,selected)
assert f['target'][0,0]==0 and f['target'][1,0]==-.5
''')
