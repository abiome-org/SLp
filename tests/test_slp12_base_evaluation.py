"""Leakage, counterfactual-control and frozen-readout evaluation contracts."""
from test_slp12_training_contracts import execute


def test_wrong_actions_cross_panels_preserve_conditions_and_exclude_noops():
    execute('''
import numpy as np
from data import batch_template
from base_evaluation import wrong_actions, action_key
b=batch_template(3,2,2,3);b['action_ids'][:]=[1,2];b['action_mask'][:]=True
b['action_mechanism'][:]=2;b['action_values'][:]=.3;b['action_known'][:]=True
b['action_features'][:]=1;b['context'][:]=17
same,available=wrong_actions(b,[b]);assert not available.any()
d={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in b.items()}
d['action_ids'][:]=[3,4];d['action_features'][:]=9
s,available=wrong_actions(b,[b,d]);assert available.all()
assert all(action_key(s,i)!=action_key(b,i) for i in range(3))
for k in b:
    if isinstance(b[k],np.ndarray) and k not in ('action_ids','action_features'):
        np.testing.assert_array_equal(b[k],s[k])
d['action_mechanism'][:]=1
assert not wrong_actions(b,[d])[1].any()
d['action_mechanism'][:]=2;d['action_mask'][:,1]=False
assert not wrong_actions(b,[d])[1].any()
''')


def test_baseline_only_uses_fitting_outcomes_and_exposes_missing_queries():
    execute('''
import numpy as np
from types import SimpleNamespace
from data import batch_template
from base_evaluation import FittingMeans,fit_means
try: fit_means(SimpleNamespace(validation=True),'x',{})
except ValueError: pass
else: raise AssertionError('accepted held labels')
b=batch_template(2,1,1,2);b['query_ids'][:]=[1,2]
b['target'][:]=[[2,4],[6,float('nan')]];b['query_mask'][1,1]=False
b['control_target']=np.zeros((2,2));b['control_anchor']=np.full((2,2),7.)
m=FittingMeans(5);m.add(b)
b['target'][:]=99999;b['query_ids'][:]=[1,4]
p=m.predict(b)
np.testing.assert_array_equal(p['target'],[[4,7],[4,7]])
np.testing.assert_array_equal(p['count'],[[2,0],[2,0]])
''')


def test_transfer_partition_is_held_only_and_disjoint():
    execute('''
from base_evaluation import split_genes
genes=['h'+str(i) for i in range(20)]
a,e=split_genes(genes,set(genes),731)
assert not a&e and a|e==set(genes)
assert (a,e)==split_genes(list(reversed(genes)),set(genes),731)
try: split_genes(genes,set(genes[:-1]),731)
except ValueError: pass
else: raise AssertionError('accepted fitting gene')
''')


def test_ridge_is_inductive_and_frozen_feature_extraction_cannot_change_base():
    execute('''
import numpy as np,torch
from transfer_probe import ridge_predict,parameter_digest
from base_evaluation import predict
from data import batch_template
from model import WorldModel,Config
torch.set_num_threads(2)
rng=np.random.default_rng(12);x=rng.normal(size=(200,4));y=x[:,0]*2-x[:,1];e=rng.normal(size=(10,4))
p=ridge_predict(x,y,e,.0001)
assert np.mean((p-(e[:,0]*2-e[:,1]))**2)<.001
np.testing.assert_allclose(p[:1],ridge_predict(x,y,np.concatenate([e[:1],e[1:]*1e6]),.0001)[:1])
assert np.isfinite(ridge_predict(x[:2],y[:2],e,.01)).all()
m=WorldModel(Config(width=32,heads=4,layers=2)).eval().requires_grad_(False)
b=batch_template(2,1,1,1)
before=parameter_digest(m);mean,features=predict(m,b,'cpu',features=True)
b['target'][:]=999
mean2,features2=predict(m,b,'cpu',features=True)
assert parameter_digest(m)==before and features.shape==(2,32)
np.testing.assert_array_equal(mean,mean2);np.testing.assert_array_equal(features,features2)
assert all(p.grad is None for p in m.parameters())
''')


def test_incompatible_bundle_is_rejected_and_pseudobulk_does_not_mix_actions():
    execute('''
import json,tempfile,numpy as np
from pathlib import Path
from types import SimpleNamespace
from base_evaluation import validate_bundle,score_case
from data import batch_template
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    (p/'corpus.json').write_text(json.dumps(dict(receipts={},held_human_genes=[],held_yeast_genes=[],molecular_scales={})))
    np.savez(p/'normalizers.npz',x=np.zeros(2))
    try: validate_bundle(SimpleNamespace(bundle=p,vocabulary=['wrong']),{})
    except ValueError: pass
    else: raise AssertionError('accepted incompatible bundle')
b=batch_template(2,1,1,2);b['action_mask'][:]=True;b['action_ids'][:,0]=[1,2]
b['baseline_count']=np.ones((2,2));b['wrong_available']=np.ones(2,bool)
try: score_case(b,{'model':b['target']},True)
except ValueError: pass
else: raise AssertionError('mixed interventions in pseudobulk')
''')
