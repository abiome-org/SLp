import importlib.util,sys
from pathlib import Path
import numpy as np

PATH=Path(__file__).parents[1]/'scripts/prepare_slp11_musl_world_pair_roster.py'
SPEC=importlib.util.spec_from_file_location('musl_roster',PATH);MOD=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=MOD;SPEC.loader.exec_module(MOD)

def test_write_snapshot_preserves_fold_order_and_excludes_uncovered(tmp_path):
    ids=np.asarray(['ENSG1','ENSG2','ENSG3']);lookup={'ENSG1':np.ones(4,np.float32),'ENSG2':np.arange(4,dtype=np.float32)}
    pairs=[('ENSG1','ENSG2')];members=[(42,0,[(0,7,('ENSG1','ENSG2')),(1,2,('ENSG1','ENSG2'))])]
    adapter=tmp_path/'adapter.npz';np.savez(adapter,query_ids=np.asarray(['q']),query_features=np.ones((1,4),np.float32),
        observed_query_mask=np.ones(1,bool),observation_indices=np.asarray([0]),control_context_values=np.asarray([2.],np.float32),control_context_mask=np.ones(1,bool))
    out=tmp_path/'out';manifest=MOD.write_snapshot(out,ids,lookup,4,pairs,members,[],[],adapter)
    with np.load(out/'seed42-fold0.npz') as a:
        np.testing.assert_array_equal(a['partition'],[0,1]);np.testing.assert_array_equal(a['source_row'],[7,2])
    with np.load(out/'k562-control.npz') as a:np.testing.assert_allclose(a['basal'],2*np.log(2),rtol=1e-6)
    assert manifest['uncoveredGenes']==['ENSG3'] and manifest['labelsPresent'] is False
