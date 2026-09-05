import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

PATH=Path(__file__).parents[1]/'scripts/extract_slp11_joint_world_pair_features.py'
SPEC=importlib.util.spec_from_file_location('joint_pair_features',PATH)
MOD=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=MOD;SPEC.loader.exec_module(MOD)

def test_canonical_alignment_and_duplicate_gene_rejection():
    ids=np.asarray([['ENSG2','ENSG1'],['ENSG3','ENSG4']]);features=np.arange(24).reshape(2,2,6)
    actual_ids,actual_features=MOD.canonicalize(ids,features)
    np.testing.assert_array_equal(actual_ids,[['ENSG1','ENSG2'],['ENSG3','ENSG4']])
    np.testing.assert_array_equal(actual_features[0],features[0,::-1])
    with pytest.raises(ValueError,match='distinct'):
        MOD.canonicalize(np.asarray([['ENSG1','ENSG1']]),np.zeros((1,2,6)))

def test_prediction_summary_is_symmetric_and_finite():
    rng=np.random.default_rng(731);base=rng.normal(size=(3,5));a=base+rng.normal(size=(3,5));b=base+rng.normal(size=(3,5))
    pair=a+b-base+.1;ab=pair+.2;ba=pair-.2;mask=np.asarray([True,False,True,True,False])
    left=MOD.prediction_summary(a,b,pair,ab,ba,base,mask)
    right=MOD.prediction_summary(b,a,pair,ba,ab,base,mask)
    np.testing.assert_allclose(left,right)
    assert left.shape==(3,30) and np.isfinite(left).all()

def test_latent_sequence_uses_total_routes_and_is_pair_symmetric():
    a=np.asarray([[1.,2.]]);b=np.asarray([[3.,4.]]);pair=np.asarray([[7.,9.]])
    after_a=np.asarray([[5.,6.]]);after_b=np.asarray([[7.,8.]])
    left=MOD.latent_pair_features(a,b,pair,after_a,after_b)
    right=MOD.latent_pair_features(b,a,pair,after_b,after_a)
    np.testing.assert_allclose(left,right)
    np.testing.assert_allclose(left[:,10:12],[[8.,10.]])
    np.testing.assert_allclose(left[:,12:14],[[4.,4.]])
