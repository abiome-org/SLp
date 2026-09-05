import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_DIR=Path(__file__).parents[1]/'modules/slp-1-1-joint-world-v2'
sys.path.insert(0,str(MODULE_DIR))
SPEC=importlib.util.spec_from_file_location('joint_v2_evaluate',MODULE_DIR/'evaluate.py')
EVAL=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(EVAL)


def test_canonical_pair_fold_ignores_action_order():
    left={'combination_rows':np.asarray([0]),'action_ids':np.asarray([['ENSG2','ENSG1']]),
          'action_mask':np.ones((1,2),bool)}
    right={'combination_rows':np.asarray([0]),'action_ids':np.asarray([['ENSG1','ENSG2']]),
           'action_mask':np.ones((1,2),bool)}
    np.testing.assert_array_equal(EVAL.canonical_pair_folds(left),EVAL.canonical_pair_folds(right))


def test_pair_report_scores_raw_and_query_centered_nonadditivity():
    truth=np.asarray([[2.,1.,4.],[4.,2.,3.]])
    additive=np.asarray([[1.,1.,1.],[2.,2.,2.]])
    values={'observedAdditive':additive,'candidate':truth.copy()}
    report=EVAL.pair_report(truth,values)
    assert report['candidate']['mse']==0.0
    assert report['candidate']['centeredNonadditivePearson']==pytest.approx(1.0)
    assert report['candidate']['finiteCenteredRows']==2


def test_context_arrays_broadcast_static_control_profile():
    data={'control_context_values':np.asarray([1.,2.]),
          'control_context_observed':np.asarray([True,False])}
    values,mask=EVAL.context_arrays(data,np.asarray([3,5]))
    assert values.shape==mask.shape==(2,2)
    np.testing.assert_array_equal(values,[[1.,2.],[1.,2.]])


def test_grouped_bootstrap_keeps_duplicate_pair_views_together():
    truth=np.zeros((3,2));autonomous=np.asarray([[1.,1.],[3.,3.],[2.,2.]])
    additive=np.asarray([[2.,2.],[2.,2.],[3.,3.]])
    result=EVAL.grouped_mse_bootstrap(
        truth,autonomous,additive,np.asarray(['A|B','A|B','C|D']),repetitions=200,seed=731)
    # A|B contributes +1 after its two views stay together; C|D contributes -5.
    assert result['mseDifference']==pytest.approx(-2.0)
    assert result['pairGroups']==2
