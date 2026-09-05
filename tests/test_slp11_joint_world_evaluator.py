import importlib.util, sys
from pathlib import Path
import numpy as np

PATH=Path(__file__).resolve().parents[1]/'scripts/evaluate_slp11_joint_world.py'
spec=importlib.util.spec_from_file_location('joint_world_evaluator',PATH); evaluator=importlib.util.module_from_spec(spec);sys.modules[spec.name]=evaluator;spec.loader.exec_module(evaluator)

def test_predicted_additive_preserves_basal_reference():
 basal=np.array([[2.,3.]])
 left=basal+np.array([[.5,-.2]]); right=basal+np.array([[-.1,.7]])
 np.testing.assert_allclose(evaluator._predicted_additive(left,right,basal),basal+(left-basal)+(right-basal),rtol=0,atol=1e-15)

def test_composition_metrics_retains_raw_and_centered_correlations():
 additive=np.zeros((3,4)); truth=np.array([[0.,1.,2.,3.],[1.,2.,4.,8.],[2.,4.,3.,9.]])
 prediction=truth+np.array([0.,1.,2.,3.])[None,:]
 metrics=evaluator.composition_metrics(truth,prediction,additive)
 assert 'nonadditivePearson' in metrics
 assert 'independentlyQueryCenteredNonadditivePearson' in metrics
 assert metrics['finiteIndependentlyQueryCenteredNonadditiveRows']>=0
