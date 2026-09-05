import importlib.util,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'scripts/aggregate_slp11_joint_world_norman_folds.py';S=importlib.util.spec_from_file_location('agg',P);M=importlib.util.module_from_spec(S);sys.modules[S.name]=M;S.loader.exec_module(M)
def test_summary_and_bootstrap_improvement():
 truth=np.array([[1.,2.,4.],[2.,4.,1.],[3.,1.,5.]])
 add=np.ones_like(truth);good=truth.copy();bad=truth+np.array([[1.,-1.,1.]])
 metrics,per=M.summaries(truth,{'observedAdditive':add,'autonomousAverage':good,'predictedAdditive':bad,'directTwoActions':good,'priorOnly':bad,'observedParentAverage':good,'observedParentPrior':bad},np.ones(3,bool))
 assert metrics['autonomousAverage']['mse']==0
 assert M.summaries(truth,{**{'observedAdditive':add,'zeroResponse':np.zeros_like(truth)}},np.ones(3,bool))[0]['zeroResponse']['mse']==np.square(truth).mean()
 boot=M.bootstrap(per,reps=100)
 assert boot['autonomousAverageVspredictedAdditive']['mseReduction95'][0]>0
