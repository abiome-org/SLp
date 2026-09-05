import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("frozen_test",ROOT/"scripts/score_slp11_gears_frozen_test.py")
MOD=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MOD)

def test_paired_bootstrap_is_deterministic_and_directional():
    candidate_r=np.asarray([.4,.5,.6]);baseline_r=candidate_r-.1
    candidate_m=np.asarray([1.,2.,3.]);baseline_m=candidate_m+.2
    a=MOD.bootstrap(candidate_r,candidate_m,baseline_r,baseline_m,reps=200)
    b=MOD.bootstrap(candidate_r,candidate_m,baseline_r,baseline_m,reps=200)
    assert a==b
    assert np.allclose(a["pearsonDeltaGain95"],[.1,.1,.1])
    assert np.allclose(a["mseReduction95"],[.2,.2,.2])

def test_condition_metrics_are_control_referenced():
    control=np.asarray([1.,1.,1.]);truth=np.asarray([[2.,1.,0.]])
    r,m=MOD.per_condition(truth,truth.copy(),control)
    assert np.allclose(r,[1.]) and np.allclose(m,[0.])
