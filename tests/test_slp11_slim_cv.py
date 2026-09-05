import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("slim_cv_test",ROOT/"scripts/run_slp11_slim_cv.py")
MOD=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MOD)

def test_fold_assignment_is_identity_deterministic():
    ids=np.asarray(["ENSG3","ENSG1","ENSG2"])
    first=MOD.folds(ids)
    lookup=dict(zip(ids,first))
    assert np.array_equal(MOD.folds(ids[::-1]),np.asarray([lookup[x] for x in ids[::-1]]))
    assert np.all((first>=0)&(first<3))

def test_normalizer_is_fitting_rows_only_and_handles_constants():
    fitting=np.asarray([[1.,4.],[3.,4.]])
    mean,scale=MOD.normalize_fit(fitting)
    assert np.allclose(mean,[2.,4.])
    assert np.allclose(scale,[1.,1.])
    held=np.asarray([[100.,9.]])
    assert np.allclose((held-mean)/scale,[[98.,5.]])
