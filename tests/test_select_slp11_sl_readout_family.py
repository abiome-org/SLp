import importlib.util
from pathlib import Path
import numpy as np
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/select_slp11_sl_readout_family.py';S=importlib.util.spec_from_file_location('readout_selector',P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)

def test_alignment_accepts_exact_roster_and_rejects_reordering():
    ids=np.array([['a','b'],['c','d'],['e','f']]);fold={'pair_indices':np.array([0,2,1]),'source_row':np.array([4,5,6]),'partition':np.array([0,1,1])}
    values={'pair_indices':np.array([2,1]),'source_row':np.array([5,6]),'pair_ids':ids[[2,1]],'blend':np.array([.2,.8])}
    M.validate_alignment(values,dict(values),fold,ids)
    changed={**values,'pair_indices':values['pair_indices'][::-1]}
    with pytest.raises(ValueError):M.validate_alignment(values,changed,fold,ids)

def test_tie_rule_prefers_v1():
    assert M.select_family(.5,.5)=='v1'
    assert M.select_family(.6,.5)=='v2'
