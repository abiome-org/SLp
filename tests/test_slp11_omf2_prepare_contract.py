import importlib.util, sys
from pathlib import Path
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/prepare_slp11_omf2_data.py'
spec=importlib.util.spec_from_file_location('slp11_omf2_prepare',PATH)
module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)

def test_partition_accepts_disjoint_ids_and_identical_queries():
 module.validate_partition(np.array(['a','b']),np.array(['c']),np.array(['q1','q2']),np.array(['q1','q2']))

def test_partition_rejects_intervention_overlap():
 with pytest.raises(ValueError,match='overlap'):
  module.validate_partition(np.array(['a','b']),np.array(['b','c']),np.array(['q']),np.array(['q']))

def test_partition_rejects_query_axis_drift():
 with pytest.raises(ValueError,match='query axes'):
  module.validate_partition(np.array(['a']),np.array(['b']),np.array(['q1','q2']),np.array(['q2','q1']))

def test_file_receipt_supports_external_output(tmp_path):
 path=tmp_path/'panel.npz'; path.write_bytes(b'panel')
 receipt=module.file_receipt(path)
 assert receipt['path']==str(path.resolve()).replace('\\','/')
 assert receipt['bytes']==5
 assert len(receipt['sha256'])==64
