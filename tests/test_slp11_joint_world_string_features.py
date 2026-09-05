import importlib.util,sys
from pathlib import Path
import numpy as np

PATH=Path(__file__).resolve().parents[1]/'scripts/augment_slp11_joint_world_features.py'; spec=importlib.util.spec_from_file_location('joint_string',PATH); module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)

def test_stable_id_join_and_missing_encoding():
 payload={'entity_id':np.array(['g2','g1']),'entity_taxon':np.array([9606,9606]),'feature_values':np.stack((np.full(64,2),np.full(64,1))).astype(np.float32),'feature_present':np.array([True,False])}
 lookup=module.string_lookup(payload); base=np.zeros((2,577),np.float32); result=module.augment_features(base,np.array(['g2','missing']),lookup)
 assert result.shape==(2,642); np.testing.assert_array_equal(result[0,577:641],2); assert result[0,-1]==1
 np.testing.assert_array_equal(result[1,577:],0)

def test_action_ids_follow_offsets_and_mask_not_row_order():
 data={'action_offsets':np.array([0,1,3]),'action_ids':np.array(['a','c','b']),'action_mask':np.array([[False,True],[True,True]])}
 result=module.row_action_ids(data); np.testing.assert_array_equal(result,np.array([['','a'],['c','b']]))
