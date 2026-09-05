import importlib.util,sys
from pathlib import Path
import numpy as np
PATH=Path(__file__).resolve().parents[1]/'scripts/prepare_slp11_joint_world_expanded_development.py';spec=importlib.util.spec_from_file_location('expanded_dev',PATH);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
def test_held_rows_preserve_views_and_remove_training_genes():
 ids=np.array(['fit','held','held','other']);roles=np.array(['validation']*4);contexts=np.array([2,2,2,3]);selected,kept=module.held_rows(ids,roles,contexts,2,{'fit'})
 np.testing.assert_array_equal(selected,[0,1,2]);np.testing.assert_array_equal(kept,[1,2])
def test_symbol_mapping_strips_ensembl_version(tmp_path):
 path=tmp_path/'x.gtf';path.write_text('1\tx\tgene\t1\t2\t.\t+\t.\tgene_id "ENSG1.4"; gene_name "ABC";\n')
 assert module.symbol_map(path)=={'ENSG1':'ABC'}
