import importlib.util, hashlib, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; CORE=ROOT/'modules/slp-1-1-response-omf2/response_model.py'
def module():
 s=importlib.util.spec_from_file_location('omf2_response_model',CORE); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m
def test_fit_save_reload_and_query_subset(tmp_path):
 m=module(); rng=np.random.default_rng(731); x=rng.normal(size=(40,8)).astype(np.float32); y=x@rng.normal(size=(8,12))
 fitted=m.fit(x,y,rank=4,alpha=10.); path=tmp_path/'model.npz'; ids=np.asarray([f'q{i}' for i in range(12)]); m.save(path,fitted,query_ids=ids,source_id='fixture'); restored=m.load(path)
 np.testing.assert_allclose(restored.predict(x),fitted.predict(x),rtol=0,atol=0); np.testing.assert_allclose(restored.predict(x,[2,5]),fitted.predict(x)[:,[2,5]],rtol=0,atol=0)

def test_standalone_inference_cli(tmp_path):
 m=module(); rng=np.random.default_rng(9); x=rng.normal(size=(20,5)).astype(np.float32); y=rng.normal(size=(20,7)); fitted=m.fit(x,y,rank=3,alpha=2.)
 for name in ('response_model.py','inference.py'): shutil.copyfile(ROOT/'modules/slp-1-1-response-omf2'/name,tmp_path/name)
 model=tmp_path/'model-k562.npz'; ids=np.asarray([f'q{i}' for i in range(7)]); m.save(model,fitted,query_ids=ids,source_id='k562')
 digest=hashlib.sha256(model.read_bytes()).hexdigest(); (tmp_path/'manifest.json').write_text(json.dumps({'contexts':{'k562':{'model':model.name,'sha256':digest}}}))
 request=tmp_path/'request.npz'; output=tmp_path/'prediction.npz'; anchor=np.zeros((3,7)); np.savez(request,features=x[:3],basal_anchor=anchor)
 subprocess.run([sys.executable,str(tmp_path/'inference.py'),'--model',str(tmp_path),'--context','k562','--input',str(request),'--output',str(output)],check=True)
 with np.load(output,allow_pickle=False) as result:
  np.testing.assert_array_equal(result['query_ids'],ids); np.testing.assert_allclose(result['predictions'],fitted.predict(x[:3]))
