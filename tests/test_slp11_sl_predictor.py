import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/"data/tooling/slp11-readout-site"
sys.path.insert(0,str(SITE))
import lightgbm as lgb


def load_predictor():
    spec=importlib.util.spec_from_file_location("slp11_sl_predictor",ROOT/"modules/slp-1-1-sl-predictor-v1/predictor.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def booster(path,width,seed):
    rng=np.random.default_rng(seed); x=rng.normal(size=(18,width)).astype(np.float32); y=np.linspace(0,1,len(x))
    model=lgb.train({"objective":"regression","verbosity":-1,"num_threads":1,"min_data_in_leaf":1,"num_leaves":5,"seed":seed},lgb.Dataset(x,label=y),num_boost_round=4)
    model.save_model(str(path)); return model


def bundle(tmp_path):
    root=tmp_path/"bundle"; models=root/"models"; models.mkdir(parents=True)
    records=[]; files={}; made={}
    for seed,fold,family,weight in ((42,0,"v1",.25),(432,1,"v2",.8)):
        prefix=f"seed{seed}-fold{fold}"; cw=1081 if family=="v1" else 3007; aw=cw+1054
        control=models/f"{prefix}-control.txt"; augmented=models/f"{prefix}-augmented.txt"
        made[(seed,fold)]=(booster(control,cw,seed),booster(augmented,aw,seed+1))
        cr=f"models/{control.name}"; ar=f"models/{augmented.name}"; files[cr]=sha(control); files[ar]=sha(augmented)
        records.append({"seed":seed,"fold":fold,"family":family,"blendWeight":weight,"control":cr,"augmented":ar})
    manifest={"schema":"slp.sl-predictor/v1","featureWidths":{"retainedBaseline":1081,"geneDescriptors":642,"world":1054},"models":records,"files":files}
    (root/"manifest.json").write_text(json.dumps(manifest)); return root,made


def test_selected_families_blend_feature_order_and_tamper(tmp_path):
    module=load_predictor(); root,made=bundle(tmp_path); predictor=module.SLPredictor(root)
    rng=np.random.default_rng(731); baseline=rng.normal(size=(3,1081)).astype(np.float32); genes=rng.normal(size=(3,2,642)).astype(np.float32); world=rng.normal(size=(3,1054)).astype(np.float32)
    ids=np.array([["a","b"],["c","d"],["e","f"]]); static=np.concatenate((genes[:,0]+genes[:,1],np.abs(genes[:,0]-genes[:,1]),genes[:,0]*genes[:,1]),1)
    for key,family,weight in (((42,0),"v1",.25),((432,1),"v2",.8)):
        control=baseline if family=="v1" else np.concatenate((baseline,static),1); augmented=np.concatenate((control,world),1)
        expected=(1-weight)*made[key][0].predict(control)+weight*made[key][1].predict(augmented)
        np.testing.assert_array_equal(predictor.predict_fold(*key,baseline,genes,world,ids),expected)
    expected=np.mean([predictor.predict_fold(*key,baseline,genes,world,ids) for key in sorted(made)],axis=0)
    np.testing.assert_array_equal(predictor.predict_ensemble(baseline,genes,world,ids),expected)
    repeated_ids=np.array([["a","b"],["b","a"],["a","b"]])
    repeated_genes=np.stack((genes[0],genes[0,::-1],genes[0]))
    repeated_baseline=np.repeat(baseline[0:1],3,axis=0); repeated_world=np.repeat(world[0:1],3,axis=0)
    repeated=predictor.predict_fold(42,0,repeated_baseline,repeated_genes,repeated_world,repeated_ids)
    np.testing.assert_array_equal(repeated,np.repeat(repeated[0],3))
    target=root/"models/seed42-fold0-control.txt"; target.write_text(target.read_text()+"\n")
    try: module.SLPredictor(root)
    except ValueError as error: assert "hash mismatch" in str(error)
    else: raise AssertionError("tampered booster accepted")


def test_paths_widths_overflow_and_empty_batch(tmp_path):
    module=load_predictor(); root,_=bundle(tmp_path); manifest_path=root/"manifest.json"
    original=json.loads(manifest_path.read_text())

    # Absolute paths remain forbidden even when they point inside the bundle.
    absolute=str((root/original["models"][0]["control"]).resolve())
    absolute_manifest=json.loads(json.dumps(original)); relative=absolute_manifest["models"][0]["control"]
    absolute_manifest["models"][0]["control"]=absolute
    absolute_manifest["files"][absolute]=absolute_manifest["files"].pop(relative)
    manifest_path.write_text(json.dumps(absolute_manifest))
    with pytest.raises(ValueError,match="relative and contained"): module.SLPredictor(root)

    # A family declaration cannot reinterpret boosters trained for other widths.
    wrong=json.loads(json.dumps(original)); wrong["models"][0]["family"]="v2"; manifest_path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match="feature width mismatch"): module.SLPredictor(root)

    manifest_path.write_text(json.dumps(original)); predictor=module.SLPredictor(root)
    empty=predictor.predict_fold(42,0,np.empty((0,1081),np.float32),np.empty((0,2,642),np.float32),np.empty((0,1054),np.float32),np.empty((0,2),dtype="U1"))
    assert empty.shape==(0,) and empty.dtype==np.float64
    huge=np.full((1,2,642),np.finfo(np.float32).max,dtype=np.float32)
    with pytest.raises(ValueError,match="composition overflowed"):
        predictor.predict_fold(432,1,np.zeros((1,1081),np.float32),huge,np.zeros((1,1054),np.float32),np.array([["a","b"]]))
