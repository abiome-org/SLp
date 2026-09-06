"""Portable inference-only synthetic-lethality research ranking predictor."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path, PurePosixPath, PureWindowsPath
import numpy as np
import lightgbm as lgb

SCHEMA="slp.sl-predictor/v1"
WIDTHS={"retainedBaseline":1081,"geneDescriptors":642,"world":1054}

def _sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1<<20),b""): digest.update(chunk)
    return digest.hexdigest()

class SLPredictor:
    def __init__(self,bundle):
        self.root=Path(bundle).resolve()
        payload=json.loads((self.root/"manifest.json").read_text(encoding="utf-8"))
        if payload.get("schema")!=SCHEMA or payload.get("featureWidths")!=WIDTHS: raise ValueError("unsupported predictor manifest")
        files=payload.get("files"); models=payload.get("models")
        if not isinstance(files,dict) or not isinstance(models,list) or not models: raise ValueError("invalid predictor manifest")
        resolved={}
        for relative,expected in files.items():
            if not isinstance(relative,str) or not isinstance(expected,str): raise ValueError("invalid file receipt")
            posix=PurePosixPath(relative); windows=PureWindowsPath(relative)
            if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or ".." in windows.parts:
                raise ValueError("receipted path must be relative and contained")
            path=(self.root/relative).resolve()
            try: path.relative_to(self.root)
            except ValueError as error: raise ValueError("receipted path escapes bundle") from error
            if not path.is_file() or _sha256(path)!=expected: raise ValueError(f"file hash mismatch: {relative}")
            resolved[relative]=path
        self.models={}
        for item in models:
            try: key=(int(item["seed"]),int(item["fold"])); family=item["family"]; weight=float(item["blendWeight"])
            except (KeyError,TypeError,ValueError) as error: raise ValueError("invalid model record") from error
            if key in self.models or family not in ("v1","v2") or not 0<=weight<=1: raise ValueError("invalid model selection")
            paths={}
            for arm in ("control","augmented"):
                relative=item.get(arm)
                if not isinstance(relative,str) or relative not in files: raise ValueError("model file is not receipted")
                paths[arm]=resolved[relative]
            self.models[key]={"family":family,"weight":weight,
                              "control":lgb.Booster(model_file=str(paths["control"])),
                              "augmented":lgb.Booster(model_file=str(paths["augmented"]))}
            expected=(1081,2135) if family=="v1" else (3007,4061)
            actual=(self.models[key]["control"].num_feature(),self.models[key]["augmented"].num_feature())
            if actual!=expected: raise ValueError(f"model feature width mismatch for seed/fold {key}: {actual} != {expected}")

    @staticmethod
    def _features(retained_baseline,raw_gene_descriptor_pairs,world_features,pair_ids):
        baseline=np.asarray(retained_baseline,np.float32); genes=np.asarray(raw_gene_descriptor_pairs,np.float32); world=np.asarray(world_features,np.float32)
        if baseline.ndim!=2 or baseline.shape[1]!=1081: raise ValueError("retained_baseline must be [B,1081]")
        b=len(baseline)
        if genes.shape!=(b,2,642) or world.shape!=(b,1054): raise ValueError("gene pairs/world must be [B,2,642]/[B,1054]")
        if not np.isfinite(baseline).all() or not np.isfinite(genes).all() or not np.isfinite(world).all(): raise ValueError("features must be finite")
        ids=None
        if pair_ids is not None:
            ids=np.asarray(pair_ids)
            if ids.shape!=(b,2) or ids.dtype.kind not in "USO": raise ValueError("pair_ids must be stable strings [B,2]")
            if ids.dtype.kind=="O" and not all(isinstance(value,str) for value in ids.flat): raise ValueError("pair_ids must contain strings")
            ids=ids.astype(str)
            if np.any(ids=="") or np.any(ids[:,0]==ids[:,1]): raise ValueError("pair IDs must be nonempty distinct genes")
        with np.errstate(over="ignore",invalid="ignore"):
            static=np.concatenate((genes[:,0]+genes[:,1],np.abs(genes[:,0]-genes[:,1]),genes[:,0]*genes[:,1]),axis=1)
        if not np.isfinite(static).all(): raise ValueError("static descriptor composition overflowed")
        return baseline,static,world,ids

    def predict_fold(self,seed,fold,retained_baseline,raw_gene_descriptor_pairs,world_features,pair_ids=None):
        try: model=self.models[(int(seed),int(fold))]
        except KeyError as error: raise ValueError("seed/fold is absent from predictor bundle") from error
        baseline,static,world,_=self._features(retained_baseline,raw_gene_descriptor_pairs,world_features,pair_ids)
        if len(baseline)==0: return np.empty((0,),np.float64)
        control=baseline if model["family"]=="v1" else np.concatenate((baseline,static),axis=1)
        augmented=np.concatenate((control,world),axis=1)
        left=np.asarray(model["control"].predict(control,num_threads=4),np.float64); right=np.asarray(model["augmented"].predict(augmented,num_threads=4),np.float64)
        score=(1-model["weight"])*left+model["weight"]*right
        if score.shape!=(len(baseline),) or not np.isfinite(score).all(): raise ValueError("nonfinite model score")
        return score

    def predict_ensemble(self,retained_baseline,raw_gene_descriptor_pairs,world_features,pair_ids=None,folds=None):
        selected=sorted(self.models) if folds is None else [(int(seed),int(fold)) for seed,fold in folds]
        if not selected or len(set(selected))!=len(selected): raise ValueError("ensemble folds must be unique and nonempty")
        scores=[self.predict_fold(seed,fold,retained_baseline,raw_gene_descriptor_pairs,world_features,pair_ids) for seed,fold in selected]
        return np.mean(scores,axis=0,dtype=np.float64)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--model",type=Path,required=True); parser.add_argument("--input",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int); parser.add_argument("--fold",type=int); parser.add_argument("--ensemble",action="store_true"); args=parser.parse_args()
    if args.ensemble==(args.seed is not None or args.fold is not None) or ((args.seed is None)!=(args.fold is None)): raise ValueError("choose --ensemble or both --seed and --fold")
    with np.load(args.input,allow_pickle=False) as pack: values={key:np.asarray(pack[key]) for key in pack.files}
    required={"retained_baseline","raw_gene_descriptor_pairs","world_features"}
    if not required<=values.keys() or values.keys()-required-{"pair_ids"}: raise ValueError("invalid request arrays")
    predictor=SLPredictor(args.model); call=predictor.predict_ensemble if args.ensemble else lambda *x,**kw:predictor.predict_fold(args.seed,args.fold,*x,**kw)
    score=call(values["retained_baseline"],values["raw_gene_descriptor_pairs"],values["world_features"],pair_ids=values.get("pair_ids"))
    output={"score":score,"score_semantics":np.asarray("research-ranking-not-calibrated-probability")}
    if "pair_ids" in values: output["pair_ids"]=values["pair_ids"]
    np.savez_compressed(args.output,**output)
if __name__=="__main__": main()
