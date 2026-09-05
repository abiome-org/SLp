"""Application-neutral reduced-rank feature-to-response model."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ReducedRankResponse:
    feature_mean: np.ndarray; feature_scale: np.ndarray; design_mean: np.ndarray
    state_projection: np.ndarray; query_loading: np.ndarray; intercept: np.ndarray; alpha: float
    @property
    def rank(self): return self.state_projection.shape[1]
    def predict(self, features, query_indices=None):
        x=np.asarray(features,np.float32)
        if x.ndim!=2 or x.shape[1]!=len(self.feature_mean) or not np.isfinite(x).all(): raise ValueError("features must be finite [N,F]")
        idx=np.arange(len(self.intercept)) if query_indices is None else np.asarray(query_indices)
        if idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer) or np.any(idx<0) or np.any(idx>=len(self.intercept)): raise ValueError("invalid query indices")
        d=(x.astype(np.float64)-self.feature_mean)/self.feature_scale
        out=self.intercept[idx]+((d-self.design_mean)@self.state_projection)@self.query_loading[:,idx]
        if not np.isfinite(out).all(): raise ValueError("nonfinite prediction")
        return out

def fit(features,targets,*,rank=32,alpha=1000.0):
    x=np.asarray(features,np.float32); y=np.asarray(targets,np.float64)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or rank<=0 or alpha<=0: raise ValueError("invalid fitting matrices")
    mean=x.mean(0,dtype=np.float64); sd=x.std(0,dtype=np.float64); scale=np.where(sd>1e-5,sd,1.)
    design=(x.astype(np.float64)-mean)/scale; dm=design.mean(0); cd=design-dm; intercept=y.mean(0); cy=y-intercept
    ev,vec=np.linalg.eigh(cd.T@cd); keep=ev>1e-8; ev=ev[keep]; vec=vec[:,keep]
    rhs=(cd@vec).T@cy; root=np.sqrt(ev+alpha); white=rhs/root[:,None]
    _,rv=np.linalg.eigh(white@white.T); rv=rv[:,-min(rank,len(rv)):]
    model=ReducedRankResponse(mean,scale,dm,(vec/root[None,:])@rv,rv.T@white,intercept,float(alpha)); model.predict(x[:1]); return model

def save(path,model,*,query_ids,source_id):
    q=np.asarray(query_ids).astype(str)
    np.savez_compressed(path,schema=np.asarray("slp.reduced-rank-response-model/v1"),source_id=np.asarray(source_id),rank=np.asarray(model.rank),alpha=np.asarray(model.alpha),query_ids=q,feature_mean=model.feature_mean,feature_scale=model.feature_scale,design_mean=model.design_mean,state_projection=model.state_projection,query_loading=model.query_loading,intercept=model.intercept)

def load(path):
    with np.load(path,allow_pickle=False) as a:
        if str(a['schema'])!="slp.reduced-rank-response-model/v1": raise ValueError("unsupported model")
        return ReducedRankResponse(*[np.asarray(a[k],np.float64) for k in ('feature_mean','feature_scale','design_mean','state_projection','query_loading','intercept')],float(a['alpha']))
