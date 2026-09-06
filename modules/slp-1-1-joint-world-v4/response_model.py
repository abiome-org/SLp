"""Application-neutral reduced-rank feature-to-response model.

Small fitting panels use leave-descriptor-group-out empirical-Bayes shrinkage:
the feature-dependent term is shrunk toward the fitting response mean.  Groups
are exact feature rows, so replicate guide views of one action cannot occur on
both sides of a fold.  Larger panels retain the established unshrunk fit.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

_SHRINKAGE_MAX_ROWS = 64


def calibrate_template_saturation(per_action_predictions, targets, basal,
                                  action_mask, fit_rows, template):
    """Estimate repeated-template subtraction on explicitly supplied fitting rows."""
    predictions=np.asarray(per_action_predictions,np.float64)
    targets=np.asarray(targets,np.float64); basal=np.asarray(basal,np.float64)
    mask=np.asarray(action_mask,bool); rows=np.asarray(fit_rows)
    template=np.asarray(template,np.float64)
    if predictions.ndim!=3 or mask.shape!=predictions.shape[:2]: raise ValueError("invalid per-action predictions or mask")
    if targets.shape!=basal.shape or targets.shape!=(len(predictions),predictions.shape[2]): raise ValueError("invalid targets or basal")
    if template.shape!=(predictions.shape[2],) or not np.isfinite(template).all(): raise ValueError("invalid template")
    if rows.ndim!=1 or not np.issubdtype(rows.dtype,np.integer) or np.any(rows<0) or np.any(rows>=len(predictions)): raise ValueError("invalid fit rows")
    if not np.isfinite(predictions).all() or not np.isfinite(targets[rows]).all() or not np.isfinite(basal[rows]).all(): raise ValueError("inputs must be finite")
    counts=mask[rows].sum(1)
    combinations=counts>1
    if not np.any(combinations): return 0.0
    selected=rows[combinations]; counts=counts[combinations]
    summed=(predictions[selected]*mask[selected,:,None]).sum(1)
    repeated=(counts-1)[:,None]*template[None,:]
    error=summed-(targets[selected]-basal[selected])
    denominator=float(np.sum(repeated*repeated))
    if denominator<=0: return 0.0
    return float(np.clip(np.sum(error*repeated)/denominator,0.,1.))


def apply_prior_increment(per_action_predictions, template, action_mask,
                          include_template, saturation):
    """Compose action priors with a learned repeated-template correction."""
    predictions=np.asarray(per_action_predictions,np.float64)
    mask=np.asarray(action_mask,bool); template=np.asarray(template,np.float64)
    include=np.asarray(include_template,bool)
    if predictions.ndim!=3 or mask.shape!=predictions.shape[:2]: raise ValueError("invalid per-action predictions or mask")
    if template.shape!=(predictions.shape[2],) or include.shape!=(len(predictions),): raise ValueError("invalid template or include_template")
    if not np.isfinite(predictions).all() or not np.isfinite(template).all() or not np.isfinite(saturation) or not 0<=saturation<=1: raise ValueError("invalid prior values")
    counts=mask.sum(1)
    summed=(predictions*mask[:,:,None]).sum(1)
    correction=saturation*(counts-include.astype(np.int64))[:,None]*template[None,:]
    output=summed-correction
    output[counts==0]=0.
    return output


@dataclass(frozen=True)
class ReducedRankResponse:
    feature_mean: np.ndarray; feature_scale: np.ndarray; design_mean: np.ndarray
    state_projection: np.ndarray; query_loading: np.ndarray; intercept: np.ndarray; alpha: float
    shrinkage: float = 1.0

    @property
    def rank(self): return self.state_projection.shape[1]

    def predict(self, features, query_indices=None):
        x=np.asarray(features,np.float32)
        if x.ndim!=2 or x.shape[1]!=len(self.feature_mean) or not np.isfinite(x).all(): raise ValueError("features must be finite [N,F]")
        idx=np.arange(len(self.intercept)) if query_indices is None else np.asarray(query_indices)
        if idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer) or np.any(idx<0) or np.any(idx>=len(self.intercept)): raise ValueError("invalid query indices")
        d=(x.astype(np.float64)-self.feature_mean)/self.feature_scale
        out=self.intercept[idx]+self.shrinkage*((d-self.design_mean)@self.state_projection)@self.query_loading[:,idx]
        if not np.isfinite(out).all(): raise ValueError("nonfinite prediction")
        return out


def _fit_base(x, y, rank, alpha, shrinkage=1.0):
    mean=x.mean(0,dtype=np.float64); sd=x.std(0,dtype=np.float64); scale=np.where(sd>1e-5,sd,1.)
    design=(x.astype(np.float64)-mean)/scale; dm=design.mean(0); cd=design-dm; intercept=y.mean(0); cy=y-intercept
    ev,vec=np.linalg.eigh(cd.T@cd); keep=ev>1e-8; ev=ev[keep]; vec=vec[:,keep]
    if not len(ev):
        return ReducedRankResponse(mean,scale,dm,np.zeros((x.shape[1],0)),
                                   np.zeros((0,y.shape[1])),intercept,float(alpha),float(shrinkage))
    rhs=(cd@vec).T@cy; root=np.sqrt(ev+alpha); white=rhs/root[:,None]
    _,rv=np.linalg.eigh(white@white.T); rv=rv[:,-min(rank,len(rv)):]
    return ReducedRankResponse(mean,scale,dm,(vec/root[None,:])@rv,rv.T@white,
                               intercept,float(alpha),float(shrinkage))


def _empirical_shrinkage(x, y, rank, alpha):
    """Fit one pooled coefficient without using a held outcome or query."""
    _, groups=np.unique(np.ascontiguousarray(x).view(
        np.dtype((np.void, x.dtype.itemsize*x.shape[1]))).reshape(-1),return_inverse=True)
    if len(x)>_SHRINKAGE_MAX_ROWS or len(np.unique(groups))<3:
        return 1.0
    feature_oof=np.empty_like(y,dtype=np.float64)
    mean_oof=np.empty_like(y,dtype=np.float64)
    for group in np.unique(groups):
        test=groups==group; train=~test
        fold=_fit_base(x[train],y[train],rank,alpha)
        feature_oof[test]=fold.predict(x[test])
        mean_oof[test]=y[train].mean(0)
    direction=feature_oof-mean_oof
    denominator=float(np.sum(direction*direction))
    if denominator<=0: return 0.0
    return float(np.clip(np.sum(direction*(y-mean_oof))/denominator,0.,1.))


def fit(features,targets,*,rank=32,alpha=1000.0):
    x=np.asarray(features,np.float32); y=np.asarray(targets,np.float64)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or rank<=0 or alpha<=0: raise ValueError("invalid fitting matrices")
    shrinkage=_empirical_shrinkage(x,y,rank,alpha)
    model=_fit_base(x,y,rank,alpha,shrinkage); model.predict(x[:1]); return model


def save(path,model,*,query_ids,source_id):
    q=np.asarray(query_ids).astype(str)
    np.savez_compressed(path,schema=np.asarray("slp.reduced-rank-response-model/v2"),source_id=np.asarray(source_id),rank=np.asarray(model.rank),alpha=np.asarray(model.alpha),shrinkage=np.asarray(model.shrinkage),query_ids=q,feature_mean=model.feature_mean,feature_scale=model.feature_scale,design_mean=model.design_mean,state_projection=model.state_projection,query_loading=model.query_loading,intercept=model.intercept)


def load(path):
    with np.load(path,allow_pickle=False) as a:
        schema=str(a['schema'])
        if schema not in ("slp.reduced-rank-response-model/v1","slp.reduced-rank-response-model/v2"): raise ValueError("unsupported model")
        values=[np.asarray(a[k],np.float64) for k in ('feature_mean','feature_scale','design_mean','state_projection','query_loading','intercept')]
        shrinkage=float(a['shrinkage']) if schema.endswith('/v2') else 1.0
        return ReducedRankResponse(*values,float(a['alpha']),shrinkage)
