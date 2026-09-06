"""Standalone molecular world inference: encode, intervene, compose, and sample."""
from __future__ import annotations
from dataclasses import dataclass,replace
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from model import CellWorld,Config


@dataclass(frozen=True)
class MolecularState:
    latent: torch.Tensor
    anchor_latent: torch.Tensor
    observed: torch.Tensor
    basal: torch.Tensor
    descriptors: torch.Tensor
    modality: torch.Tensor
    scale: torch.Tensor
    observed_mask: torch.Tensor
    assay: torch.Tensor
    taxon: torch.Tensor
    mechanism: torch.Tensor
    control: torch.Tensor


class WorldModel:
    def __init__(self,directory,device='cpu',verify=True):
        directory=Path(directory);self.device=torch.device(device)
        manifest=directory/'manifest.json'
        if verify and manifest.exists():
            for filename,digest in json.loads(manifest.read_text())['files'].items():
                path=(directory/filename).resolve()
                if not path.is_relative_to(directory.resolve()):raise ValueError('manifest path escapes bundle')
                with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
                if actual!=digest:raise ValueError(f'changed bundle payload: {filename}')
        config=Config(**json.loads((directory/'config.json').read_text()))
        self.model=CellWorld(config).to(self.device).eval()
        self.model.load_state_dict(load_file(str(directory/'model.safetensors'),device=str(self.device)))
        self.normalizer=None
        if (directory/'normalizer.npz').exists():
            with np.load(directory/'normalizer.npz',allow_pickle=False) as z:
                self.normalizer=(self.tensor(z['mean']),self.tensor(z['scale']))

    def tensor(self,value,dtype=torch.float32):
        if isinstance(value,np.ndarray) and not value.flags.writeable:value=value.copy()
        return torch.as_tensor(value,dtype=dtype,device=self.device)

    def descriptors(self,values,modality=None,normalized=False):
        value=self.tensor(values)
        if not torch.isfinite(value).all():raise ValueError('descriptors must be finite')
        if normalized:return value
        if self.normalizer is None:raise ValueError('raw descriptors require the exported feature normalizer')
        mean,scale=self.normalizer
        if modality is None:return (value-mean)/scale
        result=value.clone();rna=modality==0
        result[rna,:642]=(result[rna,:642]-mean)/scale
        return result

    @torch.inference_mode()
    def encode(self,observed,basal,query_descriptors,*,modality,scale,assay,taxon,mechanism,
               observed_mask=None,control=None,encoder_indices=None,normalized_descriptors=False):
        """Values and scales use the adapter's native assay units; taxon is NCBI ID."""
        observed=self.tensor(observed);basal=self.tensor(basal)
        if observed.ndim!=2:raise ValueError('observations must have shape [batch,queries]')
        basal=torch.broadcast_to(basal,observed.shape);b,q=observed.shape
        scale=torch.broadcast_to(self.tensor(scale),(q,))
        if not torch.isfinite(scale).all() or not (scale>0).all():raise ValueError('positive finite scales required')
        modality=self.tensor(modality,torch.long)
        mask=torch.isfinite(observed) if observed_mask is None else self.tensor(observed_mask,torch.bool)
        mask=torch.broadcast_to(mask,observed.shape)
        if not torch.isfinite(observed[mask]).all() or not torch.isfinite(basal).all():raise ValueError('observed values and basal must be finite')
        descriptors=self.descriptors(query_descriptors,modality,normalized_descriptors)
        if modality.shape!=(q,) or descriptors.shape!=(q,702):raise ValueError('query panel mismatch')
        def identifier(value):return torch.broadcast_to(self.tensor(value,torch.long),(b,))
        taxa=identifier(taxon)
        if not ((taxa==9606)|(taxa==4932)).all():raise ValueError('this checkpoint supports taxa 9606 and 4932')
        taxa=torch.where(taxa==9606,0,1);assays=identifier(assay);mechanisms=identifier(mechanism)
        control=basal if control is None else torch.broadcast_to(self.tensor(control),observed.shape)
        e=torch.arange(q,device=self.device) if encoder_indices is None else self.tensor(encoder_indices,torch.long)
        latent=self.model.encode(observed[:,e]/scale[e],basal[:,e]/scale[e],descriptors[e],modality[e],mask[:,e],
                                 assays,taxa,mechanisms,control=control[:,e])
        # Missing anchor outputs are explicitly filled by the supplied basal state.
        anchor=torch.where(mask,observed,basal)
        return MolecularState(latent,latent,anchor,basal,descriptors,modality,scale,mask,assays,taxa,mechanisms,control)

    def actions(self,state,action_descriptors,action_mask,normalized):
        mask=self.tensor(action_mask,torch.bool);actions=self.tensor(action_descriptors)
        if actions.ndim!=3 or actions.shape[:2]!=mask.shape or actions.shape[0]!=len(state.latent):raise ValueError('action batch mismatch')
        actions=torch.where(mask[...,None],actions,0.)
        return self.descriptors(actions,normalized=normalized),mask

    @torch.inference_mode()
    def intervene(self,state,action_descriptors,action_mask,*,normalized_descriptors=False):
        actions,mask=self.actions(state,action_descriptors,action_mask,normalized_descriptors)
        latent=self.model.transition(state.latent,actions,mask,state.assay,state.taxon,state.mechanism)
        return replace(state,latent=latent)

    @torch.inference_mode()
    def sample(self,state,action_descriptors,action_mask,*,seed=731,steps=16,normalized_descriptors=False):
        actions,mask=self.actions(state,action_descriptors,action_mask,normalized_descriptors)
        generator=torch.Generator(device=self.device).manual_seed(seed)
        latent=self.model.sample_state(state.latent,actions,mask,state.assay,state.taxon,state.mechanism,steps=steps,generator=generator)
        return replace(state,latent=latent)

    @torch.inference_mode()
    def decode(self,state,chunk_size=512):
        """Return anchored native-unit molecular values, reconstruction variance and support."""
        if chunk_size<1:raise ValueError('chunk size must be positive')
        values=[];variances=[]
        for start in range(0,len(state.descriptors),chunk_size):
            sl=slice(start,start+chunk_size);ctx=(state.assay,state.taxon,state.mechanism)
            predicted,logvar=self.model.decode(state.latent,state.descriptors[sl],state.modality[sl],*ctx,control=state.control[:,sl])
            anchor,_=self.model.decode(state.anchor_latent,state.descriptors[sl],state.modality[sl],*ctx,control=state.control[:,sl])
            values.append(state.observed[:,sl]+state.scale[sl]*(predicted-anchor))
            variances.append(torch.exp(logvar)*state.scale[sl].square())
        return {'values':torch.cat(values,1).cpu().numpy(),'reconstruction_variance':torch.cat(variances,1).cpu().numpy(),
                'observed_anchor_mask':state.observed_mask.cpu().numpy()}

    @torch.inference_mode()
    def reconstruct(self,state,chunk_size=512):
        """Decode the latent state itself, including coordinates withheld at encoding."""
        values=[]
        for start in range(0,len(state.descriptors),chunk_size):
            sl=slice(start,start+chunk_size)
            mean,_=self.model.decode(state.latent,state.descriptors[sl],state.modality[sl],state.assay,state.taxon,state.mechanism,control=state.control[:,sl])
            values.append(state.basal[:,sl]+state.scale[sl]*mean)
        return torch.cat(values,1).cpu().numpy()

    @torch.inference_mode()
    def generate(self,state,action_descriptors,action_mask,*,seed=731,steps=16,chunk_size=512,normalized_descriptors=False):
        """Generate processed human RNA/protein observations from learned distributions.

        The input supplies biological context. Output observation noise is drawn
        afresh; no observed-control residual is copied into the generated cell.
        RNA is Bernoulli detection times positive lognormal log-expression.
        Protein follows the continuous Gaussian decoder in its native units.
        """
        if not self.model.config.observation_likelihood:raise ValueError('checkpoint has no sparse RNA observation distribution')
        if not (((state.assay==5)|(state.assay==6))&(state.taxon==0)).all():raise ValueError('cell observation distributions were trained on human assays 5 and 6')
        if chunk_size<1:raise ValueError('chunk size must be positive')
        sampled=self.sample(state,action_descriptors,action_mask,seed=seed,steps=steps,normalized_descriptors=normalized_descriptors)
        generator=torch.Generator(device=self.device).manual_seed((seed+1)%(2**63-1))
        shape=state.observed.shape;uniform=torch.rand(shape,device=self.device,generator=generator);noise=torch.randn(shape,device=self.device,generator=generator)
        output=[];probabilities=[]
        for start in range(0,len(state.descriptors),chunk_size):
            sl=slice(start,start+chunk_size);ctx=(state.assay,state.taxon,state.mechanism)
            logit,mu,lv=self.model.observation_parameters(sampled.latent,state.descriptors[sl],state.modality[sl],*ctx,control=state.control[:,sl])
            probability=logit.float().sigmoid()
            # log1p(CP10K) is bounded by log1p(10000) for each RNA coordinate.
            positive=torch.exp((mu.float()+torch.exp(.5*lv.float())*noise[:,sl]).clamp(-15,float(np.log(np.log1p(10000)))))
            rna=positive*(uniform[:,sl]<probability)
            mean,logvar=self.model.decode(sampled.latent,state.descriptors[sl],state.modality[sl],*ctx,control=state.control[:,sl])
            continuous=state.basal[:,sl]+state.scale[sl]*(mean.float()+torch.exp(.5*logvar.float())*noise[:,sl])
            output.append(torch.where(state.modality[sl][None]==0,rna,continuous))
            probabilities.append(torch.where(state.modality[sl][None]==0,probability,torch.ones_like(probability)))
        return {'values':torch.cat(output,1).cpu().numpy(),'rna_detection_probability':torch.cat(probabilities,1).cpu().numpy(),
                'latent_state':sampled.latent.float().cpu().numpy(),
                'observation_model':'sparse RNA log-expression and continuous protein; processed measurements, not sequencing counts'}
