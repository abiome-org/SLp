"""Create a small, real molecular-control request from the admitted fitting index."""
import argparse
from pathlib import Path
import numpy as np
from data import arrays,Features


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('example output must be new')
    f=Features(a.index/'features.npz');z=arrays(a.index/'frangieh-populations.npz');q,mask=f.genes(z['rna_query_ids'])
    rng=np.random.default_rng(731);ix=np.sort(rng.choice(np.flatnonzero(mask),192,False))
    query=np.pad(q[ix],((0,0),(0,60)));protein=np.zeros((20,702),np.float32)
    for i,barcode in enumerate(z['protein_channel_ids']):
        for j,base in enumerate(str(barcode)):protein[i,642+4*j+'ACGT'.index(base)]=1.
    query=np.concatenate((query,protein));modality=np.concatenate((np.zeros(len(ix),np.int64),np.ones(20,np.int64)))
    context=list(z['control_context_ids']).index('Control')
    basal=np.concatenate((z['control_rna_targets'][context,ix],z['control_protein_targets'][context]))[None].astype(np.float32)
    gene=str(sorted(set(z['action_ids'].astype(str)))[0]);action=f.genes(np.array([[gene,'']]))[0]
    np.savez_compressed(a.output,observed=basal,basal=basal,query_descriptors=query,modality=modality,
        scale=np.concatenate((np.ones(len(ix)),np.full(20,.5))).astype(np.float32),assay=np.asarray(6),taxon=np.asarray(9606),mechanism=np.asarray(2),
        encoder_indices=np.arange(192),actions=action,action_mask=np.array([[True,False]]),action_ids=np.array([[gene,'']]),
        query_ids=np.concatenate((z['rna_query_ids'][ix],z['protein_channel_ids'])))
    print(a.output)


if __name__=='__main__':main()
