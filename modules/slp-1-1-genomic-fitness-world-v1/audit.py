"""Bounded corpus integrity, species and intervention-partition checks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from prepare import DTYPE


def main(a):
    manifest=json.loads((a.data/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        with (a.data/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=digest:raise ValueError('changed corpus payload '+name)
    with np.load(a.data/'genes.npz') as z:
        human=z['human_ids'].astype(str);yeast=z['yeast_ids'].astype(str)
        if len(set(human))!=len(human) or len(set(yeast))!=len(yeast):raise ValueError('duplicate identity')
        if not all(g.startswith('ENSG') for g in human) or not all(g.startswith('SGD:') for g in yeast):raise ValueError('species identity mismatch')
        for key in ('human_raw','yeast_raw','human_signatures','yeast_signatures'):
            if not np.isfinite(z[key]).all():raise ValueError('nonfinite gene observations')
    partitions={};counts={}
    for role in ('train','validation'):
        z=np.fromfile(a.data/f'yeast-{role}.bin',dtype=DTYPE)
        if (z['a']>=len(yeast)).any() or (z['b']>=len(yeast)).any():raise ValueError('invalid yeast query')
        if (z['a']==z['b']).any():raise ValueError('self deletion pair')
        for key in ('single_a','single_b','double'):
            if not np.isfinite(z[key]).all() or (z[key]<0).any():raise ValueError('invalid quantitative fitness')
        partitions['yeast-'+role]=set(z['a'])|set(z['b']);counts['yeast-'+role]=len(z)
        with np.load(a.data/f'human-{role}.npz') as h:
            partitions['human-'+role]=set(h['gene_indices']);counts['human-'+role]=int(h['known'].sum())
            if not np.isfinite(h['targets'][h['known']]).all():raise ValueError('nonfinite known gene effect')
    for species in ('human','yeast'):
        if partitions[species+'-train']&partitions[species+'-validation']:raise ValueError('quantitative outcome overlap')
    held=set(json.loads((a.data/'yeast-partitions.json').read_text())['held_gene_ids'])
    if set(yeast[list(partitions['yeast-train'])])&held:raise ValueError('held yeast in fitting')
    if not set(yeast[list(partitions['yeast-validation'])])<=held:raise ValueError('unheld yeast in validation')
    report={'passed':True,'counts':counts,'species_native':True,'quantitative_fitting_validation_gene_overlap':0,
        'corpus_manifest_sha256':hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest()}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
