"""Export actual weights and an application-neutral inference implementation."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from prepare import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--example',type=Path);p.add_argument('--rights-root',type=Path,default=Path('rights'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for filename in ('model.safetensors','config.json','training.json'):
        if (a.checkpoint/filename).exists():shutil.copy2(a.checkpoint/filename,a.output/filename)
    module=Path(__file__).parent
    for filename in ('model.py','inference.py','replay.py','CONTRACT.md','requirements-native.lock','requirements-linux.lock'):
        shutil.copy2(module/filename,a.output/filename)
    with np.load(a.index/'features.npz',allow_pickle=False) as z:np.savez_compressed(a.output/'normalizer.npz',mean=z['feature_mean'],scale=z['feature_scale'])
    index=json.loads((a.index/'manifest.json').read_text());shutil.copy2(a.index/'manifest.json',a.output/'training-index.json')
    protocol=a.checkpoint.parent/'protocol.json'
    if protocol.exists():shutil.copy2(protocol,a.output/'training-protocol.json')
    if a.example:shutil.copy2(a.example,a.output/'example.npz')
    rights=a.output/'rights';rights.mkdir()
    for filename in index['rights']:
        source=a.rights_root/filename
        if sha(source)!=index['sourceReceipts']['rights/'+filename]:raise ValueError('source rights changed since training-index preparation')
        shutil.copy2(source,rights/filename)
    write(a.output/'metadata.json',{'schema':'slp.cell-world-bundle/v1','checkpoint_sha256':sha(a.checkpoint/'model.safetensors'),
        'training_index_sha256':sha(a.index/'manifest.json'),'taxa':{'9606':0,'4932':1},'mechanisms':{'CRISPRi':0,'CRISPRa':1,'knockout':2},
        'assays':{'replogle_log_population_mean':0,'norman_control_z':1,'gwps_control_z':2,'nadig_control_z':3,
                  'mcf10a_mean_log_expression':4,'replogle_cell_log_expression':5,'frangieh_paired_cell':6,'yeast_mean_log_expression':7},
        'features':{'gene':642,'measurement':60,'rna_modality':0,'protein_modality':1},
        'scope':'Generative molecular research model. No fitted molecular response prior, viability head or SL classifier.',
        'generation':'Latent population coupling and computational rectified flow; no observed longitudinal cell trajectories.',
        'status':'standalone research export; not an OMF service deployment'})
    write(a.output/'manifest.json',{'files':{p.relative_to(a.output).as_posix():sha(p) for p in a.output.rglob('*') if p.is_file()}})
    print(json.dumps({'output':str(a.output),'manifest_sha256':sha(a.output/'manifest.json')}))


if __name__=='__main__':main()
