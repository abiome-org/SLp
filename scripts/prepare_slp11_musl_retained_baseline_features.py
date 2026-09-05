"""Materialize the frozen SLp-1 MuSL feature basis without benchmark labels."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/training'))
from world_model import SLPredict, pair_summary  # noqa: E402

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def public_features(path,pairs,n,triangular=False):
    source=np.load(path,allow_pickle=False); pos=np.full(n,-1,np.int32)
    pos[source['genes'].astype(np.int64)]=np.arange(len(source['genes']))
    a,b=pos[pairs[:,0]],pos[pairs[:,1]]; known=(a>=0)&(b>=0)&(a!=b)
    mean=np.zeros(len(pairs),np.float32); disagreement=np.zeros(len(pairs),np.float32)
    if triangular:
        size=len(source['genes']);lo=np.minimum(a[known],b[known]).astype(np.int64);hi=np.maximum(a[known],b[known]).astype(np.int64)
        ix=lo*(2*size-lo-1)//2+hi-lo-1;x=source['half0'][ix];y=source['half1'][ix]
    else:x=source['half0'][a[known],b[known]].astype(np.float32);y=source['half1'][a[known],b[known]].astype(np.float32)
    mean[known]=(x+y)/2;disagreement[known]=np.abs(x-y)
    return np.column_stack((mean,disagreement,known.astype(np.float32)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--roster',type=Path,default=ROOT/'data/derived/slp11-musl-world-pair-roster-v1')
    p.add_argument('--model',type=Path,default=ROOT/'results/sl_predict/native_spectral_safe_scaled_d768_z256_l8_p12_single_only_d3_t10_r3/world_model.pt')
    p.add_argument('--feature-pack',type=Path,default=ROOT/'results/sl_predict/features_spectral_safe.npz')
    p.add_argument('--meta',type=Path,default=ROOT/'data/feng2024/data/preprocessed_data/meta_table_9845.csv')
    p.add_argument('--codependency',type=Path,default=ROOT/'results/sl_predict/depmap_codependency.npz');p.add_argument('--tcga',type=Path,default=ROOT/'results/sl_predict/tcga_mutual_exclusivity.npz');p.add_argument('--silencing',type=Path,default=ROOT/'results/sl_predict/depmap_expression_silencing.npz')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=2048);p.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    roster=np.load(a.roster/'pairs.npz',allow_pickle=False);genes=np.load(a.roster/'genes.npz',allow_pickle=False)['gene_ids'].astype(str)
    meta=list(csv.DictReader(a.meta.open())); stable={r['ensembl_gene_id']:int(r['unified_id']) for r in meta};old=np.asarray([stable[x] for x in genes],np.int64);pairs=old[roster['gene_indices']]
    pack=np.load(a.feature_pack,allow_pickle=False);state=np.asarray(pack['state'],np.float32);sd=torch.load(a.model,map_location='cpu',weights_only=True)
    device=torch.device(a.device);model=SLPredict(768,256,8,sd['cell.weight'].shape[0],sd['outcome.weight'].shape[0],state.shape[1],sd['context_proj.weight'].shape[1] if 'context_proj.weight' in sd else 0).to(device);model.load_state_dict(sd);model.eval()
    with torch.inference_mode():
        raw=torch.as_tensor(state);encoded=torch.cat([model.encode(raw[i:i+2048].to(device)) for i in range(0,len(raw),2048)])
        embed_width=4*encoded.shape[1]+sd['relation.2.weight'].shape[0]+sd['outcome.weight'].shape[0]
        embed=np.lib.format.open_memmap(a.output/'embed_pairs.npy',mode='w+',dtype=np.float32,shape=(len(pairs),embed_width))
        for start in range(0,len(pairs),a.batch_size):
            sl=slice(start,min(start+a.batch_size,len(pairs)));q=torch.as_tensor(pairs[sl],device=device);left,right=encoded[q[:,0]],encoded[q[:,1]];joint,logsd=model.transition(left,right)
            embed[sl]=torch.cat(((left-right).abs(),left*right,joint,logsd,model.relation_score(left,right,joint),model.outcome(joint)),1).cpu().numpy()
    summary=np.lib.format.open_memmap(a.output/'pair_summary.npy',mode='w+',dtype=np.float32,shape=(len(pairs),40))
    for start in range(0,len(pairs),a.batch_size):
        sl=slice(start,min(start+a.batch_size,len(pairs)));summary[sl]=pair_summary(state,pairs[sl])
    public=np.column_stack((public_features(a.codependency,pairs,len(state)),public_features(a.tcga,pairs,len(state),True),public_features(a.silencing,pairs,len(state))))
    np.save(a.output/'public_relations.npy',public);np.save(a.output/'old_gene_indices.npy',old.astype(np.int32))
    files={x.name:{'bytes':x.stat().st_size,'sha256':sha(x)} for x in sorted(a.output.glob('*.npy'))}
    manifest={'schema':'slp.musl-retained-baseline-features/v1','pairs':len(pairs),'featureBlocks':{'embed_pairs':int(embed_width),'pair_summary':40,'public_relations':9},'featureDim':int(embed_width+49),'excludedLegacyBlock':'six observed_relations coordinates withdrawn because lookup lacked exact-key validation and sparse table membership is not an admissible feature','labelsAccessed':False,'threads':4,'device':a.device,'inputs':{str(x):sha(x) for x in (a.model,a.feature_pack,a.meta,a.codependency,a.tcga,a.silencing,a.roster/'manifest.json')},'files':files}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
