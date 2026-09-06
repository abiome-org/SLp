"""Build and benchmark the complete frozen-world SL application in one run."""
import argparse,subprocess,sys
from pathlib import Path


def main(a):
    # Fail before simulation if the numerical readout runtime is incomplete.
    import lightgbm,torch,safetensors,sklearn
    source=Path(__file__).resolve().parent;root=a.root.resolve();output=a.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    index=root/'data/derived/slp11-cell-world-training-v5'
    roster=root/'data/derived/slp11-musl-world-pair-roster-v1'
    labels=root/'data/models/MuSL/processed_data/data/CV3_bins_32/fold_data'
    bundle=a.bundle.resolve()
    def execute(script,*args):
        subprocess.run([sys.executable,str(source/script),*[str(x) for x in args]],check=True,cwd=root)
    for name,extra in (('world',()),('random',('--random',))):
        execute('bridge.py','--root',root,'--index',index,'--bundle',bundle,'--roster',roster,'--output',output/name,'--device',a.device,*extra)
    execute('verify.py','--bundle',bundle,'--features',output/'world','--roster',roster,'--output',output/'simulation-verification.json','--device',a.device)
    execute('decoder.py','--phase','fit','--roster',roster,'--labels',labels,'--world',output/'world','--random',output/'random','--output',output/'decoder')
    execute('audit_decoder.py','--fit',output/'decoder','--roster',roster,'--world',output/'world','--random',output/'random','--output',output/'decoder-replay.json')
    execute('decoder.py','--phase','score','--labels',labels,'--output',output/'decoder')
    execute('zero_shot.py','--root',root,'--world',output/'world','--random',output/'random','--roster',roster,'--bundle',bundle,'--index',index,'--output',output/'label-free')
    execute('predict.py','--export','--fit',output/'decoder','--bundle',bundle,'--features',output/'world','--roster',roster,
            '--gene-meta',root/'data/models/MuSL/processed_data/meta_table_7684.csv','--output',output/'predictor')
    print(f'Completed predictor: {output / "predictor"}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');main(p.parse_args())
