"""Export real functional-world payloads without application dependencies."""
import argparse,hashlib,json,shutil
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(a):
    report=json.loads((a.model/'training.json').read_text())
    if sha(a.model/'model.safetensors')!=report['weights_sha256']:raise ValueError('changed trained weights')
    a.output.mkdir(parents=True,exist_ok=False)
    for name in ('model.safetensors','normalizer.npz','training.json'):shutil.copyfile(a.model/name,a.output/name)
    for name in ('model.py','capacity_model.py','fitness_inference.py','requirements.lock','requirements-linux.lock','CONTRACT.md'):
        shutil.copyfile(Path(__file__).parent/name,a.output/name)
    (a.output/'manifest.json').write_text(json.dumps({'schema':'slp.functional-world-export/v1','architecture':report.get('architecture','conditional'),
        'parameters':report['parameters'],'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},
        'scope':'continuous species-native functional state and fitness inference; no application scoring or benchmark dependencies'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
