"""OMF artifact replay: verify and copy an already-trained standalone model."""
import argparse,hashlib,json,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.bundle.resolve()
    files=json.loads((root/'manifest.json').read_text())['files']
    for filename,digest in files.items():
        path=(root/filename).resolve()
        if not path.is_relative_to(root):raise ValueError('bundle path escapes source')
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=digest:raise ValueError(f'changed model payload: {filename}')
    a.output.mkdir(parents=True,exist_ok=False)
    for filename in [*files,'manifest.json']:
        destination=a.output/filename;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/filename,destination)
    print(json.dumps({'operation':'materialize_already_trained_world_model','optimization_steps':0,'output':str(a.output)}))


if __name__=='__main__':main()
