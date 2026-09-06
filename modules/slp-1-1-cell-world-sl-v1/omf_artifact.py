"""OMF 2 retention and numerical replay of an already-trained world-to-SL model."""
import argparse,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path


def main(a):
    if a.phase=='materialize':
        root=a.bundle.resolve();files=json.loads((root/'manifest.json').read_text())['files']
        for name,digest in files.items():
            path=(root/name).resolve()
            if not path.is_relative_to(root):raise ValueError('payload escapes artifact')
            with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=digest:raise ValueError('changed artifact '+name)
        a.output.mkdir(parents=True,exist_ok=False)
        for name in [*files,'manifest.json']:
            destination=a.output/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,destination)
    else:
        runtime=a.runtime.resolve()
        if runtime.is_file():
            destination=(a.output.parent/'readout-runtime').resolve();destination.mkdir(parents=True,exist_ok=False)
            with zipfile.ZipFile(runtime) as archive:
                for name in archive.namelist():
                    if not (destination/name).resolve().is_relative_to(destination):raise ValueError('runtime entry escapes output')
                archive.extractall(destination)
            runtime=destination
        subprocess.run([sys.executable,str(Path(__file__).parent/'replay.py'),'--bundle',str(a.bundle),'--request',str(a.request),
            '--reference',str(a.reference),'--output',str(a.output),'--device','cpu'],check=True,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(runtime)})
        report=json.loads((a.output/'report.json').read_text());benchmark=json.loads((a.bundle/'benchmark-scores.json').read_text())
        errors=report['reference_max_errors'];passed=report['passed'] and set(errors)=={'sl_score','fold_decoder_scores','label_free_response_similarity','features'}
        metrics={'passed':passed,'compatibilityPassed':passed,'sl_auroc':benchmark['macro']['world']['auroc'],
            'sl_ap':benchmark['macro']['world']['ap'],'score_max_error':errors['sl_score'],'feature_max_error':errors['features']}
        (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        if not passed:raise AssertionError('end-to-end SL replay failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['materialize','replay'],required=True)
    for name in ('bundle','output','request','reference','runtime'):p.add_argument('--'+name,type=Path,required=name in ('bundle','output'))
    main(p.parse_args())
