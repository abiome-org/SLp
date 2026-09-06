"""Expose verified standalone replay checks in OMF's evaluator result protocol."""
import argparse,json,math,os,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    subprocess.run([sys.executable,str(a.bundle/'replay.py'),'--bundle',str(a.bundle),'--request',str(a.request),
                    '--reference',str(a.reference),'--output',str(a.output),'--device','cpu'],check=True,
                   env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    report=json.loads((a.output/'report.json').read_text());metrics=json.loads((a.output/'metrics.json').read_text())
    expected={'prediction','reconstruction','generated_731','generated_732'}
    errors=report['reference_max_errors']
    compatible=set(errors)==expected and all(math.isfinite(v) and 0<=v<=1e-5 for v in errors.values())
    passed=(report['status']=='passed' and report['empty_action_max_error']==0 and report['action_permutation_max_error']<=1e-5
            and report['sample_difference']>1e-5 and compatible)
    # These are arithmetic replay results, not biological validation, signatures
    # or independent approval of a model/service release.
    metrics.update(passed=passed,compatibilityPassed=compatible)
    (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    if not passed:raise AssertionError('standalone numerical compatibility checks failed')


if __name__=='__main__':main()
