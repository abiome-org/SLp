"""Apply the fixed no-SL-training benchmark to a learned fitness observation head."""
import argparse
from pathlib import Path
import numpy as np
from bridge import sha
from zero_shot import run,normalized


def load(path):
    with np.load(path) as z:return z['gene_ids'].astype(str),normalized(z['profile'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','world','random','roster','bundle','index','output','phenotype'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.rule='positive cosine of predicted single-intervention fitness profiles after fitting-gene and context centering; 1063 fixed non-target contexts; no SL training'
    run(a,representation_loader=load,model_receipts={'phenotype_weights_sha256':sha(a.phenotype/'model.safetensors'),
        'phenotype_training_sha256':sha(a.phenotype/'training.json'),'trained_forecasts_sha256':sha(a.world),'untrained_forecasts_sha256':sha(a.random),
        'adapter_source_sha256':sha(__file__)})
