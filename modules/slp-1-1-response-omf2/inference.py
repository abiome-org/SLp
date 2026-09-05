"""Portable directory inference for the OMF2 response model."""
import argparse, hashlib
from pathlib import Path
import json, numpy as np
from response_model import load

class ResponseBundle:
    def __init__(self,path):
        self.path=Path(path); self.manifest=json.loads((self.path/'manifest.json').read_text())
        for context, record in self.manifest['contexts'].items():
            model_path=self.path/record['model']
            actual=hashlib.sha256(model_path.read_bytes()).hexdigest()
            if actual != record['sha256']:
                raise ValueError(f"model digest mismatch for {context}")
    def query_ids(self,context):
        record=self.manifest['contexts'][context]
        with np.load(self.path/record['model'],allow_pickle=False) as archive:
            return np.asarray(archive['query_ids']).astype(str)
    def predict(self,context,features,basal_anchor):
        model=load(self.path/self.manifest['contexts'][context]['model'])
        residual=model.predict(features); anchor=np.asarray(basal_anchor,np.float64)
        if anchor.shape!=residual.shape: raise ValueError('anchor and prediction must align')
        return anchor+residual

def load_bundle(path): return ResponseBundle(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--context',choices=['k562','rpe1'],required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); bundle=load_bundle(args.model)
    with np.load(args.input,allow_pickle=False) as request:
        if set(request.files) != {'features','basal_anchor'}:
            raise ValueError('request must contain exactly features and basal_anchor')
        features=np.asarray(request['features'],np.float32)
        anchor=np.asarray(request['basal_anchor'],np.float64)
    prediction=bundle.predict(args.context,features,anchor)
    query_ids=bundle.query_ids(args.context)
    if prediction.ndim != 2 or prediction.shape[1] != len(query_ids):
        raise ValueError('prediction does not align with model query axis')
    np.savez_compressed(args.output,predictions=prediction,query_ids=query_ids)

if __name__=='__main__': main()
