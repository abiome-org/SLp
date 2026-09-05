"""Predict a single CRISPRi molecular response from a local candidate package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from inference import Predictor


def predict_gene(artifact, gene, context, queries=None, num_cells=100):
    root = Path(artifact)
    ensemble = (root/"ensemble-manifest.json").exists()
    if ensemble:
        from ensemble_inference import EnsemblePredictor
        model = EnsemblePredictor(root)
        reference = model.references[0]
    else:
        model = Predictor(root)
        with np.load(root/"reference.npz", allow_pickle=False) as z:
            reference = {k:z[k] for k in z.files}
    with np.load(root/"static-features.npz", allow_pickle=False) as z:
        lookup = {(int(t), str(g)):v for t,g,v in zip(z["entity_taxon"], z["entity_id"], z["feature_values"])}
    contexts = reference["context_ids"].tolist()
    if context not in contexts:
        raise ValueError(f"supported contexts: {contexts}")
    key = (9606, gene)
    if key not in lookup:
        raise ValueError("gene is absent from packaged static features; encode its features before using the numerical API")
    query_lookup = {str(g):i for i,g in enumerate(reference["query_ids"])}
    requested = list(queries) if queries else reference["query_ids"][:32].tolist()
    missing = set(requested)-query_lookup.keys()
    if missing:
        raise ValueError(f"queries lack fitted assay descriptors: {sorted(missing)}")
    indices = np.asarray([query_lookup[g] for g in requested], dtype=np.int64)
    c = np.asarray([contexts.index(context)])
    action = lookup[key][None]
    if ensemble:
        prediction = model.predict(action, np.asarray([num_cells]), c, indices)
    else:
        means, base_scale = model.fitted_reference(action, c, indices)
        measurement_scale = model.measurement_scales([num_cells], c, indices)
        prediction = model.predict(action, reference["query_features"][indices], means, base_scale,
            measurement_scale=measurement_scale,
            context_features=reference["context_features"][None],
            context_values=reference["context_values"][c],
            context_mask=np.ones_like(reference["context_values"][c], dtype=bool))
    return {"status":"experimental-development-candidate", "intervention_gene":gene,
            "intervention":"CRISPRi", "context":context, "planned_measurement_cells":num_cells,
            "target_value_space":"author-core-control-standardized-pseudobulk-mean",
            "queries":requested, "mean":prediction["mean"][0].tolist(),
            "measurement_standard_deviation":prediction["marginal_scale"][0].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact",default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--gene",required=True)
    parser.add_argument("--context",required=True)
    parser.add_argument("--queries",nargs="+")
    parser.add_argument("--num-cells",type=int,default=100)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = predict_gene(args.artifact,args.gene,args.context,args.queries,args.num_cells)
    payload = json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.output:
        Path(args.output).write_text(payload,encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
