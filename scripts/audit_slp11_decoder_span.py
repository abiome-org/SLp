"""Check whether a frozen decoder can represent existing ridge forecasts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"results/slp11-transition/human-gwps-complete-panel-fusion-response32-seed731-v1/model"
DATA = ROOT/"data/derived/slp11-human-gwps/complete-panel-v1/development.npz"
SCREEN = ROOT/"results/slp11-transition/physical-features-ridge-screen-v1"
OUTPUT = ROOT/"results/slp11-transition/gwps-decoder-span-audit-v1"
sys.path.insert(0, str(RUN/"source"))
from inference import Predictor
from train import gene_metrics


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    inputs = {
        DATA:"006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b",
        RUN/"model.safetensors":"66f0eb42faaf310f330c3da9734531d99ae2d1ea7f1daeec97acc69701d2b97c",
        SCREEN/"predictions.npz":"c91d96b724f9b99169536ba17a3cce6f0c8578d603257b830a32a335f7e1c525",
    }
    if any(sha(path) != digest for path,digest in inputs.items()):
        raise ValueError("decoder audit input drift")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    protocol = {
        "hypothesis":"the frozen learned decoder span loses useful structure in the full physical-ridge forecasts",
        "method":"project saved baseline forecasts onto frozen molecular decoder span with least squares; no outcome used to fit projection",
        "decision":"prioritize a decoder geometry experiment if projection raises physical-ridge MSE by >=1% in any context",
        "scope":"adaptive development diagnostic, not an oracle upper bound or new world candidate",
        "inputs":{str(p.relative_to(ROOT)):h for p,h in inputs.items()},
        "script_sha256":sha(__file__), "test_accessed":False,
    }
    (OUTPUT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    with np.load(DATA,allow_pickle=False) as archive:
        data = {key:archive[key] for key in archive.files}
    if len(data["split_test"]):
        raise ValueError("development only")
    with np.load(RUN/"reference.npz",allow_pickle=False) as archive:
        reference = {key:archive[key] for key in archive.files}
    predictor = Predictor(RUN)
    model = predictor.model.eval()
    query = (reference["query_features"]-reference["query_feature_mean"])/reference["query_feature_std"]
    with torch.no_grad():
        encoded = model.query_encoder(torch.tensor(query,dtype=torch.float32))
        decoder = (encoded @ model.mean_state.weight).numpy()/np.sqrt(model.config.state_dim)
    results = {}
    with np.load(SCREEN/"predictions.npz",allow_pickle=False) as forecasts:
        for context, name in enumerate(data["context_ids"]):
            validation = data["split_validation"]
            rows = validation[data["context_index"][validation] == context]
            basis = decoder.astype(np.float64)*reference["reference_scale"][context,:,None]
            u,singular,_ = np.linalg.svd(basis,full_matrices=False)
            keep = singular > singular[0]*1e-10
            u = u[:,keep]
            context_result = {"numerical_decoder_rank":int(keep.sum())}
            for arm in ("base","physical"):
                prediction = forecasts[f"context{context}_{arm}"].astype(np.float64)
                centered = prediction-reference["reference"][context]
                projected = reference["reference"][context]+(centered@u)@u.T
                pair = {}
                for label, value in (("original",prediction),("projected",projected)):
                    metrics = gene_metrics(value,data["targets"][rows],data["observed"][rows],
                        [(9606,str(g)) for g in data["action_ids"][rows]],reference["reference"][context],
                        np.ones_like(value),value_space=str(data["target_value_space"].item()))
                    pair[label] = {key:metrics[key] for key in ("gene_macro_mse","gene_macro_profile_centroid_adjusted_pearson_mean")}
                pair["relative_mse_increase"] = pair["projected"]["gene_macro_mse"]/pair["original"]["gene_macro_mse"]-1
                pair["forecast_energy_retained"] = float(np.square(centered@u).sum()/np.square(centered).sum())
                context_result[arm] = pair
            results[str(name)] = context_result
    report = {"results":results,"prioritize_decoder_geometry":any(item["physical"]["relative_mse_increase"] >= .01 for item in results.values()), "test_accessed":False}
    (OUTPUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
