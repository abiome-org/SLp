"""Screen fixed physical features using matched ridge point forecasts only."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT/"modules/slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))
from train import gene_metrics
from transition_baselines import fit_ridge

DATA = ROOT/"data/derived/slp11-human-gwps/complete-panel-v1/development.npz"
FEATURES = ROOT/"data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz"
OUTPUT = ROOT/"results/slp11-transition/physical-features-ridge-screen-v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    started = time.monotonic()
    if sha(DATA) != "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b" or sha(FEATURES) != "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7":
        raise ValueError("pinned screen inputs changed")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT/"source").mkdir()
    for path in (Path(__file__), *MODULE.glob("*.py")):
        shutil.copyfile(path, OUTPUT/"source"/path.name)
    protocol = {
        "hypothesis":"direct physical-neighbor static features improve held-gene point forecasts beyond sequence/GO and a degree control",
        "data_sha256":sha(DATA), "features_sha256":sha(FEATURES),
        "feature_arms":{"base":"columns0:577", "degree":"base plus final2 degree/presence columns", "physical":"all1156 columns"},
        "ridge_alpha":10000, "fit":"development training rows per context only",
        "rule":"physical gene-macro MSE improves at least1% against both base and degree in each context; adjusted Pearson does not regress against either",
        "likelihood_evaluated":False, "scale_calibration":"not evaluated; screen uses point forecasts only",
        "test_accessed":False,
        "source_hashes":{p.name:sha(p) for p in (OUTPUT/"source").glob("*.py")},
    }
    (OUTPUT/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    with np.load(DATA, allow_pickle=False) as archive:
        data = {name:archive[name] for name in archive.files}
    if len(data["split_test"]):
        raise ValueError("development only")
    train, validation = data["split_train"], data["split_validation"]
    if set(data["action_ids"][train]) & set(data["action_ids"][validation]):
        raise ValueError("training/validation intervention overlap")
    with np.load(FEATURES, allow_pickle=False) as archive:
        lookup = {(int(t),str(g)):v for t,g,v in zip(archive["entity_taxon"],archive["entity_id"],archive["feature_values"])}
    features = np.stack([lookup[(9606,str(g))] for g in data["action_ids"]])
    context_results = {}
    predictions = {}
    for context, name in enumerate(data["context_ids"]):
        fitting = train[data["context_index"][train] == context]
        scoring = validation[data["context_index"][validation] == context]
        truth, mask = data["targets"][scoring], data["observed"][scoring]
        result = {}
        for arm, columns in (("base",np.arange(577)), ("degree",np.r_[np.arange(577),1154,1155]), ("physical",np.arange(1156))):
            x = features[:, columns]
            model = fit_ridge(x[fitting], data["targets"][fitting], data["observed"][fitting], 10000)
            prediction = model.predict(x[scoring])
            metrics = gene_metrics(prediction, truth, mask,
                [(9606,str(g)) for g in data["action_ids"][scoring]], model.intercept_,
                np.ones_like(truth), value_space=str(data["target_value_space"].item()))
            result[arm] = {key:metrics[key] for key in ("gene_macro_mse", "gene_macro_profile_centroid_adjusted_pearson_mean")}
            predictions[f"context{context}_{arm}"] = prediction.astype(np.float32)
        result["rule_passed"] = all(
            result["physical"]["gene_macro_mse"] <= .99*result[baseline]["gene_macro_mse"]
            and result["physical"]["gene_macro_profile_centroid_adjusted_pearson_mean"] >= result[baseline]["gene_macro_profile_centroid_adjusted_pearson_mean"]
            for baseline in ("base", "degree"))
        context_results[str(name)] = result
        print(json.dumps({"context":str(name), **result}), flush=True)
    np.savez_compressed(OUTPUT/"predictions.npz", **predictions)
    report = {"results":context_results, "rule_passed":all(item["rule_passed"] for item in context_results.values()),
              "elapsed_seconds":time.monotonic()-started, "test_accessed":False,
              "predictions_sha256":sha(OUTPUT/"predictions.npz")}
    (OUTPUT/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
