"""Test whether static graph degree and coverage explain molecular forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))
from context_transfer_scoring import collapse_gene_profiles, score_gene_profiles
from exposure_uncertainty import fit_exposure_uncertainty
from train import gene_metrics
from transition_calibration import fit_grouped_oof_ridge

DATA = ROOT / "data/derived/slp11-human-gwps/complete-panel-v1/development.npz"
FEATURES = ROOT / "data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz"
COMPARATOR = ROOT / "results/slp11-transition/human-gwps-fixed-context-minimal-control-physical-state128-response32-seed731-v1/model/report.json"
PINS = {
    "data": "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b",
    "features": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "comparator": "49333ade99f04d96e9d4c4ccc2fc01c002170b38f02d10f88fdc8559d274203d",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def degree_covariates(values: np.ndarray) -> dict[str, np.ndarray]:
    """Select known physical-pack covariates, never gene IDs or outcomes."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 1156 or not np.isfinite(values).all():
        raise ValueError("requires the pinned finite physical1156 feature pack")
    degree, coverage, peptide = values[:, -2], values[:, -1], values[:, 320]
    if np.any(degree < 0) or not np.array_equal(coverage, (degree > 0).astype(values.dtype)):
        raise ValueError("degree and coverage disagree")
    if not np.isin(peptide, [0, 1]).all():
        raise ValueError("protein presence must be Boolean-valued")
    return {
        "log_physical_degree": degree[:, None],
        "degree_graph_coverage_protein_presence": np.stack((degree, coverage, peptide), axis=1),
    }


def run(output: Path) -> dict:
    started = time.monotonic()
    for name, path in (("data", DATA), ("features", FEATURES), ("comparator", COMPARATOR)):
        if sha(path) != PINS[name]:
            raise ValueError(f"{name} pin changed")
    with np.load(DATA, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    with np.load(FEATURES, allow_pickle=False) as archive:
        keys = list(zip(archive["entity_taxon"].tolist(), archive["entity_id"].tolist()))
        values = archive["feature_values"]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate static identities")
    lookup = dict(zip(keys, range(len(keys)), strict=True))
    actions = data["action_ids"]
    covariates = degree_covariates(values[[lookup[(9606, str(gene))] for gene in actions]])
    train, validation = data["split_train"], data["split_validation"]
    if len(data["split_test"]) or set(actions[train]) & set(actions[validation]):
        raise ValueError("requires development-only intervention-disjoint source data")
    comparator = json.loads(COMPARATOR.read_text())["results"]
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for path in (Path(__file__), *MODULE.glob("*.py")):
        shutil.copyfile(path, source / path.name)
    protocol = {
        "hypothesis": "world predictions contain intervention-specific signal beyond physical degree and static coverage",
        "fixedDecision": "v2 world MSE lower and independently source-training-centroid-adjusted profile Pearson higher than both degree controls in each source context",
        "alpha": 1.0,
        "alphaRationale": "fixed weak shrinkage for one or three standardized static covariates; no hyperparameter selection",
        "arms": {name: value.shape[1] for name, value in covariates.items()},
        "pins": PINS,
        "fitting": "source split_train only, separate model per source; three gene-grouped fitting folds seed731 for uncertainty",
        "evaluation": "source development validation only, no external or test outcomes",
        "uncertainty": "model-specific grouped OOF residuals plus unchanged core-control sampling components",
        "modalities": "STRING12 direct experimental physical degree and coverage; peptide availability",
        "secondary": "prediction and truth independently query-centered across gene-averaged validation profiles",
        "sourceHashes": {path.name: sha(path) for path in source.glob("*.py")},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = {str(name): {} for name in data["context_ids"]}
    predictions = {}
    context = data["context_index"]
    for label, features in covariates.items():
        models, oof = [], np.empty_like(data["targets"][train], dtype=np.float64)
        for index in range(len(data["context_ids"])):
            positions = np.flatnonzero(context[train] == index)
            rows = train[positions]
            model, oof[positions] = fit_grouped_oof_ridge(
                features[rows], data["targets"][rows], data["observed"][rows],
                [(9606, str(gene)) for gene in actions[rows]], alpha=1.0, folds=3,
                seed=731, return_oof=True,
            )
            models.append(model)
        exposure = fit_exposure_uncertainty(
            data["targets"][train] - oof, data["observed"][train],
            data["num_cells_filtered"][train], context[train],
            control_targets=data["control_targets"], control_observed=data["control_observed"],
            control_num_cells=data["control_num_cells_filtered"],
            control_context_index=data["control_context_index"], scale_floor=.05,
        )
        for index, name in enumerate(data["context_ids"]):
            rows = validation[context[validation] == index]
            model = models[index]
            prediction = model.predict(features[rows])
            metrics = gene_metrics(
                prediction, data["targets"][rows], data["observed"][rows],
                [(9606, str(gene)) for gene in actions[rows]], model.intercept_,
                exposure.scales(data["num_cells_filtered"][rows], context[rows]),
                value_space=str(data["target_value_space"].item()),
            )
            entry = {key: metrics[key] for key in (
                "gene_macro_nll", "gene_macro_mse", "gene_macro_profile_centroid_adjusted_pearson_mean")}
            profiles = collapse_gene_profiles(prediction, data["targets"][rows],
                data["observed"][rows], actions[rows], data["record_ids"][rows])
            entry["secondaryIndependentlyCenteredPearson"] = score_gene_profiles(
                profiles, model.intercept_)["primaryIndependentlyCenteredGeneMacroProfilePearson"]
            results[str(name)][label] = entry
            predictions[f"{index}_{label}"] = prediction.astype(np.float32)
            print(json.dumps({"context": str(name), "arm": label, "metrics": entry}), flush=True)
    for name, result in results.items():
        world = comparator[name]["world"]
        result["world"] = {key: world[key] for key in (
            "gene_macro_nll", "gene_macro_mse", "gene_macro_profile_centroid_adjusted_pearson_mean")}
        result["worldExceedsBothDegreeControls"] = all(
            world["gene_macro_mse"] < result[label]["gene_macro_mse"] and
            world["gene_macro_profile_centroid_adjusted_pearson_mean"] >
            result[label]["gene_macro_profile_centroid_adjusted_pearson_mean"]
            for label in covariates)
    np.savez_compressed(output / "predictions.npz", **predictions)
    report = {"results": results, "passed": all(item["worldExceedsBothDegreeControls"] for item in results.values()),
        "elapsedSeconds": time.monotonic() - started, "protocolSha256": sha(output / "protocol.json"),
        "predictionsSha256": sha(output / "predictions.npz")}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
