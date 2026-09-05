"""Run the frozen genome-scale human development pilot and old-model comparator.

Only the complete-panel development bundle is accepted. The outer protocol is
written before training. The frozen two-context ensemble is then evaluated on
the same 7,036-query development validation panel without refitting it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b"
FEATURE_SHA256 = "a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b"
OLD_FEATURE_SHA256 = "b3de49e18d3c75676985b8790d1ce85de0d87d526bbd7c0c5b555828a1fb11a0"
OLD_ENSEMBLE_MANIFEST_SHA256 = (
    "a972d994f80c124f948b9b4a313d9e76bdd5c1a3477ebc4082c143ae96c50a70"
)
CONTEXT_IDS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
OUTPUT = ROOT / "results" / "slp11-transition" / (
    "human-gwps-complete-panel-fusion-response32-seed731-v1"
)


class GenomeScalePilotError(ValueError):
    """A frozen genome-scale development contract was violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_partitions(
    action_ids: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    records: int,
) -> None:
    """Require an exhaustive, disjoint development-only held-gene split."""

    if len(train) != 10_719 or len(validation) != 2_339 or len(test):
        raise GenomeScalePilotError("development partition counts drifted")
    combined = np.concatenate((train, validation))
    if not np.array_equal(np.sort(combined), np.arange(records)):
        raise GenomeScalePilotError("development partitions are not exhaustive and disjoint")
    if len(set(action_ids[train].tolist()) & set(action_ids[validation].tolist())):
        raise GenomeScalePilotError("intervention identity crosses train and validation")


def validate_inputs(
    data_path: Path,
    feature_path: Path,
    old_feature_path: Path,
) -> dict[str, object]:
    """Validate exact snapshots, identity coverage, and feature preservation."""

    if sha256(data_path) != DATA_SHA256:
        raise GenomeScalePilotError("complete-panel development SHA-256 drift")
    if sha256(feature_path) != FEATURE_SHA256:
        raise GenomeScalePilotError("extended static feature SHA-256 drift")
    if sha256(old_feature_path) != OLD_FEATURE_SHA256:
        raise GenomeScalePilotError("prior static feature SHA-256 drift")

    with np.load(data_path, allow_pickle=False) as archive:
        required = {
            "action_ids",
            "query_ids",
            "targets",
            "observed",
            "context_index",
            "context_ids",
            "split_train",
            "split_validation",
            "split_test",
            "control_targets",
            "control_observed",
            "control_context_index",
            "control_num_cells_filtered",
            "num_cells_filtered",
            "context_basal_expression",
            "target_value_space",
        }
        if not required.issubset(archive.files):
            raise GenomeScalePilotError("complete-panel development schema is incomplete")
        action_ids = archive["action_ids"]
        query_ids = archive["query_ids"]
        targets = archive["targets"]
        observed = archive["observed"]
        contexts = tuple(archive["context_ids"].astype(str).tolist())
        train = archive["split_train"]
        validation = archive["split_validation"]
        test = archive["split_test"]
        validate_partitions(action_ids, train, validation, test, len(action_ids))
        if targets.shape != (13_058, 7_036) or observed.shape != targets.shape:
            raise GenomeScalePilotError("complete-panel target shape drifted")
        if contexts != CONTEXT_IDS:
            raise GenomeScalePilotError("complete-panel context identity/order drifted")
        if not observed.all() or not np.isfinite(targets).all():
            raise GenomeScalePilotError("complete-panel outcomes are incomplete or nonfinite")
        if np.any(archive["context_index"] < 0) or np.any(archive["context_index"] >= 3):
            raise GenomeScalePilotError("complete-panel context index is invalid")
        if not all(str(gene).startswith("ENSG") and "." not in str(gene) for gene in action_ids):
            raise GenomeScalePilotError("action identities are not stable unversioned ENSG IDs")
        if not all(str(gene).startswith("ENSG") and "." not in str(gene) for gene in query_ids):
            raise GenomeScalePilotError("query identities are not stable unversioned ENSG IDs")
        action_set = set(action_ids.astype(str).tolist())
        query_set = set(query_ids.astype(str).tolist())
        context_train_counts = [
            int(np.count_nonzero(archive["context_index"][train] == index)) for index in range(3)
        ]
        context_validation_counts = [
            int(np.count_nonzero(archive["context_index"][validation] == index))
            for index in range(3)
        ]
        value_space = str(archive["target_value_space"].item())

    with np.load(feature_path, allow_pickle=False) as archive:
        if set(archive.files) != {"entity_taxon", "entity_id", "feature_values"}:
            raise GenomeScalePilotError("extended feature member contract drifted")
        taxon = archive["entity_taxon"]
        entity_ids = archive["entity_id"].astype(str)
        features = archive["feature_values"]
        if features.shape != (10_231, 577) or features.dtype != np.float32:
            raise GenomeScalePilotError("extended feature matrix shape or dtype drifted")
        if taxon.dtype != np.int64 or not np.all(taxon == 9606):
            raise GenomeScalePilotError("extended feature taxonomy contract drifted")
        if len(np.unique(entity_ids)) != len(entity_ids) or not np.array_equal(
            entity_ids, np.sort(entity_ids, kind="stable")
        ):
            raise GenomeScalePilotError("extended feature identities are not unique and sorted")
        if not np.isfinite(features).all():
            raise GenomeScalePilotError("extended feature values are nonfinite")
        entity_set = set(entity_ids.tolist())
        if not action_set.issubset(entity_set) or not query_set.issubset(entity_set):
            raise GenomeScalePilotError("extended features do not cover all actions and queries")
        extended_lookup = {gene: row for gene, row in zip(entity_ids.tolist(), features)}

    with np.load(old_feature_path, allow_pickle=False) as archive:
        old_taxon = archive["entity_taxon"]
        old_ids = archive["entity_id"].astype(str)
        old_features = archive["feature_values"]
        if len(old_ids) != 7_542 or not np.all(old_taxon == 9606):
            raise GenomeScalePilotError("prior feature identity contract drifted")
        for gene, row in zip(old_ids.tolist(), old_features):
            if gene not in extended_lookup or not np.array_equal(row, extended_lookup[gene]):
                raise GenomeScalePilotError(f"prior static feature row changed: {gene}")

    return {
        "records": 13_058,
        "trainRecords": 10_719,
        "validationRecords": 2_339,
        "testRecords": 0,
        "queries": 7_036,
        "featureRows": 10_231,
        "featureDimensions": 577,
        "priorRowsVerifiedByteIdentical": len(old_ids),
        "uniqueActions": len(action_set),
        "contexts": list(contexts),
        "trainRecordsByContext": context_train_counts,
        "validationRecordsByContext": context_validation_counts,
        "targetValueSpace": value_space,
    }


def _load_ensemble(ensemble_path: Path):
    source = ensemble_path / "source" / "ensemble_inference.py"
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("gwps_old_ensemble_inference", source)
    if spec is None or spec.loader is None:
        raise GenomeScalePilotError("could not load frozen old ensemble source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train_spec = importlib.util.spec_from_file_location(
        "gwps_old_train_metrics", source.parent / "train.py"
    )
    if train_spec is None or train_spec.loader is None:
        raise GenomeScalePilotError("could not load frozen metric source")
    train_module = importlib.util.module_from_spec(train_spec)
    train_spec.loader.exec_module(train_module)
    return module.EnsemblePredictor, train_module.gene_metrics


def compare_old_ensemble(
    data_path: Path,
    feature_path: Path,
    ensemble_path: Path,
    model_path: Path,
    output: Path,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate the frozen old ensemble on shared development contexts/readouts."""

    manifest = ensemble_path / "ensemble-manifest.json"
    if sha256(manifest) != OLD_ENSEMBLE_MANIFEST_SHA256:
        raise GenomeScalePilotError("old ensemble manifest SHA-256 drift")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    with np.load(feature_path, allow_pickle=False) as archive:
        feature_lookup = dict(
            zip(archive["entity_id"].astype(str).tolist(), archive["feature_values"])
        )
    Predictor, gene_metrics = _load_ensemble(ensemble_path)
    predictor = Predictor(ensemble_path, device=device)
    query_lookup = {gene: index for index, gene in enumerate(predictor.query_ids.astype(str))}
    missing_queries = [str(gene) for gene in data["query_ids"] if str(gene) not in query_lookup]
    if missing_queries:
        raise GenomeScalePilotError(f"old ensemble lacks shared queries: {missing_queries[:8]}")
    query_indices = np.asarray(
        [query_lookup[str(gene)] for gene in data["query_ids"]], dtype=np.int64
    )
    old_context_lookup = {
        context: index for index, context in enumerate(predictor.context_ids.astype(str))
    }
    shared_contexts = CONTEXT_IDS[:2]
    if any(context not in old_context_lookup for context in shared_contexts):
        raise GenomeScalePilotError("old ensemble context identities drifted")
    validation = data["split_validation"]
    selected = validation[data["context_index"][validation] < 2]
    actions = np.stack([feature_lookup[str(data["action_ids"][row])] for row in selected])
    old_context_index = np.asarray(
        [old_context_lookup[str(data["context_ids"][data["context_index"][row]])] for row in selected],
        dtype=np.int64,
    )
    prediction = np.empty((len(selected), len(query_indices)), dtype=np.float32)
    scale = np.empty_like(prediction)
    for start in range(0, len(selected), batch_size):
        local = np.arange(start, min(start + batch_size, len(selected)))
        result = predictor.predict(
            actions[local],
            data["num_cells_filtered"][selected[local]],
            old_context_index[local],
            query_indices=query_indices,
        )
        prediction[local] = result["mean"]
        scale[local] = result["marginal_scale"]

    with np.load(model_path / "reference.npz", allow_pickle=False) as archive:
        current_references = archive["reference"]
    model_report = json.loads((model_path / "report.json").read_text(encoding="utf-8"))
    value_space = str(data["target_value_space"].item())
    reports: dict[str, object] = {}
    for context_index, context_id in enumerate(shared_contexts):
        local = np.flatnonzero(data["context_index"][selected] == context_index)
        rows = selected[local]
        old_metrics = gene_metrics(
            prediction[local],
            data["targets"][rows],
            data["observed"][rows],
            [(9606, str(data["action_ids"][row])) for row in rows],
            current_references[context_index],
            scale[local],
            value_space=value_space,
        )
        new_metrics = model_report["results"][context_id]["world"]
        reports[context_id] = {
            "records": len(rows),
            "oldFrozenEnsemble": old_metrics,
            "newGenomeScaleSingleSeed": new_metrics,
            "newMinusOld": {
                "nllGainOldMinusNew": (
                    old_metrics["gene_macro_nll"] - new_metrics["gene_macro_nll"]
                ),
                "adjustedPearsonNewMinusOld": (
                    new_metrics["gene_macro_profile_centroid_adjusted_pearson_mean"]
                    - old_metrics["gene_macro_profile_centroid_adjusted_pearson_mean"]
                ),
            },
        }
    predictions_path = output / "old-ensemble-matched-panel-predictions.npz"
    np.savez_compressed(
        predictions_path,
        mean=prediction,
        scale=scale,
        record_ids=data["record_ids"][selected],
        action_ids=data["action_ids"][selected],
        context_index=data["context_index"][selected],
        query_ids=data["query_ids"],
    )
    report = {
        "schema": "slp.gwps-old-ensemble-matched-panel-development/v1",
        "label": "descriptive matched-panel neural comparator",
        "results": reports,
        "queryCount": len(query_indices),
        "validationRecords": len(selected),
        "oldEnsembleRefit": False,
        "validationUsedForFitting": False,
        "interpretation": (
            "The comparison is descriptive: it contrasts a three-seed ensemble trained on the "
            "earlier two-context corpus with one seed trained on the broader corpus. It does not "
            "isolate a data-scale effect from ensemble, optimization, or sample-composition effects."
        ),
        "predictions": {
            "path": predictions_path.name,
            "sha256": sha256(predictions_path),
        },
    }
    write_json(output / "old-ensemble-matched-panel-report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    data_path = Path(args.data).resolve(strict=True)
    feature_path = Path(args.features).resolve(strict=True)
    old_feature_path = Path(args.old_features).resolve(strict=True)
    old_ensemble = Path(args.old_ensemble).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"immutable pilot output already exists: {output}")
    input_audit = validate_inputs(data_path, feature_path, old_feature_path)
    if sha256(old_ensemble / "ensemble-manifest.json") != OLD_ENSEMBLE_MANIFEST_SHA256:
        raise GenomeScalePilotError("old ensemble manifest SHA-256 drift")
    output.mkdir(parents=True)
    source_dir = output / "source"
    source_dir.mkdir()
    runner_copy = source_dir / Path(__file__).name
    shutil.copy2(Path(__file__), runner_copy)
    training_output = output / "model"
    training_args = {
        "seed": 731,
        "queryBasisRank": 32,
        "hidden": 128,
        "stateDim": 64,
        "dropout": 0.2,
        "weightDecay": 0.1,
        "epochs": 180,
        "patience": 30,
        "maxSeconds": 1800,
        "ridgeAlpha": 10000.0,
        "referenceKind": "mean",
        "exposureAware": True,
    }
    protocol = {
        "schema": "slp.gwps-genome-scale-development-protocol/v1",
        "hypothesis": (
            "The same response-query architecture trained on the broader held-gene intervention "
            "corpus improves molecular predictions and passes the unchanged rule in all three contexts."
        ),
        "fixedRule": {
            "eachContext": {
                "geneMacroNllGainAgainstContextMean": 0.02,
                "geneMacroNllGainAgainstFullFeatureRidgeAlpha10000": 0.02,
                "geneMacroCentroidAdjustedPearson": 0.10,
            },
            "allThreeContextsRequired": True,
        },
        "inputs": {
            "development": {"path": str(data_path), "sha256": DATA_SHA256},
            "features": {"path": str(feature_path), "sha256": FEATURE_SHA256},
            "priorFeaturesForEqualityAudit": {
                "path": str(old_feature_path),
                "sha256": OLD_FEATURE_SHA256,
            },
            "oldEnsemble": {
                "path": str(old_ensemble),
                "manifestSha256": OLD_ENSEMBLE_MANIFEST_SHA256,
            },
        },
        "inputAudit": input_audit,
        "training": training_args,
        "modalities": {
            "actionsAndQueries": "frozen ESM2-t6 plus GOA-2022 fixed-basis features, 577 dimensions",
            "queryResponseDescriptors": "rank 32 fitted from split_train molecular responses only",
            "context": "measured core-control basal expression",
            "cellCounts": "likelihood uncertainty only, never mean/state inputs",
        },
        "selection": "minimum equal-context development gene-macro Gaussian NLL; one seed, no sweep",
        "oldComparator": (
            "Frozen earlier ensemble on the same 7,036-query development panel in its two original "
            "contexts; descriptive only and not an isolated data-effect estimate."
        ),
        "testArtifactAccessed": False,
        "slBenchmarkAccessed": False,
        "runner": {
            "path": str(Path(__file__)),
            "sha256": sha256(Path(__file__)),
            "copy": runner_copy.relative_to(output).as_posix(),
            "copySha256": sha256(runner_copy),
        },
    }
    write_json(output / "protocol.json", protocol)
    command = [
        sys.executable,
        str(ROOT / "modules" / "slp-1-1-world-transition-v1" / "train_human.py"),
        "--data",
        str(data_path),
        "--data-sha256",
        DATA_SHA256,
        "--features",
        str(feature_path),
        "--output",
        str(training_output),
        "--device",
        args.device,
        "--epochs",
        "180",
        "--patience",
        "30",
        "--max-seconds",
        "1800",
        "--query-basis-rank",
        "32",
        "--exposure-aware",
        "--reference-kind",
        "mean",
        "--hidden",
        "128",
        "--state-dim",
        "64",
        "--dropout",
        "0.2",
        "--weight-decay",
        "0.1",
        "--ridge-alpha",
        "10000",
        "--seed",
        "731",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    old_report = compare_old_ensemble(
        data_path,
        feature_path,
        old_ensemble,
        training_output,
        output,
        args.device,
        args.batch_size,
    )
    model_report_path = training_output / "report.json"
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    summary = {
        "schema": "slp.gwps-genome-scale-development-result/v1",
        "developmentRulePassed": model_report["development_rule_passed"],
        "results": model_report["results"],
        "bestEpoch": model_report["best_epoch"],
        "elapsedSeconds": time.monotonic() - started,
        "modelReport": {"path": "model/report.json", "sha256": sha256(model_report_path)},
        "checkpoint": {
            "path": "model/model.safetensors",
            "sha256": sha256(training_output / "model.safetensors"),
        },
        "oldComparatorReport": {
            "path": "old-ensemble-matched-panel-report.json",
            "sha256": sha256(output / "old-ensemble-matched-panel-report.json"),
        },
        "oldComparatorResults": old_report["results"],
        "testArtifactAccessed": False,
        "slBenchmarkAccessed": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=ROOT / "data/derived/slp11-human-gwps/complete-panel-v1/development.npz",
        type=Path,
    )
    parser.add_argument(
        "--features",
        default=ROOT / (
            "data/derived/slp11-human-gwps-static/ensembl116-goa2022-fixed-basis-v1/"
            "gwps-extended-static-esm-go-features.npz"
        ),
        type=Path,
    )
    parser.add_argument(
        "--old-features",
        default=ROOT / (
            "data/derived/slp11-human-static-fusion/esm2-t6-plus-go-svd-v1/"
            "human-static-esm-go-features.npz"
        ),
        type=Path,
    )
    parser.add_argument(
        "--old-ensemble",
        default=ROOT / (
            "results/slp11-transition/human-normalized-fusion-response32-ensemble731-733-v1"
        ),
        type=Path,
    )
    parser.add_argument("--output", default=OUTPUT, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        print(
            json.dumps(
                validate_inputs(
                    args.data.resolve(strict=True),
                    args.features.resolve(strict=True),
                    args.old_features.resolve(strict=True),
                ),
                sort_keys=True,
            )
        )
        return 0
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
