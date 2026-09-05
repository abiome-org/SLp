"""Audit human development targets and model scales without reading test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from transition_model import Config, TransitionWorld

CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
)
CONTROL_SUFFIX = "_non-targeting_non-targeting_non-targeting"


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return Pearson correlation, treating constant vectors as undefined."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    tolerance = 8 * np.finfo(np.float64).eps
    if np.ptp(x) <= tolerance * max(1.0, float(np.max(np.abs(x)))):
        return None
    if np.ptp(y) <= tolerance * max(1.0, float(np.max(np.abs(y)))):
        return None
    x -= x.mean()
    y -= y.mean()
    return float((x @ y) / math.sqrt(float(x @ x) * float(y @ y)))


def cp10k_log1p(raw: np.ndarray) -> np.ndarray:
    """Apply the exact shared-panel target transform."""

    values = np.asarray(raw, dtype=np.float64)
    denominator = values.sum(axis=1)
    if values.ndim != 2 or np.any(values < 0) or np.any(denominator <= 0):
        raise ValueError("raw profiles must be nonnegative nonzero rows")
    return np.log2(1.0 + 10_000.0 * values / denominator[:, None])


def calibration_summary(
    prediction: np.ndarray, truth: np.ndarray, scale: np.ndarray
) -> dict[str, float]:
    """Summarize marginal Gaussian calibration on observed values."""

    error = np.asarray(truth, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    sigma = np.broadcast_to(np.asarray(scale, dtype=np.float64), error.shape)
    standardized = error / sigma
    second_moment = float(np.square(standardized).mean())
    multiplier = math.sqrt(second_moment)
    return {
        "standardizedResidualSecondMoment": second_moment,
        "validationOptimalGlobalScaleMultiplier": multiplier,
        "coverageWithinOneScale": float(np.mean(np.abs(standardized) <= 1.0)),
        "coverageWithin1.96Scale": float(np.mean(np.abs(standardized) <= 1.96)),
        "nllGainAtValidationOptimalGlobalScale": float(
            0.5 * (second_moment - 1.0 - math.log(second_moment))
        ),
    }


def _decode(dataset: h5py.Dataset) -> list[str]:
    return [item.decode() if isinstance(item, bytes) else str(item) for item in dataset[:]]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(values, dtype=np.float64), [0, 0.1, 0.5, 0.9, 1])
    return dict(zip(("minimum", "p10", "median", "p90", "maximum"), map(float, q)))


def _duplicate_diagnostics(
    targets: np.ndarray,
    actions: np.ndarray,
    records: np.ndarray,
    cells: np.ndarray,
    basal: np.ndarray,
    high_expression: np.ndarray,
    query_ids: np.ndarray,
) -> dict[str, object]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row, action in enumerate(actions):
        groups[str(action)].append(row)
    all_correlations: list[float] = []
    high_correlations: list[float] = []
    minimum_cells: list[float] = []
    construct_pairs: dict[str, int] = defaultdict(int)
    query_lookup = {str(gene): column for column, gene in enumerate(query_ids)}
    self_left: list[float] = []
    self_right: list[float] = []
    for rows in groups.values():
        for offset, left in enumerate(rows[:-1]):
            for right in rows[offset + 1 :]:
                left_effect = targets[left] - basal
                right_effect = targets[right] - basal
                corr = pearson(left_effect, right_effect)
                high = pearson(left_effect[high_expression], right_effect[high_expression])
                if corr is not None and high is not None:
                    all_correlations.append(corr)
                    high_correlations.append(high)
                    minimum_cells.append(min(cells[left], cells[right]))
                    left_kind = str(records[left]).rsplit("_", 2)[-2]
                    right_kind = str(records[right]).rsplit("_", 2)[-2]
                    construct_pairs["/".join(sorted((left_kind, right_kind)))] += 1
                    action = str(actions[left])
                    if action in query_lookup:
                        column = query_lookup[action]
                        self_left.append(float(left_effect[column]))
                        self_right.append(float(right_effect[column]))
    return {
        "interpretation": (
            "distinct source perturbation-population summaries sharing one ENSG; "
            "not biological replicates and not single-guide replicates"
        ),
        "pairs": len(all_correlations),
        "meanPearsonAllSharedReadouts": float(np.mean(all_correlations)),
        "meanPearsonTop1000TrainingMeanReadouts": float(np.mean(high_correlations)),
        "pairPearsonVsLogMinimumCellCount": pearson(
            np.asarray(all_correlations), np.log1p(minimum_cells)
        ),
        "pairedOnTargetEffectPearson": pearson(
            np.asarray(self_left), np.asarray(self_right)
        ),
        "pairedOnTargetEffectSignAgreement": float(
            np.mean(np.sign(self_left) == np.sign(self_right))
        ),
        "pairedOnTargetBothDecreasedFraction": float(
            np.mean((np.asarray(self_left) < 0) & (np.asarray(self_right) < 0))
        ),
        "constructLabelPairs": dict(sorted(construct_pairs.items())),
    }


def source_diagnostics(
    source_path: Path,
    context_id: str,
    context: int,
    bundle: dict[str, np.ndarray],
) -> dict[str, object]:
    """Read X only for development rows and controls, never test action rows."""

    context_rows = np.flatnonzero(bundle["context_index"] == context)
    original_ids = [str(bundle["record_ids"][row]).split("|", 1)[1] for row in context_rows]
    with h5py.File(source_path, "r") as handle:
        source_ids = _decode(handle["obs/gene_transcript"])
        source_lookup = {record: row for row, record in enumerate(source_ids)}
        development_source_rows = np.asarray(
            [source_lookup[record] for record in original_ids], dtype=np.int64
        )
        control_rows = np.asarray(
            [row for row, record in enumerate(source_ids) if record.endswith(CONTROL_SUFFIX)],
            dtype=np.int64,
        )
        allowed = np.sort(np.concatenate((development_source_rows, control_rows)))
        if len(np.unique(allowed)) != len(allowed):
            raise ValueError("development and control source rows overlap")
        source_matrix = np.asarray(handle["X"][allowed, :], dtype=np.float64)
        allowed_lookup = {source_row: local for local, source_row in enumerate(allowed)}
        development_matrix = source_matrix[
            [allowed_lookup[row] for row in development_source_rows]
        ]
        controls = source_matrix[[allowed_lookup[row] for row in control_rows]]
        source_genes = _decode(handle["var/gene_id"])
        gene_lookup = {gene: column for column, gene in enumerate(source_genes)}
        shared_columns = np.asarray(
            [gene_lookup[str(gene)] for gene in bundle["query_ids"]], dtype=np.int64
        )
        core_control = np.asarray(handle["obs/core_control"][control_rows], dtype=bool)
        control_cells = np.asarray(
            handle["obs/num_cells_filtered"][control_rows], dtype=np.float64
        )
        metadata = {
            name: np.asarray(handle[f"obs/{name}"][development_source_rows], dtype=np.float64)
            for name in (
                "num_cells_filtered",
                "UMI_count_unfiltered",
                "mitopercent",
                "z_gemgroup_UMI",
                "mean_leverage_score",
                "cnv_score_z",
            )
        }

    shared = development_matrix[:, shared_columns]
    reconstructed = cp10k_log1p(shared)
    expected = bundle["targets"][context_rows]
    maximum_error = float(np.max(np.abs(reconstructed - expected)))
    control_shared = controls[:, shared_columns]
    transformed_controls = cp10k_log1p(control_shared)
    all_control_basal = transformed_controls.mean(axis=0)
    core_control_basal = transformed_controls[core_control].mean(axis=0)
    control_rmse = np.sqrt(
        np.square(transformed_controls - core_control_basal).mean(axis=1)
    )

    local_train = np.flatnonzero(np.isin(context_rows, bundle["split_train"]))
    train_targets = expected[local_train]
    train_actions = bundle["action_ids"][context_rows[local_train]]
    train_records = bundle["record_ids"][context_rows[local_train]]
    train_cells = metadata["num_cells_filtered"][local_train]
    train_mean = train_targets.mean(axis=0, dtype=np.float64)
    row_rmse = np.sqrt(np.square(train_targets - train_mean).mean(axis=1))
    high_expression = np.argsort(-train_mean, kind="stable")[:1000]

    nuisance = {
        name: pearson(row_rmse, np.log1p(value[local_train]) if name == "num_cells_filtered" else value[local_train])
        for name, value in metadata.items()
    }
    duplicate_all = _duplicate_diagnostics(
        train_targets,
        train_actions,
        train_records,
        train_cells,
        all_control_basal,
        high_expression,
        bundle["query_ids"],
    )
    duplicate_core = _duplicate_diagnostics(
        train_targets,
        train_actions,
        train_records,
        train_cells,
        core_control_basal,
        high_expression,
        bundle["query_ids"],
    )
    full_sum = development_matrix.sum(axis=1)
    shared_sum = shared.sum(axis=1)
    return {
        "source": source_path.as_posix(),
        "developmentTargetRowsRead": len(development_source_rows),
        "testTargetRowsRead": 0,
        "controlRowsRead": len(control_rows),
        "coreControlRows": int(core_control.sum()),
        "targetReconstructionMaximumAbsoluteError": maximum_error,
        "sharedPanelFractionOfFullRawUmi": _quantiles(shared_sum / full_sum),
        "fullPanelMeanUmiVsObsUmiCountUnfilteredPearson": pearson(
            full_sum, metadata["UMI_count_unfiltered"]
        ),
        "filteredCells": _quantiles(metadata["num_cells_filtered"]),
        "trainingRowRmseNuisancePearson": nuisance,
        "allVsCoreControlBasalRmse": float(
            np.sqrt(np.square(all_control_basal - core_control_basal).mean())
        ),
        "controlProfileRmseVsLogCellCount": pearson(
            control_rmse, np.log1p(control_cells)
        ),
        "coreControlProfileRmseVsLogCellCount": pearson(
            control_rmse[core_control], np.log1p(control_cells[core_control])
        ),
        "controlProfileRmse": _quantiles(control_rmse),
        "duplicateSummaryAgreementUsingAllControls": duplicate_all,
        "duplicateSummaryAgreementUsingCoreControls": duplicate_core,
    }


def model_predictions(
    bundle: dict[str, np.ndarray], feature_path: Path, run_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(feature_path, allow_pickle=False) as archive:
        lookup = {
            (int(taxon), str(entity)): row
            for row, (taxon, entity) in enumerate(
                zip(archive["entity_taxon"], archive["entity_id"])
            )
        }
        features = archive["feature_values"].astype(np.float32)
    actions = np.stack([features[lookup[(9606, str(gene))]] for gene in bundle["action_ids"]])
    queries = np.stack([features[lookup[(9606, str(gene))]] for gene in bundle["query_ids"]])
    with np.load(run_dir / "reference.npz", allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    config = Config(**json.loads((run_dir / "model-config.json").read_text()))
    model = TransitionWorld(config)
    model.load_state_dict(load_file(run_dir / "model.safetensors"))
    model.eval()
    normalized_actions = (actions - reference["feature_mean"]) / reference["feature_std"]
    normalized_queries = (queries - reference["feature_mean"]) / reference["feature_std"]
    selected = reference["context_query_indices"]
    validation = bundle["split_validation"]
    predictions: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    with torch.no_grad():
        query_tensor = torch.as_tensor(normalized_queries)
        context_features = query_tensor[selected]
        for offset in range(0, len(validation), 64):
            rows = validation[offset : offset + 64]
            contexts = bundle["context_index"][rows]
            output = model(
                torch.as_tensor(normalized_actions[rows]),
                query_tensor,
                torch.as_tensor(reference["reference"][contexts]),
                torch.as_tensor(reference["reference_scale"][contexts]),
                context_features=context_features[None].expand(len(rows), -1, -1),
                context_values=torch.as_tensor(reference["context_values"][contexts]),
                context_mask=torch.ones((len(rows), len(selected)), dtype=torch.bool),
            )
            predictions.append(output["mean"].numpy())
            scales.append(output["scale"].numpy())
    return np.concatenate(predictions), np.concatenate(scales)


def run(args: argparse.Namespace) -> dict[str, object]:
    with np.load(args.development, allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    if len(bundle["split_test"]):
        raise ValueError("development bundle contains test indices")
    source_reports = {
        context_id: source_diagnostics(path, context_id, context, bundle)
        for context, (context_id, path) in enumerate(zip(CONTEXTS, args.sources))
    }
    predictions, scales = model_predictions(bundle, args.features, args.run)
    validation = bundle["split_validation"]
    model_scale_report: dict[str, object] = {}
    for context, context_id in enumerate(CONTEXTS):
        local = np.flatnonzero(bundle["context_index"][validation] == context)
        summary = calibration_summary(
            predictions[local], bundle["targets"][validation[local]], scales[local]
        )
        summary["referenceScaleQuantiles"] = _quantiles(scales[local][0])
        summary["referenceScaleFloorFraction"] = float(np.mean(scales[local][0] <= 0.0500001))
        model_scale_report[context_id] = summary
    report = {
        "schema": "slp.human-development-scientific-audit/v1",
        "label": "development diagnostics",
        "testOnlyBundleRead": False,
        "sourceTargetRowsRead": "development train+validation only; controls also read",
        "sourceDiagnostics": source_reports,
        "modelScaleCalibration": model_scale_report,
        "runReport": (args.run / "report.json").as_posix(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        type=Path,
        default=ROOT / "data/derived/slp11-human/replogle-k562-rpe1-development-v1.npz",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT
        / "data/derived/slp11-human-sequence/esm2-t6-8m-ensembl116-full-v1"
        / "human-sequence-esm2-features.npz",
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "results/slp11-transition/human-esm2-context-seed731-v2",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        nargs=2,
        default=[
            ROOT / "data/sources/human/K562_essential_raw_bulk_01.h5ad",
            ROOT / "data/sources/human/rpe1_raw_bulk_01.h5ad",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/slp11-transition/human-scientific-audit-v1/report.json",
    )
    args = parser.parse_args()
    run(args)
    payload = args.output.read_bytes()
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(payload).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
