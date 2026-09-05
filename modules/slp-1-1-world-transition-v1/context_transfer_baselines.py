"""Target-free baselines and metrics for frozen context-transfer diagnostics.

Forecast helpers consume fitting molecular outcomes, static action features, or
control-only context descriptors.  They have no argument for target-context
perturbation outcomes.  Metric helpers are separate so outcome-dependent
centering cannot feed a forecast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


class ContextTransferError(ValueError):
    """Raised when a context-transfer baseline contract is violated."""


@dataclass(frozen=True)
class PopulationRoster:
    """Exact source-construct populations obtained from metadata only."""

    action_ids: Array
    population_ids: Array
    construct_ids: Array
    transcript_labels: Array


def population_roster(
    cell_action_ids: Array,
    cell_population_ids: Array,
    cell_construct_ids: Array,
    cell_transcript_labels: Array,
) -> PopulationRoster:
    """Collapse targeted-cell metadata to sorted exact populations."""

    arrays = [np.asarray(value).astype(str) for value in (
        cell_action_ids, cell_population_ids, cell_construct_ids, cell_transcript_labels
    )]
    if any(value.ndim != 1 for value in arrays) or not arrays[0].size:
        raise ContextTransferError("population metadata must be non-empty vectors")
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ContextTransferError("population metadata vectors must align")
    if any(np.any(value == "") for value in arrays):
        raise ContextTransferError("population metadata cannot contain empty identities")
    actions, populations, constructs, transcripts = arrays
    unique_populations = np.asarray(sorted(set(populations.tolist())))
    output = [np.empty(unique_populations.size, dtype=value.dtype) for value in (
        actions, constructs, transcripts
    )]
    for index, population in enumerate(unique_populations):
        selected = populations == population
        for source, destination, label in zip(
            (actions, constructs, transcripts), output, ("action", "construct", "transcript"),
            strict=True,
        ):
            identities = np.unique(source[selected])
            if identities.size != 1:
                raise ContextTransferError(
                    f"population {population!r} has multiple {label} identities"
                )
            destination[index] = identities[0]
    return PopulationRoster(
        action_ids=output[0], population_ids=unique_populations,
        construct_ids=output[1], transcript_labels=output[2],
    )


def control_context_distances(
    fitting_context_descriptors: Array,
    target_context_descriptor: Array,
    observed: Array,
) -> Array:
    """Euclidean distances after standardizing each context across fixed tokens."""

    fitting = np.asarray(fitting_context_descriptors, dtype=np.float64)
    target = np.asarray(target_context_descriptor, dtype=np.float64)
    mask = np.asarray(observed)
    if fitting.ndim != 2 or target.shape != (fitting.shape[1],):
        raise ContextTransferError("context descriptors must share one query axis")
    if mask.shape != (fitting.shape[1],) or mask.dtype != np.bool_ or not mask.any():
        raise ContextTransferError("observed must be a non-empty boolean query mask")
    if not np.isfinite(fitting[:, mask]).all() or not np.isfinite(target[mask]).all():
        raise ContextTransferError("observed context tokens must be finite")

    vectors = np.vstack((fitting[:, mask], target[None, mask]))
    means = vectors.mean(axis=1, keepdims=True)
    scales = vectors.std(axis=1, keepdims=True)
    if np.any(scales <= np.finfo(np.float64).eps):
        raise ContextTransferError("a context descriptor is constant on the fixed panel")
    standardized = (vectors - means) / scales
    return np.sqrt(np.square(standardized[:-1] - standardized[-1]).sum(axis=1))


def _training_arrays(
    targets: Array, observed: Array, context_index: Array, action_ids: Array
) -> tuple[Array, Array, Array, Array]:
    values = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(observed)
    contexts = np.asarray(context_index)
    actions = np.asarray(action_ids).astype(str)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ContextTransferError("fitting targets must be a non-empty matrix")
    if mask.shape != values.shape or mask.dtype != np.bool_:
        raise ContextTransferError("fitting observed must be a matching boolean matrix")
    if contexts.shape != (values.shape[0],) or contexts.dtype.kind not in "iu":
        raise ContextTransferError("context_index must be one integer per fitting row")
    if actions.shape != (values.shape[0],) or np.any(actions == ""):
        raise ContextTransferError("action_ids must contain one identity per fitting row")
    if not np.isfinite(values[mask]).all():
        raise ContextTransferError("observed fitting targets must be finite")
    return values, mask, contexts.astype(np.int64, copy=False), actions


def _gene_means(values: Array, mask: Array, actions: Array) -> tuple[Array, Array, Array]:
    genes = np.asarray(sorted(set(actions.tolist())))
    means = np.zeros((genes.size, values.shape[1]), dtype=np.float64)
    supported = np.zeros_like(means, dtype=bool)
    for index, gene in enumerate(genes):
        selected = actions == gene
        counts = mask[selected].sum(axis=0)
        supported[index] = counts > 0
        means[index] = np.divide(
            np.where(mask[selected], values[selected], 0.0).sum(axis=0),
            counts,
            out=np.zeros(values.shape[1], dtype=np.float64),
            where=counts > 0,
        )
    return genes, means, supported


def equal_source_fitting_centroid(
    targets: Array,
    observed: Array,
    context_index: Array,
    action_ids: Array,
) -> Array:
    """Return an equal-context, gene-balanced constant fitting centroid."""

    values, mask, contexts, actions = _training_arrays(
        targets, observed, context_index, action_ids
    )
    context_count = int(contexts.max()) + 1
    if set(contexts.tolist()) != set(range(context_count)):
        raise ContextTransferError("fitting contexts must be contiguous and all represented")
    context_centroids = np.zeros((context_count, values.shape[1]), dtype=np.float64)
    context_support = np.zeros_like(context_centroids, dtype=bool)
    for context in range(context_count):
        selected = contexts == context
        _, means, supported = _gene_means(values[selected], mask[selected], actions[selected])
        counts = supported.sum(axis=0)
        context_support[context] = counts > 0
        context_centroids[context] = np.divide(
            np.where(supported, means, 0.0).sum(axis=0),
            counts,
            out=np.zeros(values.shape[1], dtype=np.float64),
            where=counts > 0,
        )
    if not context_support.all():
        raise ContextTransferError("every fitting context must support every output query")
    return context_centroids.mean(axis=0)


def same_gene_source_response_forecast(
    targets: Array,
    observed: Array,
    context_index: Array,
    fitting_action_ids: Array,
    forecast_action_ids: Array,
    fallback: Array,
) -> tuple[Array, Array]:
    """Average a gene's source trajectories equally, with a constant fallback.

    Each source first averages all fitting records for the gene.  Available
    source means are then averaged equally.  A gene absent from all fitting
    contexts receives the supplied equal-source fitting centroid.
    """

    values, mask, contexts, actions = _training_arrays(
        targets, observed, context_index, fitting_action_ids
    )
    requested = np.asarray(forecast_action_ids).astype(str)
    fallback_value = np.asarray(fallback, dtype=np.float64)
    if requested.ndim != 1 or fallback_value.shape != (values.shape[1],):
        raise ContextTransferError("forecast actions or fallback shape is invalid")
    if not np.isfinite(fallback_value).all():
        raise ContextTransferError("fallback must be finite")
    by_gene: dict[str, list[tuple[Array, Array]]] = {}
    for context in range(int(contexts.max()) + 1):
        selected = contexts == context
        genes, means, supported = _gene_means(values[selected], mask[selected], actions[selected])
        for gene, mean, support in zip(genes, means, supported, strict=True):
            by_gene.setdefault(str(gene), []).append((mean, support))
    forecast = np.broadcast_to(fallback_value, (requested.size, values.shape[1])).copy()
    seen = np.zeros(requested.size, dtype=bool)
    for row, gene in enumerate(requested):
        sources = by_gene.get(str(gene))
        if sources is None:
            continue
        source_values = np.stack([item[0] for item in sources])
        source_support = np.stack([item[1] for item in sources])
        counts = source_support.sum(axis=0)
        present = counts > 0
        forecast[row, present] = (
            np.where(source_support, source_values, 0.0).sum(axis=0)[present]
            / counts[present]
        )
        seen[row] = True
    return forecast, seen


def _pearson(left: Array, right: Array) -> float | None:
    if left.size < 2:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    tolerance = np.finfo(np.float64).eps * max(1.0, float(left.size))
    if denominator <= tolerance:
        return None
    return float((left @ right) / denominator)


def gene_macro_transfer_metrics(
    prediction: Array,
    truth: Array,
    observed: Array,
    action_ids: Array,
    *,
    training_centroid: Array,
) -> dict[str, float | int | None]:
    """Compute gene-balanced point metrics for one predeclared stratum.

    The primary correlation independently subtracts gene-balanced per-query
    averages of prediction and truth, computed only inside this metric.  This
    function cannot produce a forecast and returns no fitted parameters.
    """

    means, mask, _, actions = _training_arrays(
        prediction, observed, np.zeros(np.asarray(prediction).shape[0], dtype=np.int64), action_ids
    )
    truth_values = np.asarray(truth, dtype=np.float64)
    if truth_values.shape != means.shape or not np.isfinite(truth_values[mask]).all():
        raise ContextTransferError("truth must match prediction and be finite where observed")
    reference = np.asarray(training_centroid, dtype=np.float64)
    if reference.shape != (means.shape[1],) or not np.isfinite(reference).all():
        raise ContextTransferError("training_centroid must be one finite value per query")

    genes, gene_prediction, gene_support = _gene_means(means, mask, actions)
    truth_genes, gene_truth, truth_support = _gene_means(truth_values, mask, actions)
    if not np.array_equal(genes, truth_genes) or not np.array_equal(gene_support, truth_support):
        raise ContextTransferError("prediction and truth gene support differ")
    support_counts = gene_support.sum(axis=0)
    supported_queries = support_counts > 0
    prediction_centroid = np.divide(
        np.where(gene_support, gene_prediction, 0.0).sum(axis=0), support_counts,
        out=np.zeros(means.shape[1]), where=supported_queries,
    )
    truth_centroid = np.divide(
        np.where(gene_support, gene_truth, 0.0).sum(axis=0), support_counts,
        out=np.zeros(means.shape[1]), where=supported_queries,
    )

    mse: list[float] = []
    primary: list[float] = []
    training_adjusted: list[float] = []
    uncentered: list[float] = []
    for row in range(genes.size):
        present = gene_support[row]
        mse.append(float(np.mean(np.square(gene_prediction[row, present] - gene_truth[row, present]))))
        pairs = (
            (
                gene_prediction[row, present] - prediction_centroid[present],
                gene_truth[row, present] - truth_centroid[present],
                primary,
            ),
            (
                gene_prediction[row, present] - reference[present],
                gene_truth[row, present] - reference[present],
                training_adjusted,
            ),
            (gene_prediction[row, present], gene_truth[row, present], uncentered),
        )
        for left, right, destination in pairs:
            correlation = _pearson(left, right)
            if correlation is not None:
                destination.append(correlation)
    return {
        "genes": int(genes.size),
        "geneMacroMse": float(np.mean(mse)),
        "independentlyGeneCenteredProfilePearsonMean": (
            float(np.mean(primary)) if primary else None
        ),
        "independentlyGeneCenteredDefinedGenes": len(primary),
        "independentlyGeneCenteredUndefinedGenes": int(genes.size - len(primary)),
        "trainingCentroidAdjustedProfilePearsonMean": (
            float(np.mean(training_adjusted)) if training_adjusted else None
        ),
        "trainingCentroidAdjustedDefinedGenes": len(training_adjusted),
        "uncenteredProfilePearsonMean": float(np.mean(uncentered)) if uncentered else None,
        "uncenteredDefinedGenes": len(uncentered),
        "queriesWithAnySupport": int(np.count_nonzero(supported_queries)),
    }
