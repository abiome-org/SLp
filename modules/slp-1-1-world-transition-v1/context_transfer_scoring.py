"""Frozen gene-balanced scoring for the HepG2 context-transfer diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray
_TOLERANCE_MULTIPLIER = 64.0


class ContextTransferScoringError(ValueError):
    """Raised when frozen forecast-scoring inputs do not align."""


@dataclass(frozen=True)
class GeneProfiles:
    """Missing-aware record means collapsed to one profile per stable gene."""

    gene_ids: Array
    prediction: Array
    truth: Array
    observed: Array
    gene_profile_mse: Array
    mean_construct_mse: Array
    construct_counts: Array


def collapse_gene_profiles(
    prediction: Array,
    truth: Array,
    observed: Array,
    action_ids: Array,
    record_ids: Array,
) -> GeneProfiles:
    """Collapse constructs within genes and reject duplicate source records."""

    forecast = np.asarray(prediction, dtype=np.float64)
    values = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(observed)
    actions = np.asarray(action_ids).astype(str)
    records = np.asarray(record_ids).astype(str)
    if forecast.ndim != 2 or forecast.shape[0] == 0 or forecast.shape[1] == 0:
        raise ContextTransferScoringError("prediction must be a non-empty matrix")
    if values.shape != forecast.shape or mask.shape != forecast.shape or mask.dtype != np.bool_:
        raise ContextTransferScoringError("truth and boolean observed mask must match prediction")
    if actions.shape != (forecast.shape[0],) or records.shape != (forecast.shape[0],):
        raise ContextTransferScoringError("action_ids and record_ids must align with records")
    if len(set(records.tolist())) != records.size:
        raise ContextTransferScoringError("duplicate exact record IDs are forbidden")
    if not np.isfinite(forecast[mask]).all() or not np.isfinite(values[mask]).all():
        raise ContextTransferScoringError("observed predictions and truths must be finite")
    if np.any(mask.sum(axis=1) == 0):
        raise ContextTransferScoringError("every scored construct requires observed queries")

    genes = np.asarray(sorted(set(actions.tolist())))
    shape = (genes.size, forecast.shape[1])
    gene_prediction = np.zeros(shape, dtype=np.float64)
    gene_truth = np.zeros(shape, dtype=np.float64)
    gene_observed = np.zeros(shape, dtype=bool)
    profile_mse = np.zeros(genes.size, dtype=np.float64)
    construct_mse = np.zeros(genes.size, dtype=np.float64)
    construct_counts = np.zeros(genes.size, dtype=np.int64)
    squared = np.square(forecast - values, where=mask, out=np.zeros_like(forecast))
    record_mse = squared.sum(axis=1) / mask.sum(axis=1)
    for index, gene in enumerate(genes):
        selected = actions == gene
        construct_counts[index] = int(np.count_nonzero(selected))
        counts = mask[selected].sum(axis=0)
        gene_observed[index] = counts > 0
        gene_prediction[index] = np.divide(
            np.where(mask[selected], forecast[selected], 0.0).sum(axis=0), counts,
            out=np.zeros(forecast.shape[1]), where=counts > 0,
        )
        gene_truth[index] = np.divide(
            np.where(mask[selected], values[selected], 0.0).sum(axis=0), counts,
            out=np.zeros(forecast.shape[1]), where=counts > 0,
        )
        present = gene_observed[index]
        profile_mse[index] = np.mean(
            np.square(gene_prediction[index, present] - gene_truth[index, present])
        )
        construct_mse[index] = np.mean(record_mse[selected])
    return GeneProfiles(
        gene_ids=genes,
        prediction=gene_prediction,
        truth=gene_truth,
        observed=gene_observed,
        gene_profile_mse=profile_mse,
        mean_construct_mse=construct_mse,
        construct_counts=construct_counts,
    )


def gene_balanced_query_centroids(profiles: GeneProfiles, multiplicity: Array | None = None) -> tuple[Array, Array]:
    """Return separate query-wise gene-balanced prediction/truth centroids."""

    genes = profiles.gene_ids.size
    if multiplicity is None:
        weights = np.ones(genes, dtype=np.float64)
    else:
        weights = np.asarray(multiplicity, dtype=np.float64)
        if weights.shape != (genes,) or np.any(weights < 0.0) or not np.isfinite(weights).all():
            raise ContextTransferScoringError("gene multiplicity is invalid")
    support = profiles.observed.astype(np.float64, copy=False)
    denominator = weights @ support
    if not np.any(denominator > 0.0):
        raise ContextTransferScoringError("gene population has no observed queries")
    prediction = np.divide(
        weights @ np.where(profiles.observed, profiles.prediction, 0.0), denominator,
        out=np.zeros(profiles.prediction.shape[1]), where=denominator > 0.0,
    )
    truth = np.divide(
        weights @ np.where(profiles.observed, profiles.truth, 0.0), denominator,
        out=np.zeros(profiles.truth.shape[1]), where=denominator > 0.0,
    )
    return prediction, truth


def _profile_correlations(
    profiles: GeneProfiles,
    prediction_center: Array,
    truth_center: Array,
    *,
    query_chunk: int = 256,
) -> tuple[Array, Array]:
    genes, queries = profiles.prediction.shape
    left_center = np.asarray(prediction_center, dtype=np.float64)
    right_center = np.asarray(truth_center, dtype=np.float64)
    if left_center.shape != (queries,) or right_center.shape != (queries,):
        raise ContextTransferScoringError("correlation centers must match the query axis")
    if query_chunk <= 0:
        raise ContextTransferScoringError("query_chunk must be positive")
    count = profiles.observed.sum(axis=1).astype(np.float64)
    sums = [np.zeros(genes, dtype=np.float64) for _ in range(5)]
    left_sum, right_sum, left_square, right_square, cross = sums
    max_left = np.zeros(genes, dtype=np.float64)
    max_right = np.zeros(genes, dtype=np.float64)
    for start in range(0, queries, query_chunk):
        stop = min(start + query_chunk, queries)
        present = profiles.observed[:, start:stop]
        left = np.where(
            present, profiles.prediction[:, start:stop] - left_center[None, start:stop], 0.0
        )
        right = np.where(
            present, profiles.truth[:, start:stop] - right_center[None, start:stop], 0.0
        )
        left_sum += left.sum(axis=1)
        right_sum += right.sum(axis=1)
        left_square += np.square(left).sum(axis=1)
        right_square += np.square(right).sum(axis=1)
        cross += (left * right).sum(axis=1)
        max_left = np.maximum(max_left, np.max(np.abs(left), axis=1))
        max_right = np.maximum(max_right, np.max(np.abs(right), axis=1))
    covariance = cross - left_sum * right_sum / count
    left_variance = np.maximum(left_square - np.square(left_sum) / count, 0.0)
    right_variance = np.maximum(right_square - np.square(right_sum) / count, 0.0)
    left_tolerance = np.square(
        _TOLERANCE_MULTIPLIER * np.finfo(np.float64).eps * np.maximum(max_left, 1.0)
    ) * count
    right_tolerance = np.square(
        _TOLERANCE_MULTIPLIER * np.finfo(np.float64).eps * np.maximum(max_right, 1.0)
    ) * count
    defined = (count >= 2.0) & (left_variance > left_tolerance) & (right_variance > right_tolerance)
    correlation = np.full(genes, np.nan, dtype=np.float64)
    correlation[defined] = covariance[defined] / np.sqrt(
        left_variance[defined] * right_variance[defined]
    )
    correlation[defined] = np.clip(correlation[defined], -1.0, 1.0)
    return correlation, defined


def score_gene_profiles(profiles: GeneProfiles, training_centroid: Array) -> dict[str, float | int | None]:
    """Score the frozen primary and secondary equal-gene estimands."""

    reference = np.asarray(training_centroid, dtype=np.float64)
    if reference.shape != (profiles.prediction.shape[1],) or not np.isfinite(reference).all():
        raise ContextTransferScoringError("training_centroid must be one finite value per query")
    prediction_centroid, truth_centroid = gene_balanced_query_centroids(profiles)
    primary, primary_defined = _profile_correlations(
        profiles, prediction_centroid, truth_centroid
    )
    adjusted, adjusted_defined = _profile_correlations(profiles, reference, reference)
    uncentered, uncentered_defined = _profile_correlations(
        profiles, np.zeros_like(reference), np.zeros_like(reference)
    )

    def mean_defined(values: Array, defined: Array) -> float | None:
        return float(np.mean(values[defined])) if np.any(defined) else None

    return {
        "genes": int(profiles.gene_ids.size),
        "constructs": int(profiles.construct_counts.sum()),
        "primaryGeneAveragedProfileMse": float(np.mean(profiles.gene_profile_mse)),
        "secondaryEqualGeneMeanConstructMse": float(np.mean(profiles.mean_construct_mse)),
        "primaryIndependentlyCenteredGeneMacroProfilePearson": mean_defined(
            primary, primary_defined
        ),
        "primaryPearsonDefinedGenes": int(np.count_nonzero(primary_defined)),
        "primaryPearsonUndefinedGenes": int(np.count_nonzero(~primary_defined)),
        "trainingCentroidAdjustedGeneMacroProfilePearson": mean_defined(
            adjusted, adjusted_defined
        ),
        "trainingAdjustedPearsonDefinedGenes": int(np.count_nonzero(adjusted_defined)),
        "uncenteredGeneMacroProfilePearson": mean_defined(uncentered, uncentered_defined),
        "uncenteredPearsonDefinedGenes": int(np.count_nonzero(uncentered_defined)),
    }


def bootstrap_gene_profiles(
    profiles: GeneProfiles,
    *,
    samples: int = 1000,
    seed: int = 731,
    query_chunk: int = 256,
) -> dict[str, object]:
    """Recompute query centroids in every exact gene-block bootstrap sample."""

    if type(samples) is not int or samples <= 0 or type(seed) is not int:
        raise ContextTransferScoringError("bootstrap samples and seed must be positive integers")
    genes = profiles.gene_ids.size
    rng = np.random.default_rng(seed)
    mse = np.empty(samples, dtype=np.float64)
    construct_mse = np.empty(samples, dtype=np.float64)
    correlation = np.full(samples, np.nan, dtype=np.float64)
    undefined_draws = 0
    for sample in range(samples):
        drawn = rng.integers(0, genes, size=genes)
        multiplicity = np.bincount(drawn, minlength=genes).astype(np.float64)
        prediction_centroid, truth_centroid = gene_balanced_query_centroids(
            profiles, multiplicity
        )
        per_gene, defined = _profile_correlations(
            profiles, prediction_centroid, truth_centroid, query_chunk=query_chunk
        )
        defined_weight = multiplicity * defined
        denominator = defined_weight.sum()
        if denominator > 0.0:
            correlation[sample] = float(np.nansum(per_gene * defined_weight) / denominator)
        else:
            undefined_draws += 1
        mse[sample] = float(multiplicity @ profiles.gene_profile_mse / genes)
        construct_mse[sample] = float(multiplicity @ profiles.mean_construct_mse / genes)

    def interval(values: Array) -> list[float] | None:
        finite = values[np.isfinite(values)]
        return [float(item) for item in np.quantile(finite, [0.025, 0.5, 0.975])] if finite.size else None

    return {
        "samples": samples,
        "seed": seed,
        "geneAveragedProfileMsePercentiles025_50_975": interval(mse),
        "equalGeneMeanConstructMsePercentiles025_50_975": interval(construct_mse),
        "independentlyCenteredGeneMacroProfilePearsonPercentiles025_50_975": interval(
            correlation
        ),
        "undefinedCorrelationDraws": undefined_draws,
        "centroidsRecomputedForEveryDraw": True,
        "decisionUse": "descriptive-only",
    }
