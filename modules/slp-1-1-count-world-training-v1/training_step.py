"""One application-neutral count/population training step on a native panel."""
from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

import torch

_objective_spec = importlib.util.spec_from_file_location(
    __name__ + "._molecular_mean_objective",
    Path(__file__).resolve().with_name("molecular_mean_objective.py"),
)
_objective = importlib.util.module_from_spec(_objective_spec)
_objective_spec.loader.exec_module(_objective)
normalized_profile_mse = _objective.normalized_profile_mse
population_log1p_mean = _objective.population_log1p_mean


@dataclass(frozen=True)
class CellBatch:
    actions: torch.Tensor
    action_mask: torch.Tensor
    context_index: torch.Tensor
    counts: torch.Tensor
    observed: torch.Tensor
    library: torch.Tensor


@dataclass(frozen=True)
class PopulationBatch:
    actions: torch.Tensor
    action_mask: torch.Tensor
    context_weights: torch.Tensor
    target_log1p_mean: torch.Tensor


def training_losses(
    model, query_features, basal_rate, basal_mask, cells: CellBatch,
    populations: PopulationBatch | None = None, *, mean_weight: float = 0.,
    fitting_mean_scale: float | None = None, epsilon=None,
):
    """Compute likelihood plus optional aggregate mean supervision.

    A batch belongs to one explicit measurement panel. A workload can alternate
    panels/species/contexts while sharing model parameters; it must define
    their sampling weights and evaluate each separately. Inputs carry no gene
    vocabulary or benchmark identity. This function never samples or selects
    data, updates parameters, or evaluates held outcomes.
    """
    if not model.training:
        raise ValueError("training_losses requires a model in training mode")
    if basal_mask.dtype != torch.bool or not basal_mask.all():
        raise ValueError("training requires a fully measured positive native control panel")
    if not math.isfinite(mean_weight) or mean_weight < 0:
        raise ValueError("mean weight must be finite nonnegative")
    if (populations is None) != (mean_weight == 0):
        raise ValueError("positive mean weight and population batch must be supplied together")
    index = cells.context_index
    if (index.ndim != 1 or len(index) != len(cells.actions)
            or index.dtype != torch.long or (index < 0).any()
            or (index >= len(basal_rate)).any()):
        raise ValueError("cell context indices must reference the supplied context table")
    # The pinned core's context encoder has no stochastic layers. Reuse its
    # graph within this optimizer step; never cache it across parameter updates.
    contexts = model.encode_context(query_features, basal_rate, basal_mask)
    prior = model.prior_from_context(cells.actions, cells.action_mask, contexts[index])
    count_result = model.elbo(
        cells.counts, cells.observed, cells.library, query_features,
        basal_rate[index], prior, epsilon=epsilon,
    )
    count_loss = count_result["loss_per_cell"].mean()
    normalized_mean_loss = torch.zeros_like(count_loss)
    raw_mean_loss = torch.zeros_like(count_loss)
    prediction = None
    if populations is not None:
        if fitting_mean_scale is None:
            raise ValueError("mean supervision requires a frozen fitting scale")
        population_count = len(populations.actions)
        groups = len(basal_rate)
        # Disable action dropout for the analytic population forecast, retaining
        # gradients and restoring training mode even if input validation fails.
        previous_modes = [(part, part.training) for part in model.modules()]
        model.eval()
        try:
            aggregate_prior = model.prior_from_context(
                populations.actions.repeat_interleave(groups, 0),
                populations.action_mask.repeat_interleave(groups, 0),
                contexts.repeat(population_count, 1),
            )
            rates = model.population_mean(
                aggregate_prior, query_features, basal_rate.repeat(population_count, 1),
            ).reshape(population_count, groups, len(query_features))
            prediction = population_log1p_mean(rates, populations.context_weights)
            normalized_mean_loss = normalized_profile_mse(
                prediction, populations.target_log1p_mean, fitting_mean_scale,
            )
            raw_mean_loss = normalized_mean_loss * fitting_mean_scale
        finally:
            for part, was_training in previous_modes:
                part.training = was_training
    loss = count_loss + mean_weight * normalized_mean_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite joint molecular loss")
    return {
        "loss": loss, "count_elbo": count_loss,
        "normalized_mean_mse": normalized_mean_loss, "mean_mse": raw_mean_loss,
        "population_prediction": prediction, "count_result": count_result,
        "prior": prior,
    }
