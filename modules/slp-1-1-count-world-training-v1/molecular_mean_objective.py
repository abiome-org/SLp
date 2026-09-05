"""Differentiable population-mean supervision alongside molecular likelihood."""
from __future__ import annotations

import math

import torch


def population_log1p_mean(expected_rates, context_weights):
    """Mix expected rates [P,C,Q] over contexts, then take log1p.

    The caller supplies metadata-derived weights [P,C]. Neither library sizes
    nor observed intervention outcomes determine the predicted rates here.
    Transforming each context before mixing defines a different endpoint.
    """
    if (expected_rates.ndim != 3
            or context_weights.shape != expected_rates.shape[:2]
            or min(expected_rates.shape) <= 0):
        raise ValueError("positive-sized rates [P,C,Q] and weights [P,C] required")
    if (not torch.isfinite(expected_rates).all() or (expected_rates < 0).any()
            or not torch.isfinite(context_weights).all()
            or (context_weights < 0).any()):
        raise ValueError("rates and context weights must be finite nonnegative")
    total = context_weights.sum(-1, keepdim=True)
    if not torch.isfinite(total).all() or (total <= 0).any():
        raise ValueError("each population needs positive finite context support")
    mixture = (expected_rates * (context_weights / total)[..., None]).sum(1)
    if not torch.isfinite(mixture).all():
        raise ValueError("population mean is nonfinite")
    return mixture.log1p()


def normalized_profile_mse(prediction, target, fitting_scale: float):
    """Equal-population, equal-query MSE divided by one frozen fitting scalar.

    The scalar is computed before optimization from fitting data only. It is
    never estimated from the current minibatch or from evaluation outcomes.
    The workload specifies the weight relative to its cell likelihood.
    """
    scale = float(fitting_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("fitting scale must be positive finite")
    if prediction.ndim != 2 or prediction.shape != target.shape or min(prediction.shape) <= 0:
        raise ValueError("prediction and target must share nonempty [P,Q]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("molecular means must be finite")
    return (prediction - target).square().mean() / scale
