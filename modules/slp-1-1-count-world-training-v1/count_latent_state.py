"""Control-anchored Gaussian molecular states with NB count observations.

This is an experimental conditional population model, not a fitted artifact.
Library exposure belongs to the observation model, never the state prior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    feature_dim: int
    hidden_dim: int = 128
    state_dim: int = 32
    key_dim: int = 64
    dropout: float = .1


def _mlp(inputs: int, hidden: int, outputs: int, dropout: float = 0.):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.LayerNorm(hidden),
                         nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, outputs))


def negative_binomial_log_prob(counts, log_mean, dispersion):
    """NB2 log mass using log-space means, including observed count zeros."""
    log_theta = dispersion.log()
    log_total = torch.logaddexp(log_theta, log_mean)
    return (torch.lgamma(counts + dispersion) - torch.lgamma(dispersion)
            - torch.lgamma(counts + 1) + dispersion * (log_theta - log_total)
            + counts * (log_mean - log_total))


def diagonal_gaussian_kl(q_mean, q_logvar, p_mean, p_logvar):
    return .5 * (p_logvar - q_logvar + (q_logvar - p_logvar).exp()
                 + (q_mean - p_mean).square() * (-p_logvar).exp() - 1).sum(-1)


class CountLatentState(nn.Module):
    """Feature-defined latent prior, variational encoder and queried count head.

    A positive externally estimated control rate fixes the empty-intervention
    population mean. Shared latent noise induces cross-query dependence.
    ``basal_rate`` is molecules per 10,000 source-denominator molecules;
    ``library`` is that denominator's observed count. Neither is a target label.
    """

    def __init__(self, config: Config):
        super().__init__()
        if min(config.feature_dim, config.hidden_dim, config.state_dim, config.key_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        if not 0 <= config.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.config = config
        f, h, d = config.feature_dim, config.hidden_dim, config.state_dim
        self.context_encoder = _mlp(f + 1, h, h)
        self.action_encoder = _mlp(f, h, h, config.dropout)
        self.control_prior = nn.Linear(h, 2 * d)
        self.intervention_prior = _mlp(2 * h, h, 2 * d, config.dropout)
        self.cell_keys = _mlp(f, h, config.key_dim)
        self.posterior = _mlp(config.key_dim + 2 * h, h, 2 * d, config.dropout)
        self.query_loading = _mlp(f, h, d)
        self.query_dispersion = _mlp(f, h, 1)

    def _features(self, values, label):
        if values.ndim != 2 or values.shape[1] != self.config.feature_dim:
            raise ValueError(f"{label} must be [Q,F]")
        if not torch.isfinite(values).all():
            raise ValueError(f"{label} must be finite")

    def prior(self, actions, action_mask, basal_features, basal_rate, basal_mask):
        """Encode a distribution without seeing perturbed counts or exposure."""
        context = self.encode_context(basal_features, basal_rate, basal_mask)
        return self.prior_from_context(actions, action_mask, context)

    def encode_context(self, basal_features, basal_rate, basal_mask):
        """Encode distinct controls once per optimizer step, then index cells."""
        self._features(basal_features, "basal features")
        if basal_rate.ndim != 2:
            raise ValueError("basal rates must be [C,Q]")
        if basal_rate.shape[1] != len(basal_features) or basal_mask.shape != basal_rate.shape:
            raise ValueError("basal rates and mask must be [C,Q]")
        if basal_mask.dtype != torch.bool or not basal_mask.any(-1).all():
            raise ValueError("each context requires observed basal support")
        safe_rate = torch.where(basal_mask, basal_rate, 0.)
        if not torch.isfinite(safe_rate).all() or (safe_rate < 0).any():
            raise ValueError("observed basal rates must be finite and nonnegative")
        # Factor the first linear map: never materialize repeated [C,Q,F].
        first = self.context_encoder[0]
        feature_term = F.linear(basal_features, first.weight[:, :-1], first.bias)
        context_tokens = feature_term[None] + safe_rate.log1p()[..., None] * first.weight[:, -1]
        for layer in self.context_encoder[1:]:
            context_tokens = layer(context_tokens)
        return (context_tokens * basal_mask[..., None]).sum(1) / basal_mask.sum(1)[:, None]

    def prior_from_context(self, actions, action_mask, context):
        """Use learned control embeddings indexed by the caller's cell contexts."""
        if actions.ndim != 3 or actions.shape[2] != self.config.feature_dim:
            raise ValueError("actions must be [B,A,F]")
        b = len(actions)
        if action_mask.shape != actions.shape[:2] or action_mask.dtype != torch.bool:
            raise ValueError("action mask must be Boolean [B,A]")
        if context.shape != (b, self.config.hidden_dim) or not torch.isfinite(context).all():
            raise ValueError("indexed context must be finite [B,H]")
        safe_action = torch.where(action_mask[..., None], actions, 0.)
        if not torch.isfinite(safe_action).all():
            raise ValueError("observed actions must be finite")
        action = (self.action_encoder(safe_action) * action_mask[..., None]).sum(1)
        base_mean, base_logvar = self.control_prior(context).chunk(2, -1)
        base_logvar = base_logvar.clamp(-8., 4.)
        delta_mean, delta_logvar = self.intervention_prior(torch.cat((context, action), -1)).chunk(2, -1)
        active = action_mask.any(1)[:, None]
        mean = base_mean + torch.where(active, delta_mean, 0.)
        logvar = torch.where(active, (base_logvar + delta_logvar).clamp(-8., 4.), base_logvar)
        return {"mean": mean, "logvar": logvar, "control_mean": base_mean,
                "control_logvar": base_logvar, "context": context, "action": action}

    def observation_parameters(self, query_features):
        self._features(query_features, "query features")
        return (self.query_loading(query_features) / math.sqrt(self.config.state_dim),
                F.softplus(self.query_dispersion(query_features)).squeeze(-1) + 1e-4)

    @staticmethod
    def _observation_inputs(basal_rate, library, batch, queries):
        if basal_rate.shape != (batch, queries) or library.shape != (batch,):
            raise ValueError("observation rates [B,Q] and library [B] required")
        if (not torch.isfinite(basal_rate).all() or not torch.isfinite(library).all()
                or not (basal_rate > 0).all() or not (library > 0).all()):
            raise ValueError("observation rates and library must be finite and positive")

    def log_rate(self, state, prior, query_features, basal_rate):
        """Conditional CP10k log rate with analytic control moment correction."""
        loading, dispersion = self.observation_parameters(query_features)
        if state.shape != prior["mean"].shape or basal_rate.shape != (len(state), len(loading)):
            raise ValueError("state/rate shape mismatch")
        if not torch.isfinite(basal_rate).all() or not (basal_rate > 0).all():
            raise ValueError("decoder requires externally smoothed positive basal rates")
        correction = .5 * prior["control_logvar"].exp() @ loading.square().T
        log_rate = basal_rate.log() + (state - prior["control_mean"]) @ loading.T - correction
        return log_rate, dispersion

    def population_mean(self, prior, query_features, basal_rate):
        """Exact prior expectation in CP10k units; does not require exposure."""
        loading, _ = self.observation_parameters(query_features)
        if basal_rate.shape != (len(prior["mean"]), len(loading)):
            raise ValueError("basal rate shape mismatch")
        if not torch.isfinite(basal_rate).all() or not (basal_rate > 0).all():
            raise ValueError("population means require positive finite basal rates")
        log_ratio = ((prior["mean"] - prior["control_mean"]) @ loading.T
                     + .5 * (prior["logvar"].exp() - prior["control_logvar"].exp()) @ loading.square().T)
        return basal_rate * log_ratio.exp()

    def encode_cells(self, counts, observed, library, query_features, basal_rate, prior):
        self._features(query_features, "query features")
        self._observation_inputs(basal_rate, library, len(counts), len(query_features))
        if counts.shape != basal_rate.shape or observed.shape != counts.shape or observed.dtype != torch.bool:
            raise ValueError("count observations must be [B,Q] with Boolean mask")
        if not observed.any(-1).all():
            raise ValueError("each cell requires at least one observed query")
        safe = torch.where(observed, counts, 0.)
        if not torch.isfinite(safe).all() or (safe < 0).any() or not torch.equal(safe, safe.round()):
            raise ValueError("observed counts must be finite nonnegative integers")
        if (safe.sum(-1) > library).any():
            raise ValueError("source library cannot be smaller than observed query counts")
        residual = torch.where(observed, (safe * (10000. / library[:, None])).log1p() - basal_rate.log1p(), 0.)
        keys = self.cell_keys(query_features)
        pooled = residual @ keys / observed.sum(-1, keepdim=True).sqrt()
        delta_mean, delta_logvar = self.posterior(torch.cat((pooled, prior["context"], prior["action"]), -1)).chunk(2, -1)
        return {"mean": prior["mean"] + delta_mean,
                "logvar": (prior["logvar"] + delta_logvar).clamp(-8., 4.)}

    def elbo(self, counts, observed, library, query_features, basal_rate, prior, *, epsilon=None):
        """Per-cell negative ELBO divided by observed query count, beta=1.

        ``epsilon`` permits reproducible paired checks; otherwise one posterior
        Monte Carlo sample is used. Report reconstruction and KL separately.
        """
        posterior = self.encode_cells(counts, observed, library, query_features, basal_rate, prior)
        if epsilon is None:
            epsilon = torch.randn_like(posterior["mean"])
        if epsilon.shape != posterior["mean"].shape or not torch.isfinite(epsilon).all():
            raise ValueError("posterior noise shape or finiteness mismatch")
        state = posterior["mean"] + (.5 * posterior["logvar"]).exp() * epsilon
        log_rate, dispersion = self.log_rate(state, prior, query_features, basal_rate)
        log_mean = log_rate + (library / 10000.).log()[:, None]
        safe = torch.where(observed, counts, 0.)
        log_prob = negative_binomial_log_prob(safe, log_mean, dispersion)
        nll = -torch.where(observed, log_prob, 0.).sum(-1)
        kl = diagonal_gaussian_kl(posterior["mean"], posterior["logvar"], prior["mean"], prior["logvar"])
        support = observed.sum(-1)
        loss = (nll + kl) / support
        return {"loss_per_cell": loss, "reconstruction_per_query": nll / support,
                "kl_per_cell": kl, "posterior": posterior, "log_mean": log_mean}
