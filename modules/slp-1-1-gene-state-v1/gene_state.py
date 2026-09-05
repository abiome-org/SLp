"""Explicit per-gene molecular state core with sparse local message passing.

The module consumes caller-provided stable-universe tensors. Gene identity is
represented only by positions in those tensors and the supplied graph; there
is no learned identity embedding or internal gene vocabulary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    static_features: int
    state: int = 16
    transition_hidden: int = 64
    decoder_hidden: int = 32
    message_steps: int = 2
    dropout: float = 0.0


def _mlp(input_features: int, hidden: int, output_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_features, hidden),
        nn.GELU(),
        nn.Linear(hidden, output_features),
    )


def _coalesced_row_normalized(adjacency: torch.Tensor, nodes: int, like: torch.Tensor) -> torch.Tensor:
    if adjacency.layout not in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        raise ValueError("adjacency must use a torch sparse layout")
    if adjacency.shape != (nodes, nodes):
        raise ValueError("adjacency must be sparse [N,N]")
    if adjacency.device != like.device or adjacency.dtype != like.dtype:
        raise ValueError("adjacency and node tensors must share device and dtype")
    adjacency = adjacency.to_sparse_coo().coalesce()
    weights = adjacency.values()
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("adjacency weights must be finite and nonnegative")
    row_sum = torch.zeros(nodes, device=like.device, dtype=like.dtype)
    row_sum.scatter_add_(0, adjacency.indices()[0], weights)
    nonempty = row_sum > 0
    if nonempty.any() and not torch.allclose(
        row_sum[nonempty], torch.ones_like(row_sum[nonempty]), rtol=1e-5, atol=1e-6,
    ):
        raise ValueError("each nonempty adjacency row must sum to one")
    return adjacency


def _sparse_node_messages(adjacency: torch.Tensor, node_values: torch.Tensor) -> torch.Tensor:
    """Compute adjacency @ values using one sparse MM and no edge expansion."""
    batch, nodes, state = node_values.shape
    flat = node_values.permute(1, 0, 2).reshape(nodes, batch * state)
    result = torch.sparse.mm(adjacency, flat)
    return result.reshape(nodes, batch, state).permute(1, 0, 2)


class GeneStateCore(nn.Module):
    """Control-anchored global and graph-local intervention state."""

    def __init__(self, config: Config):
        super().__init__()
        if min(
            config.static_features,
            config.state,
            config.transition_hidden,
            config.decoder_hidden,
        ) <= 0:
            raise ValueError("all dimensions must be positive")
        if config.message_steps != 2:
            raise ValueError("gene-state-v1 fixes exactly two message steps")
        if config.dropout != 0:
            raise ValueError("gene-state-v1 fixes dropout at zero for deterministic differences")
        self.config = config
        self.static_linear = nn.Linear(config.static_features, config.state)
        self.value_linear = nn.Linear(1, config.state, bias=False)
        self.observed_flag_linear = nn.Linear(1, config.state, bias=False)
        self.global_delta_mlp = _mlp(
            config.state * 2, config.transition_hidden, config.state,
        )
        self.local_initial_mlp = _mlp(
            config.state * 2, config.transition_hidden, config.state,
        )
        self.message_update_mlp = _mlp(
            config.state * 3, config.transition_hidden, config.state,
        )
        self.decoder = _mlp(
            config.state * 3, config.decoder_hidden, 1,
        )
        nn.init.normal_(self.decoder[-1].weight, std=0.002)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode(
        self,
        static_gene_features: torch.Tensor,
        basal_rna: torch.Tensor,
        basal_observed: torch.Tensor,
        action_strength: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode basal and intervention states on one fixed gene universe.

        ``adjacency[i, j]`` is the normalized weight by which node ``j`` sends
        a message to node ``i``. Rows without edges may sum to zero.
        """
        if static_gene_features.ndim != 2 or static_gene_features.shape[1] != self.config.static_features:
            raise ValueError("static gene features must be [N,F]")
        nodes = len(static_gene_features)
        if basal_rna.ndim != 2 or basal_rna.shape[1] != nodes:
            raise ValueError("basal RNA must be [B,N]")
        if basal_observed.shape != basal_rna.shape or basal_observed.dtype != torch.bool:
            raise ValueError("basal observed mask must be Boolean [B,N]")
        if action_strength.shape != basal_rna.shape:
            raise ValueError("action strength must align with basal RNA [B,N]")
        if not (
            static_gene_features.device
            == basal_rna.device
            == basal_observed.device
            == action_strength.device
        ):
            raise ValueError("all dense inputs must share one device")
        if not (static_gene_features.dtype == basal_rna.dtype == action_strength.dtype):
            raise ValueError("static, basal and action tensors must share one floating dtype")
        if not torch.is_floating_point(static_gene_features):
            raise ValueError("state inputs must be floating point")
        if not torch.isfinite(static_gene_features).all() or not torch.isfinite(action_strength).all():
            raise ValueError("static features and action strengths must be finite")
        safe_basal = torch.where(basal_observed, basal_rna, 0.0)
        if not torch.isfinite(safe_basal).all():
            raise ValueError("observed basal RNA values must be finite")
        observed_count = basal_observed.sum(1)
        if not (observed_count > 0).all():
            raise ValueError("every record requires at least one observed basal gene")
        adjacency = _coalesced_row_normalized(adjacency, nodes, static_gene_features)

        # This is the sole static feature projection in an encode call. Its
        # result is reused by basal, global-action, local-action and query paths.
        static_state = self.static_linear(static_gene_features)
        basal_node_state = torch.nn.functional.gelu(
            static_state[None]
            + self.value_linear(safe_basal[..., None])
            + self.observed_flag_linear(basal_observed[..., None].to(basal_rna.dtype))
        )
        global_basal_state = (
            (basal_node_state * basal_observed[..., None]).sum(1)
            / observed_count[:, None]
        )
        global_action = torch.einsum("bn,ns->bs", action_strength, static_state)
        has_action = (action_strength != 0).any(1)
        raw_global_delta = self.global_delta_mlp(
            torch.cat((global_basal_state, global_action), dim=-1),
        )
        global_delta = torch.where(has_action[:, None], raw_global_delta, 0.0)

        local_proposal = self.local_initial_mlp(
            torch.cat((basal_node_state, static_state[None].expand(len(basal_rna), -1, -1)), dim=-1),
        )
        initial_local_delta = action_strength[..., None] * local_proposal
        local_delta = initial_local_delta
        zeros = torch.zeros_like(local_delta)
        null_update = self.message_update_mlp(
            torch.cat((zeros, zeros, basal_node_state), dim=-1),
        )
        for _ in range(self.config.message_steps):
            message = _sparse_node_messages(adjacency, local_delta)
            update = self.message_update_mlp(
                torch.cat((local_delta, message, basal_node_state), dim=-1),
            ) - null_update
            local_delta = local_delta + update

        return {
            "static_state": static_state,
            "basal_node_state": basal_node_state,
            "global_basal_state": global_basal_state,
            "global_action": global_action,
            "global_delta": global_delta,
            "global_state": global_basal_state + global_delta,
            "initial_local_delta": initial_local_delta,
            "local_delta": local_delta,
            "local_state": basal_node_state + local_delta,
            "has_action": has_action,
        }

    def _decode(
        self,
        local_state: torch.Tensor,
        global_state: torch.Tensor,
        query_static_state: torch.Tensor,
    ) -> torch.Tensor:
        batch, queries, _ = local_state.shape
        decoder_input = torch.cat(
            (
                local_state,
                global_state[:, None].expand(-1, queries, -1),
                query_static_state[None].expand(batch, -1, -1),
            ),
            dim=-1,
        )
        return self.decoder(decoder_input).squeeze(-1)

    def observe(
        self,
        encoded: dict[str, torch.Tensor],
        query_node_indices: torch.Tensor,
        control_mean: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode physical RNA queries selected by caller-provided node indices."""
        if query_node_indices.ndim != 1 or query_node_indices.dtype != torch.long:
            raise ValueError("query node indices must be int64 [Q]")
        static_state = encoded["static_state"]
        if query_node_indices.device != static_state.device:
            raise ValueError("query indices and encoded state must share one device")
        if (query_node_indices < 0).any() or (query_node_indices >= len(static_state)).any():
            raise ValueError("query node index outside graph universe")
        batch, queries = len(encoded["global_state"]), len(query_node_indices)
        if control_mean.shape != (batch, queries) or not torch.isfinite(control_mean).all():
            raise ValueError("control mean must be finite [B,Q]")
        if amplitude.shape != (queries,) or not torch.isfinite(amplitude).all() or not (amplitude > 0).all():
            raise ValueError("amplitude must be finite positive [Q]")
        if control_mean.device != static_state.device or amplitude.device != static_state.device:
            raise ValueError("observation inputs and encoded state must share one device")
        if control_mean.dtype != static_state.dtype or amplitude.dtype != static_state.dtype:
            raise ValueError("observation inputs and encoded state must share one dtype")
        query_static = static_state.index_select(0, query_node_indices)
        after = self._decode(
            encoded["local_state"].index_select(1, query_node_indices),
            encoded["global_state"],
            query_static,
        )
        before = self._decode(
            encoded["basal_node_state"].index_select(1, query_node_indices),
            encoded["global_basal_state"],
            query_static,
        )
        delta = (after - before) * amplitude[None]
        delta = torch.where(encoded["has_action"][:, None], delta, 0.0)
        return {"mean": control_mean + delta, "delta": delta}


def profile_synthetic_cuda(
    *,
    nodes: int = 24_000,
    batch: int = 32,
    static_features: int = 577,
    query_count: int = 1_024,
    repeats: int = 3,
    seed: int = 731,
) -> dict[str, float | int]:
    """Explicit opt-in CUDA profile; never called during import or construction."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the synthetic profile")
    if min(nodes, batch, static_features, query_count, repeats) <= 0 or query_count > nodes:
        raise ValueError("invalid synthetic profile dimensions")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    model = GeneStateCore(Config(static_features=static_features)).to(device)
    static = torch.randn(nodes, static_features, device=device)
    basal = torch.randn(batch, nodes, device=device)
    observed = torch.ones(batch, nodes, device=device, dtype=torch.bool)
    strength = torch.zeros(batch, nodes, device=device)
    strength[torch.arange(batch, device=device), torch.arange(batch, device=device) % nodes] = 1.0
    rows = torch.arange(nodes, device=device)
    columns = (rows - 1) % nodes
    with torch.sparse.check_sparse_tensor_invariants():
        adjacency = torch.sparse_coo_tensor(
            torch.stack((rows, columns)), torch.ones(nodes, device=device), (nodes, nodes),
        ).coalesce()
    queries = torch.arange(query_count, device=device)
    control = torch.zeros(batch, query_count, device=device)
    amplitude = torch.ones(query_count, device=device)
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(repeats):
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        encoded = model.encode(static, basal, observed, strength, adjacency)
        prediction = model.observe(encoded, queries, control, amplitude)["mean"]
        prediction.square().mean().backward()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
    return {
        "nodes": nodes,
        "batch": batch,
        "static_features": static_features,
        "query_count": query_count,
        "repeats": repeats,
        "mean_forward_backward_seconds": sum(elapsed) / len(elapsed),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
