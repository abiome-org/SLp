"""Assay-specific decoder and intervention-composition architectures.

Every module here retains its historical parameter layout so existing decoder
and endpoint checkpoints remain loadable.
"""

import torch
from torch import nn


class ResidualEndpoint(nn.Module):
    def __init__(self, world, legacy_decoder, residual_dim=64):
        super().__init__()
        self.world = world.requires_grad_(False)
        self.legacy_decoder = legacy_decoder.requires_grad_(False)
        latent = world.gene.out_features
        self.residual_decoder = nn.Sequential(
            nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, residual_dim)
        )

    def forward(self, action, second=None, context=None, context_state=None, sample=None):
        mean, log_std = self.world.transition(action, second, context=context, context_state=context_state)
        state = mean if sample is None else mean + log_std.exp() * sample
        return torch.cat((self.legacy_decoder(mean), self.residual_decoder(state)), 1), mean, log_std


class SourceEndpoint(nn.Module):
    def __init__(self, sources=5, latent=128, state=32):
        super().__init__()
        self.decoders = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, state))
            for _ in range(sources)
        )
        for head in self.decoders:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, latent, source):
        all_state = torch.stack([head(latent) for head in self.decoders], 1)
        return all_state[torch.arange(len(latent), device=latent.device), source]


def dependency_landscape_head(latent=128, state=64):
    head = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, state))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class DependencyActionAdapter(nn.Module):
    def __init__(self, state_dim=1816, latent=128, target=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, 64, bias=False), nn.GELU(), nn.Linear(64, latent, bias=False)
        )
        nn.init.zeros_(self.adapter[-1].weight)
        self.decoder = dependency_landscape_head(latent, target)

    def action(self, state, base):
        return base + self.adapter(state)


def pair_transition_adapter(latent=128, bottleneck=16):
    module = nn.Sequential(
        nn.LayerNorm(latent), nn.Linear(latent, bottleneck), nn.GELU(), nn.Linear(bottleneck, latent, bias=False)
    )
    nn.init.zeros_(module[-1].weight)
    return module


class DiagonalActionCalibration(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(latent))

    def forward(self, action):
        return action * torch.exp(.1 * torch.tanh(self.log_scale))


class LowRankActionRotation(nn.Module):
    def __init__(self, latent=128, rank=8):
        super().__init__()
        self.down = nn.Linear(latent, rank, bias=False)
        self.up = nn.Linear(rank, latent, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, action):
        delta = torch.tanh(self.up(nn.functional.gelu(self.down(nn.functional.layer_norm(action, (action.shape[-1],))))))
        return action + .1 * action.square().mean(1, keepdim=True).sqrt() * delta


class SymmetricPairFusion(nn.Module):
    def __init__(self, latent=128, relations=6, rank=8):
        super().__init__()
        self.down = nn.Linear(3 * latent + relations, rank, bias=False)
        self.up = nn.Linear(rank, 96, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, a, b, joint, relation):
        features = torch.cat((a * b, (a - b).abs(), joint, relation), 1)
        return .25 * torch.tanh(self.up(nn.functional.gelu(self.down(nn.functional.layer_norm(features, (features.shape[-1],))))))


class SymmetricPairLatentFusion(nn.Module):
    def __init__(self, latent=128, relations=6, rank=8):
        super().__init__()
        self.down = nn.Linear(3 * latent + relations, rank, bias=False)
        self.up = nn.Linear(rank, latent, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, a, b, joint, relation):
        features = torch.cat((a * b, (a - b).abs(), joint, relation), 1)
        delta = torch.tanh(self.up(nn.functional.gelu(self.down(nn.functional.layer_norm(features, (features.shape[-1],))))))
        return .1 * joint.square().mean(1, keepdim=True).sqrt() * delta


class MultiActionComposition(nn.Module):
    def __init__(self, latent=128, rank=16, max_actions=8):
        super().__init__()
        width = 3 * latent + max_actions
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, latent, bias=False)
        self.max_actions = max_actions
        nn.init.zeros_(self.up.weight)

    def forward(self, actions, mask, joint):
        weight = mask[..., None]
        count = weight.sum(1).clamp_min(1)
        mean = (actions * weight).sum(1) / count
        variance = (actions.square() * weight).sum(1) / count - mean.square()
        cardinality = nn.functional.one_hot(mask.sum(1).clamp(1, self.max_actions) - 1, self.max_actions).float()
        delta = torch.tanh(self.up(nn.functional.gelu(self.down(self.norm(torch.cat((joint, mean, variance, cardinality), 1))))))
        return .1 * joint.square().mean(1, keepdim=True).sqrt() * delta


class GraphHead(nn.Module):
    def __init__(self, dim, views=6):
        super().__init__()
        self.input = nn.Linear(dim, 256)
        self.skip = nn.Linear(256, 128)
        self.one = nn.ModuleList(nn.Linear(256, 256) for _ in range(views))
        self.two = nn.ModuleList(nn.Linear(256, 128) for _ in range(views))
        self.att = nn.Parameter(torch.zeros(views))
        self.scale = nn.Parameter(torch.tensor(2.3))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, features, graphs):
        base = nn.functional.gelu(self.input(features))
        states = []
        for graph, one, two in zip(graphs, self.one, self.two):
            states.append(two(torch.sparse.mm(graph, nn.functional.gelu(one(torch.sparse.mm(graph, base))))))
        return nn.functional.normalize(self.skip(base) + sum(weight * state for weight, state in zip(self.att.softmax(0), states)), dim=1)

    def score(self, embeddings, pairs):
        return (embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]]).sum(1) * self.scale.exp() + self.bias


class SymmetricHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encode = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 384), nn.GELU(), nn.Dropout(.1), nn.Linear(384, 128))
        self.score = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU(), nn.Dropout(.2), nn.Linear(256, 1))

    def forward(self, features, pairs):
        a, b = self.encode(features[pairs[:, 0]]), self.encode(features[pairs[:, 1]])
        return self.score(torch.cat(((a - b).abs(), a * b), 1)).squeeze(1)


class SymmetricQGIHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gene = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 64))
        self.pair = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, features, pairs):
        a, b = self.gene(features[pairs[:, 0]]), self.gene(features[pairs[:, 1]])
        return self.pair(torch.cat(((a - b).abs(), a * b), 1)).squeeze(1)
