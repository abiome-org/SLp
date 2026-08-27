"""Checkpoint compatibility and provenance for the frozen v1 contract."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from .world import MODEL_VERSION, SLPredict, WorldModelConfig


def _world_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract a base world from either a world or residual-endpoint checkpoint."""

    if any(key.startswith("world.") for key in state_dict):
        return {key.removeprefix("world."): value for key, value in state_dict.items() if key.startswith("world.")}
    return dict(state_dict)


def world_config_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> WorldModelConfig:
    """Infer a v1 configuration without loading data or importing a runner."""

    world = _world_state_dict(state_dict)
    projection_keys = sorted(
        (key for key in world if key.startswith("proj.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if not projection_keys or "gene.weight" not in world or "cell.weight" not in world:
        raise ValueError("Checkpoint is not compatible with the v1 world-model contract.")
    layer_ids = {
        int(key.split(".")[2])
        for key in world
        if key.startswith("core.layers.") and key.split(".")[2].isdigit()
    }
    return WorldModelConfig(
        d=int(world["cls"].shape[0]),
        latent=int(world["gene.weight"].shape[0]),
        layers=len(layer_ids),
        contexts=int(world["cell.weight"].shape[0]),
        outcomes=int(world["outcome.weight"].shape[0]),
        state_dim=sum(int(world[key].shape[1]) for key in projection_keys),
        context_dim=int(world["context_proj.weight"].shape[1]) if "context_proj.weight" in world else 0,
    )


def load_world_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> SLPredict:
    """Load an immutable v1 world checkpoint without a data dependency."""

    state_dict = torch.load(Path(path), map_location="cpu", weights_only=True)
    world_state = _world_state_dict(state_dict)
    config = world_config_from_state_dict(world_state)
    model = SLPredict(**asdict(config)).to(device)
    model.load_state_dict(world_state, strict=strict)
    return model.eval()


def checkpoint_manifest(path: str | Path) -> dict[str, object]:
    """Return a portable, content-addressed manifest for a v1 checkpoint."""

    checkpoint = Path(path)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return {
        "model_version": MODEL_VERSION,
        "checkpoint": checkpoint.name,
        "sha256": digest,
        "config": asdict(world_config_from_state_dict(state_dict)),
    }


def write_checkpoint_manifest(path: str | Path, output: str | Path | None = None) -> Path:
    """Persist the model identity next to a checkpoint (or at `output`)."""

    checkpoint = Path(path)
    destination = Path(output) if output is not None else checkpoint.with_suffix(".manifest.json")
    destination.write_text(json.dumps(checkpoint_manifest(checkpoint), indent=2, sort_keys=True) + "\n")
    return destination
