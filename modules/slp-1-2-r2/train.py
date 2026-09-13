"""Finite, resumable training primitives. Research launch is deliberately separate."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F


def objective(output, batch):
    """Student-t(4) for admitted processed numeric scores; BCE for human SL.

    Raw counts require preprocessing/admission to their own likelihood and must
    not be mislabeled processed scores. Average coordinates within each sampled
    experimental unit; the sampler sets the family mixture, rather than a cell
    count or the number of requested output genes setting its importance.
    """
    target = batch["target"].float()
    valid = batch["query_mask"]
    binary = batch["query_kind"] == 4
    scale = output["log_scale"].exp()
    residual = (target - output["location"]) / scale
    numeric = output["log_scale"] + 2.5 * torch.log1p(residual.square() / 4.0)
    classification = F.binary_cross_entropy_with_logits(
        output["sl_logit"], target, reduction="none"
    )
    loss = torch.where(binary, classification, numeric)
    per_unit = (loss * valid).sum(1) / valid.sum(1).clamp_min(1)
    if not bool(valid.any(1).all()):
        raise ValueError("Empty measurement unit")
    return per_unit.mean()


def identity_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save_checkpoint(path, model, optimizer, sampler, *, update, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": "slp.r2-checkpoint/v1",
        "config": model.configuration(),
        "identity": identity,
        "identity_sha256": identity_digest(identity),
        "update": update,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def resume(path, model, optimizer, sampler, *, identity, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if (
        state.get("schema") != "slp.r2-checkpoint/v1"
        or state["identity_sha256"] != identity_digest(identity)
        or state["identity_sha256"] != identity_digest(state["identity"])
    ):
        raise ValueError("Resume data/fold/recipe identity mismatch")
    if state["config"] != model.configuration():
        raise ValueError("Resume model configuration mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    sampler.load_state_dict(state["sampler"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda_rng"]) != torch.cuda.device_count()
        ):
            raise ValueError("Exact resume requires the same accelerator RNG topology")
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
    random.setstate(state["python_rng"])
    return state["update"]


def optimize(
    model,
    sampler,
    builder,
    optimizer,
    *,
    updates,
    batch_size,
    max_seconds,
    device,
    identity,
    checkpoint_path,
    start_update=0,
    checkpoint_every=1000,
    learning_rate=2e-4,
    warmup=100,
    max_queries=128,
    gradient_clip=1.0,
    bf16=True,
    context_dropout=0.0,
    retain_updates=(),
):
    if (
        updates <= start_update
        or max_seconds <= 0
        or batch_size < 1
        or not 0 <= context_dropout < 1
    ):
        raise ValueError("A finite positive fitting budget is required")
    start = time.monotonic()
    model.train()
    history = []
    step = start_update

    def checkpoint(at):
        save_checkpoint(
            checkpoint_path, model, optimizer, sampler, update=at, identity=identity
        )
        if at in retain_updates:
            path = Path(checkpoint_path)
            retained = path.with_name(f"checkpoint-u{at:06d}.pt")
            if not retained.exists():
                os.link(path, retained)

    try:
        for step in range(start_update + 1, updates + 1):
            if time.monotonic() - start >= max_seconds:
                step -= 1
                break
            # If the repeat budget is exhausted mid-batch, do not fit a partial
            # batch under a different mixture or pretend the update happened.
            units = [sampler.draw(max_queries) for _ in range(batch_size)]
            batch = {k: v.to(device) for k, v in builder(units).items()}
            if context_dropout:
                keep = torch.rand(batch_size, device=device) >= context_dropout
                batch["context_known"] &= keep
                batch["context"] = torch.where(keep[:, None], batch["context"], 0.0)
                batch["observation_mask"] &= keep[:, None]
            warm = min(1.0, step / max(warmup, 1))
            progress = max(0.0, (step - warmup) / max(updates - warmup, 1))
            lr = (
                learning_rate
                * warm
                * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=bf16 and torch.device(device).type == "cuda",
            ):
                loss = objective(model(batch), batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            if step == start_update + 1 or step % 20 == 0:
                item = {
                    "update": step,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "elapsed_seconds": time.monotonic() - start,
                    "lr": lr,
                }
                print(json.dumps(item), flush=True)
                history.append(item)
            if step % checkpoint_every == 0:
                checkpoint(step)
    except StopIteration:
        step -= 1
    checkpoint(step)
    return {
        "completed_update": step,
        "requested_updates": updates,
        "elapsed_seconds": time.monotonic() - start,
        "history": history,
    }
