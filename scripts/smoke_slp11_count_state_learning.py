"""Bounded independent Poisson-teacher smoke for the untrained count core."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "modules/slp-1-1-count-latent-state-v1/count_latent_state.py"
OUTPUT = ROOT / "results/slp11-transition/count-state-independent-teacher-smoke-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(CORE) != "75df347a82151074c0ce6f4c732106e70ed17126aff07d017294894421d30bac":
        raise ValueError("core drift")
    OUTPUT.mkdir(parents=True)
    protocol = {
        "schema": "slp.count-state-independent-teacher-smoke/v1",
        "teacher": "Independent Poisson query counts with rate basal_q*exp(.7*x*q0). 64 fitting scalar x grid points in [-1,1]; 63 held midpoints. No learned latent teacher or model decoder used.",
        "role": "Synthetic training/inference check, not biology or out-of-distribution evidence.",
        "objective": "Original beta=1 normalized ELBO; B64 half empty controls, half targeting, CPU2, seed731,500updates,AdamWlr.003/decay.01,clip1,finalonly.",
        "gate": "Held midpoint analytic-prior rate MSE improves at least30% over fixed control rates; empty means exact; finite losses/gradients.",
        "maximumSeconds": 90,
        "sourceSha256": sha(CORE), "runnerSha256": sha(Path(__file__)),
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    shutil.copy2(CORE, OUTPUT / CORE.name)
    shutil.copy2(Path(__file__), OUTPUT / "runner.py")
    spec = importlib.util.spec_from_file_location("count_smoke_core", CORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    torch.manual_seed(731)
    torch.use_deterministic_algorithms(True)
    model = module.CountLatentState(module.Config(4, hidden_dim=32, state_dim=4, key_dim=8, dropout=0.))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    q0 = torch.linspace(-1., 1., 12)
    query = torch.stack((q0, q0.square(), q0.sin(), torch.ones_like(q0)), -1)
    basal = torch.linspace(8., 20., 12)[None]
    x_fit = torch.linspace(-1., 1., 64)
    x_held = (x_fit[:-1] + x_fit[1:]) / 2

    def features(x):
        return torch.stack((x, x.square(), x.sin(), torch.ones_like(x)), -1)[:, None]

    def teacher(x):
        return basal * (.7 * x[:, None] * q0[None]).exp()

    observed = torch.ones(64, 12, dtype=torch.bool)
    library = torch.full((64,), 10000.)
    mask = torch.zeros(64, 1, dtype=torch.bool)
    mask[32:] = True
    losses = []
    started = time.perf_counter()
    for step in range(500):
        if time.perf_counter() - started > 90:
            raise TimeoutError("synthetic smoke exceeded cap")
        x = x_fit[torch.randint(64, (64,))]
        x[:32] = 0
        counts = torch.poisson(teacher(x))
        context = model.encode_context(query, basal, torch.ones_like(basal, dtype=torch.bool))
        prior = model.prior_from_context(features(x), mask, context.expand(64, -1))
        output = model.elbo(counts, observed, library, query, basal.expand(64, -1), prior)
        loss = output["loss_per_cell"].mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite synthetic loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0 or step == 499:
            losses.append({"step": step + 1, "loss": float(loss.detach())})
    model.eval()
    with torch.no_grad():
        context = model.encode_context(query, basal, torch.ones_like(basal, dtype=torch.bool)).expand(len(x_held), -1)
        active = torch.ones(len(x_held), 1, dtype=torch.bool)
        prior = model.prior_from_context(features(x_held), active, context)
        prediction = model.population_mean(prior, query, basal.expand(len(x_held), -1))
        empty = model.prior_from_context(features(x_held), ~active, context)
        empty_mean = model.population_mean(empty, query, basal.expand(len(x_held), -1))
        truth = teacher(x_held)
        mse = float((prediction - truth).square().mean())
        baseline = float((basal - truth).square().mean())
        exact_empty = torch.equal(empty_mean, basal.expand_as(empty_mean))
    report = {"schema": protocol["schema"], "protocolSha256": sha(OUTPUT / "protocol.json"),
              "seconds": time.perf_counter() - started, "heldPriorRateMse": mse,
              "controlRateMse": baseline, "relativeImprovement": 1 - mse / baseline,
              "emptyMeanBitExact": exact_empty, "passes": bool(exact_empty and mse <= .7 * baseline),
              "lossTrace": losses, "biologicalDataAccessed": False}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
