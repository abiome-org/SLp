"""Fixed fitting-only prior forecast diagnostic; never changes a trained model."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/slp11-transition/k562-essential-count-latent-state-seed731-v1"
RIDGE = ROOT / "results/slp11-transition/k562-essential-count-anchored-static-ridge-seed731-v1"
OUT = ROOT / "results/slp11-transition/k562-essential-count-prior-fitting-audit-v2"
PREDECESSOR = ROOT / "results/slp11-transition/k562-essential-count-prior-fitting-audit-v1/protocol.json"
FREEZE = ARTIFACT / "FROZEN-BEFORE-DEVELOPMENT-V2.json"
MOMENTS = ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1/fitting-action-moments.npz"
STATIC = ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz"
PINS = {
    MOMENTS: "a1f44a15a42c5b56e4ce897fde6ebba97298fc296105c6c870ee0e740331694e",
    STATIC: "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659",
    RIDGE / "model.npz": "dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3",
    RIDGE / "source/count_static_ridge.py": "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
    ARTIFACT / "source/runner.py": "9d6668ceb61a3bb0b9dc540a42430b523632b86ddcf547ec2175bfb2fe155920",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main():
    if (OUT / "report.json").exists():
        raise FileExistsError("immutable fitting audit already complete")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "hypothesis": "The frozen count-state prior learns fitting intervention aggregate means beyond the fitting anchored mean; this distinguishes failed fit from failed held-gene transfer.",
        "rule": "Descriptive audit only: report all metrics without selecting or changing a model. Compare prior MSE with frozen final-fitting ridge and mean; report total expected CP10k quantiles to diagnose unconstrained library mass.",
        "population": "128 fitting action genes selected by SHA256(slp11-count-prior-fit-audit-v1|ENSG) ordering, before reading count moments",
        "endpoint": "ln1p(equal-cell mean CP10k), all 8563 queries, gene GEM composition matched to training cells",
        "cpuThreads": 2, "maxSeconds": 600,
        "developmentRead": False, "testRead": False,
        "pins": {str(path.relative_to(ROOT)): value for path, value in PINS.items()},
        "scriptSha256": digest(Path(__file__)),
        "predecessorProtocolSha256": digest(PREDECESSOR),
        "executionAmendment": "Accept the explicit v2 numerical replay freeze instead of the original failed absolute-tolerance freeze filename; same checkpoint, fitting population, endpoint and diagnostic, no refit or model choice.",
    }
    if (OUT / "protocol.json").exists():
        if json.loads((OUT / "protocol.json").read_text()) != protocol:
            raise ValueError("frozen audit protocol changed")
    else:
        write(OUT / "protocol.json", protocol)
    if "--prepare" in sys.argv:
        return
    if not FREEZE.exists():
        raise RuntimeError("wait for final checkpoint freeze")
    frozen = json.loads(FREEZE.read_text())
    if (frozen["modelSha256"] != digest(ARTIFACT / "model.safetensors")
            or not frozen["isolatedCpuVerification"]["passes"]):
        raise ValueError("model differs from replay-verified final checkpoint")
    started = time.perf_counter()
    torch.set_num_threads(2)
    for path, expected in PINS.items():
        if digest(path) != expected:
            raise ValueError(f"input mismatch: {path}")
    with np.load(MOMENTS, allow_pickle=False) as values:
        genes = values["action_ids"].astype(str)
        selected = np.asarray(sorted(range(len(genes)), key=lambda i: hashlib.sha256(
            ("slp11-count-prior-fit-audit-v1|" + genes[i]).encode()).digest())[:128])
        # Row selection is fixed from identities before quantitative members open.
        truth = np.log1p(values["cp10k_sum"][selected] / values["cell_count"][selected, None])
        gem_counts = values["gem_cell_count"][selected]
        query_ids = values["query_ids"].astype(str)
    static, baseline = load(STATIC), load(RIDGE / "model.npz")
    lookup = {key: i for i, key in enumerate(static["entity_id"].astype(str))}
    features = static["feature_values"][[lookup[key] for key in genes[selected]]]
    core = module(RIDGE / "source/count_static_ridge.py", "audit_count_ridge")
    score = module(ARTIFACT / "source/runner.py", "audit_count_score").profile_metrics
    inference = module(ARTIFACT / "source/inference.py", "audit_count_inference")
    predictor = inference.Predictor(ARTIFACT, device="cpu")
    if not np.array_equal(query_ids, predictor.query_ids):
        raise ValueError("fitting query axis mismatch")
    with threadpool_limits(2), torch.no_grad():
        anchor = core.control_anchor(baseline["basal_rate"], gem_counts)
        ridge = core.absolute_prediction(anchor, core.predict_residual(baseline, features, str(baseline["selected_alpha"])))
        mean = core.absolute_prediction(anchor, np.broadcast_to(baseline["target_mean"], anchor.shape))
        pieces = []
        for left in range(0, len(features), 8):
            if time.perf_counter() - started > 600:
                raise TimeoutError("fitting audit exceeded 600 seconds")
            pieces.append(predictor.predict(features[left:left+8], gem_counts[left:left+8])["mean_cp10k"])
        cp10k = np.concatenate(pieces)
    report = {
        "protocolSha256": digest(OUT / "protocol.json"),
        "modelSha256": digest(ARTIFACT / "model.safetensors"),
        "seconds": time.perf_counter() - started,
        "fittingGenes": genes[selected].tolist(),
        "metrics": {name: score(value, truth, anchor) for name, value in
                    (("prior", np.log1p(cp10k)), ("ridge", ridge), ("mean", mean), ("control", anchor))},
        "priorTotalCp10kQuantiles": np.quantile(cp10k.sum(1), [0, .25, .5, .75, 1]).tolist(),
        "observedTotalCp10k": 10000,
        "developmentRead": False, "testRead": False,
    }
    write(OUT / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "fittingGenes"}, allow_nan=False))


if __name__ == "__main__":
    main()
