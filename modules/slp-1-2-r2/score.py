"""Explicit, row-preserving SL scoring; inner selection and outer reporting differ."""

import argparse
import json
from pathlib import Path
import time
import torch
from artifact import export
from baselines import PairBaseline
from benchmarks import Benchmark
from data import Features, Scales, BatchBuilder, digest
from evaluate import binary_metrics
from model import Config, WorldModel
from train import identity_digest


def main(args):
    if args.scope == "outer" and not args.allow_outer_test:
        raise ValueError("Official outer scoring requires explicit --allow-outer-test")
    features = Features(args.features)
    benchmark = Benchmark(args.fold, features)
    fold = benchmark.fold(args.scope)
    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    identity = state["identity"]
    if state["identity_sha256"] != identity_digest(identity):
        raise ValueError("Checkpoint identity checksum mismatch")
    if (
        identity.get("phase") != "adapt"
        or identity.get("scope") != args.scope
        or identity.get("fold") != benchmark.identity
    ):
        raise ValueError("Scoring requires the matching adapted fold checkpoint")
    if identity["features_sha256"] != digest(Path(args.features) / "manifest.json"):
        raise ValueError("Scoring static inputs changed")
    if set(identity["human_fitted_intervention_genes"]) & fold.forbidden:
        raise ValueError("Scoring checkpoint has forbidden human exposure")
    c = state["config"]
    model = (
        PairBaseline(Config(**c["model"]), c["baseline"])
        if "baseline" in c
        else WorldModel(Config(**c))
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    if identity.get("basal_sha256") != digest(args.basal):
        raise ValueError(
            "Scoring basal observations changed or lack a training checksum"
        )
    basal = json.loads(Path(args.basal).read_text())
    vocabulary = identity["vocabulary"]
    if args.export:
        export(
            args.export,
            model,
            args.features,
            vocabulary=vocabulary,
            basal=basal,
            provenance=identity,
        )
    rows = benchmark.records(args.scope, "evaluate")
    builder = BatchBuilder(features, Scales([]), vocabulary, model.config, basal)
    scores = []
    started = time.monotonic()
    with torch.inference_mode():
        for lo in range(0, len(rows), args.batch_size):
            batch = {
                k: v.to(args.device)
                for k, v in builder(
                    [[r] for r in rows[lo : lo + args.batch_size]]
                ).items()
            }
            scores.extend(model(batch)["sl_logit"][:, 0].float().cpu().tolist())
    metrics = binary_metrics([r.value for r in rows], scores)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("w") as file:
        for row, score in zip(rows, scores, strict=True):
            file.write(
                json.dumps(
                    {
                        "record_id": row.record_id,
                        "source_locator": row.raw_locator,
                        "targets": row.targets,
                        "context": row.context,
                        "label": row.value,
                        "logit": score,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    report = {
        **metrics,
        "partition": args.scope,
        "benchmark": benchmark.spec["benchmark"],
        "fold": benchmark.spec["name"],
        "checkpoint": args.candidate,
        "checkpoint_sha256": digest(args.checkpoint),
        "forbidden_human_exposures": 0,
        "forbidden_genes": sorted(fold.forbidden),
        "identity_sha256": state["identity_sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "protocol_sha256": benchmark.identity,
        "source": {
            p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "predictions_sha256": digest(output / "predictions.jsonl"),
    }
    (output / "metrics.json").write_text(json.dumps(report, sort_keys=True))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "features", "fold", "basal", "output", "candidate"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--scope", choices=("inner", "outer"), required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--allow-outer-test", action="store_true")
    p.add_argument("--export")
    main(p.parse_args())
