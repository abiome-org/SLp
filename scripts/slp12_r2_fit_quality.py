"""Read-only training/inner SL comparison; never opens the outer partition."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch


def main(args):
    sys.path.insert(0, str(args.root))
    from baselines import PairBaseline
    from benchmarks import Benchmark, verify_training_labels
    from data import Features, Scales, BatchBuilder, digest
    from evaluate import binary_metrics
    from model import Config, WorldModel
    from train import identity_digest

    torch.set_num_threads(4)
    features = Features(args.root / "features")
    benchmark = Benchmark(args.root / "protocols/musl-s42-f0-fold.json", features)
    fold = benchmark.fold("inner")
    rows = {"fit": benchmark.records("inner", "fit"),
            "validation": benchmark.records("inner", "evaluate")}
    verify_training_labels(rows["fit"], fold)
    candidates = {
        "feature_mlp": args.root / "runs/musl-s42-f0/feature_mlp/adapt",
        "human8k-sl-only": args.root / "diagnosis/sl-only/human-u008000",
        "no-pretraining-sl-only": args.root / "diagnosis/sl-only/no-pretraining",
    }
    report = {"schema": "slp.r2-sl-fit-diagnostic/v1", "source_sha256": digest(__file__),
              "scope": "first inner fold fitting and validation only", "models": {}}
    for name, directory in candidates.items():
        if time.time() > args.deadline - 30:
            raise TimeoutError("Fitting diagnostic deadline")
        checkpoint = directory / "checkpoint.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        identity = saved["identity"]
        assert saved["identity_sha256"] == identity_digest(identity)
        assert identity["scope"] == "inner" and identity["fold"] == benchmark.identity
        assert not set(identity["human_fitted_intervention_genes"]) & fold.forbidden
        assert identity["basal_sha256"] == digest(directory / "basal.json")
        config = saved["config"]
        model = (PairBaseline(Config(**config["model"]), config["baseline"])
                 if "baseline" in config else WorldModel(Config(**config)))
        model.load_state_dict(saved["model"], strict=True)
        model.cuda().eval()
        builder = BatchBuilder(features, Scales([]), identity["vocabulary"], model.config,
                               json.loads((directory / "basal.json").read_text()))
        result = {"checkpoint_sha256": digest(checkpoint), "forbidden_human_exposures": 0}
        for partition, values in rows.items():
            scores = []
            for lo in range(0, len(values), 64):
                batch = {k: v.cuda() for k, v in builder([[r] for r in values[lo:lo+64]]).items()}
                with torch.inference_mode():
                    scores.extend(model(batch)["sl_logit"][:, 0].cpu().tolist())
            result[partition] = binary_metrics([r.value for r in values], scores)
        report["models"][name] = result
        args.output.write_text(json.dumps(report, indent=2))
        del model, saved
        torch.cuda.empty_cache()
    report["completed_at"] = time.time()
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/workspace/slp-r2"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline", type=float, required=True)
    main(p.parse_args())
