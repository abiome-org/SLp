"""Restore published outer bundles on the pod and replay their inference API."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    request = json.loads(args.receipt.read_text())
    if digest(args.root / "campaign/plan.json") != request["plan_sha256"]:
        raise ValueError("Replay references a different campaign")
    if set(request["families"]) != {"feature_mlp", "no_pretraining", "mixed_pretraining"}:
        raise ValueError("Require all three published model families")
    if args.limit < 1 or not math.isfinite(args.deadline) or args.deadline <= time.time():
        raise ValueError("Invalid bounded replay request")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.root))
    import cloud_io

    spec = importlib.util.spec_from_file_location("replay_transport", args.transport)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    os.environ["SLP_TRANSFER_TICKETS"] = str(args.tickets)
    report = {"schema": "slp.r2-published-inference-replay/v1", "started_at": time.time(),
              "fold": request["fold"], "plan_sha256": request["plan_sha256"],
              "driver_sha256": digest(__file__), "request_sha256": digest(args.receipt),
              "device": "cpu", "atol_probability": 1e-5, "families": {}}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def restore(receipt, directory):
        if time.time() >= args.deadline - 60:
            raise TimeoutError("Replay download allowance exhausted")
        artifact = transport.retry_io(lambda: cloud_io.download_directory(receipt["job"], directory))
        files = {name: {"sha256": item["sha256"], "bytes": item["bytes"]}
                 for name, item in artifact["files"].items()}
        if artifact["job"] != receipt["job"] or files != receipt["files"]:
            raise ValueError("Restored files differ from the published campaign receipt")
        return artifact

    save()
    try:
        for family, artifacts in request["families"].items():
            directory = args.output / family
            directory.mkdir()
            bundle, scores = directory / "bundle", directory / "scores"
            restored = restore(artifacts["bundle"], bundle)
            restore(artifacts["scores"], scores)
            rows = []
            with (scores / "predictions.jsonl").open() as file:
                for line in file:
                    rows.append(json.loads(line))
                    if len(rows) == args.limit:
                        break
            if not rows:
                raise ValueError("No published predictions to replay")
            groups = {}
            for index, row in enumerate(rows):
                groups.setdefault(row["context"], []).append((index, row))
            actual = [None] * len(rows)
            for group_index, (context, group) in enumerate(groups.items()):
                pairs = directory / f"pairs-{group_index}.json"
                predictions = directory / f"replayed-{group_index}.json"
                pairs.write_text(json.dumps([row["targets"] for _, row in group]))
                environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                               "CUDA_VISIBLE_DEVICES": ""}
                environment.pop("PYTHONPATH", None)
                remaining = args.deadline - time.time()
                if remaining < 60:
                    raise TimeoutError("Replay inference allowance exhausted")
                # A fresh process imports only the exported code from its own directory.
                with (directory / f"inference-{group_index}.log").open("w") as log:
                    subprocess.run([sys.executable, str(bundle / "inference.py"),
                                    "--bundle", str(bundle), "--pairs", str(pairs),
                                    "--context", context, "--device", "cpu", "--output", str(predictions)],
                                   cwd=bundle, env=environment, check=True, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=min(300, remaining - 20))
                values = json.loads(predictions.read_text())
                if len(values) != len(group):
                    raise ValueError("Inference changed the number of queried pairs")
                for (index, row), value in zip(group, values, strict=True):
                    if value["targets"] != row["targets"]:
                        raise ValueError("Inference changed pair order")
                    actual[index] = value["sl_probability"]
            expected = []
            for row in rows:
                logit = float(row["logit"])
                if not math.isfinite(logit):
                    raise ValueError("Nonfinite published logit")
                expected.append(1 / (1 + math.exp(-logit)) if logit >= 0
                                else math.exp(logit) / (1 + math.exp(logit)))
            if any(not isinstance(value, (int, float)) or not math.isfinite(value)
                   or not 0 <= value <= 1 for value in actual):
                raise ValueError("Invalid inference probability")
            errors = [abs(a - b) for a, b in zip(actual, expected, strict=True)]
            result = {"bundle_job": artifacts["bundle"]["job"], "scores_job": artifacts["scores"]["job"],
                      "n": len(rows), "contexts": list(groups), "max_absolute_probability_error": max(errors),
                      "mean_absolute_probability_error": sum(errors) / len(errors),
                      "bundle_manifest_sha256": digest(bundle / "bundle.json"),
                      "published_predictions_sha256": digest(scores / "predictions.jsonl"),
                      "restored_files": len(restored["files"]), "verified_at": time.time(),
                      "passed": max(errors) <= report["atol_probability"]}
            report["families"][family] = result
            save()
            if not result["passed"]:
                raise ValueError("Standalone inference does not reproduce published scores")
            # Remove only this verified disposable restore; R2 originals remain intact.
            shutil.rmtree(bundle)
            shutil.rmtree(scores)
        report["completed_at"] = time.time()
    except Exception as error:
        report["failed_at"] = time.time()
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        save()
        args.tickets.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "receipt", "tickets", "output", "transport"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--limit", type=int, default=64)
    main(parser.parse_args())
