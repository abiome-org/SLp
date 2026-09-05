"""Apply the known nonnegative measurement domain equally to frozen forecasts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("nonfinite molecular prediction")
    return np.maximum(values, 0)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = root / "results/slp11-transition"
    destination = runs / "frangieh-nonnegative-support-diagnostic-v1"
    if destination.exists():
        raise FileExistsError(destination)
    paths = {
        "cell": runs / "frangieh-cell-state-ae-latent-ridge-seed731-v1/predictions.npz",
        "prior": runs / "frangieh-paired-state-physical1156-seed731-v2/predictions.npz",
        "static": runs / "frangieh-specieswide-physical-ridge-v1/predictions.npz",
        "reference": runs / "frangieh-paired-state-physical1156-seed731-v2/reference.npz",
        "prior_report": runs / "frangieh-paired-state-vs-static-scoring-v1/report.json",
        "metrics": runs / "frangieh-cell-state-ae-latent-ridge-seed731-v1/source/frangieh_basal_ridge.py",
    }
    expected = {
        "cell": "a5cc6724ad55c5d3f2ad709be36a5fcbcb77e7255d7f79af29145556e2a24b96",
        "prior": "36ebe74677f7bb75e467bf8f225cc313417590772de356f98470d32a5e26b50b",
        "static": "1e342a75e4a1cc67d6d0a6e3c1e4acefb95d7a51fad7a1bf47fcbff978c7abfe",
        "reference": "8b82e4781b73a721f995dd218ef341ea8324b87d3c9189bfe40644d436800e73",
        "prior_report": "d0c577e093198e9060a582cc5852b0db61246daa5772ae0c1e8451addc584b90",
    }
    hashes = {key: digest(path) for key, path in paths.items()}
    if any(hashes[key] != value for key, value in expected.items()):
        raise ValueError("frozen input mismatch")
    spec = importlib.util.spec_from_file_location("support_metrics", paths["metrics"])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    destination.mkdir()
    protocol = {"inputSha256": hashes, "scriptSha256": digest(Path(__file__)),
                "rule": "Project every method to max(prediction,0), fixed by both processed measurement definitions; no target-dependent threshold or refitting.",
                "claimLimit": "Post-hoc support diagnostic; projection of population forecasts is not a nonlinear cell observation model and does not establish averaging commutation."}
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (destination / "scoring-source.py").write_bytes(Path(__file__).read_bytes())
    contexts = ("Co-culture", "Control", "IFNγ")
    report = {"contexts": {}, "claimLimit": protocol["claimLimit"]}
    old = json.loads(paths["prior_report"].read_text())
    with np.load(paths["cell"], allow_pickle=False) as cell, np.load(paths["prior"], allow_pickle=False) as prior, np.load(paths["static"], allow_pickle=False) as static, np.load(paths["reference"], allow_pickle=False) as reference:
        for ci, context in enumerate(contexts):
            key = context.replace("-", "_").replace("γ", "gamma")
            report["contexts"][context] = {}
            rows = cell["context_ids"] == context
            genes = cell["action_ids"][rows]
            lookup = {(str(g), int(c)): i for i, (g, c) in enumerate(zip(prior["action_ids"], prior["context_index"], strict=True))}
            indices = [lookup[(str(g), ci)] for g in genes]
            for head in ("rna", "protein"):
                sh = "adt" if head == "protein" else head
                truth = cell[f"{head}_truth"][rows]
                if np.any(truth < 0) or not np.isfinite(truth).all():
                    raise ValueError("measurement support disagrees with source definition")
                if not np.array_equal(genes, static[f"{key}_{sh}_action_ids"]) or not np.array_equal(truth, static[f"{key}_{sh}_truth"]) or not np.array_equal(truth, prior[f"{head}_truth"][indices]):
                    raise ValueError("frozen comparator alignment differs")
                methods = {"cellState": cell[f"{head}_prediction"][rows], "priorPaired": prior[f"{head}_prediction"][indices],
                           "mean": np.broadcast_to(reference[f"{head}_means"][ci], truth.shape),
                           "base577": static[f"{key}_{sh}_base577"], "physical1156": static[f"{key}_{sh}_physical1156"]}
                result = {}
                for name, values in methods.items():
                    before = helper.metrics(values, truth, np.ones(truth.shape[1]))
                    after = helper.metrics(project(values), truth, np.ones(truth.shape[1]))
                    if after["raw_mse"] > before["raw_mse"] + 1e-14:
                        raise AssertionError("projection increased squared error")
                    result[name] = {"negativeFraction": float(np.mean(values < 0)), "minimum": float(values.min()),
                                    "mse": before["raw_mse"], "projectedMse": after["raw_mse"],
                                    "projectedCenteredPearson": after["query_centroid_adjusted_profile_pearson"]}
                if not np.isclose(result["mean"]["mse"], old["contexts"][context]["heads"][head]["baselines"]["mean"]["raw_mse"], rtol=1e-6):
                    raise ValueError("frozen mean reconstruction mismatch")
                candidate = result["cellState"]
                comparisons = {name: {"mseGain": 1 - candidate["projectedMse"] / base["projectedMse"],
                                     "rNonregression": not np.isfinite(base["projectedCenteredPearson"]) or candidate["projectedCenteredPearson"] >= base["projectedCenteredPearson"]}
                               for name, base in result.items() if name != "cellState"}
                passed = candidate["projectedCenteredPearson"] >= .1 and all(c["mseGain"] >= .01 and c["rNonregression"] for c in comparisons.values())
                report["contexts"][context][head] = {"methods": result, "comparisons": comparisons, "sameForecastRulePassed": bool(passed)}
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return clean(value.item())
        return value
    (destination / "report.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({c: {h: v["sameForecastRulePassed"] for h, v in heads.items()} for c, heads in report["contexts"].items()}))


if __name__ == "__main__":
    main()
