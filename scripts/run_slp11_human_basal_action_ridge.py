"""Matched source3 BP ridge test of action-aligned basal abundance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
PHYSICAL = ROOT / "data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz"
BP = ROOT / "data/derived/slp11-human-go-bp/goa-2022-09-19-ensembl108-source3-fit-svd128-v1/human-go-bp-source3-fit-svd128-features.npz"
BASAL = ROOT / "data/derived/slp11-action-aligned-basal-v1/human-source3-action-basal.npz"
OLD_RIDGE = ROOT / "results/slp11-transition/human-gwps-bp-ridge-source3-seed731-v2/development-predictions.npz"
OLD_RIDGE_REPORT = ROOT / "results/slp11-transition/human-gwps-bp-ridge-source3-seed731-v2/report.json"
OLD_KERNEL = ROOT / "results/slp11-transition/human-gwps-bp-nystrom-rbf512-seed731-v1/development-predictions.npz"
OLD_KERNEL_REPORT = ROOT / "results/slp11-transition/human-gwps-bp-nystrom-rbf512-seed731-v1/report.json"
CORE = ROOT / "modules/slp-1-1-basal-action-ridge-v1/basal_action_ridge.py"
OUTPUT = ROOT / "results/slp11-transition/human-source3-bp-action-basal-ridge-seed731-v1"
PINS = {
    "development": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "physical": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "bp": "b29cbd70f08e227cddfc013e66cd1032212c8cb62e6e25162965a57101cd1fac",
    "basal": "57957e763d9f284ae6770dca8c114c2805ccd439a4032bcaf3e6ba23fdf39de3",
    "oldRidgePredictions": "f88efe29faccddbe93a7af1c3e95210b615d9235a3f9ad7d6f9de8530fec498f",
    "oldRidgeReport": "8a3d1ba2265dc09bf6856c97c7a791775ef3282594beed269f708f353d895a0a",
    "oldKernelPredictions": "1434a0c572728142dc91ac7b1ffb06ddd994badc1f040ef0a1b66f055f7e7725",
    "oldKernelReport": "d8259c864460a21f9a13718b2190aad926ca58dc01409c0fab1220a6fbbd276c",
}
CONTEXTS = (
    "replogle-2022-k562-essential-day-6",
    "replogle-2022-rpe1-essential-day-7",
    "replogle-2022-k562-gwps-day-8",
)
CANDIDATES = ("0.1", "1", "10", "100", "1000", "10000", "100000", "1e+06", "mean-limit")
ARMS = ("presence-control", "basal-value")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_python(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fold(gene: str, seed: int = 731) -> int:
    digest = hashlib.sha256(f"slp11-bp-ridge-v1|{seed}|global-inner-fold|9606|{gene}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 3


def collapse(rows, action_ids, static, basal, observed, targets):
    genes = np.asarray(sorted(set(action_ids[rows].astype(str))))
    x, b, m, y, counts = [], [], [], [], []
    for gene in genes:
        selected = rows[action_ids[rows] == gene]
        if not np.all(static[selected] == static[selected[0]]) or not np.all(basal[selected] == basal[selected[0]]) or not np.all(observed[selected] == observed[selected[0]]):
            raise ValueError("within-context action features differ within a gene")
        x.append(static[selected[0]])
        b.append(basal[selected[0]])
        m.append(observed[selected[0]])
        y.append(targets[selected].mean(axis=0, dtype=np.float64))
        counts.append(len(selected))
    return genes, np.asarray(x, np.float32), np.asarray(b, np.float32), np.asarray(m, bool), np.asarray(y, np.float32), np.asarray(counts, np.int64)


def fit_arm(core, static, basal, observed, targets, arm):
    normalizer = core.fit_design_normalizer(static, basal, observed)
    design = core.transform_design(static, basal, observed, normalizer, include_basal_value=arm == "basal-value")
    state = core.fit_state(design, targets)
    return normalizer, state


def predict_arm(core, static, basal, observed, normalizer, state, arm, candidate):
    design = core.transform_design(static, basal, observed, normalizer, include_basal_value=arm == "basal-value")
    return core.predict_state(state, design, candidate)


def choose(core, genes, static, basal, observed, targets, arm):
    folds = np.asarray([fold(gene) for gene in genes])
    totals = {candidate: 0.0 for candidate in CANDIDATES}
    reports = []
    for index in range(3):
        fitting, held = folds != index, folds == index
        normalizer, state = fit_arm(core, static[fitting], basal[fitting], observed[fitting], targets[fitting], arm)
        scale = np.maximum(targets[fitting].std(axis=0, dtype=np.float64), 0.05)
        scores = {}
        for candidate in CANDIDATES:
            prediction = predict_arm(core, static[held], basal[held], observed[held], normalizer, state, arm, candidate)
            score = float(np.mean(np.square((prediction - targets[held]) / scale), dtype=np.float64))
            totals[candidate] += score * int(held.sum())
            scores[candidate] = score
        reports.append({
            "fold": index, "fittingGenes": int(fitting.sum()), "heldGenes": int(held.sum()),
            "observedBasalFittingGenes": int(observed[fitting].sum()),
            "basalMean": float(normalizer["basal_mean"]), "basalScale": float(normalizer["basal_scale"]),
            "staticMeanSha256": hashlib.sha256(normalizer["static_mean"].tobytes()).hexdigest(),
            "staticScaleSha256": hashlib.sha256(normalizer["static_scale"].tobytes()).hexdigest(),
            "scaledMse": scores,
        })
    mean_scores = {candidate: value / len(genes) for candidate, value in totals.items()}
    selected = min(CANDIDATES, key=lambda candidate: (mean_scores[candidate], CANDIDATES.index(candidate)))
    reports.append({"meanScaledMse": mean_scores, "selected": selected})
    return selected, reports


def score(core, prediction, truth):
    pred_centered = core.independently_query_center(prediction)
    truth_centered = core.independently_query_center(truth)
    independent = [core.profile_pearson(a, b) for a, b in zip(pred_centered, truth_centered, strict=True)]
    ordinary = [core.profile_pearson(a, b) for a, b in zip(prediction, truth, strict=True)]
    finite_i = [x for x in independent if x is not None]
    finite_o = [x for x in ordinary if x is not None]
    return {
        "geneProfileMse": float(np.mean(np.square(prediction - truth), dtype=np.float64)),
        "independentlyQueryCenteredPearson": float(np.mean(finite_i)) if finite_i else None,
        "independentUndefinedGenes": len(independent) - len(finite_i),
        "ordinaryPearson": float(np.mean(finite_o)) if finite_o else None,
        "ordinaryUndefinedGenes": len(ordinary) - len(finite_o),
        "genes": len(truth),
    }


def save_model(path, normalizer, state, selected, arm, query_ids):
    np.savez_compressed(path, **normalizer, **state, selected_alpha=np.asarray(selected), arm=np.asarray(arm), query_ids=query_ids)


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def verify_artifact(output: Path):
    output = output.resolve()
    manifest = json.loads((output / "artifact-manifest-before-validation.json").read_text())
    for relative, expected in manifest["hashes"].items():
        if sha(output / relative) != expected:
            raise ValueError(f"artifact hash mismatch: {relative}")
    core = load_python(output / "source/basal_action_ridge.py", "saved_basal_action_ridge")
    maximum = 0.0
    with np.load(output / "target-free-probe.npz", allow_pickle=False) as probe:
        for context in range(3):
            for arm in ARMS:
                model = load_model(output / f"model-{arm}-context-{context}.npz")
                actual = predict_arm(core, probe[f"static_{context}"], probe[f"basal_{context}"], probe[f"observed_{context}"], model, model, arm, str(model["selected_alpha"].item()))
                expected = probe[f"expected_{arm}_{context}"]
                maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    result = {"maximumAbsoluteDifference": maximum, "tolerance": 1e-6, "passes": maximum <= 1e-6}
    if not result["passes"]:
        raise ValueError("fresh saved-artifact replay failed")
    print(json.dumps(result, allow_nan=False))
    return result


def load_inputs(include_targets: bool):
    for name, path in (("development", DATA), ("physical", PHYSICAL), ("bp", BP), ("basal", BASAL), ("oldRidgePredictions", OLD_RIDGE), ("oldRidgeReport", OLD_RIDGE_REPORT), ("oldKernelPredictions", OLD_KERNEL), ("oldKernelReport", OLD_KERNEL_REPORT)):
        if sha(path) != PINS[name]:
            raise ValueError(f"input hash drift: {name}")
    with np.load(DATA, allow_pickle=False) as z:
        names = ["action_ids", "context_index", "context_ids", "query_ids", "record_ids", "split_train", "split_validation", "split_test"]
        data = {name: z[name] for name in names}
        if include_targets:
            data["targets"] = z["targets"].astype(np.float32)
            data["observed"] = z["observed"]
    if len(data["split_test"]) or tuple(data["context_ids"].astype(str)) != CONTEXTS:
        raise ValueError("development split/context drift")
    with np.load(PHYSICAL, allow_pickle=False) as z:
        physical = {str(g): v for g, v in zip(z["entity_id"], z["feature_values"], strict=True)}
    with np.load(BP, allow_pickle=False) as z:
        bp = {str(g): (v, p) for g, v, p in zip(z["entity_id"], z["feature_values"], z["annotation_present"], strict=True)}
    actions = data["action_ids"].astype(str)
    static = np.stack([np.concatenate((physical[g], bp[g][0], np.asarray([bp[g][1]], np.float32))) for g in actions]).astype(np.float32)
    with np.load(BASAL, allow_pickle=False) as z:
        if not np.array_equal(z["population_context_index"], data["context_index"]) or not np.array_equal(z["action_ids"][z["population_action_index"]].astype(str), actions):
            raise ValueError("basal sidecar population alignment drift")
        basal = z["population_basal_value"].astype(np.float32)
        basal_observed = z["population_basal_observed"].astype(bool)
    return data, static, basal, basal_observed


def prepare_profile(output: Path):
    if output.exists():
        raise FileExistsError("refusing to overwrite experiment")
    output.mkdir(parents=True)
    (output / "source").mkdir()
    shutil.copy2(Path(__file__), output / "source/runner.py")
    shutil.copy2(CORE, output / "source/basal_action_ridge.py")
    protocol = {
        "schema": "slp.human-source3-bp-action-basal-ridge-protocol/v1",
        "hypothesis": "The intervention gene's measured matched-control RNA abundance adds unseen-gene forecast information beyond static BP features and measurement availability.",
        "arms": {"presence-control": "BP1285 + zero scalar + measured presence", "basal-value": "BP1285 + fitting-only z-scored basal scalar + identical measured presence"},
        "normalization": "static BP1285 mean/SD exactly refit on context-local unique fitting genes inside each fold; basal mean/SD uses only observed context-local unique fitting genes inside each fold; missing scalar remains zero after transform; presence is raw; no clipping or mixed-tail renormalization",
        "folds": "three global intervention-gene folds from SHA256(slp11-bp-ridge-v1|731|global-inner-fold|9606|ENSG), shared contexts/arms",
        "selection": {"candidates": list(CANDIDATES), "criterion": "equal-gene mean fitting-query-SD-scaled MSE; SD floor .05", "meanFallback": True},
        "gate": "basal-value must reduce raw gene-profile MSE >=1% and not regress finite independently query-centered Pearson versus presence-control in every context",
        "strata": "report observed versus unobserved validation action basal without changing primary all-gene score",
        "comparators": "prior BP linear ridge and BP Nyström kernel, descriptive only",
        "inputs": {name: {"path": str(path), "sha256": PINS[name]} for name, path in (("development", DATA), ("physical", PHYSICAL), ("bp", BP), ("basal", BASAL), ("oldRidgePredictions", OLD_RIDGE), ("oldRidgeReport", OLD_RIDGE_REPORT), ("oldKernelPredictions", OLD_KERNEL), ("oldKernelReport", OLD_KERNEL_REPORT))},
        "sourceHashes": {"runner": sha(Path(__file__)), "core": sha(CORE)},
        "limits": {"cpuThreads": 2, "seconds": 900},
        "testHepJurkatBenchmarkAccess": False,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    data, static, basal, observed = load_inputs(True)
    core = load_python(CORE, "basal_action_ridge_profile")
    rows = data["split_train"][data["context_index"][data["split_train"]] == 2]
    genes, x, b, m, y, _ = collapse(rows, data["action_ids"], static, basal, observed, data["targets"])
    folds = np.asarray([fold(g) for g in genes])
    fitting = folds != 0
    started = time.perf_counter()
    with threadpool_limits(2):
        for arm in ARMS:
            normalizer, state = fit_arm(core, x[fitting], b[fitting], m[fitting], y[fitting], arm)
            prediction = predict_arm(core, x[~fitting], b[~fitting], m[~fitting], normalizer, state, arm, "10000")
            if not np.isfinite(prediction).all():
                raise ValueError("profile prediction nonfinite")
    seconds = time.perf_counter() - started
    profile = {"largestContextFittingGenes": len(genes), "profileFoldFittingGenes": int(fitting.sum()), "twoArmFitPredictSeconds": seconds, "conservativeProjectedSeconds": seconds * 12 + 90, "passes": seconds * 12 + 90 < 900}
    (output / "resource-profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    return protocol, profile


def execute(output: Path):
    if (output / "FROZEN-BEFORE-VALIDATION.json").exists() or (output / "report.json").exists():
        raise FileExistsError("immutable result already exists")
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["sourceHashes"] != {"runner": sha(Path(__file__)), "core": sha(CORE)}:
        raise ValueError("executing source differs from frozen protocol")
    started = time.perf_counter()
    data, static, basal, observed = load_inputs(True)
    if not data["observed"][data["split_train"]].all() or not data["observed"][data["split_validation"]].all():
        raise ValueError("source3 complete-query contract drift")
    core = load_python(CORE, "basal_action_ridge_execute")
    predictions = {arm: np.empty((len(data["split_validation"]), len(data["query_ids"])), np.float32) for arm in ARMS}
    reports, models, probe = {}, {}, {}
    with threadpool_limits(2):
        for context, name in enumerate(CONTEXTS):
            train_rows = data["split_train"][data["context_index"][data["split_train"]] == context]
            train = collapse(train_rows, data["action_ids"], static, basal, observed, data["targets"])
            genes, x, b, m, y, counts = train
            reports[name] = {"arms": {}}
            probe[f"static_{context}"], probe[f"basal_{context}"], probe[f"observed_{context}"] = x[:2], b[:2], m[:2]
            for arm in ARMS:
                selected, cv = choose(core, genes, x, b, m, y, arm)
                normalizer, state = fit_arm(core, x, b, m, y, arm)
                model_path = output / f"model-{arm}-context-{context}.npz"
                save_model(model_path, normalizer, state, selected, arm, data["query_ids"])
                models[arm, context] = (normalizer, state, selected)
                probe[f"expected_{arm}_{context}"] = predict_arm(core, x[:2], b[:2], m[:2], normalizer, state, arm, selected)
                reports[name]["arms"][arm] = {"selectedAlpha": selected, "crossValidation": cv, "modelSha256": sha(model_path), "trainGenes": len(genes), "trainObservedBasalGenes": int(m.sum()), "constructCountRange": [int(counts.min()), int(counts.max())]}
            if time.perf_counter() - started > 900:
                raise TimeoutError("runtime cap exceeded before validation freeze")
    np.savez_compressed(output / "target-free-probe.npz", **probe)
    hashes = {path.name: sha(path) for path in output.glob("model-*.npz")}
    hashes["target-free-probe.npz"] = sha(output / "target-free-probe.npz")
    hashes["source/runner.py"] = sha(output / "source/runner.py")
    hashes["source/basal_action_ridge.py"] = sha(output / "source/basal_action_ridge.py")
    manifest = {"protocolSha256": sha(protocol_path), "hashes": hashes}
    (output / "artifact-manifest-before-validation.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    replay = subprocess.run([sys.executable, str(output / "source/runner.py"), "--verify-artifact", str(output)], check=True, capture_output=True, text=True, timeout=120)
    freeze = {"protocolSha256": sha(protocol_path), "artifactManifestSha256": sha(output / "artifact-manifest-before-validation.json"), "freshReplay": json.loads(replay.stdout)}
    (output / "FROZEN-BEFORE-VALIDATION.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    # The final models and target-free replay are frozen. Validation target rows
    # are indexed for the first time below.
    for context, name in enumerate(CONTEXTS):
        val_rows = data["split_validation"][data["context_index"][data["split_validation"]] == context]
        local_val = np.flatnonzero(data["context_index"][data["split_validation"]] == context)
        val_genes, vx, vb, vm, vy, val_counts = collapse(
            val_rows, data["action_ids"], static, basal, observed, data["targets"]
        )
        reports[name]["strata"] = {}
        for arm in ARMS:
            normalizer, state, selected = models[arm, context]
            pred_gene = predict_arm(core, vx, vb, vm, normalizer, state, arm, selected)
            gene_index = {g: i for i, g in enumerate(val_genes)}
            predictions[arm][local_val] = np.stack(
                [pred_gene[gene_index[g]] for g in data["action_ids"][val_rows]]
            )
            reports[name]["arms"][arm].update({
                "scores": score(core, pred_gene, vy),
                "validationGenes": len(val_genes),
                "validationObservedBasalGenes": int(vm.sum()),
                "validationConstructCountRange": [int(val_counts.min()), int(val_counts.max())],
            })
        for label, mask in (("observed", vm), ("unobserved", ~vm)):
            reports[name]["strata"][label] = {
                arm: score(
                    core,
                    predict_arm(
                        core, vx[mask], vb[mask], vm[mask],
                        models[arm, context][0], models[arm, context][1], arm,
                        models[arm, context][2],
                    ),
                    vy[mask],
                )
                for arm in ARMS
            }
        control_score = reports[name]["arms"]["presence-control"]["scores"]
        basal_score = reports[name]["arms"]["basal-value"]["scores"]
        finite = (
            basal_score["independentlyQueryCenteredPearson"] is not None
            and control_score["independentlyQueryCenteredPearson"] is not None
        )
        reports[name]["gate"] = {
            "mseAtLeastOnePercentBetter": basal_score["geneProfileMse"]
            <= 0.99 * control_score["geneProfileMse"],
            "correlationsFinite": finite,
            "independentRNonregression": finite
            and basal_score["independentlyQueryCenteredPearson"]
            >= control_score["independentlyQueryCenteredPearson"],
        }
        reports[name]["gate"]["passes"] = all(reports[name]["gate"].values())
    # Expose prior frozen comparators only after the new models are frozen.
    comparators = {}
    for label, path, key in (("priorBpRidge", OLD_RIDGE, "physical1156_bp128_present1"), ("priorBpKernel", OLD_KERNEL, "mean")):
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z["record_ids"], data["record_ids"][data["split_validation"]]) or not np.array_equal(z["query_ids"], data["query_ids"]):
                raise ValueError("comparator identity drift")
            values = z[key]
        comparators[label] = {}
        for context, name in enumerate(CONTEXTS):
            local = data["context_index"][data["split_validation"]] == context
            val_rows = data["split_validation"][local]
            genes = np.asarray(sorted(set(data["action_ids"][val_rows].astype(str))))
            pred = np.stack([values[local][data["action_ids"][val_rows] == gene].mean(0) for gene in genes])
            truth = np.stack([data["targets"][val_rows][data["action_ids"][val_rows] == gene].mean(0) for gene in genes])
            comparators[label][name] = score(core, pred, truth)
    np.savez_compressed(output / "development-predictions.npz", **predictions, record_ids=data["record_ids"][data["split_validation"]], action_ids=data["action_ids"][data["split_validation"]], context_index=data["context_index"][data["split_validation"]], query_ids=data["query_ids"])
    report = {"schema": "slp.human-source3-bp-action-basal-ridge-result/v1", "contexts": reports, "comparators": comparators, "passesAllContexts": all(reports[name]["gate"]["passes"] for name in CONTEXTS), "runtimeSeconds": time.perf_counter() - started, "protocolSha256": sha(protocol_path), "freezeSha256": sha(output / "FROZEN-BEFORE-VALIDATION.json"), "predictionsSha256": sha(output / "development-predictions.npz"), "limitations": "Control abundance is observed for roughly two thirds of actions; missing actions remain in primary scores. Linear point forecasts do not establish a world model or assay equivalence."}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-profile", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify-artifact", type=Path)
    args = parser.parse_args()
    if args.verify_artifact:
        verify_artifact(args.verify_artifact)
    elif args.prepare_profile:
        print(json.dumps(dict(zip(("protocol", "profile"), prepare_profile(args.output), strict=True)), indent=2, sort_keys=True))
    elif args.run:
        print(json.dumps(execute(args.output), indent=2, sort_keys=True, allow_nan=False))
    else:
        raise SystemExit("choose --prepare-profile, --run, or --verify-artifact")


if __name__ == "__main__":
    main()
