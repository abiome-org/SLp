"""Fitting-only decomposition of the frozen count-state prior forecast.

This diagnostic never fits, selects, or changes model parameters.  It uses a
stable identity-selected subset of fitting intervention genes and fitting
aggregate moments only.  Development, test, and reconstruction-held counts
are outside its input contract.
"""
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
MOMENTS = ROOT / "data/derived/slp11-human-k562-essential-fitting-action-moments-v1/fitting-action-moments.npz"
STATIC = ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz"
OUT = ROOT / "results/slp11-transition/k562-essential-count-prior-variance-audit-v1"
SELECTION_DOMAIN = "slp11-count-prior-fit-audit-v1|"
GENES = 128
PINS = {
    MOMENTS: "a1f44a15a42c5b56e4ce897fde6ebba97298fc296105c6c870ee0e740331694e",
    STATIC: "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659",
    RIDGE / "model.npz": "dbb669d2eb8d844ec9be7c88a2ed21f5592de434d1b2e916412bda4a52fe1cf3",
    RIDGE / "source/count_static_ridge.py": "1032eeff59382fae3874da9a389033192e113e0f5ac2c8d01f09f8441d969e62",
    ARTIFACT / "protocol.json": "a85d2ab7cb83760a818614f20ab28d2936c3604c4f9236293c18b355391b89e7",
    ARTIFACT / "source/runner.py": "9d6668ceb61a3bb0b9dc540a42430b523632b86ddcf547ec2175bfb2fe155920",
    ARTIFACT / "source/count_latent_state.py": "75df347a82151074c0ce6f4c732106e70ed17126aff07d017294894421d30bac",
    ARTIFACT / "source/inference.py": "8922c4516e2356a875c737f4e16ba444838f18a7ff0101308f7fcdc0ef6adaaf",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: np.asarray(values[key]) for key in values.files}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def selected_rows(gene_ids: np.ndarray, count: int = GENES) -> np.ndarray:
    """Return the exact identity-hash selection shared with the root audit."""
    genes = np.asarray(gene_ids, dtype=str)
    if genes.ndim != 1 or len(set(genes.tolist())) != len(genes) or count <= 0 or count > len(genes):
        raise ValueError("gene IDs must be a unique one-dimensional roster")
    return np.asarray(
        sorted(
            range(len(genes)),
            key=lambda row: hashlib.sha256(
                (SELECTION_DOMAIN + genes[row]).encode("utf-8")
            ).digest(),
        )[:count],
        dtype=np.int64,
    )


def log_ratio_terms(
    delta_mean: np.ndarray, delta_variance: np.ndarray, loading: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return W*delta_mu and 0.5*W^2*delta_variance."""
    mean = np.asarray(delta_mean, dtype=np.float64)
    variance = np.asarray(delta_variance, dtype=np.float64)
    weights = np.asarray(loading, dtype=np.float64)
    if (
        mean.ndim != 2
        or variance.shape != mean.shape
        or weights.ndim != 2
        or weights.shape[1] != mean.shape[1]
        or not np.isfinite(mean).all()
        or not np.isfinite(variance).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("invalid latent decomposition arrays")
    return mean @ weights.T, 0.5 * variance @ np.square(weights).T


def population_from_terms(
    basal_rate: np.ndarray,
    mean_term: np.ndarray,
    variance_term: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full and variance-neutralized population rates."""
    basal = np.asarray(basal_rate, dtype=np.float64)
    mean = np.asarray(mean_term, dtype=np.float64)
    variance = np.asarray(variance_term, dtype=np.float64)
    if basal.shape != mean.shape or variance.shape != mean.shape:
        raise ValueError("basal and decomposition terms must share shape")
    if not np.isfinite(basal).all() or np.any(basal <= 0):
        raise ValueError("basal rates must be finite and positive")
    neutral = basal * np.exp(mean)
    full = basal * np.exp(mean + variance)
    return full, neutral


def protocol() -> dict[str, object]:
    return {
        "schema": "slp.k562-count-prior-variance-audit-protocol/v1",
        "hypothesis": "The deployable frozen prior may encode fitting response through intervention variance or unconstrained total CP10k mass rather than its intervention mean.",
        "decision": "Descriptive fitting-only diagnostic; no model, checkpoint, feature, or hyperparameter selection.",
        "population": f"The exact {GENES} fitting genes selected by ascending SHA256({SELECTION_DOMAIN!r} + ENSG), identical to audit_slp11_count_prior_fitting.py.",
        "endpoint": "ln1p(equal-cell mean CP10k), all 8,563 queries; fitting reconstruction-training action moments only.",
        "arms": {
            "fullPrior": "Frozen prior marginal mean with learned intervention mean and variance.",
            "varianceNeutralized": "Same prior mean with intervention log variance replaced by its matched control log variance; no refit.",
        },
        "decomposition": {
            "mean": "W @ (mu_action - mu_control)",
            "variance": "0.5 * W^2 @ (var_action - var_control)",
            "rmsWeighting": "For every selected gene, weight its 48 action-GEM log-ratio terms by that gene's fitting cell fraction; average squared terms equally over genes and all 8,563 queries.",
            "mass": "Quantiles across gene-level GEM-mixture sums of predicted CP10k over all 8,563 queries; reference 10,000.",
        },
        "metrics": "Raw full-query MSE and anchor-subtracted independently query-centered Pearson from the frozen runner profile_metrics implementation.",
        "cpuThreads": 2,
        "maxSeconds": 600,
        "developmentRead": False,
        "reconstructionHeldRead": False,
        "testRead": False,
        "pins": {str(path.relative_to(ROOT)): value for path, value in PINS.items()},
        "scriptSha256": digest(Path(__file__).resolve()),
    }


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = protocol()
    path = OUT / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != frozen:
            amendment_path = OUT / "execution-amendment.json"
            amended = dict(frozen)
            amended["scriptSha256"] = existing.get("scriptSha256")
            if not amendment_path.exists() or existing != amended:
                raise ValueError("frozen variance-audit protocol changed")
            amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
            if (
                amendment.get("originalProtocolSha256") != digest(path)
                or amendment.get("originalScriptSha256") != existing.get("scriptSha256")
                or amendment.get("amendedScriptSha256") != digest(Path(__file__).resolve())
                or amendment.get("scientificProtocolChanged") is not False
            ):
                raise ValueError("variance-audit execution amendment mismatch")
    else:
        write_json(path, frozen)


def verify_artifact() -> dict[str, object]:
    freeze_path = ARTIFACT / "FROZEN-BEFORE-DEVELOPMENT-V2.json"
    manifest_path = ARTIFACT / "artifact-manifest.json"
    if not freeze_path.exists() or not manifest_path.exists():
        raise RuntimeError("wait for the final checkpoint and artifact manifest freeze")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if digest(ARTIFACT / "protocol.json") != manifest["protocolSha256"]:
        raise ValueError("artifact protocol does not match its manifest")
    if (
        freeze.get("schema") != "slp.k562-count-latent-final-checkpoint-freeze/v2"
        or freeze.get("originalProtocolSha256") != manifest["protocolSha256"]
        or freeze.get("modelSha256") != manifest["sha256"]["model.safetensors"]
        or freeze.get("referenceSha256") != manifest["sha256"]["reference.npz"]
        or freeze.get("developmentCountMembersOpened") is not False
        or freeze.get("testOpened") is not False
    ):
        raise ValueError("authoritative v2 pre-development freeze does not match the artifact")
    for name in ("model.safetensors", "reference.npz", "source/count_latent_state.py", "source/inference.py"):
        if digest(ARTIFACT / name) != manifest["sha256"][name]:
            raise ValueError(f"artifact member checksum mismatch: {name}")
    return manifest


def run() -> dict[str, object]:
    prepare()
    if (OUT / "report.json").exists():
        raise FileExistsError("immutable variance audit already complete")
    started = time.perf_counter()
    torch.set_num_threads(2)
    for path, expected in PINS.items():
        if digest(path) != expected:
            raise ValueError(f"frozen input mismatch: {path}")
    manifest = verify_artifact()

    moments = load_npz(MOMENTS)
    genes = moments["action_ids"].astype(str)
    rows = selected_rows(genes)
    truth = np.log1p(
        moments["cp10k_sum"][rows].astype(np.float64)
        / moments["cell_count"][rows, None].astype(np.float64)
    )
    gem_count = moments["gem_cell_count"][rows].astype(np.float64)
    gem_weight = gem_count / gem_count.sum(1, keepdims=True)
    query_ids = moments["query_ids"].astype(str)

    static = load_npz(STATIC)
    lookup = {gene: row for row, gene in enumerate(static["entity_id"].astype(str))}
    raw_action = static["feature_values"][[lookup[gene] for gene in genes[rows]]]

    inference = load_module(ARTIFACT / "source/inference.py", "count_prior_variance_inference")
    predictor = inference.Predictor(ARTIFACT, device="cpu")
    if not np.array_equal(query_ids, predictor.query_ids):
        raise ValueError("fitting moment and registered query axes differ")
    reference = predictor.reference
    normalized = (
        (raw_action.astype(np.float32) - reference["feature_mean"])
        / reference["feature_scale"]
    ).astype(np.float32)
    groups = len(predictor.gem_group_ids)
    if gem_weight.shape != (GENES, groups):
        raise ValueError("fitting GEM weights do not match registered contexts")

    model = predictor.model
    with threadpool_limits(2), torch.no_grad():
        query = torch.as_tensor(reference["query_features"])
        basal = torch.as_tensor(reference["basal_rate"])
        basal_mask = torch.as_tensor(reference["basal_observed"])
        context = model.encode_context(query, basal, basal_mask)
        actions = torch.as_tensor(np.repeat(normalized[:, None, :], groups, axis=0))
        mask = torch.ones(actions.shape[:2], dtype=torch.bool)
        prior = model.prior_from_context(actions, mask, context.repeat(GENES, 1))
        loading, _ = model.observation_parameters(query)
        delta_mean = prior["mean"] - prior["control_mean"]
        delta_variance = prior["logvar"].exp() - prior["control_logvar"].exp()
        weights = torch.as_tensor(gem_weight.reshape(GENES * groups, 1), dtype=torch.float64)

        full_parts: list[np.ndarray] = []
        neutral_parts: list[np.ndarray] = []
        mean_square = 0.0
        variance_square = 0.0
        cross = 0.0
        for left in range(0, len(query), 1024):
            if time.perf_counter() - started > 600:
                raise TimeoutError("variance audit exceeded 600 seconds")
            local_loading = loading[left : left + 1024]
            mean_term = delta_mean @ local_loading.T
            variance_term = 0.5 * delta_variance @ local_loading.square().T
            local_basal = basal[:, left : left + 1024]
            local_basal = local_basal.unsqueeze(0).expand(GENES, -1, -1).reshape(GENES * groups, -1)
            full_rate = local_basal * (mean_term + variance_term).exp()
            neutral_rate = local_basal * mean_term.exp()
            width = mean_term.shape[1]
            full_parts.append(
                (full_rate.double() * weights).reshape(GENES, groups, width).sum(1).cpu().numpy()
            )
            neutral_parts.append(
                (neutral_rate.double() * weights).reshape(GENES, groups, width).sum(1).cpu().numpy()
            )
            mean_square += float((mean_term.double().square() * weights).sum())
            variance_square += float((variance_term.double().square() * weights).sum())
            cross += float((mean_term.double() * variance_term.double() * weights).sum())

    full_cp10k = np.concatenate(full_parts, axis=1)
    neutral_cp10k = np.concatenate(neutral_parts, axis=1)
    if not np.isfinite(full_cp10k).all() or not np.isfinite(neutral_cp10k).all():
        raise FloatingPointError("nonfinite prior population mean")

    ridge_core = load_module(RIDGE / "source/count_static_ridge.py", "count_prior_variance_ridge")
    baseline = load_npz(RIDGE / "model.npz")
    anchor = ridge_core.control_anchor(baseline["basal_rate"], gem_count)
    score = load_module(ARTIFACT / "source/runner.py", "count_prior_variance_score").profile_metrics
    denominator = GENES * len(query_ids)
    report = {
        "schema": "slp.k562-count-prior-variance-audit/v1",
        "protocolSha256": digest(OUT / "protocol.json"),
        "executionAmendmentSha256": digest(OUT / "execution-amendment.json"),
        "modelSha256": manifest["sha256"]["model.safetensors"],
        "referenceSha256": manifest["sha256"]["reference.npz"],
        "seconds": time.perf_counter() - started,
        "fittingGenes": genes[rows].tolist(),
        "metrics": {
            "fullPrior": score(np.log1p(full_cp10k), truth, anchor),
            "varianceNeutralized": score(np.log1p(neutral_cp10k), truth, anchor),
        },
        "weightedLogRatioRms": {
            "meanTerm": float(np.sqrt(mean_square / denominator)),
            "varianceTerm": float(np.sqrt(variance_square / denominator)),
            "combinedTerm": float(np.sqrt((mean_square + variance_square + 2 * cross) / denominator)),
            "meanVarianceCrossMean": float(cross / denominator),
        },
        "totalCp10kQuantiles": {
            "fullPrior": np.quantile(full_cp10k.sum(1), [0, .25, .5, .75, 1]).tolist(),
            "varianceNeutralized": np.quantile(neutral_cp10k.sum(1), [0, .25, .5, .75, 1]).tolist(),
            "reference": 10000.0,
        },
        "fullMinusNeutralized": {
            "cp10kRms": float(np.sqrt(np.mean((full_cp10k - neutral_cp10k) ** 2))),
            "log1pCp10kRms": float(np.sqrt(np.mean((np.log1p(full_cp10k) - np.log1p(neutral_cp10k)) ** 2))),
        },
        "developmentRead": False,
        "reconstructionHeldRead": False,
        "testRead": False,
    }
    write_json(OUT / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "fittingGenes"}, allow_nan=False))
    return report


def main() -> None:
    prepare()
    if "--prepare" not in sys.argv:
        run()


if __name__ == "__main__":
    main()
