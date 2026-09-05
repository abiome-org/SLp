"""Compare pooled-count and equal-cell control means using fitting NT only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/slp11-transition/raw-count-control-estimand-audit-v1"
INPUTS = {
    "K562": ("data/derived/slp11-human-k562-essential-raw-cells-v2/control-gem-moments.npz",
             "51f4b53f1e24df5299e39c7d3354784c5da0cc7cd00995630d618f824e1c25c2"),
    "RPE1": ("data/derived/slp11-human-rpe1-essential-raw-cells-v1/control-gem-moments.npz",
             "5aceba5fb4874811aac797be14d1947a9fca866d11178d5f8fe2bdc534df6f61"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("immutable control estimand audit already exists")
    OUT.mkdir(parents=True)
    result = {
        "scope": "Fitting non-targeting controls only; no intervention outcomes, refit or model choice",
        "inputs": INPUTS,
        "scriptSha256": digest(Path(__file__)),
        "derivation": "With a fixed scalar Poisson rate r and observed library L_i, maximizing sum_i log Pois(y_i; L_i*r/10000) gives r=10000*sum_i y_i/sum_i L_i. This is a library-weighted mean of cell CP10k, generally different from the equal-cell average. This exact identity is for the scalar Poisson model, not an asserted closed-form optimum of the fitted latent NB model.",
        "contexts": {},
    }
    for label, (filename, expected_hash) in INPUTS.items():
        path = ROOT / filename
        if digest(path) != expected_hash:
            raise ValueError("control input mismatch")
        with np.load(path, allow_pickle=False) as data:
            raw = data["raw_count_sum"].astype(np.float64)
            library = data["library_count_sum"].astype(np.float64)
            n = data["num_cells"].astype(np.float64)
            equal = data["sum_cp10k"] / n[:, None]
        pooled = 10000 * raw / library[:, None]
        smooth = 10000 * (raw + .5) / (library[:, None] + .5 * raw.shape[1])
        if not np.allclose(equal.sum(1), 10000, atol=1e-7, rtol=0):
            raise ValueError("equal-cell CP10k mass does not close")
        error = (np.log1p(pooled) - np.log1p(equal)) ** 2
        smooth_error = (np.log1p(smooth) - np.log1p(equal)) ** 2
        result["contexts"][label] = {
            "controlCells": int(n.sum()), "groups": len(n), "queries": raw.shape[1],
            "pooledVsEqualCellLog1pMseMacroGem": float(error.mean()),
            "smoothedPooledVsEqualCellLog1pMseMacroGem": float(smooth_error.mean()),
            "smoothedPooledVsEqualCellLog1pMseCellWeighted": float(np.average(smooth_error.mean(1), weights=n)),
            "smoothingOnlyLog1pMseMacroGem": float(np.mean((np.log1p(smooth) - np.log1p(pooled)) ** 2)),
            "pooledVsEqualCellGemMseQuantiles": np.quantile(error.mean(1), [0, .25, .5, .75, 1]).tolist(),
            "excludedFromThisClaim": "Does not estimate the latent NB model optimum or explain its forecasting gap; no control reference is changed.",
        }
    (OUT / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["contexts"], allow_nan=False))


if __name__ == "__main__":
    main()
