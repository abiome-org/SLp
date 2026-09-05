#!/usr/bin/env python3
"""Target-free fresh-process verification of one frozen mean-objective arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", type=Path)
    args = parser.parse_args()
    arm = args.arm.resolve(strict=True)
    sys.path.insert(0, str(arm.parent / "source"))
    from four_context_mean_inference import FrozenMeanArm, empty_identity_audit

    audit = empty_identity_audit(arm)
    frozen = FrozenMeanArm(arm)
    feature_count = len(frozen.reference["feature_mean"])
    result = frozen.predict(
        np.zeros((1, feature_count), dtype=np.float32),
        np.zeros(1, dtype=np.int64),
    )
    audit["nonemptyFinite"] = bool(np.isfinite(result["mean"]).all())
    with np.load(arm / "target-free-probe.npz", allow_pickle=False) as probe:
        replay = frozen.predict(probe["raw_action_features"], probe["context_index"])
        difference = np.abs(replay["mean"] - probe["expected_mean"])
        audit["probeRows"] = len(probe["context_index"])
        audit["probeContexts"] = sorted(set(probe["context_index"].tolist()))
        audit["probeMaximumAbsoluteDifference"] = float(difference.max())
        audit["probeMeanWithin1e5"] = bool(difference.max() <= 1e-5)
    audit["sourceDirectoryOnly"] = True
    if not (
        audit["meanBitExact"]
        and audit["deltaNonzero"] == 0
        and audit["latentDeltaNonzero"] == 0
        and audit["nonemptyFinite"]
        and audit["probeMeanWithin1e5"]
        and audit["probeRows"] == 8
        and audit["probeContexts"] == [0, 1, 2, 3]
    ):
        raise RuntimeError(f"portable verification failed: {audit}")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
