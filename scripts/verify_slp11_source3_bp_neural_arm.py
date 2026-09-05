#!/usr/bin/env python3
"""Target-free fresh-process verification for a three-context mean arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def verify(arm: Path) -> dict[str, object]:
    sys.path.insert(0, str(arm.parent / "source"))
    from four_context_mean_inference import FrozenMeanArm, empty_identity_audit

    audit = empty_identity_audit(arm)
    frozen = FrozenMeanArm(arm)
    with np.load(arm / "target-free-probe.npz", allow_pickle=False) as probe:
        replay = frozen.predict(probe["raw_action_features"], probe["context_index"])
        difference = np.abs(replay["mean"] - probe["expected_mean"])
        audit.update(
            {
                "probeRows": len(probe["context_index"]),
                "probeContexts": sorted(set(probe["context_index"].tolist())),
                "probeMaximumAbsoluteDifference": float(difference.max()),
                "probeMeanWithin1e5": bool(difference.max() <= 1e-5),
                "nonemptyFinite": bool(np.isfinite(replay["mean"]).all()),
                "sourceDirectoryOnly": True,
            }
        )
    required = (
        audit["meanBitExact"]
        and audit["deltaNonzero"] == 0
        and audit["latentDeltaNonzero"] == 0
        and audit["contextsChecked"] == 3
        and audit["queriesChecked"] == 7036
        and audit["probeRows"] == 6
        and audit["probeContexts"] == [0, 1, 2]
        and audit["probeMeanWithin1e5"]
        and audit["nonemptyFinite"]
    )
    if not required:
        raise RuntimeError(f"portable verification failed: {audit}")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.arm.resolve(strict=True)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
