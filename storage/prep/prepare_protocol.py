"""Continue from verified identity outputs; prepare Harle and official protocols."""

import json
import traceback
from pathlib import Path
from inspect_corpus import save, request
import prepare_benchmarks

phase = "identity-receipt"
try:
    identity = "identity-r2-20260912-v5"
    roster = json.load(request(f"/outputs/{identity}/musl-roster.json", prep=True))
    features = json.load(
        request(f"/outputs/{identity}/sequence-inputs-manifest.json", prep=True)
    )
    save(
        "identity-report",
        {
            "sequence_input_job": identity,
            "feature_genes": features["rows"],
            "original_musl_rows": roster["original_rows"],
            "unresolved_musl_rows": roster["unresolved"],
            "source_manifests_complete": True,
        },
    )
    phase = "harle"
    harle = json.load(
        request("/outputs/protocol-r2-20260912-v2/harle-complete.json", prep=True)
    )
    save("harle-complete", {**harle, "prepared_job": "protocol-r2-20260912-v2"})
    phase = "benchmarks"
    prepare_benchmarks.main()
except Exception as exc:
    save(
        "failed",
        {
            "phase": phase,
            "benchmark_phase": prepare_benchmarks.PHASE,
            "type": type(exc).__name__,
            "http_status": getattr(exc, "code", None),
            "detail": str(exc)[:400]
            if isinstance(exc, (ValueError, KeyError, AttributeError))
            else None,
            "frames": [
                {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                for f in traceback.extract_tb(exc.__traceback__)
            ],
        },
    )
    raise SystemExit(1) from None
