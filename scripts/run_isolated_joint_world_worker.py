"""Launch the portability worker under Python -S with an explicit import set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import sysconfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--method", choices=("predict", "predict_latent_rollout"), default="predict"
    )
    parser.add_argument("--python-runtime", required=True)
    parser.add_argument("--worker-output", required=True)
    parser.add_argument("--isolation-output", required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ.update(
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4"
    )
    verifier = Path(args.verifier).resolve()
    model = Path(args.model).resolve()
    admitted = [str(model), str(verifier.parent), args.site_packages]
    stdlib = Path(sysconfig.get_path("stdlib"))
    runtime_paths = [str(stdlib), str(stdlib / "lib-dynload")]
    sys.path[:] = admitted + runtime_paths
    if any(importlib.util.find_spec(name) is not None for name in ("omf", "openfoundry")):
        raise RuntimeError("A factory package is importable in isolated portability worker")
    isolation = {
        "pythonNoSite": bool(sys.flags.no_site),
        "omfImportable": False,
        "admittedPaths": admitted,
        "runtimePaths": runtime_paths,
        "executable": sys.executable,
        "threadLimit": 4,
        "method": args.method,
    }
    Path(args.isolation_output).write_text(json.dumps(isolation, indent=2, sort_keys=True) + "\n")
    sys.argv = [
        str(verifier),
        "--model",
        str(model),
        "--checkpoint",
        args.checkpoint,
        "--method",
        args.method,
        "--python-runtime",
        args.python_runtime,
        "--worker",
        "--worker-output",
        args.worker_output,
        "--output",
        str(Path(args.worker_output).parent / "unused"),
    ]
    runpy.run_path(str(verifier), run_name="__main__")


if __name__ == "__main__":
    main()
