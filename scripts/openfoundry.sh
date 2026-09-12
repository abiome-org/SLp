#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RUNTIME_DIR="${PROJECT_ROOT}/data/tooling/openfoundry-runtime"
PYTHON="${RUNTIME_DIR}/bin/python"
OPENFOUNDRY="${RUNTIME_DIR}/bin/openfoundry"

if [[ ! -x "${PYTHON}" || ! -x "${OPENFOUNDRY}" ]]; then
  printf 'Pinned OpenFoundry runtime is absent; run bash scripts/bootstrap_openfoundry.sh first.\n' >&2
  exit 1
fi
"${PYTHON}" -c \
  'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)' || {
    printf 'Pinned OpenFoundry runtime must use Python 3.11 or 3.12.\n' >&2
    exit 1
  }
"${PYTHON}" - "${PROJECT_ROOT}" "${RUNTIME_DIR}" <<'PY'
import json
import sys
from importlib.metadata import version
from pathlib import Path

root, runtime = map(Path, sys.argv[1:])
pin = json.loads((root / "openfoundry-version.json").read_text())
receipt = runtime / "toolchain-pin.json"
if (not receipt.is_file() or json.loads(receipt.read_text()) != pin
        or version("openfoundry") != pin["version"]
        or (root / pin["runtime"]["path"]).resolve() != runtime.resolve()):
    sys.exit("OpenFoundry pin changed; run bash scripts/bootstrap_openfoundry.sh")
PY

export PATH="${RUNTIME_DIR}/bin:${PATH}"
exec "${OPENFOUNDRY}" --project "${PROJECT_ROOT}" "$@"
