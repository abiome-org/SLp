#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RUNTIME="${PROJECT_ROOT}/data/tooling/slp-joint-world-linux"
LOCK="${PROJECT_ROOT}/modules/slp-1-1-joint-world-v1/requirements-linux.lock"

if [[ ! -x /usr/bin/python3.12 ]]; then
  printf 'Python 3.12 is required at /usr/bin/python3.12.\n' >&2
  exit 1
fi
if [[ ! -x "${RUNTIME}/bin/python" ]]; then
  /usr/bin/python3.12 -m venv "${RUNTIME}"
fi
"${RUNTIME}/bin/python" -m pip install --require-hashes -r "${LOCK}"
CUDA_VISIBLE_DEVICES='' "${RUNTIME}/bin/python" - <<'PY'
import numpy
import safetensors
import threadpoolctl
import torch

assert torch.__version__ == "2.11.0+cu128"
assert numpy.__version__ == "2.2.6"
assert safetensors.__version__ == "0.6.2"
assert threadpoolctl.__version__ == "3.6.0"
value = torch.tensor([1.0, 2.0], device="cpu").square().sum().item()
assert value == 5.0
print({
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "safetensors": safetensors.__version__,
    "threadpoolctl": threadpoolctl.__version__,
    "cpuTensorSmoke": value,
    "cudaProbePerformed": False,
})
PY
