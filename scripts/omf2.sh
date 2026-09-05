#!/usr/bin/env bash
set -euo pipefail

OMF_VERSION="2.0.0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RUNTIME_DIR="${PROJECT_ROOT}/data/tooling/omf2-runtime"
PYTHON="${RUNTIME_DIR}/bin/python"
OMF="${RUNTIME_DIR}/bin/omf"

if [[ ! -x "${PYTHON}" || ! -x "${OMF}" ]]; then
  printf 'Pinned OMF 2 runtime is absent; run bash scripts/bootstrap_omf2.sh first.\n' >&2
  exit 1
fi
"${PYTHON}" -c \
  'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)' || {
    printf 'Pinned OMF 2 runtime must use Python 3.11 or 3.12.\n' >&2
    exit 1
  }
INSTALLED_VERSION="$("${PYTHON}" -c \
  'from importlib.metadata import version; print(version("open-model-factory"))')"
if [[ "${INSTALLED_VERSION}" != "${OMF_VERSION}" ]]; then
  printf 'Pinned OMF version mismatch: expected %s, found %s\n' \
    "${OMF_VERSION}" "${INSTALLED_VERSION}" >&2
  exit 1
fi

export PATH="${RUNTIME_DIR}/bin:${PATH}"
exec "${OMF}" --project "${PROJECT_ROOT}" "$@"
