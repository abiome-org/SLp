#!/usr/bin/env bash
set -euo pipefail

OMF_VERSION="2.0.0"
OMF_REPOSITORY="https://github.com/abiome-org/OpenModelFactory.git"
OMF_REVISION="75f002b4226b32dd428f5fec0efe9b950db0c6d5"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SOURCE_DIR="${PROJECT_ROOT}/data/tooling/omf-2.0-75f002b"
RUNTIME_DIR="${PROJECT_ROOT}/data/tooling/omf2-runtime"

usage() {
  printf 'Usage: %s [--diagnostics]\n' "${0##*/}"
}

RUN_DIAGNOSTICS=0
case "${1:-}" in
  "") ;;
  --diagnostics) RUN_DIAGNOSTICS=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

select_python() {
  local candidate
  for candidate in /usr/bin/python3.12 /usr/bin/python3.11; do
    if [[ -x "${candidate}" ]] && "${candidate}" -c \
      'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)'
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf 'OMF 2 requires /usr/bin/python3.12 or /usr/bin/python3.11.\n' >&2
  return 1
}

if [[ -e "${SOURCE_DIR}" && ! -d "${SOURCE_DIR}/.git" ]]; then
  printf 'Pinned source path exists but is not a Git checkout: %s\n' "${SOURCE_DIR}" >&2
  exit 1
fi
if [[ ! -e "${SOURCE_DIR}" ]]; then
  mkdir -p -- "$(dirname -- "${SOURCE_DIR}")"
  git init --quiet "${SOURCE_DIR}"
  git -C "${SOURCE_DIR}" remote add origin "${OMF_REPOSITORY}"
  git -C "${SOURCE_DIR}" fetch --quiet --depth 1 origin "${OMF_REVISION}"
  git -C "${SOURCE_DIR}" checkout --quiet --detach "${OMF_REVISION}"
fi

ACTUAL_REVISION="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_REVISION}" != "${OMF_REVISION}" ]]; then
  printf 'OMF source revision mismatch: expected %s, found %s\n' \
    "${OMF_REVISION}" "${ACTUAL_REVISION}" >&2
  exit 1
fi
if [[ -n "$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=all)" ]]; then
  printf 'Pinned OMF source checkout is dirty; refusing to discard or install it.\n' >&2
  exit 1
fi

if [[ -x "${RUNTIME_DIR}/bin/python" ]]; then
  PYTHON="${RUNTIME_DIR}/bin/python"
  "${PYTHON}" -c \
    'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)' || {
      printf 'Existing OMF runtime uses an unsupported Python version.\n' >&2
      exit 1
    }
else
  if [[ -e "${RUNTIME_DIR}" ]]; then
    printf 'Runtime path exists but is not a usable virtual environment: %s\n' \
      "${RUNTIME_DIR}" >&2
    exit 1
  fi
  SYSTEM_PYTHON="$(select_python)"
  "${SYSTEM_PYTHON}" -m venv "${RUNTIME_DIR}"
  PYTHON="${RUNTIME_DIR}/bin/python"
fi

"${PYTHON}" -m pip install --require-hashes \
  -r "${SOURCE_DIR}/requirements.runtime.lock" \
  -r "${SOURCE_DIR}/requirements.build.lock"
"${PYTHON}" -m pip install --force-reinstall --no-deps --no-build-isolation \
  "${SOURCE_DIR}"

OMF="${RUNTIME_DIR}/bin/omf"
export PATH="${RUNTIME_DIR}/bin:${PATH}"
"${OMF}" --version
INSTALLED_VERSION="$("${PYTHON}" -c \
  'from importlib.metadata import version; print(version("open-model-factory"))')"
if [[ "${INSTALLED_VERSION}" != "${OMF_VERSION}" ]]; then
  printf 'Installed OMF version mismatch: expected %s, found %s\n' \
    "${OMF_VERSION}" "${INSTALLED_VERSION}" >&2
  exit 1
fi

if (( RUN_DIAGNOSTICS )); then
  bash "${SCRIPT_DIR}/omf2.sh" doctor
  bash "${SCRIPT_DIR}/omf2.sh" agent context
  bash "${SCRIPT_DIR}/omf2.sh" agent capabilities
fi
