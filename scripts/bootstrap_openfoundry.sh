#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PIN="${PROJECT_ROOT}/openfoundry-version.json"

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
  for candidate in "${OPENFOUNDRY_PYTHON:-}" python3.12 python3.11; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c \
      'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)'
    then
      command -v "${candidate}"
      return 0
    fi
  done
  printf 'OpenFoundry requires Python 3.11/3.12 on PATH or OPENFOUNDRY_PYTHON.\n' >&2
  return 1
}

SYSTEM_PYTHON="$(select_python)"
pin_value() {
  "${SYSTEM_PYTHON}" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); print(value[sys.argv[2]][sys.argv[3]] if len(sys.argv)>3 else value[sys.argv[2]])' \
    "${PIN}" "$@"
}
OPENFOUNDRY_VERSION="$(pin_value version)"
OPENFOUNDRY_REPOSITORY="$(pin_value source repository)"
OPENFOUNDRY_REVISION="$(pin_value source revision)"
SOURCE_DIR="${PROJECT_ROOT}/$(pin_value source path)"
RUNTIME_DIR="${PROJECT_ROOT}/$(pin_value runtime path)"

if [[ -e "${SOURCE_DIR}" && ! -d "${SOURCE_DIR}/.git" ]]; then
  printf 'Pinned source path exists but is not a Git checkout: %s\n' "${SOURCE_DIR}" >&2
  exit 1
fi
if [[ ! -e "${SOURCE_DIR}" ]]; then
  mkdir -p -- "$(dirname -- "${SOURCE_DIR}")"
  git init --quiet "${SOURCE_DIR}"
  git -C "${SOURCE_DIR}" remote add origin "${OPENFOUNDRY_REPOSITORY}"
  git -C "${SOURCE_DIR}" fetch --quiet --depth 1 origin "${OPENFOUNDRY_REVISION}"
  git -C "${SOURCE_DIR}" checkout --quiet --detach "${OPENFOUNDRY_REVISION}"
fi

ACTUAL_REVISION="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_REVISION}" != "${OPENFOUNDRY_REVISION}" ]]; then
  printf 'OpenFoundry source revision mismatch: expected %s, found %s\n' \
    "${OPENFOUNDRY_REVISION}" "${ACTUAL_REVISION}" >&2
  exit 1
fi
if [[ -n "$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=all)" ]]; then
  printf 'Pinned OpenFoundry source checkout is dirty; refusing to discard or install it.\n' >&2
  exit 1
fi

if [[ -x "${RUNTIME_DIR}/bin/python" ]]; then
  PYTHON="${RUNTIME_DIR}/bin/python"
  "${PYTHON}" -c \
    'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)' || {
      printf 'Existing OpenFoundry runtime uses an unsupported Python version.\n' >&2
      exit 1
    }
else
  if [[ -e "${RUNTIME_DIR}" ]]; then
    printf 'Runtime path exists but is not a usable virtual environment: %s\n' \
      "${RUNTIME_DIR}" >&2
    exit 1
  fi
  "${SYSTEM_PYTHON}" -m venv "${RUNTIME_DIR}"
  PYTHON="${RUNTIME_DIR}/bin/python"
fi

"${PYTHON}" -m pip install --only-binary=:all: --require-hashes \
  -r "${SOURCE_DIR}/requirements.runtime.lock" \
  -r "${SOURCE_DIR}/requirements.build.lock"
"${PYTHON}" -m pip install --force-reinstall --no-deps --no-build-isolation \
  "${SOURCE_DIR}"

OPENFOUNDRY="${RUNTIME_DIR}/bin/openfoundry"
export PATH="${RUNTIME_DIR}/bin:${PATH}"
"${OPENFOUNDRY}" --version
INSTALLED_VERSION="$("${PYTHON}" -c \
  'from importlib.metadata import version; print(version("openfoundry"))')"
if [[ "${INSTALLED_VERSION}" != "${OPENFOUNDRY_VERSION}" ]]; then
  printf 'Installed OpenFoundry version mismatch: expected %s, found %s\n' \
    "${OPENFOUNDRY_VERSION}" "${INSTALLED_VERSION}" >&2
  exit 1
fi

# The renamed upstream still reports 2.0.0, so version alone is insufficient.
cp -- "${PIN}" "${RUNTIME_DIR}/toolchain-pin.json"

if (( RUN_DIAGNOSTICS )); then
  bash "${SCRIPT_DIR}/openfoundry.sh" doctor
  bash "${SCRIPT_DIR}/openfoundry.sh" agent context
  bash "${SCRIPT_DIR}/openfoundry.sh" agent capabilities
fi
