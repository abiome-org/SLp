#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run slbench export-public data/release/slb >/dev/null
SLB_BENCH=data/release/slb uv run slbench verify >/dev/null
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  --pax-option=delete=atime,delete=ctime -cf - -C data/release slb \
  | gzip -n > data/release/slb-public.tar.gz
sha256sum data/release/slb-public.tar.gz > reference/public.sha256
