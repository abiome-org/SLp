#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run slpbench export-public data/release/slb1.3 >/dev/null
SLB_BENCH=data/release/slb1.3 uv run slpbench verify >/dev/null
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  --pax-option=delete=atime,delete=ctime -cf - -C data/release slb1.3 \
  | gzip -n > data/release/slb1.3-public.tar.gz
sha256sum data/release/slb1.3-public.tar.gz > reference/slb1.3-public.sha256
