# SLB data

`uv run slpbench build` creates the private benchmark under `data/bench/slb1.3`. The source
files are pinned by [`reference/raw_sha256sums.txt`](../reference/raw_sha256sums.txt); retrieval
locations and manual supplement extraction steps are in `notes/data/*.fetch.tsv`. The core
downloader is `uv run python -m slpbench.fetch`.

Verify a built copy with `uv run slpbench verify`; add `--raw` to verify every pinned raw file.
The test labels and fitness propensities live under `data/bench/slb1.3/hidden/` and are required
only on the evaluator's machine. Generate a shareable model input bundle with:

```bash
uv run slpbench export-public data/release/slb1.3
```

The export includes train/dev labels, dev propensity weights, test inputs, context and single-gene
data, the family map, and a checksum lock. It contains no test labels or test propensities. Set
`SLB_BENCH=data/release/slb1.3` to verify the bundle and score dev predictions locally. A model can
generate test predictions from this bundle; the private evaluator scores them by `example_id`.
`bash scripts/package_public.sh` creates a deterministic archive and its SHA-256 pin in
`reference/slb1.3-public.sha256`.
