# Brief for agents expanding SLB (2026-09-24)

Repo: /home/abiome/projects/slp. Read README.md and BENCHMARK.md first (10 minutes well spent).
SLB-1.2 is a synthetic-lethality benchmark built only from measured combinatorial screens, with
gene families (paralogs + cross-species orthologs) held out into train/dev/test. Labels only come
from sources that pass a reproducibility audit (REPLICATION.md). The headline metric is a
context x screen stratified AUROC that is fitness-balanced (fitness.py): single-gene sickness
earns nothing.

Machine: 96 cores, 183 GB RAM, 1.5 TB disk, ONE RTX 3090 (24 GB) shared by several agents.
Python via `uv` (no conda). Docker is available for legacy stacks (TF1, old torch).

## Coordination
- Message board: `board post -c slp --as <your-agent-name> "..."` / `board read -c slp`.
  Post when you start, on major findings, blockers, and when done.
- GPU: post `gpu:claim <name> <est minutes>` before using CUDA and `gpu:release <name>` after.
  Read the channel first; if someone holds it, work on CPU tasks meanwhile. Keep jobs < ~60 min.
- Never run two agents' heavy jobs so that RAM exceeds ~150 GB total; check `free -g`.

## Hard rules
- Do NOT edit existing files under src/slpbench/ (build.py, fetch.py, families.py, evaluate.py, ...),
  BENCHMARK.md, LEADERBOARD.md, leaderboard.yaml. The lead integrates. Create NEW files only, in
  the locations your task names.
- Do NOT read data/bench/*/hidden/ and do NOT run `slpbench eval --split test`. Dev only.
- Do NOT commit to git. The lead commits.
- Downloads: data/raw/<source_key>/ (gitignored). Record URL + sha256 of every file you fetch.
- Be honest in reports: what ran, what did not, and why. No invented numbers.

## Measurement schema (for data parsers)
See src/slpbench/sources/__init__.py (MEASUREMENT_SCHEMA, finalize, neutral_mask) and the existing
parsers in src/slpbench/sources/*.py for the house style. One row per (source, context, unordered
pair). label 1 = strong negative GI / SL / synthetic sick, 0 = tested and clearly neutral,
null = ambiguous. Positives AND negatives must be measured; never "untested = negative".

## Leakage contract (for models)
A model scored on SLB must be fitted only on SLB `train.parquet` labels (plus permitted single-gene
data: DepMap, expression, sequence, GO, PPI, pretrained embeddings). Models whose released weights
were trained on SynLethDB / SLKB / other SL labels are "leaky": they may be scored as-is, but must
be labelled leaky. Retraining on SLB train is the goal wherever the code allows it.

## Addendum (2026-09-24): the model battery must run across species
The purpose of the model zoo is a battery of standard SL baselines we re-run on every future SLB
version, including new species (cele, dmel, ecol, bsub, spne, ...). So:
1. Every adapter must be callable as `scripts/models/<name>/run.sh` with env SLB_BENCH (benchmark
   dir) and SLB_SPLIT (default dev), writing results/models/<name>[__<variant>]_<split>.parquet.
   It must not hard-code species, file versions or gene lists.
2. Shared per-species feature bundle (built once, reused by all): scripts/models/_common/bundle.py
   -> data/interim/bundle/<species>/{go.parquet, ppi.parquet, esm2.parquet, fitness.parquet}
   (GO with IGI evidence dropped; STRING without experimental/textmining channels + BioGRID physical
   only; ESM-2 per-gene mean embeddings; single-gene fitness from slpbench.fitness / new species'
   *_single()). models-mechanistic already has mech_common/build_go.py and models-features-fm is
   computing ESM-2 embeddings: extend/merge those rather than duplicating. Owner: models-mechanistic.
3. Where a model's architecture only needs a gene graph + node features (most graph/KG/MF models,
   ESM4SL-style, GO/PPI models), add an `allspecies` variant trained per species (or jointly) on SLB
   train using the bundle, in addition to the faithful human-only variant. Where it genuinely needs
   human-only data (DepMap/TCGA/cell-line omics), say so in the notes: that is a coverage fact, not
   a failure.
4. notes/models/<name>.md gets a line: `battery: <variants>; species: <list>; needs: <inputs>`.
Results (*.parquet, *.txt, *.json) go in results/models/, never notes/models/.
