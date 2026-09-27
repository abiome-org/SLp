# Brief: adding a human combinatorial screen to SLB (read fully before starting)

Repo: /home/abiome/projects/slp/slb (Python via `uv run`, run from this directory). SLB = family-held-out
synthetic-lethality benchmark. Read README.md sections "Label quality", "Label rules", and
src/slbench/sources/human.py (existing human parsers: chou2025, spidr2025 re-scored from counts,
harle2025, flister2025) and src/slbench/sources/__init__.py (MEASUREMENT_SCHEMA, finalize).
Useful helpers: src/slbench/replication.py (replicate_agreement, split_rule_auroc, cross_study),
slbench.contexts.lookup(cell_line_name) -> Cellosaurus record (must not return None for your lines).

## You may ONLY write to
- data/raw/<key>/                     downloaded raw files (keep original file names where sensible)
- src/slbench/sources/staging/<key>.py   your parser module (one per source key)
- data/interim/new_human/<key>.json   your checks + summary (schema below)
- data/interim/new_human/<key>.md     short notes (design, decisions, caveats)
Do NOT edit build.py, fetch.py, human.py, data/slb, data/interim/measurements, or anything else.
Other agents work in parallel on other sources. Cap threads at 8 (OMP_NUM_THREADS etc.). No GPU.
Downloads: use curl with -fL --retry 5 -A slbench/0.1; files up to ~300 MB are fine; don't fetch FASTQ/BAM.

## Parser contract (staging/<key>.py)
- `def load() -> pl.DataFrame` returning MEASUREMENT_SCHEMA rows via `finalize(df, "human")`
  (from slbench.sources import finalize). One row per (source, context, unordered pair).
- species "human"; source = your key; context = a cell-line name that contexts.lookup resolves
  (for isogenic/query-KO clones, context is the PARENTAL line, e.g. "HAP1"; the query gene is gene_a/b);
  mechanism one of CRISPR-KO | CRISPR-Cas12a | CRISPRi | CRISPRa; score = native interaction score where
  NEGATIVE = synthetic sick/lethal (flip signs if needed); signif = FDR/p if available.
- Only drug-free / standard-medium / unirradiated arms (the fitness GI of the double loss itself).
- Also provide `def replicates() -> pl.DataFrame` with columns gene_a, gene_b, (context,) and per-replicate
  GI columns gi_rep1, gi_rep2, ... when the data allow it (from raw counts: LFC end vs start per replicate,
  centred on control x control, f_g = median LFC of gene x control, GI = LFC - f_a - f_b, median over
  guide pairs — the spidr_replicate_gi recipe). If the source has independent maps/enzymes/orientations
  instead, provide those as the replicate columns and say so.
- Gene symbols go through finalize's HGNC resolver; report how many symbols fail to resolve.

## Label rule (be conservative; SLB prefers fewer, replicable positives)
- Positive: a STRONG negative interaction. Prefer the authors' own call if it is stringent; otherwise
  re-score (e.g. z <= -3 on a GI z-score, or an FDR < 0.05 call with a strong effect). Positive rate
  should be plausible (roughly 0.3–5% of tested pairs); report it.
- Negative: tested, not called, and in the neutral band (|z| < 1, or |score| below that screen's median
  and non-significant).
- Everything else: label null (ambiguous).
State exactly which rule you used and why.

## Checks to run and report (the SLB inclusion gate)
1. Own replication: AUROC >= 0.65 when labels defined from one replicate/map/orientation are scored by
   another (use replication.split_rule_auroc on replicate GI columns, both directions, and report
   replicate_agreement). Also AUROC of each single replicate for YOUR final labels.
2. Cross-study: if another study measured the same pairs in the same cell line (existing SLB measurements
   in data/interim/measurements/{slkb,ryanlab_zdlfc,chou2025,spidr2025,harle2025,flister2025}.parquet,
   whose `context` values you map with contexts.lookup(...)['cellosaurus_name']), report AUROC of the
   other study's score for your labels and of your score for their labels, with n positives.
   Different cell line overlaps can be reported separately as weaker evidence.
3. Hit rate and fitness confounding: AUROC of -(f_a + f_b) (single-gene effects, e.g. DepMap via
   data/slb/gene_single_effects.parquet or the screen's own singles) for your labels. If single-gene
   fitness alone predicts labels at > 0.8, flag it.
4. Counts: pairs, labelled pos/neg per context, unique genes, fraction of pairs that are paralogs
   (use data/slb/features/paralogs.parquet via slbench.features.read if convenient).

## Output JSON (data/interim/new_human/<key>.json)
{"key":..., "citation":..., "doi":..., "license":..., "files":[{"name","url","bytes","sha256"}],
 "contexts":[...], "mechanism":..., "label_rule":..., "n_pairs":..., "n_pos":..., "n_neg":...,
 "pos_rate":..., "paralog_fraction":..., "own_replication":{...}, "cross_study":{...},
 "fitness_auroc":..., "verdict":"include"|"exclude"|"measurements-only", "verdict_reason":...}
Verdict rule: include only if own-replication or cross-study AUROC >= 0.65, labels are not
contradicted by cross-study evidence (< 0.6 with >= 20 overlapping positives is a contradiction),
and the hit rate is plausible. Be honest; an exclude with evidence is a good result.
Final message: a compact summary of the JSON plus anything surprising.
