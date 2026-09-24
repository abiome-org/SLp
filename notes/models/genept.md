# genept: GenePT gene-text embeddings + pair classifier (LLM-embedding SL baseline)

battery: genept__slbtrain; species: human; needs: GenePT embeddings (Zenodo 10833191, CC BY 4.0; data/raw/genept, sha256 in SOURCES.tsv; unpacked in external/models/genept)

PPI/KG: none (NCBI gene summary text embeddings).

- Source: Chen Y, Zou J. GenePT: a simple but hard-to-beat foundation model for genes and cells built from ChatGPT. bioRxiv 2023, doi:10.1101/2023.10.16.562533. Gene embedding = OpenAI text-embedding-ada-002 of the NCBI gene summary (1536-d). GenePT itself is not an SL model; this is the standard "LLM gene embedding + pair classifier" recipe used by LLM-based SL predictors (e.g. SLAMR's text branch).
- Pair model: PCA(128) of all gene embeddings, features [a*b, |a-b|, cosine], LightGBM (400 trees, lr 0.05, 31 leaves) fitted on SLB human train rows.
- Leakage: none of the 33,703 NCBI summaries mentions "synthetic lethal" (checked); SLB train labels only => **not leaky** (caveat: ada-002 pretraining corpus unknown).
- Status: acquired / adapted / dev-scored. Coverage 14,874/15,048 human dev rows.
- Dev (SLB-1.2): SLB 0.5061, H. sapiens 0.5246. Runtime ~3 min CPU.

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| genept__slbtrain | 0.5216 | 0.5648 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5323 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
