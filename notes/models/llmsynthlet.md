# llmsynthlet: zero-shot open-weights LLM SL prediction (Prosz et al. 2026) + BASIS A1 question

battery: llmsynthlet__released (CPU), llmsynthlet__qwen25_7b, llmsynthlet__qwen25_7b_context, basis_a1__qwen25_7b, basis_a1__qwen25_7b_allspecies (GPU, vLLM); species: human (basis_a1 allspecies: every species); needs: Qwen2.5 weights (HF), vLLM venv external/models/llmsynthlet/.venv

PPI/KG: none (zero-shot prompt; pretraining contamination noted under Leakage).

- Paper: Prosz A, Sztupinszki Z, Diossy M, Zimon B, Csabai I, Szallasi Z. Zero-shot biological reasoning with open-weights large language models reproduces CRISPR screen based prediction of synthetic lethal interactions. bioRxiv 2026, doi:10.64898/2026.01.28.702211. Repo https://github.com/Paureel/LLMsynthlet (commit 5014f87, MIT): inference script only.
- Released data (Google Drive folder 1tS3s9dauWEEm_GElWve36PhaCMFdVn0K; data/raw/llmsynthlet, sha256 in SOURCES.tsv): `sorted_hypotheses.jsonl` = Qwen2.5-32B-Instruct scores for 398,277 "clinically relevant" human pairs.
- Prompt: Supplementary Section 1.1 (supplement PDF in data/raw/llmsynthlet), extracted verbatim to scripts/models/llmsynthlet/{system_prompt,user_template}.txt.
- BASIS (Wu et al., ACM BCB 2026, doi:10.1145/3807503.3819483; https://github.com/davidwushi1145/BASIS commit f305b8f): SL-Bench task A1 asks "Is there a synthetic lethal relationship between A and B?". Its SL-Agent uses GraphRAG over an SL knowledge graph (leaky) and a proprietary/API LLM; only the A1 question format is reused here, with a local model and P(Yes) from first-token log-probabilities.

## Leakage
Zero-shot: no SL-label fitting. But the LLMs were pretrained on the scientific literature, which contains the SL screens themselves (Horlbeck 2018, Zhao 2018, Dede 2020, Parrish 2021, ...) and SL databases. Per the contract ("any SL label from papers" not allowed) these are **leaky (pretraining contamination)**, not rankable.

## Status / results
- __released: acquired / dev-scored. Coverage only 60/15,048 human dev rows (the release covers clinically relevant pairs, rarely SLB paralog pairs) -> SLB 0.4998, human 0.4992: uninformative.
- GPU variants: adapters written (scripts/models/llmsynthlet/run_llm.sh, llm_run.py, finalize_llm.py; MODES=yesno|yesno_all|synthlet|synthlet_context, MODEL/TAG overridable); Qwen2.5-7B-Instruct downloaded to external/models/_hf (HF revision a09a354); vLLM 0.30 installed. NOT RUN: withdrawn from the shared-GPU queue on 2026-09-24 because the variants are leaky by construction (pretraining contamination) and the lead prioritised scoring existing non-leaky adapters on SLB-1.3; estimated GPU time ~45 min (7B, both splits). Status: acquired / env built / adapted / not dev-scored.

## SLB-1.3 dev results (results/models/slb1.3/; SLB_BENCH=data/bench/slb1.3)
Headline = mean of human/scer/spom; auxiliary columns are the per-species strata (bsub/cele/dmel have < 20 dev positives, not in the headline). Species a model does not score get a constant (AUROC 0.5); `native` lists the species actually scored (from <name>_dev.coverage.json).

| model | SLB | human | scer | spom | bsub | cele | dmel | paralog | native |
|---|---|---|---|---|---|---|---|---|---|
| llmsynthlet__released | 0.4998 | 0.4995 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.4997 | human |

Reference on SLB-1.3 dev: lgbm 0.5315 (human 0.558), paralog_identity 0.5392 (human 0.608), codependency 0.5149, random 0.4854.
