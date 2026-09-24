# models-features catalogue: acquired but not (separately) scored


battery: none of the models in this file is in the battery (acquired / reference only; reasons per row).
Repos are under external/models/<dir> (gitignored). Status ladder: acquired / env built / ran original / adapted / dev-scored.
Scored models have their own note (dekegel2021, dennler2025, ryan2026_context, daisy, isle, statsl_common, discoversl,
sinatra, slidr, deltadep, exp2sl, llmsynthlet, misl, slant; sub-agent notes: elisl, sbsl, esm4sl, cilantro_sl, synleaf, musl).

| model | dir, commit | license | what it is | leakage of released artefacts | status and reason not scored |
|---|---|---|---|---|---|
| PromptCASL (JieZheng lab 2025) | promptcasl, 260d81d | none stated | BioBERT prompt-tuning on per-cell-line gene omics text templates + KG-neighbour embeddings, cell-line SL | trained on SLKB cell-line labels; KG neighbours from SynLethKG (SL edges) | acquired. Datasets and KG-neighbour files not released (`data/datasets` absent); templates for SLB lines could be regenerated from DepMap, but the KG branch needs an SL-edge-free KG. Not ported (time). |
| LLM4SL (Zhu 2026) | llm4sl, c12132e | none stated | GATv2 on STRING v12 + ESM-2 + DeepSeek-NER weak labels from 1,844 PubMed SL papers | literature-mined SL labels + SynLethDB => leaky | acquired. Graph model whose distinctive input (LLM-mined SL literature) is forbidden by the contract; without it it reduces to GAT+ESM-2 (covered by esm4sl / graph agents). Not run. |
| BASIS SL-Agent (Wu et al. BCB 2026) | basis, f305b8f | none stated | LLM agent: GraphRAG over an SL KG + solver/verifier | KG contains SL edges => leaky; needs API LLMs | acquired. Only its task-A1 question format is used (basis_a1__* in llmsynthlet). |
| PARIS (Benfatto 2021) | paris, 179b67a | MIT | per-dependency random forest on DepMap omics; feature importance = inferred SL | none (DepMap only) | acquired. One RF per dependency gene over ~19k expression features; the SL signal it extracts (dependency ~ partner loss) is what the functional-examination / depmap_ols scores compute directly. Not run (compute). |
| slipt / SLIPT (Kelly & Black) | slipt, e864aed | GPL-2 | expression-based mutual exclusivity chi-square | none | acquired. Equivalent to statsl sof_expr (SoF on expression tertiles). Not separately scored. |
| SLIDE-VIP slideCell / slidePat (Szczurek lab 2022) | slidecell f6ee058, slidepat 4389384 | GPL (DESCRIPTION) | cell-line Wilcoxon tests (SPID/SPEA) + patient tests (exprSL, SoF, SurvLRT, iSurvLRT) | none | acquired. Each test has a statsl equivalent (ess_*, coexp, sof_*, surv_*); dev results in statsl_common.md. |
| mslp (Shao) | mslp, f3213c5 | GPL-3 | SL partners of tumour mutations: GENIE3 + rank products + screens | none | acquired. Mutation-centric like MiSL (few SLB genes are recurrent drivers). Not run. |
| ccSL (Liao 2026) | ccsl, 75534bd | MIT | DepMap dependency with co-mutation confounding correction for 12 tumour suppressors | none | acquired. Defined for 12 TSG drivers only; generic form = deltadep/depmap_ols. Not run. |
| Uhler lab kernel regression (Cai et al.) | uhlerlab_sl, 4ffcfbf | none stated | per-KO kernel regression on DepMap expression/mutation, gradient feature importance | none | acquired. Needs its processed .hkl files (not in repo). Not run. |
| MOSL | mosl, 9f738ea | none | TabNet on genomic sequence features (student project) | unclear labels | acquired. No released data/labels/pipeline to reproduce. Not run. |
| SL-DeployBench | sl_deploybench, 38b95d5 | see repo | deployment-oriented audit of SL pipelines (not a model) | n/a | acquired for reference. |
| Dandage & Landry 2019 | dandage_landry2019, 4d1da07 | none | analysis of paralog dependency robustness (not a predictor) | n/a | acquired for reference. |
| Struct2SL, MiT4SL | (models-graph clones Struct2SL, MiT4SL) | | | | handed to models-graph (board #9). |
| Medea (mims-harvard) | not cloned | | therapeutic-reasoning agent | | out of scope (no SL pair predictor). |
