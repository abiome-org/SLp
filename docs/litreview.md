



Scope and central conclusion  

Current scientific scope (September4,2026): the active SLp-1.1 objective is
held-intervention molecular-state prediction, with SL as a downstream test.
Earlier benchmark-first positioning below records a proposed publication
direction and is not a training or selection instruction. The implemented
models and completed failures are described in `MODEL_CARD.md` and
`docs/results.md`; literature ambitions are not empirical results.
  
Synthetic lethality (SL) is a context-dependent interaction in which each single perturbation is compatible with viability but their combination is lethal. More general negative genetic interactions or synergistic fitness defects should not be called SL unless the single-perturbation tolerance and lethality criteria are met. Most computational SL systems treat SL as supervised link prediction: encode two genes, optionally add a cancer or cell-line context, and learn a binary or ranking score from known SL pairs. The SL-Predict formulation would be different. It would learn an intervention-conditioned cellular transition, then decode the predicted state into molecular measurements and viability. SL would be measured as a non-additive viability consequence of the predicted double perturbation rather than as the direct training target of the transition model.  
  
Literature search current through August 25, 2026. “To our knowledge” claims are limited to the papers, supplements, repositories, and database releases inspected by that date. The literature supports this direction, but it narrows the novelty claim. SL-Predict would not be the first cellular world model, the first model to compose perturbations, or the first method to derive SL without pair labels. Constraint-based metabolic models have inferred lethal combinations from simulated physiology for more than a decade. Boolean cancer models simulate combinations, and recent systems including State, AlphaCell, X-Cell, Lingshu-Cell, Chreode, and VCWorld already use virtual-cell or world-model language. GEARS predicts double-perturbation transcriptomes, including both-targets-unseen settings. Cilantro-sl uses foundation-model knockout deltas and single-gene viability pretraining for SL prediction.  
  
The primary intended contribution is empirical: SL-Predict should establish a new state of the art across the full benchmark matrix, with strict cold-start performance as the headline result. That means outperforming the strongest applicable baselines on pair holdout, one-new-gene, two-new-gene, natural-prevalence ranking, context transfer, and independent experimental CDKO tests rather than winning one convenient split.  
  
SL-Predict aims to establish benchmark-wide state-of-the-art synthetic-lethality prediction, including strict two-new-gene and intervention-isolated cold start. Its action-conditioned cellular transition model is the proposed reason it can generalize beyond the pair-label interpolation learned by direct classifiers.  
  
The world-model formulation is the methodological contribution supporting the SOTA claim. Until the experiments are complete, the paper should describe this as the target claim or central hypothesis rather than an established result.  
  
Terminology  
  
A direct SL classifier learns f(gene A, gene B, context) -> SL score. Its inputs can include omics, sequence, a protein interaction graph, a knowledge graph, or foundation-model embeddings, but the output remains directly supervised by pair labels.  
  
An endpoint perturbation predictor learns f(control state, intervention, context) -> post-intervention state. scGen, CPA, GEARS, biolord, PerturbNet, and CellOT fit this category. Most generate one endpoint and do not support stable recursive rollout.  
  
A cellular transition or world model should represent a state, an explicit intervention operator, context and elapsed time; produce a distribution over possible next states; support simultaneous or sequential actions; expose uncertainty; and be evaluated on counterfactual outcomes. A frozen cell encoder plus a pair classifier is not a world model.  
  
Cold-start labels are inconsistent across papers. This review uses explicit descriptions:  
  
- Pair holdout: the test pair is absent from training, but both genes may appear in other training pairs.  
- One-new-gene: exactly one test gene is absent from supervised SL training.  
- Two-new-gene: both test genes are absent from supervised SL training.  
- Intervention-isolated: one or both genes are also absent from perturbation-response pretraining and phenotype-derived features.  
- Context holdout: the cell line, lineage, donor, or assay is absent from training.  
  
CV1, CV2, CV3, C1, C2, C3, “nonoverlap,” and “gene holdout” should never be reported without spelling out the actual gene-overlap rule.  
  
1. Computational synthetic-lethality prediction  
  
1.1 Statistical and matrix methods  
  
Early cancer-specific methods inferred SL from conditional essentiality, patient selection, co-expression, survival, and mutual exclusivity rather than from a learned cellular transition.  
  
- DAISY (2014)  
  - Core idea: Combines genomic survival of co-inactivation states, conditional essentiality from shRNA screens, and co-expression.  
  - Main evidence: Jerby-Arnon et al., “Predicting cancer-specific vulnerability via data-driven detection of synthetic lethality,” Cell. DOI: [https://doi.org/10.1016/j.cell.2014.07.027](https://doi.org/10.1016/j.cell.2014.07.027)  
  - Cold-start limitation: Candidate discovery and biological validation, not a gene-disjoint benchmark.  
- MiSL (2017)  
  - Core idea: Mines mutation-specific amplification/deletion and survival-selection patterns in TCGA.  
  - Main evidence: Sinha et al., Nature Communications 8:15580. DOI: [https://doi.org/10.1038/ncomms15580](https://doi.org/10.1038/ncomms15580)  
  - Cold-start limitation: Context-specific association, not intervention simulation.  
- ISLE (2018)  
  - Core idea: Filters screened candidates using conditional essentiality, patient co-inactivation, survival, and phylogenetic profiles.  
  - Main evidence: Lee et al., Nature Communications 9:2546. DOI: [https://doi.org/10.1038/s41467-018-04647-1](https://doi.org/10.1038/s41467-018-04647-1)  
  - Cold-start limitation: Candidate filtering does not yield a fully inductive pair model.  
- DiscoverSL (2019)  
  - Core idea: Random forest on differential expression, correlation, mutual exclusivity, and shared pathway features.  
  - Main evidence: Das et al., Bioinformatics. DOI: [https://doi.org/10.1093/bioinformatics/bty673](https://doi.org/10.1093/bioinformatics/bty673); code: [https://github.com/shaoli86/DiscoverSL](https://github.com/shaoli86/DiscoverSL)  
  - Cold-start limitation: Pair-level cross-validation; no strict two-new-gene test.  
- SL²MF (2019/2020)  
  - Core idea: Logistic matrix factorization of the SL adjacency, regularized by PPI and GO similarity.  
  - Main evidence: Liu et al., IEEE/ACM TCBB. DOI: [https://doi.org/10.1109/TCBB.2019.2909908](https://doi.org/10.1109/TCBB.2019.2909908); code: [https://github.com/stephenliu0423/SL2MF](https://github.com/stephenliu0423/SL2MF)  
  - Cold-start limitation: Primarily transductive; weak when both genes lack SL topology.  
- GRSMF (2019)  
  - Core idea: Graph-regularized self-representative matrix factorization.  
  - Main evidence: Huang et al., BMC Bioinformatics 20:657. DOI: [https://doi.org/10.1186/s12859-019-3197-3](https://doi.org/10.1186/s12859-019-3197-3)  
  - Cold-start limitation: Strong historical matrix baseline, but topology-dependent.  
- CMF-W (2020)  
  - Core idea: Weighted collective factorization of SL, expression, SCNA, essentiality, pathway, PPI, complex, and other matrices.  
  - Main evidence: Liany et al., Bioinformatics 36:2209-2216. DOI: [https://doi.org/10.1093/bioinformatics/btz893](https://doi.org/10.1093/bioinformatics/btz893); code: [https://github.com/lianyh/CMF-W](https://github.com/lianyh/CMF-W)  
  - Cold-start limitation: Unseen entities require auxiliary-relation coverage; poor strict cold start in later benchmarks.  
- EXP2SL (2020)  
  - Core idea: Semi-supervised model using LINCS shRNA-induced expression for cell-line-specific SL.  
  - Main evidence: Wan et al., Frontiers in Pharmacology 11:112. DOI: [https://doi.org/10.3389/fphar.2020.00112](https://doi.org/10.3389/fphar.2020.00112)  
  - Cold-start limitation: Relevant use of perturbation response, but still a direct SL predictor on fixed genes.  
  
  
These methods remain important baselines. A transition model should beat gene degree, essentiality, pathway distance, matrix factorization, and simple omics classifiers before architectural complexity is credited.  
  
1.2 Graph, knowledge-graph, contrastive, and transformer models  
  
Graph-based SL prediction dominates the modern field. The central limitation is that better gene representations do not change the supervised task. Most systems still learn the SL edge itself.  
  
- DDGCN (2020)  
  - Architecture and inputs: GCN over the SL graph with node and edge dropout.  
  - Reported evidence: Original SynLethDB AUROC 0.8782 and AUPR 0.3442; DOI: [https://doi.org/10.1093/bioinformatics/btaa211](https://doi.org/10.1093/bioinformatics/btaa211)  
  - Relevance to SL-Predict: Cannot naturally represent a gene absent from SL topology.  
- GCATSL (2021)  
  - Architecture and inputs: Multiple PPI/GO-derived graphs, node- and feature-level attention, MLP fusion.  
  - Reported evidence: Balanced random five-fold AUROC/AUPR 0.9375/0.9483 in PT-GNN’s comparison; DOI: [https://doi.org/10.1093/bioinformatics/btab110](https://doi.org/10.1093/bioinformatics/btab110); code: [https://github.com/lichenbiostat/GCATSL](https://github.com/lichenbiostat/GCATSL)  
  - Relevance to SL-Predict: Strong random-pair model that degrades sharply under strict cold start and realistic ranking.  
- SLMGAE (2021)  
  - Architecture and inputs: Multi-view graph autoencoders reconstruct SL, PPI, GO, expression, pathway, and related views, then combine them with attention.  
  - Reported evidence: Best overall model in Feng et al.’s 2024 standardized benchmark; DOI: [https://doi.org/10.1109/JBHI.2021.3079302](https://doi.org/10.1109/JBHI.2021.3079302); code: [https://github.com/DiNg1011/SLMGAE](https://github.com/DiNg1011/SLMGAE)  
  - Relevance to SL-Predict: Required reproducible graph baseline, including CV3.  
- KG4SL (2021)  
  - Architecture and inputs: Message passing over SynLethKG followed by an SL pair classifier.  
  - Reported evidence: In PiLSL’s setup, AUROC/AUPR fell from 0.9427/0.9504 on C1 to 0.5040/0.5176 on strict C3; DOI: [https://doi.org/10.1093/bioinformatics/btab271](https://doi.org/10.1093/bioinformatics/btab271)  
  - Relevance to SL-Predict: Illustrates how a rich KG can still collapse at two-new-gene prediction.  
- PT-GNN (2022)  
  - Architecture and inputs: Pretrains GNNs on PPI/GO reconstruction and fine-tunes for SL links.  
  - Reported evidence: Random balanced AUROC/AUPR 0.9525/0.9551; DOI: [https://doi.org/10.1093/bioinformatics/btac100](https://doi.org/10.1093/bioinformatics/btac100)  
  - Relevance to SL-Predict: High random-split performance does not establish cold start.  
- PiLSL (2022)  
  - Architecture and inputs: Pair-specific enclosing subgraphs from SynLethKG, attentive propagation, and explicit multi-omics.  
  - Reported evidence: C1 0.9538/0.9594; C2 0.7944/0.8156; strict C3 0.6659/0.6709 AUROC/AUPR; DOI: [https://doi.org/10.1093/bioinformatics/btac476](https://doi.org/10.1093/bioinformatics/btac476); code: [https://github.com/JieZheng-ShanghaiTech/PiLSL](https://github.com/JieZheng-ShanghaiTech/PiLSL)  
  - Relevance to SL-Predict: Important historical two-new-gene baseline.  
- NSF4SL (2022)  
  - Architecture and inputs: Negative-sample-free contrastive learning for partner ranking.  
  - Reported evidence: DOI: [https://doi.org/10.1093/bioinformatics/btac462](https://doi.org/10.1093/bioinformatics/btac462)  
  - Relevance to SL-Predict: Avoids treating every unknown pair as negative, but remains trained on known SL relations.  
- MVGCN-iSL (2022/2023)  
  - Architecture and inputs: Cell-specific multiview GCN using omics and biological networks.  
  - Reported evidence: Fan et al., Frontiers in Genetics. DOI: [https://doi.org/10.3389/fgene.2022.1103092](https://doi.org/10.3389/fgene.2022.1103092)  
  - Relevance to SL-Predict: Context-aware direct classifier.  
- SLGNN (2023)  
  - Architecture and inputs: Factor-aware KG embeddings combined with known SL-graph aggregation.  
  - Reported evidence: Strong random-CV claims; DOI: [https://doi.org/10.1093/bioinformatics/btad015](https://doi.org/10.1093/bioinformatics/btad015); code: [https://github.com/zy972014452/SLGNN](https://github.com/zy972014452/SLGNN)  
  - Relevance to SL-Predict: Known-SL topology creates major inductive and leakage concerns.  
- MSGT-SL (2023)  
  - Architecture and inputs: Multi-omics sampling GNN plus transformer attention for long-range dependencies.  
  - Reported evidence: Explicit leave-gene-out experiments; arXiv: [https://arxiv.org/abs/2310.11082](https://arxiv.org/abs/2310.11082)  
  - Relevance to SL-Predict: Relevant transformer baseline, but split nomenclature differs from standardized CV3.  
- ELISL (2024)  
  - Architecture and inputs: Early integration within feature families and late ensemble across sequence, expression, CNV, mutation, dependency, and survival features.  
  - Reported evidence: Tests gene holdout, label-source transfer, and cross-cancer transfer; DOI: [https://doi.org/10.1093/bioinformatics/btad764](https://doi.org/10.1093/bioinformatics/btad764); code: [https://github.com/joanagoncalveslab/ELISL](https://github.com/joanagoncalveslab/ELISL)  
  - Relevance to SL-Predict: Strong precedent for selection-bias and source-disjoint evaluation.  
- MPASL (2024)  
  - Architecture and inputs: Multi-perspective KG attention, ripple propagation, and contrastive layer alignment.  
  - Reported evidence: Random CV AUROC/AUPR 0.9656/0.9798; “leave-out SL” 0.8766/0.8941; DOI: [https://doi.org/10.3389/fphar.2024.1398231](https://doi.org/10.3389/fphar.2024.1398231)  
  - Relevance to SL-Predict: “Leave-out SL” is not automatically equivalent to two-new-gene CV3.  
- MLEC-iSL (2024)  
  - Architecture and inputs: Predicts gene-level SL connectivity with gene, graph, and transformer encoders, then predicts pairs with logistic regression.  
  - Reported evidence: K562 nonoverlap AUPR 0.719, AUROC 0.727, Precision at top 10% 0.819. Prospective 22Rv1 selected panel: AUROC 0.415, AUPR 0.424, top-decile precision 0.418. DOI: [https://doi.org/10.1093/bib/bbae425](https://doi.org/10.1093/bib/bbae425); PMCID: [https://pmc.ncbi.nlm.nih.gov/articles/PMC11361842/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11361842/)  
  - Relevance to SL-Predict: Important cell-specific and prospective precedent. An audit of the inspected repository version found that connectivity targets were computed before the gene split, which can transmit held-out-edge information into training-gene targets. This should be confirmed against a frozen commit and corrected with fold-local connectivity in reproduction.  
- Struct2SL (2025)  
  - Architecture and inputs: Protein sequence, AlphaFold structure, PPI node2vec, SL-graph embeddings, MLP.  
  - Reported evidence: Random pan-cancer AUROC/AUPR 0.9812/0.9798; DOI: [https://doi.org/10.1016/j.csbj.2025.04.012](https://doi.org/10.1016/j.csbj.2025.04.012)  
  - Relevance to SL-Predict: Structure is useful for unseen genes, but the random split and SL-graph input do not establish CV3.  
- SLGRN (2025)  
  - Architecture and inputs: Graph recurrent network with a global context state.  
  - Reported evidence: DOI: [https://doi.org/10.1007/s11427-023-2618-y](https://doi.org/10.1007/s11427-023-2618-y); code: [https://github.com/jyygit/SLGRN](https://github.com/jyygit/SLGRN)  
  - Relevance to SL-Predict: Context-aware direct pair predictor.  
- MiT4SL (2025)  
  - Architecture and inputs: Gene-gene-cell triplets with cell-specific PPI, sequence, and KG representations for transfer to label-poor cell lines.  
  - Reported evidence: DOI: [https://doi.org/10.1101/2025.04.20.649694](https://doi.org/10.1101/2025.04.20.649694); code: [https://github.com/JieZheng-ShanghaiTech/MiT4SL](https://github.com/JieZheng-ShanghaiTech/MiT4SL)  
  - Relevance to SL-Predict: Addresses context transfer more directly than gene cold start.  
- ESM4SL (2025)  
  - Architecture and inputs: ESM-2 protein embeddings with cell-line-specific SL supervision.  
  - Reported evidence: DOI: [https://doi.org/10.1109/EMBC58623.2025.11254319](https://doi.org/10.1109/EMBC58623.2025.11254319)  
  - Relevance to SL-Predict: Sequence enables unseen-gene features, but gene-holdout performance drops sharply.  
- KG-SLomics (2025)  
  - Architecture and inputs: Relational KG attention plus cancer/cell-line multi-omics and symmetric pair features.  
  - Reported evidence: Reports strong random-pair and unseen-gene results and an external 22Rv1 test; IEEE TCBB record 11239444  
  - Relevance to SL-Predict: A recent context-aware baseline whose exact unseen-gene rule must be matched before comparison.  
- DGIB4SL (2025)  
  - Architecture and inputs: Motif-based high-order KG GNN with graph information bottleneck and diverse explanations.  
  - Reported evidence: DOI: [https://doi.org/10.1093/bib/bbaf142](https://doi.org/10.1093/bib/bbaf142)  
  - Relevance to SL-Predict: Useful ranking and interpretability comparator, still direct KG link prediction.  
- Cilantro-sl (2026)  
  - Architecture and inputs: Geneformer in-silico KO delta embeddings, Gene2vec conditioning, DepMap single-KO viability pretraining, direct pair classifier, conformal uncertainty.  
  - Reported evidence: Gene-holdout AUPR 0.7148; DOI: [https://doi.org/10.64898/2026.02.25.708096](https://doi.org/10.64898/2026.02.25.708096); code: [https://github.com/kaileyhh/Cilantro-SL](https://github.com/kaileyhh/Cilantro-SL)  
  - Relevance to SL-Predict: Closest direct SL precedent. It lacks a measured action-conditioned state transition and still trains a pair classifier.  
- SynLeaF (2026)  
  - Architecture and inputs: Cross-VAE product-of-experts fusion of expression, mutation, methylation, and CNV; RGCN over SynLethKG; adaptive knowledge-distillation or ensemble fusion.  
  - Reported evidence: Pan-cancer AUROC/AUPR: CV1 0.9652/0.9669, CV2 0.8624/0.8754, strict CV3 0.7407/0.7611. arXiv: [https://arxiv.org/abs/2603.22369](https://arxiv.org/abs/2603.22369); code: [https://github.com/Jmpax404/SynLeaF](https://github.com/Jmpax404/SynLeaF)  
  - Relevance to SL-Predict: Strongest inspected balanced CV3 AUPR claim. It is a v1 preprint using balanced labels and full test-gene features.  
- SLAMR (2026)  
  - Architecture and inputs: Cell-line-conditioned LLM gene summaries, text embeddings, SL-edge-stripped KG, ESM-2, contrastive fusion, BPR ranking.  
  - Reported evidence: Strict K562 both-cold MRR 0.09 and Recall@20 0.25; DOI: [https://doi.org/10.1145/3807503.3819499](https://doi.org/10.1145/3807503.3819499); code: [https://github.com/Rrrrachellll/SLAMR](https://github.com/Rrrrachellll/SLAMR)  
  - Relevance to SL-Predict: Recent recommendation-focused CV3 baseline with low absolute performance and model-version contamination concerns.  
  
  
Study status note: Cilantro-sl, SynLeaF, AlphaCell, X-Cell, Lingshu-Cell, Chreode, State, and several other 2025-2026 systems were preprints at the inspected cutoff unless a peer-reviewed venue is explicitly stated. Report author claims separately from independently reproduced results. MGE4SL, KR4SL, EFOL-SL, and rapidly appearing 2026 methods remain a watchlist where code, final versions, or strict split definitions were not fully verified.  
  
1.3 What “SOTA” currently means  
  
There is no single SOTA score.  
  
- Random, balanced pan-cancer pair classification is saturated. Many papers report AUROC/AUPR around 0.95 to 0.98. These numbers are highly sensitive to pair overlap, generated negatives, and use of the SL graph.  
- In Feng et al.’s 2024 standardized comparison, SLMGAE was the strongest overall reproducible baseline across 12 methods, three split types, four class ratios, and three negative-sampling strategies.  
- For a balanced two-new-gene label-cold-start task, SynLeaF reports the strongest inspected AUROC/AUPR claim, 0.7407/0.7611, but it uses randomly generated unknown pairs as negatives when recorded negatives are insufficient and permits sequence, omics, and non-target-relation knowledge-graph features for test genes.  
- PiLSL is a strong historical CV3 baseline at 0.6659/0.6709 AUROC/AUPR in its own setup.  
- Cilantro-sl is the closest foundation-model baseline. Its gene-holdout AUPR is 0.7148, though KG4SL reaches 0.7325 in that particular comparison while Cilantro-sl leads on F1 and uncertainty-aware prediction.  
- Recommendation under realistic imbalance remains weak. SLAMR reports K562 strict both-cold MRR 0.09 and Recall@20 0.25.  
- Prospective experimental discrimination remains unresolved. MLEC-iSL’s second 22Rv1 screen yielded 462 called positives among 987 selected pairs, but the ranking itself had AUROC 0.415 and AUPR 0.424. The 46.8% selected-panel hit rate is not a genome-wide precision estimate.  
  
Every SOTA statement should name the dataset version, context, gene-overlap rule, class prevalence, negative definition, task, metric, and feature access.  
  
2. Benchmark audit  
  
2.1 The 2024 standardized benchmark  
  
Feng et al., “Benchmarking machine learning methods for synthetic lethality prediction in cancer,” Nature Communications 15:9058 (2024), DOI: [https://doi.org/10.1038/s41467-024-52900-7](https://doi.org/10.1038/s41467-024-52900-7), code: [https://github.com/JieZheng-ShanghaiTech/SL_benchmark](https://github.com/JieZheng-ShanghaiTech/SL_benchmark), data: [https://doi.org/10.5281/zenodo.13691648](https://doi.org/10.5281/zenodo.13691648).  
  
It compares 12 methods across pair-random, one-new-gene, and two-new-gene splits; positive-to-negative ratios of 1:1, 1:5, 1:20, and 1:50; random, expression-informed, and dependency-informed negative sampling; and both classification and partner ranking. The benchmark contains 9,845 genes and 35,913 SynLethDB positives, including 9,322 computational predictions, plus an independent K562 panel.  
  
Its main result is methodological rather than architectural. Model rankings change with label provenance, negative sampling, class balance, and evaluation task. SLMGAE performs best overall. Removing computationally inferred positives improves most models. Classification degrades as the test distribution becomes more imbalanced. Nearly every method’s CV3 NDCG@10 falls below 0.01 in the cited setting. This is stronger evidence about the field than a new model’s balanced random-split AUROC.  
  
Adjacent benchmark templates  
  
CausalBench evaluates network inference from more than 200,000 CRISPRi single-cell observations and found that access to interventional data did not automatically make interventional methods superior to observational ones. Chevalley et al., Communications Biology (2025), DOI: [https://doi.org/10.1038/s42003-025-07764-y](https://doi.org/10.1038/s42003-025-07764-y); code: [https://github.com/causalbench/causalbench](https://github.com/causalbench/causalbench).  
  
DREAM3/4/5 network-inference challenges provide hidden-network and held-out-perturbation templates, though their systems are too small or non-human for an SL claim. Archive: [https://gnw.sourceforge.net/dreamchallenge.html](https://gnw.sourceforge.net/dreamchallenge.html).  
  
The AstraZeneca-Sanger DREAM drug-combination challenge contains 11,576 experiments, 910 drug combinations, and 85 molecularly characterized cancer cell lines, including unseen-combination evaluation. It is a useful design template, not an SL benchmark. Menden et al., Nature Communications (2019), DOI: [https://doi.org/10.1038/s41467-019-09799-2](https://doi.org/10.1038/s41467-019-09799-2).  
  
2.2 Label and negative-sampling problems  
  
Most pan-cancer databases mix direct assays, computational predictions, literature mining, and statistical inference. A random unknown gene pair is not a verified non-SL. Treating unknowns as negatives can inflate or distort both AUROC and AUPR. Synthetic rescue should be stored as a separate interaction class rather than merged with non-SL.  
  
SLKB shows that the label problem begins before model training. It integrates 11 combinatorial CRISPR experiments across 22 cell lines, 16,059 SL and 264,424 non-SL pairs, and five scoring methods. Only 1.21% of the methods’ top-decile calls overlap. Source: Gökbağ et al., “SLKB: synthetic lethality knowledge base,” Nucleic Acids Research 52:D1418-D1428, DOI: [https://doi.org/10.1093/nar/gkad806](https://doi.org/10.1093/nar/gkad806); data: [https://doi.org/10.6084/m9.figshare.22902839](https://doi.org/10.6084/m9.figshare.22902839); pipeline: [https://doi.org/10.5281/zenodo.8274172](https://doi.org/10.5281/zenodo.8274172).  
  
A unified benchmark should preserve raw guide counts, quality controls, single-perturbation fitness, double-perturbation fitness, continuous interaction score, uncertainty, binary threshold, scorer, context, time, and provenance. Binary labels should be derived only after the raw data and scoring choices are frozen.  
  
2.3 Leakage taxonomy  
  
Pair-level leakage: reversed duplicates, aliases, replicate pairs, and time points cross folds.  
  
Gene-identity leakage: test genes appear in other supervised pairs. This is expected in pair holdout but forbidden in two-new-gene evaluation.  
  
Label-derived feature leakage: gene degree, SL connectivity, neighbor counts, embeddings, or subgraphs are computed from the complete SL graph before splitting. The inspected MLEC-iSL repository version appears to have this problem; the final paper should cite the audited commit and reproduction script.  
  
Knowledge-graph leakage: test SL, non-SL, synthetic-rescue, or computational-prediction relations remain in the KG. Removing only the target relation type is insufficient if a derived or reciprocal relation preserves the same claim.  
  
Perturbation-pretraining leakage: a gene is “unseen” in SL labels but has single-KO viability, double-KO, transcriptomic, or phenotype supervision in pretraining. This is valid for a label-cold-start track but must not be called intervention-zero-shot.  
  
Foundation-model contamination: a protein or cell model may have seen a held-out screen, a derivative dataset, or an annotation produced from it. Pretraining cutoff dates and dataset manifests are required.  
  
Context leakage: cells from the same donor, cell line, clone, batch, or study appear on both sides of the split.  
  
Test-aware preprocessing: graphs, features, thresholds, or negative pools are built using the full benchmark before folds are defined.  
  
2.4 MLEC-iSL and SynLeaF caveats  
  
MLEC-iSL uses Horlbeck’s K562 and Jurkat CDKO screens. K562 contains 100,128 measured pairs and 1,523 positives at GI < -3, about 1.5% prevalence; Jurkat contains 74,691 pairs and 373 positives, about 0.5%. The released training sets balance positives and negatives 1:1, making random AUPR 0.5 rather than the screen prevalences. “Precision@10” means the top 10% of scored pairs, not the top ten pairs. A strict reproduction should compute connectivity inside each fold, build graphs without test-aware edge removal, evaluate natural prevalence, and publish fixed pair lists.  
  
SynLeaF’s released code correctly uses gene partitions for CV3 and removes SL, non-SL, and synthetic-rescue relation types from SynLethKG before graph construction. Its pan-cancer benchmark contains 33,746 positives but only 3,509 recorded negative or rescue pairs after filtering, so most negatives are randomly sampled unknowns. Each split is balanced 1:1. Test genes retain UniProt sequence, cancer-omics features, and knowledge-graph neighborhoods from relation types other than SL, non-SL, and synthetic rescue. Its result is therefore a strong feature-accessible, SL-label-cold-start score, not evidence for two biologically unseen interventions.  
  
3. Core data resources and remaining accession audit  
  
This section identifies the main resources used by the reviewed models. It is not yet a download-complete inventory: every benchmark release must receive a direct accession or download URL, version or retrieval date, license, checksum where possible, and a model-usage map before the dataset table is treated as exhaustive.  
  
3.1 SL labels and combinatorial viability  
  
- SynLethDB 1.0  
  - Contents: Multi-species SL database; reported 19,952 human pairs.  
  - Best use: Historical pan-cancer graph studies.  
  - Direct access: DOI: [https://doi.org/10.1093/nar/gkv1108](https://doi.org/10.1093/nar/gkv1108)  
- SynLethDB 2.0  
  - Contents: 50,868 SL pairs across species, 35,943 human; 5,798 non-SL and 16,207 synthetic-rescue pairs; SynLethKG.  
  - Best use: Reproducing KG/link-prediction work with provenance flags.  
  - Direct access: [https://synlethdb.sist.shanghaitech.edu.cn/v2](https://synlethdb.sist.shanghaitech.edu.cn/v2) ; DOI: [https://doi.org/10.1093/database/baac030](https://doi.org/10.1093/database/baac030)  
- SynLethDB 3.0 live site  
  - Contents: Live site reports expanded experimental, computational, non-SL, and KG content, but no inspected archival 3.0 methods paper.  
  - Best use: Watchlist only until an export is frozen and checksummed.  
  - Direct access: [https://www.synlethdb.com/](https://www.synlethdb.com/)  
- SLKB  
  - Contents: 11 CDKO studies, 22 cell lines, 280,483 reported pairs, 16,059 SL, 264,424 non-SL; raw counts for ten studies; five scoring pipelines.  
  - Best use: Primary cell-specific benchmark source.  
  - Direct access: [https://slkb.osubmi.org/](https://slkb.osubmi.org/) ; data: [https://doi.org/10.6084/m9.figshare.22902839](https://doi.org/10.6084/m9.figshare.22902839)  
- Horlbeck K562/Jurkat  
  - Contents: 222,784 systematically perturbed pairs across two lines; common GI threshold benchmark.  
  - Best use: Cell-specific training and external testing with natural imbalance.  
  - Direct access: Cell 2018: [https://doi.org/10.1016/j.cell.2018.06.010](https://doi.org/10.1016/j.cell.2018.06.010)  
- MLEC 22Rv1  
  - Contents: First 50-gene all-pairs screen and second 151-gene model-selected panel; raw reads/counts.  
  - Best use: Prospective-style external validation and selection-bias study.  
  - Direct access: GEO: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE262953](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE262953) ; processed: [https://data.mendeley.com/datasets/7shf34snd3/2](https://data.mendeley.com/datasets/7shf34snd3/2)  
- Harle pan-cancer CDKO  
  - Contents: 472 candidate pairs across 27 melanoma, pancreatic, and lung cancer cell lines; 117 unique hits and 882 line-specific interactions.  
  - Best use: Context transfer and external validation.  
  - Direct access: DOI: [https://doi.org/10.1186/s13059-025-03737-w](https://doi.org/10.1186/s13059-025-03737-w) ; ENA PRJEB60853; data: [https://doi.org/10.6084/m9.figshare.25954027.v4](https://doi.org/10.6084/m9.figshare.25954027.v4)  
- De Kegel and paralog screens  
  - Contents: Thousands of paralog pairs with low positive prevalence; scoring-method disagreement is substantial.  
  - Best use: Family-disjoint and biologically hard test sets.  
  - Direct access: See Ajmal et al. 2025: [https://doi.org/10.1093/nargab/lqaf129](https://doi.org/10.1093/nargab/lqaf129)  
- ELISL compilation  
  - Contents: Cancer-specific labels from 25 studies via DiscoverSL, ISLE, EXP2SL, and related sources for eight cancers.  
  - Best use: Source-disjoint and cross-cancer evaluation.  
  - Direct access: Data: [https://doi.org/10.6084/m9.figshare.23607558](https://doi.org/10.6084/m9.figshare.23607558) ; code: [https://github.com/joanagoncalveslab/ELISL](https://github.com/joanagoncalveslab/ELISL)  
  
  
Other CDKO sources catalogued by SLKB include Wong 2016 OVCAR8; Han 2017 K562; Shen 2017 HeLa, 293T, and A549; Najm 2018 Big Papi across six lines; Zhao 2018 HeLa/A549; Ito 2021 across 11 lines; Parrish 2021 pgPEN in PC9/HeLa; Diehl 2021 RPE1; and Thompson 2021 A375/MEWO/RPE1. These screens should be reprocessed from counts rather than pooled as if their published binary calls were interchangeable.  
  
3.2 Omics, dependency, and biological structure  
  
- CCLE and DepMap provide expression, mutation, CNV, methylation where available, and single-gene dependency/essentiality. Direct portal: [https://depmap.org/portal/download/](https://depmap.org/portal/download/)  
- TCGA/cBioPortal provides tumor expression, mutation, CNV, methylation, survival, and cancer context. Direct portal: [https://www.cbioportal.org/](https://www.cbioportal.org/)  
- LINCS L1000 and Connectivity Map provide large-scale drug and genetic perturbation signatures. Direct portal: [https://clue.io/](https://clue.io/)  
- BioGRID provides physical and genetic interactions: [https://thebiogrid.org/](https://thebiogrid.org/)  
- STRING provides scored functional associations: [https://string-db.org/](https://string-db.org/)  
- Reactome provides curated pathways: [https://reactome.org/download-data](https://reactome.org/download-data)  
- KEGG provides pathway maps and relations: [https://www.genome.jp/kegg/download/](https://www.genome.jp/kegg/download/)  
- NCI/Nature Pathway Interaction Database is a legacy pathway source: [https://academic.oup.com/nar/article/37/suppl_1/D674/2506012](https://academic.oup.com/nar/article/37/suppl_1/D674/2506012)  
- OmniPath, DoRothEA, TRRUST, Human1, Recon3D, and COBRA resources support signaling, regulatory, and metabolic baselines.  
- UniProt reviewed human proteins and ESM-type sequence embeddings can define gene representations without SL labels: [https://www.uniprot.org/](https://www.uniprot.org/)  
- AlphaFold structures can provide structural features, but their contribution must be separated from sequence, PPI degree, and SL-graph leakage.  
  
3.3 Perturbation-response data  
  
- Norman et al. 2019 CRISPRa in K562 contains single and paired perturbations with transcriptomic phenotypes. GEARS commonly uses 105 singles and 131 doubles. DOI: [https://doi.org/10.1126/science.aax4438](https://doi.org/10.1126/science.aax4438)  
- Replogle et al. 2022 genome-scale Perturb-seq covers roughly 9,500 genetic perturbations in K562 and RPE1. DOI: [https://doi.org/10.1016/j.cell.2022.05.013](https://doi.org/10.1016/j.cell.2022.05.013)  
- Adamson and Dixit Perturb-seq datasets provide earlier single-gene intervention data.  
- sci-Plex 3 contains 581,777 cells from A549, MCF7, and K562 treated with 188 compounds at four doses.  
- Tahoe-100M contains about 100 million single-cell transcriptomes, 1,100 small molecules, 50 cancer lines, and about 60,000 drug-context experiments. DOI: [https://doi.org/10.1101/2025.02.20.639398](https://doi.org/10.1101/2025.02.20.639398)  
- Parse-PBMC, used by State, contains about 10 million cells, 90 cytokine perturbations, 12 donors, and 18 cell types.  
- X-Atlas/Pisces, used by X-Cell, contains 25.6 million CRISPRi profiles across 16 contexts. Release and licensing require confirmation before benchmark inclusion.  
- Arc Virtual Cell Atlas aggregates observational and perturbational datasets: [https://arcinstitute.org/tools/virtualcellatlas](https://arcinstitute.org/tools/virtualcellatlas)  
- scPerturBench provides processed datasets, 27 method implementations, 29 datasets, and context- and perturbation-generalization evaluation. Code: [https://github.com/bm2-lab/scPerturBench](https://github.com/bm2-lab/scPerturBench)  
  
4. Perturbation prediction and virtual cells  
  
4.1 Endpoint perturbation models  
  
- scGen  
  - Transition formulation: VAE plus an average latent treatment vector transferred between populations. DOI: [https://doi.org/10.1038/s41592-019-0494-8](https://doi.org/10.1038/s41592-019-0494-8)  
  - Generalization evidence: Cell type, study, infection, and species transfer.  
  - Limitation for SL: Mean latent shift; no interaction-aware double perturbation, time, or viability.  
- CPA  
  - Transition formulation: Adversarial autoencoder separates basal state from perturbation, dose, and covariates; perturbation embeddings add compositionally. DOI: [https://doi.org/10.15252/msb.202211517](https://doi.org/10.15252/msb.202211517); code: [https://github.com/theislab/CPA](https://github.com/theislab/CPA)  
  - Generalization evidence: Unseen doses, cell types, times, species, drugs, and some held-out combinations.  
  - Limitation for SL: Categorical embeddings do not handle wholly unseen genes well; one-step endpoint.  
- chemCPA  
  - Transition formulation: CPA with chemical representations and transfer from large bulk screens. arXiv: [https://arxiv.org/abs/2204.13545](https://arxiv.org/abs/2204.13545)  
  - Generalization evidence: Novel compounds and contexts.  
  - Limitation for SL: Chemical generalization does not solve unseen genetic operators.  
- CellOT  
  - Transition formulation: Neural optimal transport maps control-cell distributions to treated distributions. DOI: [https://doi.org/10.1038/s41592-023-01969-x](https://doi.org/10.1038/s41592-023-01969-x); code: [https://github.com/bunnech/cellot](https://github.com/bunnech/cellot)  
  - Generalization evidence: Held-out patients, species, and contexts.  
  - Limitation for SL: Usually learns a treatment-specific map; the unpaired coupling is not identifiable.  
- scVIDR  
  - Transition formulation: VAE with cell-type-aware perturbation vectors and linear dose response. DOI: [https://doi.org/10.1016/j.patter.2023.100817](https://doi.org/10.1016/j.patter.2023.100817)  
  - Generalization evidence: Cross-cell-type, multi-dose, cross-study, and cross-species transfer.  
  - Limitation for SL: Mainly chemical single-perturbation endpoints.  
- biolord  
  - Transition formulation: Disentangles known attributes from residual state and generates unseen attribute combinations. DOI: [https://doi.org/10.1038/s41587-023-02079-x](https://doi.org/10.1038/s41587-023-02079-x)  
  - Generalization evidence: Unseen genetic, chemical, and multimodal combinations.  
  - Limitation for SL: Attribute shifting does not by itself establish causal or recursive dynamics.  
- SAMS-VAE  
  - Transition formulation: Sparse additive mechanism shifts plus sample-specific latent state. arXiv: [https://arxiv.org/abs/2311.02794](https://arxiv.org/abs/2311.02794)  
  - Generalization evidence: OOD combinatorial reasoning under limited data.  
  - Limitation for SL: Additive operators may miss genuine interaction mechanisms.  
- CRADLE-VAE  
  - Transition formulation: Counterfactual latent modulation separates perturbation effects from technical artifacts. arXiv: [https://arxiv.org/abs/2409.05484](https://arxiv.org/abs/2409.05484)  
  - Generalization evidence: Genetic perturbation prediction with artifact disentanglement.  
  - Limitation for SL: Endpoint generator; no viability or stable rollout.  
- PerturbNet  
  - Transition formulation: scVI state model plus conditional invertible network from perturbation embeddings to state distributions. DOI: [https://doi.org/10.1038/s44320-025-00131-3](https://doi.org/10.1038/s44320-025-00131-3)  
  - Generalization evidence: Protein/GO/chemical features support unseen interventions. On one Norman setup, median R² across all genes/DEGs was 0.942/0.629.  
  - Limitation for SL: Strong action-encoder precedent, but score is benchmark-specific and the model is one-step.  
- GEARS  
  - Transition formulation: Expression encoder, GO graph, and set-valued perturbation representation predict multigene transcriptomes and genetic-interaction scores. DOI: [https://doi.org/10.1038/s41587-023-01905-6](https://doi.org/10.1038/s41587-023-01905-6); code: [https://github.com/snap-stanford/GEARS](https://github.com/snap-stanford/GEARS)  
  - Generalization evidence: Norman splits include zero, one, or both perturbed genes unseen as individual interventions. The paper reports its largest relative improvement in the 2/2-unseen class, but the model and split should be compared using the exact original metrics rather than treating the percentage as a cross-paper SOTA score.  
  - Limitation for SL: Closest double-perturbation precursor, but later benchmarks show additive models can outperform it.  
- Latent Causal Diffusion  
  - Transition formulation: Stochastic diffusion model for expression distributions with unseen intervention combinations. arXiv: [https://arxiv.org/abs/2601.15341](https://arxiv.org/abs/2601.15341)  
  - Generalization evidence: Combination prediction and direct-effect estimation under model assumptions.  
  - Limitation for SL: Transcriptomes rather than calibrated viability; stationarity and identification assumptions are strong.  
  
  
4.2 Foundation models are components, not simulators  
  
Geneformer, scGPT, scFoundation, UCE, CellPLM, and GeneCompass can supply cell or gene encoders. Their pretraining scale does not establish intervention prediction.  
  
- Geneformer is a masked transformer on rank-ordered gene lists. In-silico deletion measures embedding change, which is not the same as predicting a measured post-KO transcriptome. DOI: [https://doi.org/10.1038/s41586-023-06139-9](https://doi.org/10.1038/s41586-023-06139-9)  
- scGPT is a generative single-cell transformer pretrained on more than 33 million cells and fine-tuned for perturbation tasks. DOI: [https://doi.org/10.1038/s41592-024-02201-0](https://doi.org/10.1038/s41592-024-02201-0)  
- scFoundation is a 100-million-parameter model trained on more than 50 million cells. Its perturbation task uses learned gene embeddings in a GEARS-like downstream model. DOI: [https://doi.org/10.1038/s41592-024-02305-7](https://doi.org/10.1038/s41592-024-02305-7)  
- UCE is a 650-million-parameter cross-species cell encoder trained on 36 million cells, using ESM2 gene features. It has no native transition head. DOI: [https://doi.org/10.1038/s41586-026-10689-z](https://doi.org/10.1038/s41586-026-10689-z)  
- CellPLM models cell-cell context and spatial structure but has not established action-conditioned dynamics. OpenReview: [https://openreview.net/forum?id=BKXvPDekud](https://openreview.net/forum?id=BKXvPDekud)  
- GeneCompass uses cross-species regulatory priors, but modifying an input expression value and measuring embedding change is not equivalent to learning the biological transition. DOI: [https://doi.org/10.1038/s41422-024-01034-y](https://doi.org/10.1038/s41422-024-01034-y)  
  
Ahlmann-Eltze et al., “Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines,” Nature Methods (2025), DOI: [https://doi.org/10.1038/s41592-025-02772-6](https://doi.org/10.1038/s41592-025-02772-6), is essential negative evidence. On Norman double perturbations, every tested deep model had higher error than the additive single-perturbation baseline. The authors identified 5,035 expression-level interactions at 5% FDR, but no model predicted them well. Observational atlas pretraining contributed little; perturbational pretraining mattered more. SL-Predict must therefore include additive, no-change, mean, linear, low-rank, and perturbation-pretrained linear baselines.  
  
4.3 Current virtual-cell and world-model systems  
  
- State  
  - Evidence: Adduri et al., bioRxiv 2025, DOI: [https://doi.org/10.1101/2025.06.26.661135](https://doi.org/10.1101/2025.06.26.661135); code: [https://github.com/ArcInstitute/state](https://github.com/ArcInstitute/state)  
  - What it establishes: State Embedding trained on 167 million observational cells; State Transition trained on more than 100 million perturbed cells across 70 contexts; predicts treated cell populations using set attention and MMD.  
  - Missing for SL-Predict: Mostly one-step endpoints; no double-KO SL or decoded viability benchmark.  
- AlphaCell  
  - Evidence: Chuai et al., bioRxiv 2026, DOI: [https://doi.org/10.64898/2026.03.02.709176](https://doi.org/10.64898/2026.03.02.709176)  
  - What it establishes: Full-transcriptome latent state and optimal-transport conditional flow matching for perturbation-induced transitions.  
  - Missing for SL-Predict: No inspected recursive rollout or strict two-new-gene SL result.  
- X-Cell  
  - Evidence: Wang et al., bioRxiv 2026, DOI: [https://doi.org/10.64898/2026.03.18.712807](https://doi.org/10.64898/2026.03.18.712807)  
  - What it establishes: Diffusion language model up to 4.9B parameters trained on 25.6M perturbed profiles with text, protein, network, dependency, and morphology priors.  
  - Missing for SL-Predict: “Causal” naming does not prove identified mechanisms; no SL viability evaluation.  
- Lingshu-Cell  
  - Evidence: arXiv: [https://arxiv.org/abs/2603.25240](https://arxiv.org/abs/2603.25240)  
  - What it establishes: Masked discrete diffusion over about 18,000 genes conditioned on cell, donor, and perturbation.  
  - Missing for SL-Predict: No recursive rollout, viability, or double-KO SL evidence.  
- Chreode  
  - Evidence: arXiv: [https://arxiv.org/abs/2605.28111](https://arxiv.org/abs/2605.28111)  
  - What it establishes: scVI encoder plus DiT and structured residual flow for one-step temporal and perturbation prediction.  
  - Missing for SL-Predict: Title correctly limits the claim to one-step dynamics.  
- VCWorld  
  - Evidence: arXiv: [https://arxiv.org/abs/2512.00306](https://arxiv.org/abs/2512.00306); code: [https://github.com/GENTEL-lab/VCWorld](https://github.com/GENTEL-lab/VCWorld)  
  - What it establishes: Structured biological knowledge plus iterative language-model reasoning for differential-expression membership and direction.  
  - Missing for SL-Predict: Closer to a knowledge-driven gene-level simulator than a continuous state transition.  
- A World Model of the Virtual Cell  
  - Evidence: Xing and Song, 2026 report: [https://genbio.ai/research/virtual-cell-may-3.pdf](https://genbio.ai/research/virtual-cell-may-3.pdf)  
  - What it establishes: Explicit multimodal encoder, action-conditioned transition core, and decoder for p(x′  
  - Missing for SL-Predict: x, a, environment), including simultaneous and sequential intervention requirements.  
- AI Virtual Cell priorities  
  - Evidence: Bunne et al., Cell 2024, DOI: [https://doi.org/10.1016/j.cell.2024.11.015](https://doi.org/10.1016/j.cell.2024.11.015)  
  - What it establishes: Defines multimodal, multiscale simulation across states, perturbations, disease, and environment.  
  - Missing for SL-Predict: Broad agenda, not an implemented SL model.  
  
  
5. Mechanistic and causal precedents  
  
5.1 Constraint-based metabolism already derives SL without pair labels  
  
These papers rule out any claim that SL-Predict is the first model to derive lethality from simulated cellular function.  
  
- Suthers et al., “Genome-scale gene/reaction essentiality and synthetic lethality analysis,” Molecular Systems Biology (2009), DOI: [https://doi.org/10.1038/msb.2009.56](https://doi.org/10.1038/msb.2009.56). MILP/FBA enumerates lethal gene and reaction sets.  
- Folger et al., “Predicting selective drug targets in cancer through metabolic networks,” Molecular Systems Biology (2011), DOI: [https://doi.org/10.1038/msb.2011.35](https://doi.org/10.1038/msb.2011.35). Cancer-specific metabolic models use single and joint removals; loss of biomass feasibility identifies lethal combinations.  
- Fast-SL enumerates double and higher-order lethal sets efficiently. Pratapa et al., Bioinformatics (2015), DOI: [https://doi.org/10.1093/bioinformatics/btv352](https://doi.org/10.1093/bioinformatics/btv352); code: [https://github.com/RamanLab/FastSL](https://github.com/RamanLab/FastSL)  
- Apaolaza et al., “An in-silico approach to predict and exploit synthetic lethality in cancer metabolism,” Nature Communications (2017), DOI: [https://doi.org/10.1038/s41467-017-00555-y](https://doi.org/10.1038/s41467-017-00555-y). Genetic minimal cut sets are filtered by cancer expression and experimentally tested in multiple myeloma.  
- gMCS scales minimal gene-cut computation to Recon2/Recon3D. DOI: [https://doi.org/10.1093/bioinformatics/bty656](https://doi.org/10.1093/bioinformatics/bty656)  
- Barrena et al., npj Systems Biology and Applications (2023), DOI: [https://doi.org/10.1038/s41540-023-00296-3](https://doi.org/10.1038/s41540-023-00296-3), integrates Human1 metabolism with signed regulatory paths from OmniPath, DoRothEA, and TRRUST.  
- SL-scan builds cancer-specific iMAT metabolic models and compares simulated dependencies with CRISPR, shRNA, and pharmacological data. DOI: [https://doi.org/10.1038/s41598-023-42992-4](https://doi.org/10.1038/s41598-023-42992-4)  
  
Their assumptions are restrictive: steady state, fixed stoichiometry, known gene-protein-reaction rules, medium specification, and biomass as a viability proxy. They remain necessary mechanistic baselines wherever gene coverage permits.  
  
5.2 Logical cancer models simulate combinations  
  
Boolean models provide explicit intervention semantics and native double perturbations. Fumiã and Martins (PLOS ONE 2013, DOI: [https://doi.org/10.1371/journal.pone.0069008](https://doi.org/10.1371/journal.pone.0069008)) integrate major cancer signaling pathways and simulate targeted attacks. Flobak et al. (PLOS Computational Biology 2015, DOI: [https://doi.org/10.1371/journal.pcbi.1004426](https://doi.org/10.1371/journal.pcbi.1004426)) predicted five gastric-cancer drug synergies, four experimentally confirmed. Béal et al. (Molecular Systems Biology 2020, DOI: [https://doi.org/10.15252/msb.20188664](https://doi.org/10.15252/msb.20188664)) personalize logic models with biopsy screening. Montagud et al. (eLife 2022, DOI: [https://doi.org/10.7554/eLife.72626](https://doi.org/10.7554/eLife.72626)) simulate single and double mutants across prostate tumor and cell-line models and score Bliss-like synergy.  
  
These are context-conditioned combination simulators, but binary states and hand-specified logic suppress dose, time, uncertainty, and cell heterogeneity. Synergy in a proliferation node is not automatically synthetic lethality.  
  
5.3 Learned dynamics and causal models  
  
TrajectoryNet, RNAForecaster, and scNODE learn temporal evolution from destructive single-cell snapshots, but do not define genetic actions. CellOT adds perturbation maps. PerturbODE encodes an interpretable GRN in a neural ODE and simulates unseen interventions (arXiv: [https://arxiv.org/abs/2501.02409](https://arxiv.org/abs/2501.02409)). Latent Causal Diffusion models stochastic expression dynamics and unseen perturbation combinations (arXiv: [https://arxiv.org/abs/2601.15341](https://arxiv.org/abs/2601.15341)). PDGrapher predicts target sets that move a disease state toward a desired state (DOI: [https://doi.org/10.1038/s41551-025-01481-x](https://doi.org/10.1038/s41551-025-01481-x)), but solves an inverse planning problem rather than forward SL simulation.  
  
The main methodological caution is identifiability. Neural ODEs infer continuous paths from cross-sectional measurements. Optimal transport yields a plausible control-treated coupling rather than the true same-cell path. Additive latent operators can fit double endpoints while missing non-additive mechanisms. Perturbational data support causal questions, but the learned representation is not automatically causal.  
  
5.4 Pathway-informed combination models  
  
DrugCell uses a Gene Ontology-guided visible network for cancer genotype and a chemical-structure branch. It was trained directly on drug response and used internal subsystem activations to propose combinations validated by CRISPR, drug screens, and xenografts. It is a major structured-cell precedent but not a transition model. Kuenzi et al., Cancer Cell (2020), DOI: [https://doi.org/10.1016/j.ccell.2020.09.014](https://doi.org/10.1016/j.ccell.2020.09.014); code: [https://github.com/idekerlab/DrugCell](https://github.com/idekerlab/DrugCell).  
  
TranSynergy combines drug targets, PPI propagation, expression, and attention for directly supervised drug-synergy prediction. DOI: [https://doi.org/10.1371/journal.pcbi.1008653](https://doi.org/10.1371/journal.pcbi.1008653). MOViDA extends biologically informed drug-activity prediction with expression and CNV pathway activity. DOI: [https://doi.org/10.1093/bioinformatics/btad432](https://doi.org/10.1093/bioinformatics/btad432). PDGrapher predicts target sets that move a disease state toward a desired state and can be treated as a planner baseline after a forward transition model has been validated. DOI: [https://doi.org/10.1038/s41551-025-01481-x](https://doi.org/10.1038/s41551-025-01481-x).  
  
Biological structure must be tested against randomized and degree-matched graphs. Attention weights, pathway labels, or visible units do not establish a mechanism without intervention-level validation.  
  
6. Transferable world-model ideas  
  
The useful object from the RL literature is the modular transition model, not the visual-generation machinery.  
  
- Ha and Schmidhuber’s World Models (arXiv: [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)) separate observation encoding, stochastic action-conditioned dynamics, and a downstream controller.  
- PlaNet (arXiv: [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)) uses a recurrent state-space model with deterministic memory and stochastic latent state, observation and reward models, and latent planning.  
- Dreamer ([https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603)), DreamerV2 ([https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193)), and DreamerV3 ([https://doi.org/10.1038/s41586-025-08744-2](https://doi.org/10.1038/s41586-025-08744-2)) train downstream behavior from imagined rollouts. The transferable lesson is separation between dynamics learning and downstream objectives.  
- MuZero (DOI: [https://doi.org/10.1038/s41586-020-03051-4](https://doi.org/10.1038/s41586-020-03051-4)) learns a transition sufficient for planning without reconstructing every observation. In biology this can be dangerous: a viability-sufficient latent may discard mechanisms and hide a direct pair classifier inside the transition.  
- Decision Transformer ([https://arxiv.org/abs/2106.01345](https://arxiv.org/abs/2106.01345)) and Trajectory Transformer ([https://arxiv.org/abs/2106.02039](https://arxiv.org/abs/2106.02039)) show sequence conditioning and trajectory tokenization, but Decision Transformer is a policy model rather than a world model.  
- TransDreamer ([https://arxiv.org/abs/2202.09481](https://arxiv.org/abs/2202.09481)), IRIS ([https://arxiv.org/abs/2209.00588](https://arxiv.org/abs/2209.00588)), and STORM ([https://arxiv.org/abs/2310.09615](https://arxiv.org/abs/2310.09615)) show how attention, discrete tokens, and stochastic latent dynamics support long-range or efficient rollout.  
- I-JEPA ([https://arxiv.org/abs/2301.08243](https://arxiv.org/abs/2301.08243)) and V-JEPA ([https://arxiv.org/abs/2404.08471](https://arxiv.org/abs/2404.08471)) motivate prediction in representation space rather than raw observations. A biological model still needs action conditioning and observation-space grounding so that low-abundance but decisive changes are not ignored.  
- GAIA-1 ([https://arxiv.org/abs/2309.17080](https://arxiv.org/abs/2309.17080)), UniSim ([https://arxiv.org/abs/2310.06114](https://arxiv.org/abs/2310.06114)), and Genie ([https://arxiv.org/abs/2402.15391](https://arxiv.org/abs/2402.15391)) show multimodal action conditioning and heterogeneous dataset mixtures. Visual realism has no biological analogue; evaluation must use interventions and measured outcomes.  
  
Direct transfer fails because single-cell assays are destructive, same-cell trajectories are rare, observation times are sparse, contexts are heterogeneous, interventions have uncertain efficacy, and there is no cheap online environment. The model should therefore predict distributions over populations and use assay-aware likelihoods, flow matching, diffusion, optimal transport, or related objectives rather than assume dense Markov trajectories.  
  
7. Proposed SL-Predict formulation  
  
7.1 Modules  
  
Observation encoder E: maps pre-intervention RNA, protein, morphology, mutation, CNV, methylation, lineage, donor, and environment into a latent state or belief state. Each modality should retain an assay-specific likelihood and explicit missingness.  
  
Context c: stable or slow variables such as cell line, tissue, genotype, donor, disease state, batch, medium, dose regime, and perturbation platform.  
  
Action encoder A: represents gene KO/KD/OE, drug, target, dose, efficacy, duration, timing, and combination membership. Simultaneous perturbations should use a permutation-invariant set representation. Sequential interventions require order and elapsed time.  
  
Transition core F: models p(z′ | z, action, context, Δt). A practical starting point is a stochastic state-space or distributional set model with deterministic memory plus stochastic population variation. Candidate backbones include a set transformer, conditional flow matching, diffusion, or a neural ODE when time-resolved data justify it.  
  
Observation decoder D: generates distributions over post-intervention RNA and, where data exist, protein, morphology, pathway activity, and other measurements.  
  
Outcome heads H: estimate viability, proliferation, apoptosis, and assay-specific phenotypes from the evolved state. These heads should remain separate from F. Pair labels must not train the transition core if the paper claims emergent SL.  
  
Uncertainty: aleatoric uncertainty represents cell heterogeneity and assay noise; epistemic uncertainty represents unsupported genes, actions, contexts, and combinations. The model should be able to abstain.  
  
Planner: optional. Once the transition is validated, an experiment selector can choose combinations by expected information gain, uncertainty, or therapeutic value. Planning is not required to establish the world model.  
  
7.2 Training objectives  
  
- Assay-aware masked reconstruction of observational cell states.  
- Latent predictive or JEPA-style objectives for pathway-scale structure.  
- Distributional control-to-perturbed state prediction.  
- Time-conditioned transition loss where time courses exist.  
- Single- and double-perturbation objectives, with explicit comparison against additive composition.  
- Cross-modal consistency for paired or partially paired assays.  
- Composition tests among simultaneous F(z, a ⊕ b), sequential F(F(z, a), b), and reversed F(F(z, b), a). Equality should not be assumed; disagreement should be measured and calibrated.  
- Viability and phenotype heads trained first from single-perturbation outcomes. Any use of double-perturbation viability for calibration should be reported as a separate supervision regime, use pairs disjoint from evaluation, and not be described as fully emergent SL.  
- Calibration and selective-risk objectives for abstention.  
  
7.3 What counts as “emergent SL”  
  
The transition core may learn from molecular responses to double perturbations if the claim concerns absence of binary SL supervision, but the paper must say so. A stronger test holds out all double responses and predicts them from singles. The strongest intervention-zero-shot track also removes single-gene response and dependency supervision for test targets.  
  
A viability head trained directly on double-KO SL labels can recreate the supervised pair task even if F is self-supervised. The cleanest experiment trains F on molecular perturbation response, calibrates H on single-perturbation viability and a separate non-overlapping calibration set, and reserves double-KO interaction labels for evaluation.  
  
8. Unified evaluation plan  
  
8.1 Unit of observation and provenance  
  
Each record should store:  
  
(gene A, gene B, context, perturbation technology, dose/efficacy, time, study, replicate, guide identities, raw counts, single-A fitness, single-B fitness, double fitness, continuous interaction score, binary call, scorer, provenance).  
  
Gene pairs are unordered unless intervention order is part of the task. Aliases, paralogs, and pair orientation must be normalized before splitting.  
  
8.2 Split matrix  
  
1. Known genes, unseen pair.  
2. One gene absent from SL-label training.  
3. Both genes absent from SL-label training.  
4. Two-new-gene plus held-out protein family or pathway.  
5. Both genes absent from perturbation-response training.  
6. Held-out cell line.  
7. Held-out cancer lineage.  
8. Held-out donor or patient-derived model.  
9. Held-out laboratory, screen, or perturbation technology.  
10. New pair and new context jointly.  
11. Temporal external test using only resources released before the test screen.  
12. Prospective model-selected wet-lab validation.  
  
Two CV3 tracks should be published. Feature-accessible CV3 permits sequence, structure, generic PPI/pathway annotations, and observational omics for held-out genes. Intervention-isolated CV3 also removes perturbation responses, dependency features, computational SL predictions, and phenotype-tuned checkpoints involving held-out targets.  
  
8.3 Test labels  
  
Primary negatives should be experimentally measured non-SL pairs. High-confidence cross-scorer consensus negatives can be secondary. Random unknown pairs belong in ranking candidate universes, not as verified negatives. Preserve continuous GI scores and scorer disagreement.  
  
8.4 Baselines  
  
Simple controls:  
  
- Global prevalence.  
- Per-gene SL degree or fold-local connectivity.  
- Single-gene essentiality.  
- No-change prediction.  
- Mean perturbation and mean context.  
- Additive transcriptomic composition.  
- Multiplicative or Bliss-style viability composition.  
- Low-rank linear and Stable-Shift-style models.  
- Sequence similarity, PPI distance, pathway distance, and nearest-neighbor transfer.  
- Constraint-based Fast-SL/gMCS/SL-scan where coverage permits.  
- Logical-model baselines for curated pathways.  
  
Direct SL models:  
  
- SL²MF, GCATSL, SLMGAE, KG4SL, PiLSL, NSF4SL, ELISL, MLEC-iSL with fold-local connectivity, KG-SLomics, ESM4SL, Cilantro-sl, SynLeaF, and SLAMR.  
  
Perturbation models:  
  
- CPA, SAMS-VAE, GEARS, biolord or PerturbNet, State, and a stochastic-dynamics model such as Latent Causal Diffusion if code and scale permit.  
  
The decisive comparison is a direct pair classifier with the same encoder, features, training genes, label budget, and tuning budget as SL-Predict. This separates the value of transition learning from extra data or capacity.  
  
8.5 Metrics  
  
For molecular state prediction:  
  
- Pearson or cosine similarity on perturbation-induced change, not raw expression alone.  
- MSE on prespecified highly variable and differentially expressed genes.  
- DEG AUPRC and precision at fixed N.  
- Effect-size calibration.  
- Perturbation discrimination.  
- MMD, energy distance, Sinkhorn/Wasserstein distance, or another population metric.  
- Genetic-interaction residual error relative to the additive single-perturbation expectation.  
  
For viability and SL:  
  
- AUPRC at natural prevalence and normalized lift over prevalence.  
- AUROC as a secondary metric.  
- NDCG@10/50, Recall@10/50, and Precision@10/50 for partner screening.  
- Continuous GI-score Spearman correlation.  
- Brier score or log loss, expected calibration error, and reliability diagrams.  
- Hit-rate enrichment over the screened-panel baseline.  
- Selective risk versus coverage for abstaining models.  
- Macro averages over genes, contexts, and studies, plus pooled pair-level results.  
  
Report 95% confidence intervals. Use nested validation, fixed public partitions, hierarchical bootstrap over studies, cell lines, and pairs, and paired tests across matched folds or contexts. Individual pairs are not independent experimental replicates.  
  
8.6 Mandatory ablations  
  
- Transition learning versus direct SL supervision.  
- Single-perturbation only versus combination-response training.  
- Viability head with and without any double-pair labels.  
- Fold-local versus global connectivity.  
- Each omics modality.  
- Sequence and structure priors.  
- Graph priors with direct SL/non-SL/rescue edges removed.  
- Real, randomized, and degree-matched biological graphs.  
- Context, time, and dose tokens.  
- Stochasticity versus deterministic transition.  
- Observational pretraining versus perturbational pretraining.  
- Pretraining cutoff date.  
- Performance after regressing out gene degree.  
- Simultaneous versus sequential intervention composition.  
- Uncertainty and abstention.  
  
9. Equity and ancestry  
  
Existing SL datasets do not support a strong ancestry-fairness claim. Most labels come from cell lines or mixed databases without reliable donor ancestry, and ancestry is confounded with cancer lineage, genotype, laboratory history, and assay. The paper can motivate diverse data collection but should not imply that TCGA/CCLE training ensures ancestry generalization.  
  
A defensible study would use models with documented or genetically inferred ancestry, hold out entire donors or models, report calibration and AUPRC lift by group only when sample sizes are adequate, match or adjust for lineage and platform, report worst-group uncertainty, and validate prospectively in diverse patient-derived organoid or iPSC panels. Small groups should be described as exploratory.  
  
10. Novelty map and claims  
  
Direct precedents  
  
- FBA, Fast-SL, gMCS, and SL-scan already derive lethal combinations from modeled metabolic feasibility without pair-label training.  
- Boolean cancer models already simulate context-specific single and double interventions.  
- DrugCell and related pathway-informed models already use structured cellular representations to propose combinations, though they remain supervised response predictors.  
- GEARS predicts double-perturbation transcriptomes and genetic-interaction scores, including both-genes-unseen tests.  
- Cilantro-sl already combines foundation-model in-silico KO deltas with single-gene viability pretraining for SL classification.  
- State and the 2026 virtual-cell models already frame perturbation prediction as cellular transition modeling.  
  
Strong adjacent precedents  
  
CPA, SAMS-VAE, biolord, PerturbNet, CellOT, PerturbODE, Latent Causal Diffusion, and PDGrapher establish compositional perturbation embeddings, population transitions, action-conditioned dynamics, and intervention planning. Assay-aware observation-model precedents include scVI (Lopez et al., 2018, [https://doi.org/10.1038/s41592-018-0229-2](https://doi.org/10.1038/s41592-018-0229-2)), totalVI (Gayoso et al., 2021, [https://doi.org/10.1038/s41592-020-01050-x](https://doi.org/10.1038/s41592-020-01050-x)), MOFA+ (Argelaguet et al., 2020, [https://doi.org/10.1186/s13059-020-02015-1](https://doi.org/10.1186/s13059-020-02015-1)), and MultiVI (Ashuach et al., 2023, [https://doi.org/10.1038/s41592-023-01909-9](https://doi.org/10.1038/s41592-023-01909-9)). These models can supply probabilistic encoders and decoders but do not learn p(state′ | state, action) by themselves.  
  
Primary contribution claim: benchmark-wide SOTA  
  
If the experiments support it, the paper’s strongest contribution is:  
  
SL-Predict establishes a new state of the art across the complete preregistered synthetic-lethality benchmark suite, including pair holdout, one-new-gene, strict two-new-gene, intervention-isolated cold start, context holdout, natural-prevalence partner ranking, and independent experimental CDKO validation.  
  
The architecture-level contribution explains that result:  
  
SL-Predict treats synthetic lethality as a downstream viability consequence of learned cellular response dynamics rather than only as a directly supervised relation. The transition model is trained without synthetic-lethal pair labels, lethality is read from a separate viability model, and evaluation controls graph, omics, dependency, and perturbation-pretraining leakage.  
  
A useful abstract should lead with benchmark-wide SOTA, quantify the margin on strict cold start, and then present the world-model formulation as the mechanism and scientific explanation. “SOTA across all benchmarks” is defensible only if every named benchmark, split, metric, negative definition, and baseline is preregistered and reported, including losses or ties.  
  
Claims to avoid  
  
- “The first cell world model.”  
- “The first model to derive synthetic lethality without pair labels.”  
- “From first principles.”  
- “Mechanistic” because the model uses attention, a pathway graph, or interpretable latent factors.  
- “Causal” merely because the training data contain perturbations.  
- “Emergent SL” if a viability head or transition objective sees the same double-KO labels.  
- “Generalizes to unseen genes” without stating where those genes remain visible.  
- “State of the art” without a task, split, prevalence, negative definition, metric, and data version.  
  
11. Draft related-work synthesis  
  
Computational synthetic-lethality prediction has developed mainly as supervised pair inference. Early systems such as DAISY, MiSL, ISLE, and DiscoverSL combined conditional essentiality with tumor co-occurrence, expression, survival, or phylogenetic evidence. Matrix-factorization methods then treated known SL relations as an adjacency matrix regularized by PPI, GO, expression, and other gene similarities. Recent work has shifted toward multiview graph neural networks, biomedical knowledge graphs, contrastive objectives, protein-language models, and multimodal omics. Representative systems include GCATSL, SLMGAE, KG4SL, PT-GNN, PiLSL, SLGNN, MLEC-iSL, ELISL, KG-SLomics, Cilantro-sl, and SynLeaF. These models improve gene representations, context conditioning, or gene-holdout performance, but their final objective remains direct prediction of an SL label or partner ranking.  
  
Reported scores cannot be compared without reconstructing the evaluation protocol. Random pair splits allow both genes in a test pair to appear throughout training and often yield AUROC or AUPR above 0.95 on balanced data. Feng et al.’s standardized benchmark showed that performance falls with gene-disjoint splits, realistic class imbalance, independent cell-specific testing, and stricter label provenance. SLMGAE was strongest overall in that benchmark, yet nearly all methods had CV3 NDCG@10 below 0.01 in the cited ranking setting. SynLeaF reports AUROC/AUPR of 0.7407/0.7611 on a balanced two-new-gene pan-cancer split, but test genes retain multimodal and knowledge-graph features and most negatives are sampled from unknown pairs. MLEC-iSL reports strong balanced K562 nonoverlap performance but its prospective 22Rv1 selected panel had AUROC 0.415 and AUPR 0.424. These results leave strict experimental cold start unresolved.  
  
Perturbation-response modeling addresses a different problem: predicting a post-intervention molecular state. scGen and CPA introduced latent perturbation arithmetic and compositional covariates; CellOT modeled population transport; biolord, SAMS-VAE, CRADLE-VAE, and PerturbNet developed disentangled or distributional counterfactual generators; GEARS used gene-graph priors and set-valued interventions to predict unseen multigene transcriptomes. State and newer virtual-cell systems scale action-conditioned endpoint prediction to very large perturbational corpora using population attention, flow matching, diffusion, and multimodal priors. These systems provide the closest architectural precedents for SL-Predict, but most are evaluated on one-step transcriptomic reconstruction or differential-expression recovery rather than viability interactions. Independent benchmarking also shows that additive and linear baselines can outperform deep models on double-perturbation expression and interaction recovery.  
  
Mechanistic modeling provides an older parallel line of evidence. Constraint-based metabolic models derive synthetic-lethal sets by simulating single and joint reaction or gene loss, while Boolean cancer models propagate combination interventions through curated signaling logic. These methods show that lethality can emerge from a simulated phenotype without pair-label supervision. Their fixed networks, steady-state assumptions, coarse state spaces, and limited molecular coverage restrict their use as general cell models, but they are direct precedents and important baselines.  
  
SLp-1.1 connects these lines of work by treating a molecular experiment as an explicit perturbation–readout–context query. Variable-size basal and action sets condition sparse molecular readout queries, so data sources need not share one fixed expression panel. The primary empirical objective is held-intervention molecular prediction across species and contexts, not benchmark-wide SL performance. Synthetic lethality remains a downstream stress test after model and readout selection lock. A convincing result requires context and intervention holdouts, perturbation-specific metrics that remove systematic average shifts, simple baselines, source- and species-level preservation, and proof that SL labels do not enter the world-model or reward path.

The 2025 large perturbation model provides a direct precedent for separating perturbation, readout and context when pooling heterogeneous experiments. Systema shows why conventional control-referenced correlations can reward systematic shifts instead of perturbation-specific biology, and therefore motivates perturbed-centroid and landscape-reconstruction evaluation. The independent Nature Methods comparison showing that deep perturbation predictors did not beat simple linear baselines makes those baselines release gates rather than optional context. For yeast, the Costanzo global SGA release provides roughly 23 million double-mutant measurements and broad gene coverage, but its quantitative epsilon scores must remain in their original species and assay context. Cross-species transfer should be learned through sequence, annotation, phylogeny and orthology structure, then tested separately.
  
12. Priority source list  
  
1. Feng et al. 2024 SL benchmark: [https://www.nature.com/articles/s41467-024-52900-7](https://www.nature.com/articles/s41467-024-52900-7)  
2. Benchmark code and data: [https://github.com/JieZheng-ShanghaiTech/SL_benchmark](https://github.com/JieZheng-ShanghaiTech/SL_benchmark) ; [https://doi.org/10.5281/zenodo.13691648](https://doi.org/10.5281/zenodo.13691648)  
3. SLKB: [https://doi.org/10.1093/nar/gkad806](https://doi.org/10.1093/nar/gkad806) ; [https://slkb.osubmi.org/](https://slkb.osubmi.org/)  
4. SynLethDB 2.0: [https://doi.org/10.1093/database/baac030](https://doi.org/10.1093/database/baac030)  
5. Horlbeck K562/Jurkat CDKO map: [https://doi.org/10.1016/j.cell.2018.06.010](https://doi.org/10.1016/j.cell.2018.06.010)  
6. MLEC-iSL: [https://pmc.ncbi.nlm.nih.gov/articles/PMC11361842/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11361842/)  
7. SynLeaF: [https://arxiv.org/abs/2603.22369](https://arxiv.org/abs/2603.22369)  
8. Cilantro-sl: [https://doi.org/10.64898/2026.02.25.708096](https://doi.org/10.64898/2026.02.25.708096)  
9. SLAMR: [https://doi.org/10.1145/3807503.3819499](https://doi.org/10.1145/3807503.3819499)  
10. SLMGAE: [https://doi.org/10.1109/JBHI.2021.3079302](https://doi.org/10.1109/JBHI.2021.3079302)  
11. PiLSL: [https://doi.org/10.1093/bioinformatics/btac476](https://doi.org/10.1093/bioinformatics/btac476)  
12. ELISL: [https://doi.org/10.1093/bioinformatics/btad764](https://doi.org/10.1093/bioinformatics/btad764)  
13. GEARS: [https://doi.org/10.1038/s41587-023-01905-6](https://doi.org/10.1038/s41587-023-01905-6)  
14. State: [https://doi.org/10.1101/2025.06.26.661135](https://doi.org/10.1101/2025.06.26.661135)  
15. AlphaCell: [https://doi.org/10.64898/2026.03.02.709176](https://doi.org/10.64898/2026.03.02.709176)  
16. X-Cell: [https://doi.org/10.64898/2026.03.18.712807](https://doi.org/10.64898/2026.03.18.712807)  
17. Lingshu-Cell: [https://arxiv.org/abs/2603.25240](https://arxiv.org/abs/2603.25240)  
18. Chreode: [https://arxiv.org/abs/2605.28111](https://arxiv.org/abs/2605.28111)  
19. VCWorld: [https://arxiv.org/abs/2512.00306](https://arxiv.org/abs/2512.00306)  
20. AI Virtual Cell priorities: [https://doi.org/10.1016/j.cell.2024.11.015](https://doi.org/10.1016/j.cell.2024.11.015)  
21. Ahlmann-Eltze et al. perturbation benchmark: [https://doi.org/10.1038/s41592-025-02772-6](https://doi.org/10.1038/s41592-025-02772-6)  
22. scPerturBench: [https://doi.org/10.1038/s41592-025-02980-0](https://doi.org/10.1038/s41592-025-02980-0)  
23. Norman double perturbations: [https://doi.org/10.1126/science.aax4438](https://doi.org/10.1126/science.aax4438)  
24. Replogle genome-scale Perturb-seq: [https://doi.org/10.1016/j.cell.2022.05.013](https://doi.org/10.1016/j.cell.2022.05.013)  
25. Fast-SL: [https://doi.org/10.1093/bioinformatics/btv352](https://doi.org/10.1093/bioinformatics/btv352)  
26. Cancer metabolic gMCS: [https://doi.org/10.1038/s41467-017-00555-y](https://doi.org/10.1038/s41467-017-00555-y)  
27. Integrated metabolic-regulatory gMCS: [https://doi.org/10.1038/s41540-023-00296-3](https://doi.org/10.1038/s41540-023-00296-3)  
28. Latent Causal Diffusion: [https://arxiv.org/abs/2601.15341](https://arxiv.org/abs/2601.15341)  
29. World Models: [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)  
30. PlaNet: [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)  
31. Dreamer: [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603)  
32. MuZero: [https://doi.org/10.1038/s41586-020-03051-4](https://doi.org/10.1038/s41586-020-03051-4)  
33. V-JEPA: [https://arxiv.org/abs/2404.08471](https://arxiv.org/abs/2404.08471)
34. Large perturbation model: [https://doi.org/10.1038/s43588-025-00870-1](https://doi.org/10.1038/s43588-025-00870-1)
35. Systema perturbation-specific evaluation: [https://doi.org/10.1038/s41587-025-02777-8](https://doi.org/10.1038/s41587-025-02777-8)
36. Costanzo global yeast genetic-interaction map and downloads: [https://thecellmap.org/yeast/costanzo2016/](https://thecellmap.org/yeast/costanzo2016/)

September4,2026 implementation-focused update

[Scouter](https://doi.org/10.1038/s43588-025-00912-8) is a directly relevant
lightweight baseline: it compresses a control transcriptome, concatenates
fixed GenePT text embeddings and decodes a perturbed transcriptome. Its
published training samples individual control/perturbed cells and fits models
separately by dataset. The reported comparisons emphasize top20 differentially
expressed genes. A pseudobulk adaptation with protein features would change
both inputs and sampling and must not be called an exact reproduction. Our
current aggregate source data and perturbation-specific full-panel metrics
therefore require an explicit adapted comparison, or acquisition of the
matching single-cell data. Its primary code is
[PancakeZoy/scouter](https://github.com/PancakeZoy/scouter).

[Frangieh2021 Perturb-CITE-seq](https://doi.org/10.1038/s41588-021-00779-1)
provides paired RNA and surface-protein measurements under control,
interferon-gamma and T-cell co-culture conditions. This is useful for testing
observation heads and within-study environmental transfer. Public processed
data are listed at Broad SCP1064; raw data have a separate DUOS access route.
Acquisition does not imply that differently normalized RNA and protein values
can share one measurement decoder.


### 2026-09-05: context breadth and chemical intervention data

Tahoe-100M's [publisher dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M)
provides RNA counts, source-local Ensembl/token relations, chemical structures,
dose-bearing sample records and plate-matched DMSO controls. The pinned revision
is `2dc57900b7981cfcf5e211527169a0b006546a95`, with publisher CC0 terms.
Its 50-cell-line design motivates a separate typed chemical-intervention arm;
chemical exposure is not interchangeable with genetic deletion or CRISPRi.
Locally acquired metadata do not yet constitute a training dataset.

[Arc's 2026 challenge announcement](https://arcinstitute.org/news/virtual-cell-challenge-2026)
emphasizes perturbation prediction in unseen cell contexts from measured
controls. This is aligned with our stated scientific question. Challenge
outcomes have not been acquired or used for model selection here. Competition
participation is not required for the current development experiments.

[Nadal-Ribelles et al. 2025](https://www.nature.com/articles/s41467-025-57600-4)
profiles yeast deletion strains in control and 0.4 M NaCl for 15 minutes.
The methods describe within-condition mutant-versus-WT Wilcoxon testing and
log2 fold changes. The acquired `FC_genotype.Rdata` preserves the authors'
`logfoldchanges` column, but the supplied summary script starts from upstream
CSV files and does not establish the numerical estimator or assay transform.
Consequently the current adapter preserves a source-specific endpoint rather
than assuming compatibility with other log-normalized expression datasets.
The [author archive](https://zenodo.org/records/14062629) also includes unrelated
fitness and third-party comparisons; those are not part of this RNA acquisition.

### 2026-09-05: current virtual-cell comparison scope

[Arc's current project page](https://arcinstitute.org/virtual-cell-initiative)
now lists STATE version 1.0 in Cell. This supersedes the older preprint-status
note for that specific release. The publisher full text was not retrievable
in this check, so this is a verified release-status update, not a new independent
architecture or performance audit. The existing State embedding/transition
precedent remains relevant; neither molecular-state representation nor
population-conditioned forecasting is a new category introduced by SLp.

The current [Systema paper](https://www.nature.com/articles/s41587-025-02777-8)
and [27-method, 29-dataset benchmark](https://www.nature.com/articles/s41592-025-02980-0)
continue to support testing perturbation-specific effects and cellular-context
generalization separately. Their published rankings cannot establish how an
unreproduced comparator performs on our current raw-count normalization,
static modalities and gene splits. The new count-state prototype is therefore
an internal experiment with matched count-derived controls and ridge, not a
published-method reproduction or a SOTA result.

### 2026-09-05 — World-model reassessment: composition must be grounded

The return to frozen SLp-1 identifies a distinction between supporting a
rollout API and learning an action operator from observed perturbed backgrounds.
SLp-1's measured doubles supervised simultaneous endpoints; its sequential
consistency compared model-generated states. The new pilot therefore tests
observed single-to-double endpoint relations before expanding model capacity.
The endpoints are not measured sequential trajectories, so this factorization
must not be described as recovered temporal dynamics.

Recent primary work considered for the redesign:

- [U-Pert: Unbalanced perturbation dynamics for cell-fate design](https://www.biorxiv.org/content/10.64898/2026.06.30.735555v1)
  (2026 preprint; [author project](https://qiangweipeng.github.io/UPT/)) uses
  condition- and context-conditioned flow matching with velocity and growth
  fields. Its author figures report genetic response, held drug/context/dose
  tuples, and held-donor cytokine analyses. This motivates explicitly modeling
  how actions depend on state. It does not by itself establish genetic
  combination emergence. Endpoint recovered-cell abundance mixes biological
  growth with capture and sampling; SLp will not label a learned count head
  as viability without an appropriate experimental measurement model.
- [MultiFlow](https://github.com/liuq-lab/MultiFlow)
  ([August 2026 preprint](https://doi.org/10.64898/2026.08.20.746112)) provides
  coupled flow matching for paired RNA–ATAC perturbation responses. It is
  relevant to a future multiomic observation model; this round does not have
  the paired RNA–ATAC inputs needed for a matched reproduction. No numerical
  advantage over SLp is inferred from its abstract or repository description.
- [Perturbation response decomposition enables biologically aligned
  generalization to unseen perturbations and cellular contexts](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1.full)
  (July 2026 preprint) separates shared response from perturbation-specific
  structure and cautions that capacity alone need not improve generalization.
  The actionable evaluation principle is to remove additive singles and the
  cross-combination residual average when measuring nonadditive prediction.
  An improvement in total profile correlation alone is insufficient.

These are architecture and evaluation precedents, not copied implementations
or independently reproduced results. For the first test, the simpler
SLp-1-style attention core is sufficient to isolate observed-background
supervision. Fitting a large flow model before establishing this capability
would confound the scientific comparison.
