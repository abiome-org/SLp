# 2026 synthetic-lethality model landscape (audited 2026-09-24)

This is the release-date audit for the SLB-1.3 battery. We searched 2026
preprints, papers, official repositories and released checkpoints for methods
that score gene pairs for synthetic lethality or closely related genetic
interaction outcomes. A **test-scored** row means an SLB adapter was fitted or
run under the benchmark's family holdout and leakage contract; it does not mean
we reproduced the paper's original dataset, model weights or reported score.
The score files and native coverage are in [MODELS_TEST.md](../../MODELS_TEST.md).

| 2026 method | Primary source | SLB status | Reason / evaluated branch |
|---|---|---|---|
| [SynLeaF](https://arxiv.org/abs/2603.22369) | [code](https://github.com/Jmpax404/SynLeaF) | Test-scored | Filtered KG/RGCN across species and omics branch on human, both retrained on SLB train. |
| [GiGCN](https://doi.org/10.1093/bib/bbag470) | [code](https://github.com/wqz2469/GIGCN) | Test-scored | Binary DINES architecture adaptation; released GO similarity matrix is absent, so SLB's filtered GO features are substituted. [Audit](gigcn.md). |
| [MuSL](https://doi.org/10.1109/JBHI.2026.3698476) | [code](https://github.com/JieZheng-ShanghaiTech/MuSL) | Test-scored | Clean GNN across species and human multimodal branch. |
| [Cilantro-SL](https://doi.org/10.64898/2026.02.25.708096) | [code](https://github.com/kaileyhh/Cilantro-SL) | Test-scored | Geneformer perturbation and pair classifier retrained on SLB train; human only. |
| [Ryan context-specific paralog SL](https://doi.org/10.64898/2026.01.19.700065) | [code](https://github.com/cancergenetics/context_specific_paralog_SL) | Test-scored | Clean refits; released weights are separately flagged because their original GEMINI training labels include SLB screens. |
| [PAGAN](https://doi.org/10.64898/2026.01.27.26344931) | [code](https://github.com/RausellLab/PAGAN) | Test-scored | Genes-to-pairs GraphSAGE trained on single-gene essentiality and a filtered SLB graph; human and budding yeast. |
| [SL-Predict](https://github.com/j8ckfi/sl-predict) | [released MAE checkpoint](https://huggingface.co/potteryrage/sl-predict) | Test-scored | Frozen DepMap single-gene encoder and SLB-trained LightGBM adaptation; source model is a 2026 repository/manuscript, not a peer-reviewed publication. |
| [SLxGO / SLxGO+](https://doi.org/10.64898/2026.07.31.742101) | [code](https://github.com/sshameer/SLxGO2026) | Diagnostic test score, unranked | GO-PCA branch only; GO evidence provenance is absent, and full released model's SLant-derived network ANN and SL classifiers use outside interaction labels. |
| [LLMsynthlet](https://github.com/Paureel/LLMsynthlet) | [code](https://github.com/Paureel/LLMsynthlet) | Dev-scored, unranked | Zero-shot LLM weights can contain benchmark literature. Its released scores target a narrow clinical subset. |
| [SLAMR](https://doi.org/10.1145/3807503.3819499) | [model note](slamr.md) | Acquired, unranked | Released embeddings cover 13.7% of SLB human dev; the supplied graph contains SL edges and per-context text inputs are unavailable. |
| [MCKG-SL](https://doi.org/10.1016/j.artmed.2026.103519) | [code](https://github.com/Qian0711/MCKG-SL) | Acquired, unranked | Numeric KG/omics entity IDs have no released gene-symbol mapping for SLB queries; the repository does supply C1/C2/C3 folds. |
| [MGANSL](https://doi.org/10.1186/s12859-025-06345-4) | [code](https://github.com/lijinxinchina/MGANSL) | Acquired, unranked | Released GO/PPI/Gaussian pair-feature matrices include SL-label-derived features; they must be regenerated using SLB train before a clean score. [Audit](mgansl.md). |
| [MSITN](https://doi.org/10.1016/j.eswa.2026.132323) | [code](https://github.com/dlmu-qxl/MSITN-main) | Acquired, unranked | Repository identifies the code as IMSI, runs TensorFlow 1.13 and consumes SynLethKG with SL edges. |
| [Medea](https://doi.org/10.64898/2026.01.16.696667) | [code](https://github.com/mims-harvard/Medea) | Acquired, unranked | Has an SL task; LLM API and provenance-safe literature inputs needed for an SLB submission. |
| [LLM4SL](https://github.com/tjogzt/LLM4SL) | [model catalogue](./_models-features_catalogue.md) | Acquired, unranked | Its literature-mined SL labels and SynLethDB-derived inputs overlap held-out screen knowledge. |
| [BASIS SL-Agent](https://github.com/davidwushi1145/BASIS) | [model catalogue](./_models-features_catalogue.md) | Acquired, unranked | Agent retrieval graph contains SL edges; its question format is separately exercised in the LLM battery. |
| [ccSL](https://github.com/drliaochengcheng-tech/ccSL) | [model catalogue](./_models-features_catalogue.md) | Acquired, represented by controls | Its 12-driver mutation/dependency statistic is covered by the DepMap and delta-dependency controls. |
| [DepMine](https://doi.org/10.1093/bioinformatics/btag337) | [code](https://github.com/UOSbioinformaticslab/depmine) | Reviewed, represented by controls | Interactive biomarker-profile/target dependency mining rather than unrestricted pair scoring; its dependency-switch statistic is represented by SLB's mutation and delta-dependency controls. |
| [paralogSL delta dependency](https://github.com/tjogzt/paralogSL) | [model note](deltadep.md) | Dev-scored | Mutation-stratified single-gene dependency approach; very narrow native coverage. |
| [OCellus](https://doi.org/10.64898/2026.07.08.737248) | preprint | Reviewed, unranked | Multi-task LLM fine-tunes on synthetic-lethality labels; no provenance-safe SLB checkpoint or stand-alone score interface identified. |
| [Xu et al. *E. coli* graph embedding](https://doi.org/10.53941/emicrobe.2026.100006) | [article](https://www.sciltp.com/journals/eMicrobe/articles/2601002815) | Reviewed, outside scored species | Targets *E. coli* SL; SLB keeps conflicting *E. coli* screens as measurements only, without scored labels. No code release was identified in the article. |

The [2024 controlled CV1/CV2/CV3 study](https://www.nature.com/articles/s41467-024-52900-7)
compares 12 older methods on its own SynLethDB dataset. Its results are plotted
in [Figure 1](../../figures/01_gene_exposure.svg). Its pair labels are withheld
in all three splits; CV1 permits *gene identities* to recur across training
pairs. The paper's F1 and NDCG values are not directly comparable with SLB's
fitness-balanced AUROC. SLB offers a family-held-out test only.

This table records methods found in a dated search, rather than a claim that
every preprint anywhere has been discovered. A newly released method is eligible
for the same test protocol when its training labels and features can be audited.
