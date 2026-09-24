# KR4SL (knowledge-graph reasoning for explainable SL prediction)

battery: kr4sl (human); species: human; needs: KG (GO + pathways) + training SL facts, CODER text embeddings of entity names

- Paper: Bioinformatics 39(Supplement_1):i158 (ISMB 2023), https://academic.oup.com/bioinformatics/article/39/Supplement_1/i158/7210467.
- Repo: https://github.com/JieZheng-ShanghaiTech/KR4SL @ 61b5c844839434da5fbf902b81995ee5727ff302 (MIT).
- Released weights: results/trans/trans_best_0fold_model.pkl (transductive, SynLethDB 2.0: leaky; entity set differs, not used).
- Original training data: SynLethDB 2.0 + SynLethKG gene-pathway/GO relations (+ OntoProtein GO-GO).
- Leakage status: **clean** for the SLB-trained variants (fit on SLB human train only; no SL/GI edges or GI-derived inputs; dev labels never read), except where noted below. PPI / KG sources and how GI evidence was removed: no PPI; KG = GO DAG + GAF gene-GO annotations with IGI removed + SynLethKG gene-pathway edges; SL facts only from SLB fit pairs.
- What we changed:
  - KG rebuilt for SLB genes: GO-GO from go-basic.obo, gene-GO from goa_human.gaf with IGI dropped, gene-pathway from KR4SL's kg.txt; fit SL pairs split 50/50 into graph facts and training queries; validation = held-aside families.
  - Released inductive evaluate() references an undefined `filters` (NameError); validation NDCG@50 recomputed with the released cal_ndcg. Positives only (softmax over genes); measured negatives unused by design.
  - Dev pairs scored by querying (a, SL, ?) and (b, SL, ?) in inductive mode over the KG + all fit SL facts; entities not reached in 3 hops score 0.
  - Environment fixes for torch 2.4 / numpy: pretrained entity embeddings kept on the GPU (torch >= 2 refuses CPU-tensor indexing with CUDA indices), ragged answer lists built with dtype=object, numpy < 2 (np.in1d). CODER embeddings of 59,896 entity names computed on the GPU (KR4SL's extract_pretrain_emb.py recipe). SLB-1.3: 15 epochs, best validation NDCG@50 0.28, 31 min on the RTX 3090. Not run on SLB-1.2 (GPU time).
- Adapter: scripts/models/kr4sl/run.sh (build.py, embed.py, run.py); env SLB_BENCH, SLB_SPLIT.

## Results (rows a model does not score get its median score before `slpbench eval`, as `--allow-missing` does)
### SLB1.3 dev
- kr4sl: **SLB 0.5067**; H. sapiens 0.5200, S. cerevisiae 0.5000, S. pombe 0.5000, B. subtilis n/a, C. elegans n/a, D. melanogaster n/a, M. musculus n/a. Human ancestry: AFR 0.5190, EAS 0.5097, EUR 0.5314, unknown 0.5181. Rows scored by the model: 19,149 (the rest get the model's median score).
### SLB1.2 dev
- not scored on this version.
- Status: acquired, env built, adapted, dev-scored.
