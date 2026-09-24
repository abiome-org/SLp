"""Generate notes/models/<name>.md for the non-Feng graph models run by models-graph (MVGCN-iSL, MSGT-SL, MLEC-iSL,
Struct2SL, LukePi, KR4SL) from the facts below + results/models/<name>[__variant]_dev.txt."""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
M = {
    "mvgcn_isl": dict(
        ppi="human: BioGRID physical (rows with Experimental System Type = physical of the repo's BIOGRID-9606.csv), CCLE co-expression and DepMap co-essentiality graphs (single-gene data); the BioGRID genetic-interaction view (PPI-genetic) is dropped; no STRING. allspecies: bundle ppi.parquet = BioGRID physical + STRING neighborhood/fusion/cooccurence/coexpression/database channels only (no experimental, no textmining)",
        title="MVGCN-iSL (multi-view GCN for cancer cell-specific SL)",
        battery="mvgcn_isl (human, DepMap cell-line omics), mvgcn_isl__allspecies (bundle views + fitness/ESM-2 features)",
        species="human (faithful); all benchmark species with a bundle (allspecies)",
        needs="gene graphs (PPI, co-expression, co-essentiality, training SL graph) + per-gene node features",
        paper="Fan et al., Frontiers in Genetics (2023), doi:10.3389/fgene.2022.1103092",
        repo="https://github.com/kunjiefan/MVGCNiSL @ 85b7b880be9a86ae0b721bef0b26cc4363026882 (MIT)",
        weights="none", data="Horlbeck 2018 CRISPRi GI maps (K562, Jurkat; GI <= -3 = SL) shipped as *_GI_scores.csv; these screens are SLB sources, so the original training data overlap SLB",
        changes=["Contexts pooled with shared weights: every epoch, each context's pairs are decoded from embeddings computed with that context's node features (DepMap 24Q4 exp / damaging-mut / CN / Chronos of that line; pan-line mean for hTERT-RPE1 and C092); the original trains one model per cell line.",
                 "Views: PPI-physical, co-exp, co-ess (all shipped with the repo) + SLB fit SL graph; the default `PPI-genetic` view (BioGRID genetic interactions) is dropped as leaky; no per-line expression filter of graph nodes (pooled contexts).",
                 "Split: SLB fit families train, held-aside train families validate (balanced, as the original); early stopping on validation loss (patience 150, max 500 epochs, AdamW 1e-4, batch 512).",
                 "Scores averaged over both gene orders (the decoder is asymmetric).", "Wall-clock cap MV_MAXTIME (default 1800 s) because the GPU is shared: on SLB-1.3 training stopped at epoch 63 (best validation loss at epoch 5; patience would have stopped it at epoch 155), best-validation weights used. allspecies: per-species cap 240 s (MV_MAXTIME_SPECIES); views = SL, bundle PPI, bundle STRING co-expression (>= 400); node features = bundle fitness + 16 PCs of ESM-2 650M."],
        script="scripts/models/mvgcn_isl/run.sh (run.py, run_allspecies.py)"),
    "msgt_sl": dict(
        ppi="only the training SL graph is used by the released transformer model (see below); no PPI/KG",
        title="MSGT-SL (multi-omics sampling-based graph transformer)",
        battery="msgt_sl (human)", species="human", needs="SL graph + cell-line omics node features",
        paper="arXiv 2310.11082 (2023)", repo="https://github.com/MSGT-SL/MSGT-SL @ a6ca9c33379190c107c0d1314cddd5edee9ca816 (no license, no README)",
        weights="none", data="Horlbeck 2018 K562 / Jurkat GI maps (SLB sources)",
        changes=["Released code is work-in-progress: main.py is MVGCN-iSL's GCN_pool; the transformer model (GCN_transformer_pool) is in main_version02.py, which keeps only the first input graph (the training SL graph) and computes random-walk neighbour samples but then overwrites them with an empty list, so the transformer attends over the nodes of each 50-pair batch only. We reproduce exactly that released behaviour.",
                 "Contexts pooled as for MVGCN-iSL (per-context DepMap node features); max 100 epochs (released default 1), patience 5 (SLB-1.3 run early-stopped at epoch 8, 3.5 min on GPU), AdamW 1e-4; modified_transformer made device-safe (the released one concatenates a CPU tensor)."],
        script="scripts/models/msgt_sl/run.sh"),
    "mlec_isl": dict(
        ppi="BioGRID physical (repo file, physical rows) + Opticon pathway network (Mendeley data); PPI-genetic view dropped; no STRING; population features from DepMap expression / Chronos",
        title="MLEC-iSL (SL connectivity + graph transformer, cell-specific)",
        battery="mlec_isl (human)", species="human", needs="Opticon pathway + BioGRID physical graphs, DepMap population PCA features, cell-line omics",
        paper="Fan et al., Briefings in Bioinformatics (2024), doi:10.1093/bib/bbae425",
        repo="https://github.com/kunjiefan/MLEC-iSL @ 3cce41d8c31d6ee7ef88720c85ae6e54143aa32f (MIT); data Mendeley 10.17632/7shf34snd3 (Opticon_networks.csv fetched, sha256 in MANIFEST)",
        weights="none", data="Horlbeck 2018 K562 / Jurkat (SLB sources)",
        changes=["Connectivity targets (number of SL partners per gene) are computed from SLB fit rows only, per context (the audited repo computes them before the gene split, leaking held-out edges into training targets).",
                 "Contexts pooled (one optimiser step per context per epoch with that line's omics); PPI-genetic view dropped; node universe restricted to SLB genes + graph genes with mean DepMap log2(TPM+1) > 1 so full attention fits a 24 GB GPU; max 300 epochs / patience 50 (original 1000 / 200) and a 1800 s wall-clock cap (MLEC_MAXTIME; SLB-1.3 run stopped at epoch 32, validation MSE still improving slowly).",
                 "Pair model: logistic regression on (connectivity_a, connectivity_b), trained on true fit connectivity (balanced 1:1, as the original) and applied to predicted connectivity of dev genes; symmetrised."],
        script="scripts/models/mlec_isl/run.sh"),
    "struct2sl": dict(
        ppi="SLB-trained variant: BioGRID physical from the bundle (node2vec rebuilt); released variant: STRING v12 physical links incl. textmining (leaky)",
        title="Struct2SL (AlphaFold2 structure + sequence + PPI embeddings, MLP)",
        battery="struct2sl (human, SLB-retrained, leak-free PPI block), struct2sl__released (authors' weights + STRING-derived features, LEAKY)", species="human",
        needs="per-gene AlphaFold2-contact-map node2vec, SeqVec, STRING-physical node2vec embeddings (17,180 human genes)",
        paper="Computational and Structural Biotechnology Journal (2025), doi:10.1016/j.csbj.2025.04.012",
        repo="https://github.com/hyr-hit/Struct2SL @ ac6a825e1d8bd297ece53dcdf31f9c00cfb5456f (no license stated; figshare data Apache-2.0)",
        weights="figshare 33439123 bestmodel.pt (trained on SynLethDB 2.0 filtered pairs: leaky)",
        data="SynLethDB 2.0 filtered (23,749 positives, computational-only removed) + Human_nonSL + random negatives",
        changes=["Structure (AlphaFold2 contact-map node2vec) and sequence (SeqVec) features: the authors' released per-gene files, unchanged. PPI feature: the released one is node2vec on STRING v12 *physical links*, whose combined score includes the experiments and textmining channels (lead's STRING warning); for the SLB-trained variant it is REBUILT as node2vec (repo defaults: 128-d, walk 80, 10 walks, window 10, p=q=1; PyG Node2Vec, scripts/models/struct2sl/ppi_node2vec.py) on BioGRID physical edges of the shared bundle (data/interim/bundle/human/ppi.parquet). Pairs with a gene outside the feature tables are left out (median-filled).",
                 "slbtrain: MLP trained on SLB fit pairs with measured negatives, early stopping on SLB valid; released: authors' checkpoint, z-score statistics re-estimated from their training pairs (not released).",
                 "Released code applies BCEWithLogits to an already-sigmoided output (double sigmoid); kept as is."],
        script="scripts/models/struct2sl/run.sh"),
    "lukepi": dict(
        ppi="PrimeKG (authors' pre-built HeteroData), not filterable",
        title="LukePi (self-supervised universal KG embedding, PrimeKG HGT)",
        battery="lukepi (human)", species="human (PrimeKG is human)", needs="PrimeKG HeteroData + pre-trained HGT checkpoint",
        paper="Tao et al., bioRxiv 2025 / IEEE (record 11189877)", repo="https://github.com/JieZheng-ShanghaiTech/LukePi @ f4ee6d790ba64ac19cc409cd63e8689325574b94 (MIT)",
        weights="pre-trained HGT Primekg_HGT_0.2_0.001 + kgdata.pkl from the authors' Google Drive (self-supervised on PrimeKG, no SL labels; sha256 in MANIFEST)",
        data="SynLethDB-derived C1/C2/C3 splits (for the fine-tuning head only)",
        changes=["Fine-tuning head trained on SLB fit pairs (measured labels) exactly as src/finetune_LukePi.py + test_LukePi.sh (frozen encoder, 50 epochs, lr1 0.003, last epoch kept); validation AUROC only logged.",
                 "Released src/model.py does not import (IndentationError at `class GIN`); the HGT class is taken from the file text. The checkpoint needs the pre-2.3 PyG HGTConv (torch 1.13.1 + PyG 2.2.0).",
                 "SLB genes mapped to PrimeKG gene/protein nodes by symbol (PrimeKG node table from the MiT4SL data release)."],
        comment="Validation AUROC (held-aside train families, pairs pooled over contexts, unadjusted) reached 0.65 at epoch 50, yet the dev ranking is inverted (human 0.375, unadjusted within-stratum AUROC 0.30-0.47): what the head learns on pooled pairs does not transfer to within-screen, fitness-balanced ranking of held-out genes.",
        leak_extra="POSSIBLY LEAKY: PrimeKG's protein_protein edges integrate several PPI resources (the PrimeKG paper lists STRING among them) and its GO edges are not evidence-filtered, so genetic-interaction-derived edges (STRING experimental channel, GO IGI) may be present; the self-supervised encoder was pre-trained on that graph by the authors and cannot be re-trained here. Treat the lukepi score as 'clean labels, possibly leaky inputs'.",
        script="scripts/models/lukepi/run.sh"),
    "kr4sl": dict(
        ppi="no PPI; KG = GO DAG + GAF gene-GO annotations with IGI removed + SynLethKG gene-pathway edges; SL facts only from SLB fit pairs",
        title="KR4SL (knowledge-graph reasoning for explainable SL prediction)",
        battery="kr4sl (human)", species="human", needs="KG (GO + pathways) + training SL facts, CODER text embeddings of entity names",
        paper="Bioinformatics 39(Supplement_1):i158 (ISMB 2023), https://academic.oup.com/bioinformatics/article/39/Supplement_1/i158/7210467",
        repo="https://github.com/JieZheng-ShanghaiTech/KR4SL @ 61b5c844839434da5fbf902b81995ee5727ff302 (MIT)",
        weights="results/trans/trans_best_0fold_model.pkl (transductive, SynLethDB 2.0: leaky; entity set differs, not used)",
        data="SynLethDB 2.0 + SynLethKG gene-pathway/GO relations (+ OntoProtein GO-GO)",
        changes=["KG rebuilt for SLB genes: GO-GO from go-basic.obo, gene-GO from goa_human.gaf with IGI dropped, gene-pathway from KR4SL's kg.txt; fit SL pairs split 50/50 into graph facts and training queries; validation = held-aside families.",
                 "Released inductive evaluate() references an undefined `filters` (NameError); validation NDCG@50 recomputed with the released cal_ndcg. Positives only (softmax over genes); measured negatives unused by design.",
                 "Dev pairs scored by querying (a, SL, ?) and (b, SL, ?) in inductive mode over the KG + all fit SL facts; entities not reached in 3 hops score 0.", "Environment fixes for torch 2.4 / numpy: pretrained entity embeddings kept on the GPU (torch >= 2 refuses CPU-tensor indexing with CUDA indices), ragged answer lists built with dtype=object, numpy < 2 (np.in1d). CODER embeddings of 59,896 entity names computed on the GPU (KR4SL's extract_pretrain_emb.py recipe). SLB-1.3: 15 epochs, best validation NDCG@50 0.28, 31 min on the RTX 3090. Not run on SLB-1.2 (GPU time)."],
        script="scripts/models/kr4sl/run.sh (build.py, embed.py, run.py)"),
}

def bench_lines(names):
    """Results lines for every bench and variant (slb1.3 first)."""
    import re as _re
    out = []
    for bench in ["slb1.3", "slb1.2"]:
        D = ROOT / "results/models" / ("" if bench == "slb1.2" else bench)
        got = []
        for n in names:
            f = D / f"{n}_dev.txt"
            if not f.exists():
                continue
            t = f.read_text()
            slb = _re.search(r"SLB score: ([0-9.]+)", t).group(1)
            sp = ", ".join(f"{a} {b}" for a, b in _re.findall(r"([A-Z]\. [a-z]+)=([0-9.]+|n/a)", t))
            anc = ", ".join(f"{a} {x}" for a, x in _re.findall(r"human ancestry=(\w+)\s+[0-9,]+\s+[0-9,]+\s+([0-9.]+)", t))
            import pandas as pd
            nsc = len(pd.read_parquet(D / f"{n}_dev.parquet"))
            got.append(f"- {n}: **SLB {slb}**; {sp}. Human ancestry: {anc}. Rows scored by the model: {nsc:,} (the rest get the model's median score).")
        out.append(f"### {bench.upper()} dev")
        out += got if got else ["- not scored on this version."]
    return out


def res(name):
    f = ROOT / "results/models" / f"{name}_dev.txt"
    if not f.exists():
        return None
    t = f.read_text()
    r = {"slb": re.search(r"SLB score: ([0-9.]+)", t).group(1)}
    r.update(dict(re.findall(r"(H\. sapiens|S\. cerevisiae|S\. pombe|S\. pneumoniae)=([0-9.]+|nan)", t)))
    r["anc"] = re.findall(r"human ancestry=(\w+)\s+[0-9,]+\s+[0-9,]+\s+([0-9.]+)", t)
    import pandas as pd
    p = pd.read_parquet(ROOT / "results/models" / f"{name}_dev.parquet"); r["n"] = f"{len(p):,} of 132,722"
    return r


for name, d in M.items():
    L = [f"# {d['title']}", "", f"battery: {d['battery']}; species: {d['species']}; needs: {d['needs']}", "",
         f"- Paper: {d['paper']}.", f"- Repo: {d['repo']}.", f"- Released weights: {d['weights']}.", f"- Original training data: {d['data']}.",
         "- Leakage status: **clean** for the SLB-trained variants (fit on SLB human train only; no SL/GI edges or GI-derived inputs; dev labels never read), except where noted below. PPI / KG sources and how GI evidence was removed: " + d.get("ppi", "see What we changed") + "." + (" " + d["leak_extra"] if d.get("leak_extra") else ""),
         "- What we changed:"] + [f"  - {c}" for c in d["changes"]] + [f"- Adapter: {d['script']}; env SLB_BENCH, SLB_SPLIT.", "", "## Results (rows a model does not score get its median score before `slpbench eval`, as `--allow-missing` does)"]
    names = [name, name + "__allspecies", name + "__released"]
    L += bench_lines(names)
    any_ = any((ROOT / "results/models" / b / f"{v}_dev.txt").exists() for b in ["", "slb1.3"] for v in names)
    if d.get("comment"):
        L.append(f"- Comment: {d['comment']}")
    L.append("- Status: acquired, env built, adapted" + (", dev-scored." if any_ else "; not yet scored (see blockers in the final report)."))
    (ROOT / "notes/models" / f"{name}.md").write_text("\n".join(L) + "\n")
    print("wrote", name)
