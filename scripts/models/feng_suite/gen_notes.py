"""Generate notes/models/<name>.md for the Feng-suite models from the facts below + results/models/<name>_dev.txt and the
run logs. Re-run after new results: `external/models/SL_benchmark/.venv-prep/bin/python scripts/models/feng_suite/gen_notes.py`"""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "external/models/_slb_work/slb1.2/feng/logs"
M = {
    "sl2mf": dict(title="SL2MF (logistic matrix factorisation with GO/PPI regularisation)", feng="SL2MF",
        paper="Liu et al., IEEE/ACM TCBB, doi:10.1109/TCBB.2019.2909908",
        repo="https://github.com/stephenliu0423/SL2MF @ 101a97938eaefb17d195e2227892579816acf3ee (no license stated; data on Google Drive)",
        data="SynLethDB (SL_Human_FinalCheck), GO BP/CC semantic similarity, PPI topology similarity",
        inputs="GO BP + GO CC Wang similarity and PPI-topology (cosine) similarity as graph-Laplacian regularisers (kNN 45); importance weight 50 on training SL pairs",
        allsp=True, extra="Transductive in form (a latent vector per gene), but held-out genes are placed by the GO/PPI Laplacian terms, so they get informative scores."),
    "grsmf": dict(title="GRSMF (graph-regularised self-representative matrix factorisation)", feng="GRSMF",
        paper="Huang et al., BMC Bioinformatics 20:657 (2019), doi:10.1186/s12859-019-3197-3",
        repo="code: GCATSL repo, baseline methods/GRSMF (https://github.com/lichenbiostat/GCATSL @ 1ad960e03a0b9bc4f0c3f35db66b5b45d53bad3b, MIT)",
        data="SynLethDB v1 (6,375 genes), GO BP/CC similarity, BioGRID PPI",
        inputs="GO BP, GO CC similarity and PPI adjacency as graph regularisers",
        allsp=True, extra=""),
    "cmfw": dict(title="CMF-W (weighted collective matrix factorisation)", feng="CMFW",
        paper="Liany et al., Bioinformatics 36:2209-2216 (2020), doi:10.1093/bioinformatics/btz893",
        repo="https://github.com/lianyh/CMF-W @ 8ac0ceedd82a88d8146af320f1887af517b39b7e (no license stated)",
        data="332-gene breast-cancer matrices (SL, co-expression, mutual exclusivity, pathway, PPI, complex); Feng's version uses GO BP/CC + PPI",
        inputs="SL matrix (training pairs) jointly factorised with GO BP, GO CC and PPI matrices (TensorFlow)",
        allsp=True, extra=""),
    "ddgcn": dict(title="DDGCN (dual-dropout GCN)", feng="DDGCN",
        paper="Cai et al., Bioinformatics (2020), doi:10.1093/bioinformatics/btaa211",
        repo="https://github.com/CXX1113/Dual-DropoutGCN @ 8bde3877614a3805bcc0498b9875d8c5bc8401d9 (no license stated)",
        data="SynLethDB v1 (SL_Human_Approved.txt, 6,375 genes)",
        inputs="identity node features + the training SL graph only (no side information)",
        allsp=True, extra="Structurally cannot represent a gene without training SL edges: held-out genes keep their random initial identity embedding, so dev scores are nearly constant (SLB-1.3: 0.70703-0.70711, 449 distinct values) and their ranking is a fixed random gene order. The SLB-1.3 human 0.600 is carried by the AFR group (0.72 on 26 positives from 3 lines) and should be read as chance. SLB-1.3 run stopped at epoch ~840 of max 2000 (CPU time budget); the best-validation-F1 checkpoint matrix saved by the code was used."),
    "gcatsl": dict(title="GCATSL (graph contextualised attention network)", feng="GCATSL",
        paper="Long et al., Bioinformatics (2021), doi:10.1093/bioinformatics/btab110",
        repo="https://github.com/lichenbiostat/GCATSL @ 1ad960e03a0b9bc4f0c3f35db66b5b45d53bad3b (MIT; Zenodo 10.5281/zenodo.4522679)",
        data="SynLethDB v1 (6,375 genes) with BioGRID PPI, GO BP, GO CC feature graphs",
        inputs="PPI, GO BP, GO CC as node features (PCA-128) + local (training SL graph) and global (random-walk-with-restart on it) neighbourhoods, node- and feature-level attention",
        allsp=True, extra="The released train_gcatsl.py only builds its global random-walk matrix on the first call and skips training (`if build_premat==1: continue`); run_model.sh therefore calls it twice. RWR made sparse with early exit (same fixed point). CPU: 187 s/epoch (x200); SLB-1.3 was run on the GPU image (15 s/epoch) with a 40-min cap: stopped at epoch 82 of 200, best-validation-F1 score matrix used. Not scored on SLB-1.2 (CPU too slow; GPU slot used for SLB-1.3)."),
    "slmgae": dict(title="SLMGAE (multi-view graph auto-encoder)", feng="SLMGAE",
        paper="Hao et al., IEEE JBHI (2021), doi:10.1109/JBHI.2021.3079302",
        repo="https://github.com/DiNg1011/SLMGAE @ ba0c7016922fbec569f8adda5153c28a5392ec58 (no license stated)",
        data="SynLethDB v1 (6,375 genes) + GO BP/CC similarity + BioGRID PPI; breast-cancer set as CMF-W",
        inputs="views: GO BP kNN-45 graph, GO CC kNN-45 graph, PPI graph (support views) + training SL graph (main view)",
        allsp=True, extra="Best overall model in Feng et al. 2024."),
    "kg4sl": dict(title="KG4SL (knowledge-graph neural network)", feng="KG4SL",
        paper="Wang et al., Bioinformatics 37:i418 (2021), doi:10.1093/bioinformatics/btab271",
        repo="https://github.com/JieZheng-ShanghaiTech/KG4SL @ be3626a6970479affb95fdf6b1d84a6244e52637 (MIT)",
        data="SynLethDB 2.0 (~36k pairs) + SynLethKG (its kg2id.txt already excludes SL relations)",
        inputs="SynLethKG without SL/SR/non-SL relations (1-hop, 64 sampled neighbours, dim 256); pairs scored for all 78.5M gene pairs at the end",
        allsp=False, extra="SynLethKG is a human KG: human-only (coverage fact). On SLB-1.2 it was trained with the Feng default for this pipeline (all measured negatives, 1:46): the loss fell to the base-rate entropy and every pair got the same score (0.02146), i.e. SLB 0.500. The SLB-1.3 run uses KG4SL's original 1:1 protocol (fit negatives subsampled to the number of positives, SLB_BALANCE=1, scripts/models/kg4sl/run.sh default)."),
    "slgnn": dict(title="SLGNN (factor-aware knowledge-graph neural network)", feng="SLGNN",
        paper="Zhu et al., Bioinformatics (2023), doi:10.1093/bioinformatics/btad015",
        repo="https://github.com/zy972014452/SLGNN @ c56e23fbd20e825c5571ef3e94377944c669ee80 (MIT)",
        data="SynLethDB 2.0 + SynLethKG (24 relation types, no SL)",
        inputs="SynLethKG without SL relations (factor-aware aggregation, 4 factors, 3 hops) + training SL graph",
        allsp=False, extra="Human-only (SynLethKG).", comment="The Feng implementation's final step scores all N(N-1)/2 = 78.5 M gene pairs (~4 h on 8 CPU threads); patched to score only the evaluated pairs (fit/valid train pairs + dev/test-input pairs, scripts/models/feng_suite/make_score_pairs.py), which gives identical scores for those pairs. (A GPU run was not possible: the dgl-cu117 wheel needs CUDA-11 libraries missing from the CUDA-12 NGC image.)"),
    "nsf4sl": dict(title="NSF4SL (negative-sample-free contrastive learning)", feng="NSF4SL",
        paper="Wang et al., Bioinformatics (2022, ECCB suppl.), doi:10.1093/bioinformatics/btac462",
        repo="https://github.com/JieZheng-ShanghaiTech/NSF4SL @ de000dec9993286e3228331c51a543668f6c832d (MIT)",
        data="SynLethDB 2.0 (CV1/CV2/CV3 splits) + SynLethKG TransE embeddings (shipped)",
        inputs="per-gene TransE_l2 (dim 400) SynLethKG embeddings, re-trained here on the SL-free KG (scripts/models/feng_suite/transe.py); BUIR-style online/target encoders trained on positive pairs only",
        allsp=False, extra="Uses positives only (negative-sample-free); measured negatives are ignored by design. map_genes() patched to index all nodes (the original only works when every node has an SL pair)."),
    "ptgnn": dict(title="PT-GNN (pre-trained GNN for biomedical link prediction)", feng="PTGNN",
        paper="Long et al., Bioinformatics (2022), doi:10.1093/bioinformatics/btac100",
        repo="https://github.com/longyahui/PT-GNN @ 4767a8be6f3f82318dc20060591f16520ac2741d (no license; no SL fine-tuning script, pre-training data absent)",
        data="SynLethDB v1 + PPI/GO pre-training graphs",
        inputs="protein-sequence 3-mer word encodings (CNN) + GAT over the training SL graph; Feng's fine-tuning does not restore the pre-trained checkpoint (the restore is commented out), so it is trained from scratch here too",
        allsp=False, extra="Needs UniProt protein sequences (human encodings only prepared)."),
    "pilsl": dict(title="PiLSL (pairwise interaction learning GNN on enclosing subgraphs)", feng="PiLSL",
        paper="Liu et al., Bioinformatics (2022, ECCB suppl.), doi:10.1093/bioinformatics/btac476",
        repo="https://github.com/JieZheng-ShanghaiTech/PiLSL @ 12d23be6ba3d0e75eeb33e0bdd0dc6428deff149 (MIT)",
        data="SynLethDB (32,561 pairs, 9,516 genes) + SynLethKG + 600-d omics features",
        inputs="3-hop enclosing subgraphs over SynLethKG (no SL relations) + relation 0 = training SL pairs (original PiLSL design; Feng's release used all 48M gene pairs as relation 0); learnable entity embeddings; node features random (as Feng: add_feat_emb=False)",
        allsp=False, extra="Human-only (SynLethKG).", comment="NOT SCORED (time): inputs are built (scripts/models/feng_suite/build_pilsl_inputs.py: 76,797 train/valid/dev/test pairs to extract 3-hop enclosing subgraphs for, fit SL pairs as relation-0 edges, re-mapped entity types) and the code is patched (graph-vs-extraction pair lists, pair->db index, worker cap), but enclosing-subgraph extraction over the 2.2 M-triple KG plus 30 epochs of CPU-only dgl-0.4 R-GCN training was estimated at many hours and was not started within the session budget. Run: `SLB_BENCH=... bash scripts/models/pilsl/run.sh`."),
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
    out = {"slb": re.search(r"SLB score: ([0-9.]+)", t).group(1)}
    out.update(dict(re.findall(r"(H\. sapiens|S\. cerevisiae|S\. pombe|S\. pneumoniae)=([0-9.]+|nan)", t)))
    out["anc"] = dict((a, v) for a, v in re.findall(r"human ancestry=(\w+)\s+[0-9,]+\s+[0-9,]+\s+([0-9.]+)", t))
    out["par"] = re.findall(r"pair=(same family \(paralogs\)|different families)\s+[0-9,]+\s+[0-9,]+\s+([0-9.]+)", t)
    return out


def runtime(name):
    f = LOGS / f"{name}.log"
    if f.exists():
        m = re.findall(r"runtime_sec (\d+)", f.read_text())
        if m:
            return int(m[-1])
    return None


for name, d in M.items():
    r, ra = res(name), res(name + "__allspecies")
    rt = runtime(name)
    status = "acquired, env built, adapted" + (", dev-scored (see Results)" if (r or (ROOT / "results/models/slb1.3" / f"{name}_dev.txt").exists()) else " (not scored: see Comment)")
    lines = [f"# {d['title']}", "",
             f"battery: {name} (human, faithful inputs)" + (f", {name}__allspecies (bundle inputs, every benchmark species)" if d["allsp"] else "") +
             f"; species: {'human + all bundle species' if d['allsp'] else 'human'}; needs: {d['inputs'].split(';')[0]}", "",
             f"- Paper: {d['paper']}.",
             f"- Original repo: {d['repo']}. Run here through the Feng et al. 2024 unified implementation (`{d['feng']}`), see notes/models/feng_suite.md for the shared pipeline, patches and leakage audit.",
             f"- Released weights: none. Original training data: {d['data']} (SynLethDB labels include SLB source screens, so no released model could be scored anyway).",
             f"- Inputs on SLB: {d['inputs']}.",
             "- Leakage status: **clean**. Trained only on SLB human train pairs (fit families; valid families for model selection); dev labels never read. PPI / KG sources and GI removal: PPI = BioGRID 5.0.261 physical rows only (no STRING, no genetic rows); GO similarity from NCBI gene2go with IGI and NOT annotations removed; KG (KG models only) = Feng's SynLethKG copy with the SL_GsG / SR_GsrG / NONSL_GnsG relations verified absent; its gene-gene relations are Hetionet INTERACTS_GiG (physical interactome), COVARIES_GcG (evolutionary rate covariation) and REGULATES_GrG (LINCS knock-down signatures), no STRING; residual risk: SynLethKG's gene-GO edges are not evidence-filtered (possible IGI). allspecies variants: bundle ppi.parquet (BioGRID physical + STRING non-experimental, non-textmining channels) and bundle GO (IGI/ND/NOT dropped).",
             f"- Status: {status}.",
             f"- Adapter: scripts/models/{name}/run.sh (SLB_BENCH, SLB_SPLIT, SLB_VARIANTS) -> scripts/models/feng_suite/run_model.sh."]
    if d["extra"]:
        lines.append(f"- Notes: {d['extra']}")
    lines += ["", "## Results (rows a variant does not score get the model's median score before `slpbench eval`, i.e. what `--allow-missing` does; for human-only variants the other species therefore sit at 0.5)"]
    lines += bench_lines([name] + ([name + "__allspecies"] if d["allsp"] else []))
    if rt:
        lines.append(f"- Runtime (SLB-1.2, CPU container, 8-16 threads): {rt/60:.0f} min.")
    if d.get("comment"):
        lines.append(f"- Comment: {d['comment']}")
    (ROOT / "notes/models" / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("wrote", name)
