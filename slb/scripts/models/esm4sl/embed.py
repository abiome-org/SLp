"""ESM-2 (esm2_t33_650M_UR50D) embeddings, adapted from ESM4SL data_preprocess/esm2_gen.py.

usage: python embed.py <seqs.tsv> <mean_dir> <perres_dir> [perres_species]

* mean_dir/<species>.parquet : per-gene mean of layer-33 residue representations (BOS/EOS excluded) over the first
  1022 residues (shared-bundle recipe agreed with models-mechanistic).  float32, columns gene, e0..e1279.
* perres_dir/<species>/<gene>.pt : per-residue layer-33 representations (fp16, [L, 1280]) for genes of
  `perres_species` (default: human) that occur in the benchmark, L capped at 2000 (= ESM4SL's collate max_len).
  Sequences > 1022 aa are embedded in consecutive 1022-residue windows and concatenated (ESM-2 was trained on
  <= 1024 tokens; the original script ran full length, which is infeasible for titin-sized proteins).
Resumable: genes already present are skipped.  Species order: perres species, other benchmark species, (only if
ESM_SPECIES lists them) non-benchmark proteomes.  ESM_BENCH_ONLY=1 restricts to genes that occur in the benchmark.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import esm

W = 1022
MAXRES = 2000
TOKS = int(os.environ.get("ESM_TOKS_PER_BATCH", 16000))


def main(seqs_tsv, mean_dir, perres_dir, perres_species="human"):
    mean_dir, perres_dir = Path(mean_dir), Path(perres_dir)
    mean_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(seqs_tsv, sep="\t").dropna(subset=["seq"])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))
    import slb
    bench_by_sp: dict[str, set] = {}
    for split in ["train", "dev", "test"]:
        try:
            d = slb.load(split)
        except FileNotFoundError:
            continue
        for sp_, dd in d.groupby("species"):
            bench_by_sp.setdefault(sp_, set()).update(set(dd.gene_a.astype(str)) | set(dd.gene_b.astype(str)))
    bench = bench_by_sp.get(perres_species, set())
    bench_only = os.environ.get("ESM_BENCH_ONLY") == "1"   # embed only benchmark genes (fast); else whole proteome

    wdir = Path(__file__).resolve().parents[3] / "external/models/esm4sl/_weights"
    model = bc = None

    # benchmark species first (they gate the models), then the rest of the proteomes
    bench_sp = set(bench_by_sp)
    only = os.environ.get("ESM_SPECIES")
    order = sorted(df.species.unique(), key=lambda s: (s != perres_species, s not in bench_sp, s))
    if only:
        order = [s for s in order if s in only.split(",")]
    if os.environ.get("ESM_SPECIES") is None:
        order = [s for s in order if s in bench_sp]   # default: only species present in the benchmark
    for sp in order:
        g = df[df.species == sp]
        out = mean_dir / f"{sp}.parquet"
        done = set(pd.read_parquet(out, columns=["gene"]).gene) if out.exists() else set()
        pdir = perres_dir / sp
        need_pr = set()
        if sp == perres_species:
            pdir.mkdir(parents=True, exist_ok=True)
            have = {p.stem for p in pdir.glob("*.pt")}
            need_pr = {x for x in bench if x not in have}
        todo = g[~g.gene.isin(done) | g.gene.isin(need_pr)]
        if bench_only:
            todo = todo[todo.gene.isin(bench_by_sp.get(sp, set()))]
        if todo.empty:
            continue
        if model is None:   # load lazily: a fully cached run needs no GPU
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(wdir / "esm2_t33_650M_UR50D.pt"))
            if os.environ.get("ESM_DEVICE", "cuda") == "cpu":   # small top-ups without the (shared) GPU
                torch.set_num_threads(8)
                model = model.eval()
            else:
                model = model.eval().half().cuda()
            bc = alphabet.get_batch_converter()
        # jobs: (gene, window index, subsequence)
        jobs = []
        for gene, seq in zip(todo.gene, todo.seq):
            nwin = (min(len(seq), MAXRES) + W - 1) // W if gene in need_pr else 1
            for k in range(nwin):
                sub = seq[k * W:(k + 1) * W] if gene in need_pr else seq[:W]
                if gene in need_pr and k == nwin - 1:
                    sub = seq[k * W:min(len(seq), MAXRES)]
                jobs.append((gene, k, sub))
        nexp = {}
        for b in jobs:
            nexp[b[0]] = nexp.get(b[0], 0) + 1
        jobs.sort(key=lambda j: len(j[2]))
        means, parts = {}, {}
        t0 = time.time()
        i = 0
        while i < len(jobs):
            L = len(jobs[i][2]) + 2
            j = i
            while j < len(jobs) and (j - i + 1) * (len(jobs[j][2]) + 2) <= TOKS:
                j += 1
            j = max(j, i + 1)
            batch = jobs[i:j]
            _, _, toks = bc([(f"{b[0]}#{b[1]}", b[2]) for b in batch])
            with torch.no_grad():
                rep = model(toks.to(next(model.parameters()).device), repr_layers=[33])["representations"][33]
            for b, r in zip(batch, rep):
                x = r[1:len(b[2]) + 1].float()
                if b[1] == 0:
                    means[b[0]] = x.mean(0).cpu().numpy()
                if b[0] in need_pr:
                    d = parts.setdefault(b[0], {})
                    d[b[1]] = x.half().cpu()
                    if len(d) == nexp[b[0]]:
                        torch.save(torch.cat([d[k] for k in sorted(d)]).clone(), pdir / f"{b[0]}.pt")
                        parts[b[0]] = None
            i = j
            if len(means) and len(means) % 2000 < len(batch):
                print(f"{sp}: {i}/{len(jobs)} windows, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        new = pd.DataFrame(np.stack(list(means.values())), columns=[f"e{k}" for k in range(1280)])
        new.insert(0, "gene", list(means.keys()))
        if out.exists():
            old = pd.read_parquet(out)
            new = pd.concat([old[~old.gene.isin(new.gene)], new])
        new.to_parquet(out, index=False)
        print(f"{sp}: {len(means)} mean embeddings, {len(parts)} per-residue files, {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
