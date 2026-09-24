"""Zero-shot open-weights LLM SL scoring on SLB with vLLM (GPU). No SL-label training.

Modes
  synthlet  LLMsynthlet protocol (Prosz et al. 2026): system + user prompt verbatim from the paper's Supplementary
            Section 1.1 (scripts/models/llmsynthlet/{system_prompt,user_template}.txt), gene pair appended as
            'GENE1, GENE2'; free-text reasoning; score = the last 'SCORE: x' in [-1, 1] (fallback VERDICT).
            --context: one prompt per (cell line, pair) with the SLB cell line name and up to 25 damaging-mutated
            genes of that line (DepMap 24Q4, most recurrently mutated first); otherwise one prompt per pair with
            '<cell_line_name>' = 'human cancer' and no mutation list.
  yesno     BASIS SL-Bench task A1 question ('Is there a synthetic lethal relationship between A and B?'),
            answered with one token; score = P(Yes) / (P(Yes) + P(No)) from first-token log-probabilities.
Outputs a CSV (key/context, score, raw text for synthlet) that run_llm.sh turns into results parquet.
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_common"))
import slb  # noqa: E402

SCORE = re.compile(r"SCORE:\s*\**\s*([-–−]?\s*[01](?:\.\d+)?)")


def parse_score(t: str) -> float:
    m = SCORE.findall(t or "")
    if m:
        v = m[-1].replace("–", "-").replace("−", "-").replace(" ", "")
        try:
            return max(-1.0, min(1.0, float(v)))
        except ValueError:
            pass
    u = (t or "").upper()
    if "VERDICT: SYNTHETIC-LETHAL" in u or "VERDICT: SYNTHETIC LETHAL" in u:
        return 0.5
    if "VERDICT: SYNTHETIC-RESCUE" in u:
        return -0.5
    if "VERDICT: NEUTRAL" in u:
        return 0.0
    return float("nan")


ORGANISM = {"human": "human cells", "scer": "the budding yeast Saccharomyces cerevisiae",
            "spom": "the fission yeast Schizosaccharomyces pombe", "spne": "the bacterium Streptococcus pneumoniae",
            "cele": "the nematode Caenorhabditis elegans", "dmel": "Drosophila melanogaster cells",
            "mmus": "mouse cells", "ecol": "the bacterium Escherichia coli", "bsub": "the bacterium Bacillus subtilis"}


def display_names(sp: str) -> dict:
    """Canonical SLB gene ID -> the name an LLM is most likely to know (standard yeast names)."""
    if sp == "scer":
        t = pd.read_csv(slb.RAW / "ids/SGD_features.tab", sep="\t", header=None, usecols=[1, 3, 4])
        t = t[(t[1] == "ORF") & t[4].notna()]
        return dict(zip(t[3], t[4]))
    if sp == "spom":
        t = pd.read_csv(slb.RAW / "ids/pombase_gene_IDs_names_products.tsv", sep="\t")
        t = t[t.gene_name.notna()]
        return dict(zip(t.gene_systematic_id, t.gene_name))
    return {}


def line_mutations(max_n=25):
    import omics
    mut = omics.ccle_mut() > 0
    freq = mut.mean(0)
    ctx = slb.contexts().set_index("context_id").depmap_id
    out = {}
    for c, dm in ctx.items():
        if isinstance(dm, str) and dm in mut.index:
            g = mut.columns[mut.loc[dm].to_numpy()]
            g = sorted(g, key=lambda x: -freq[x])[:max_n]
            out[c] = ", ".join(g) if g else "none reported"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthlet", "yesno"], required=True)
    ap.add_argument("--split", default=slb.SPLIT)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--context", action="store_true")
    ap.add_argument("--allspecies", action="store_true", help="yesno mode: score every species (organism-aware)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    d = slb.load(a.split)
    h = d[d.species == "human"].copy() if not (a.allspecies and a.mode == "yesno") else d.copy()
    h["key"] = (h.species + ":" if a.allspecies and a.mode == "yesno" else "") + slb.pair_key(h.gene_a, h.gene_b).values
    if a.mode == "synthlet" and a.context:
        items = h[["key", "context_id"]].drop_duplicates()
        muts = line_mutations()
    else:
        items = h[["key"]].drop_duplicates()
        items["context_id"] = ""
    done = set()
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        done = set(zip(prev.key, prev.context_id.fillna("")))
    items = items[[(k, c) not in done for k, c in zip(items.key, items.context_id)]]
    if a.limit:
        items = items.head(a.limit)
    print(f"{len(items):,} prompts to run", file=sys.stderr)
    if items.empty:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    system = (HERE / "system_prompt.txt").read_text().strip()
    template = (HERE / "user_template.txt").read_text().strip()
    prompts = []
    names = {}
    for k, c in zip(items.key, items.context_id):
        g1, g2 = k.split(":", 1)[-1].split("|")
        if a.mode == "synthlet":
            name = c.split(":", 1)[1] if c else "human cancer"
            mut = muts.get(c, "none reported") if c else "none specified"
            user = template.replace("<cell_line_name>", name).replace("<genes_with_deletorious_mutation>", mut)
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": f"{user}\n{g1}, {g2}"}]
        elif a.allspecies:
            spc, pair = k.split(":", 1)
            g1, g2 = pair.split("|")
            nm = names.setdefault(spc, display_names(spc))
            q = (f"In {ORGANISM.get(spc, spc)}, is there a synthetic lethal (or synthetic sick) genetic interaction between "
                 f"{nm.get(g1, g1)} and {nm.get(g2, g2)}? Answer with a single word: Yes or No.")
            msgs = [{"role": "user", "content": q}]
        else:
            q = (f"Is there a synthetic lethal relationship between {g1} and {g2}? "
                 "Answer with a single word: Yes or No.")
            msgs = [{"role": "user", "content": q}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    if a.mode == "synthlet":
        sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05, max_tokens=a.max_tokens,
                            seed=0)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    yes_ids = {i for t in ["Yes", " Yes", "yes", "YES"] for i in tok.encode(t, add_special_tokens=False)[:1]}
    no_ids = {i for t in ["No", " No", "no", "NO"] for i in tok.encode(t, add_special_tokens=False)[:1]}
    B = 512
    rows = list(zip(items.key, items.context_id))
    for s in range(0, len(prompts), B):
        outs = llm.generate(prompts[s:s + B], sp)
        recs = []
        for (k, c), o in zip(rows[s:s + B], outs):
            if a.mode == "synthlet":
                txt = o.outputs[0].text
                recs.append({"key": k, "context_id": c, "score": parse_score(txt), "text": txt})
            else:
                lp = o.outputs[0].logprobs[0]
                py = sum(math.exp(v.logprob) for t, v in lp.items() if t in yes_ids)
                pn = sum(math.exp(v.logprob) for t, v in lp.items() if t in no_ids)
                sc = py / (py + pn) if py + pn > 0 else float("nan")
                recs.append({"key": k, "context_id": c, "score": sc, "text": o.outputs[0].text})
        pd.DataFrame(recs).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
        print(f"{min(s + B, len(prompts)):,}/{len(prompts):,}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
