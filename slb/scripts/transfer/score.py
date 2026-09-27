"""Score the cross-species transfer track (see PROTOCOL.md) and write results/transfer/report.{json,md}.

Human dev and test are both evaluation data here, because no arm selects on human labels. Every test
readout is appended to results/test_evals.jsonl, as `slbench eval` does.

    uv run python scripts/transfer/score.py [--reps 1000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from slbench import evaluate as E

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/transfer"
SPLITS = ("dev", "test")
MARGIN = 0.03  # non-inferiority margin, fixed in PROTOCOL.md
MODELS = ("ontotype", "gbm_nocode", "gbm_xs", "gbm")
ARMS = ["full", "h0", "human_only"] + [f"dose_p{d:02d}_s{s}" for s in (0, 1) for d in (1, 3, 10, 30)]
BASELINES = {  # label-free human baselines, plus the ortholog-open ceiling
    "paralog_identity": "results/slb/paralog_identity_{s}.parquet",
    "depmap_ols": "results/models/slb/depmap_ols__loss_{s}.parquet",
    "codependency": "results/slb/codependency_{s}.parquet",
    "fitness": "results/slb/fitness_{s}.parquet",
    "ortholog_gi_transfer": "results/transfer/_baselines/ortholog_gi_transfer_{s}.parquet",
}
LEADERBOARD = {  # the adapters' own predictions, to tie the harness to the leaderboard
    "lb:ontotype__pooled": "results/models/slb/ontotype__pooled_{s}.parquet",
    "lb:go_ppi_gbm__pooled": "results/models/slb/go_ppi_gbm__pooled_{s}.parquet",
}


class Split:
    def __init__(self, split: str, reps: int):
        self.split = split
        self.gold = E.load_gold(split)
        self.ids = E.input_ids(split)
        self.hmask = pl.Series((self.gold["species"] == "human").to_numpy())
        h = self.gold.filter(self.hmask)
        self.pos = int(h.filter(pl.col("ancestry_group") != "unknown")["label"].sum())
        ia, ib, n = E._family_index(h)
        rng = np.random.default_rng(0 if split == "dev" else 1)
        self.w = [E._family_weights(rng.poisson(1.0, n), ia, ib) for _ in range(reps)]

    def score(self, path: Path) -> dict:
        preds = E.read_predictions(path)
        if self.split == "test":
            E._log_test_eval("test", preds)
        df, missing = E.validated_join(self.gold, preds, allow_missing=True, inputs=self.ids)
        head, parts = E.headline(df)
        h = df.filter(self.hmask)
        boot = np.array([E.species_score(h, "human", w) for w in self.w])
        # human AUROC within paralog (same-family) pairs and within the rest, all contexts pooled
        fam = {k: E._auc(h.filter(pl.col("same_family") == v))[0] for k, v in (("paralog", True), ("other", False))}
        return {"slb": head, "species": parts, "human": parts["human"], "boot": boot, "missing": missing,
                "human_by_pair": fam}


def ci(x: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    sp = {s: Split(s, a.reps) for s in SPLITS}
    wd = sp["dev"].pos / (sp["dev"].pos + sp["test"].pos)
    entries: dict[str, dict] = {}

    def add(name: str, pattern: str):
        paths = {s: ROOT / pattern.format(s=s) for s in SPLITS}
        if not all(p.exists() for p in paths.values()):
            return
        r = {s: sp[s].score(p) for s, p in paths.items()}
        pooled_boot = wd * r["dev"]["boot"] + (1 - wd) * r["test"]["boot"]
        entries[name] = {"r": r, "pooled": wd * r["dev"]["human"] + (1 - wd) * r["test"]["human"],
                         "pooled_boot": pooled_boot}
        print(f"{name:32s} human dev {r['dev']['human']:.3f} test {r['test']['human']:.3f} "
              f"pooled {entries[name]['pooled']:.3f}", flush=True)

    for n, p in {**BASELINES, **LEADERBOARD}.items():
        add(n, p)
    for arm in ARMS:
        for m in MODELS:
            add(f"{arm}/{m}", f"results/transfer/{arm}/{m}_{{s}}.parquet")
    for arm in ("full", "h0"):
        for p in sorted((RES / arm).glob("gbm_nocode_seed*_test.parquet")):
            add(f"{arm}/{p.name[:-len('_test.parquet')]}", str(p.relative_to(ROOT)).replace("_test.parquet", "_{s}.parquet"))
    for p in sorted((RES / "ablation").glob("*/*_test.parquet")):
        add(f"ablation:{p.parent.name}/{p.name[:-len('_test.parquet')]}",
            str(p.relative_to(ROOT)).replace("_test.parquet", "_{s}.parquet"))

    def diff(b: str, a_: str) -> dict:
        eb, ea = entries[b], entries[a_]
        d = eb["pooled_boot"] - ea["pooled_boot"]
        return {"delta": eb["pooled"] - ea["pooled"], "ci95": ci(d), "p_not_better": float(np.mean(d <= 0))}

    out = {"reps": a.reps, "margin": MARGIN, "dev_weight": wd,
           "human_pos": {s: sp[s].pos for s in SPLITS}, "entries": {}, "comparisons": {}, "verdicts": {}}
    for n, e in entries.items():
        out["entries"][n] = {
            "human_pooled": e["pooled"], "human_pooled_ci95": ci(e["pooled_boot"]),
            **{f"human_{s}": e["r"][s]["human"] for s in SPLITS},
            **{f"human_{s}_ci95": ci(e["r"][s]["boot"]) for s in SPLITS},
            **{f"slb_{s}": e["r"][s]["slb"] for s in SPLITS},
            **{f"species_{s}": e["r"][s]["species"] for s in SPLITS},
            **{f"human_by_pair_{s}": e["r"][s]["human_by_pair"] for s in SPLITS},
            "missing_filled": {s: e["r"][s]["missing"] for s in SPLITS}}
    free = [b for b in ("paralog_identity", "depmap_ols", "codependency", "fitness") if b in entries]
    for m in MODELS:
        h0, hp = f"h0/{m}", f"full/{m}"
        if h0 not in entries or hp not in entries:
            continue
        c = {"h0_minus_full": diff(h0, hp)}
        for b in free:
            c[f"h0_minus_{b}"] = diff(h0, b)
            c[f"full_minus_{b}"] = diff(hp, b)
        if f"human_only/{m}" in entries:
            c["full_minus_human_only"] = diff(hp, f"human_only/{m}")
        for arm in ARMS[3:]:
            if f"{arm}/{m}" in entries:
                c[f"{arm}_minus_full"] = diff(f"{arm}/{m}", hp)
        for k in [k for k in entries if k.startswith(f"ablation:h0/{m}__no_")]:
            c[f"{k.split('/')[-1]}_minus_h0"] = diff(k, h0)
        for k in [k for k in entries if k.startswith(f"ablation:full/{m}__no_")]:
            c[f"full:{k.split('/')[-1]}_minus_full"] = diff(k, hp)
        out["comparisons"][m] = c
        best_free = max(free, key=lambda b: entries[b]["pooled"])
        noninf = c["h0_minus_full"]["ci95"][0] > -MARGIN
        beats = all(entries[h0]["pooled"] > entries[b]["pooled"] for b in free) and \
            c["h0_minus_depmap_ols"]["ci95"][0] > 0
        informative = c[f"full_minus_{best_free}"]["ci95"][0] > 0
        out["verdicts"][m] = {"non_inferior": bool(noninf), "beats_label_free": bool(beats),
                              "pass": bool(noninf and beats), "h_plus_beats_best_label_free": bool(informative),
                              "best_label_free": best_free}
    # refit noise: the same arm refitted with GBM seeds 0..N (bagging and internal-validation genes)
    seeds = {arm: [k for k in entries if k == f"{arm}/gbm_nocode" or k.startswith(f"{arm}/gbm_nocode_seed")]
             for arm in ("full", "h0")}
    if min(len(v) for v in seeds.values()) > 1:
        vals = {arm: [entries[k]["pooled"] for k in ks] for arm, ks in seeds.items()}
        paired = [entries[h]["pooled"] - entries[f]["pooled"] for h, f in zip(seeds["h0"], seeds["full"])]
        out["seed_noise"] = {"models": seeds, "human_pooled": vals,
                             "sd": {arm: float(np.std(v, ddof=1)) for arm, v in vals.items()},
                             "h0_minus_full_by_seed": paired,
                             "h0_mean_minus_full_mean": float(np.mean(vals["h0"]) - np.mean(vals["full"]))}
    (RES / "report.json").write_text(json.dumps(out, indent=1))
    (RES / "report.md").write_text(report(out))
    print(report(out))


def f3(x):
    return "n/a" if x is None or np.isnan(x) else f"{x:.3f}"


def report(o: dict) -> str:
    E_ = o["entries"]
    L = ["# Cross-species transfer track", "",
         (f"Human positives scored (ancestry groups with >= 20): dev {o['human_pos']['dev']}, test "
          f"{o['human_pos']['test']}. Pooled human = {o['dev_weight']:.2f} x dev + {1 - o['dev_weight']:.2f} x test. "
          f"Family-cluster bootstrap, {o['reps']} reps. Protocol: `scripts/transfer/PROTOCOL.md`."), "",
         "## Verdicts", "", "| model | H0 − H+ (pooled human, 95% CI) | non-inferior (> −0.03) | beats label-free | PASS |",
         "|---|---|---|---|---|"]
    for m, v in o["verdicts"].items():
        c = o["comparisons"][m]["h0_minus_full"]
        L.append(f"| {m} | {c['delta']:+.3f} ({c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f}) | {v['non_inferior']} | "
                 f"{v['beats_label_free']} | **{v['pass']}** |")
    L += ["", "## Human score by arm", "",
          "| entry | human pooled (95% CI) | human dev | human test | test paralog pairs | test other pairs | SLB test | scer test | spom test |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, e in E_.items():
        st = e["species_test"]
        L.append(f"| {n} | {f3(e['human_pooled'])} ({f3(e['human_pooled_ci95'][0])}–{f3(e['human_pooled_ci95'][1])}) | "
                 f"{f3(e['human_dev'])} | {f3(e['human_test'])} | {f3(e['human_by_pair_test']['paralog'])} | "
                 f"{f3(e['human_by_pair_test']['other'])} | {f3(e['slb_test'])} | {f3(st.get('scer'))} | "
                 f"{f3(st.get('spom'))} |")
    if "seed_noise" in o:
        sn = o["seed_noise"]
        L += ["", "## Refit noise (gbm_nocode, GBM seeds)", "",
              (f"Pooled human SD across seeds: full {sn['sd']['full']:.3f}, h0 {sn['sd']['h0']:.3f}. "
               f"H0 − H+ by seed: {', '.join(f'{x:+.3f}' for x in sn['h0_minus_full_by_seed'])}; "
               f"difference of seed means {sn['h0_mean_minus_full_mean']:+.3f}.")]
    L += ["", "## Paired comparisons (pooled human)", "", "| model | comparison | Δ | 95% CI |", "|---|---|---|---|"]
    for m, cs in o["comparisons"].items():
        for k, c in cs.items():
            L.append(f"| {m} | {k} | {c['delta']:+.3f} | {c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
