"""`slbench battery`: score every model in the SL model battery on one benchmark split, one table.

    uv run slbench battery --split dev              # writes results/reports/models_dev.md
    uv run slbench battery --split dev --boot 100   # adds family-cluster 95% CIs

Reads battery.yaml and every prediction file results/models/<benchmark>/<model>[__variant]_<split>.parquet,
plus the reference baselines in results/<benchmark>/<baseline>_<split>.parquet. Every file is re-scored
with the same evaluator; rows a model cannot score are filled with its median (a tie), and the table
reports the share of each species' rows the model actually scored. Producing the predictions is the
adapters' job: scripts/models/run_battery.sh runs every scripts/models/<model>/run.sh.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from slbench import evaluate as E

REGISTRY = Path("battery.yaml")


def _files(split: str) -> list[tuple[str, str, Path]]:
    reg = yaml.safe_load(REGISTRY.read_text())["models"]
    out = []
    for root in (Path("results") / E.BENCH.name, Path("results/models") / E.BENCH.name):
        for p in sorted(root.glob(f"*_{split}.parquet")):
            name = p.name.removesuffix(f"_{split}.parquet")
            model, _, variant = name.partition("__")
            if model in reg and (root.parent.name == "models") == (reg[model]["family"] != "baseline"):
                out.append((model, variant, p))
    return out


def _leak(entry: dict, variant: str) -> tuple[str, str]:
    lv = entry.get("leaky_variants", {})
    if variant in lv:
        return "yes", lv[variant]
    v = entry.get("leaky", False)
    return ("possibly" if v == "possibly" else "yes" if v else "no"), entry.get("why", "")


def _score(job: tuple) -> dict:
    model, variant, path, split, boot = job
    import os

    os.environ.setdefault("POLARS_MAX_THREADS", "4")
    gold = E.load_gold(split)
    p = E.read_predictions(path)
    # The registry fixes the benchmark directory, and validated_join still rejects
    # duplicate/unknown IDs. Some published methods score only human or a small
    # subset of pairs, so the battery permits sparse files and reports the fill.
    df, missing = E.validated_join(gold, p, allow_missing=True, max_missing_fraction=1.0)
    coverage_path = path.with_suffix(".coverage.json")
    if coverage_path.exists():
        native = json.loads(coverage_path.read_text())
        cov = {sp: v["scored"] / v["rows"] for sp, v in native.items()}
    else:
        raw = p.join(gold.select("example_id", "species"), on="example_id", how="left", validate="1:1")
        cov = dict(raw.with_columns((pl.col("score").is_not_null() & ~pl.col("score").is_nan()).alias("_s"))
                   .group_by("species").agg(pl.col("_s").mean()).iter_rows())
    _, aux = E.species_tiers()
    score, parts = E.headline(df)
    auxs = E.headline(df, species=aux)[1]
    r = {"model": model, "variant": variant or "-", "slb": score, **parts, **auxs,
         "coverage": cov, "missing_filled": missing}
    if boot:
        r["ci"] = E.bootstrap(df, boot)
    return r


def report_path(split: str) -> Path:
    return E.REPORTS / f"models_{split}.md"


def run(split: str = "dev", boot: int = 0, out: Path | None = None) -> pl.DataFrame:
    out = out or report_path(split)
    reg = yaml.safe_load(REGISTRY.read_text())["models"]
    jobs = [(m, v, p, split, boot) for m, v, p in _files(split)]
    with ProcessPoolExecutor(8, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_score, jobs))
    main, aux = E.species_tiers()
    table = []
    for r in rows:
        leak, why = _leak(reg[r["model"]], r["variant"] if r["variant"] != "-" else "")
        cov = r["coverage"]
        table.append({"model": r["model"], "variant": r["variant"], "family": reg[r["model"]]["family"],
                      "leaky": leak, "SLB": r["slb"], "ci": r.get("ci"),
                      **{s: r.get(s, float("nan")) for s in main + aux},
                      "covered": ", ".join(f"{s} {cov[s]:.0%}" for s in main + aux if cov.get(s, 0) > 0.005),
                      "missing_filled": r["missing_filled"], "why": why})
    df = pl.DataFrame(table, infer_schema_length=None).sort(
        pl.col("leaky") != "no", -pl.col("SLB"))
    _write(df, split, boot, main, aux, out)
    (Path("results/models") / E.BENCH.name / f"battery_{split}.json").write_text(
        json.dumps(df.to_dicts(), indent=1, default=float))
    return df


def _f(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def _write(df: pl.DataFrame, split: str, boot: int, main: list, aux: list, out: Path) -> None:
    head = ["#", "model", "variant", "family", f"SLB ({split})"] + main + [f"*{a}*" for a in aux] + ["native coverage", "filled"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rank = 0
    for r in df.iter_rows(named=True):
        if r["leaky"] == "no":
            rank += 1
        ci = f" ({r['ci'][0]:.3f}–{r['ci'][1]:.3f})" if r["ci"] else ""
        tag = "" if r["leaky"] == "no" else (" ⚠ leaky" if r["leaky"] == "yes" else " ⚠ possibly leaky")
        lines.append("| " + " | ".join([str(rank) if r["leaky"] == "no" else "–", f"**{r['model']}**{tag}", r["variant"],
                                         r["family"], _f(r["SLB"]) + ci] + [_f(r[s]) for s in main + aux]
                                        + [r["covered"], str(r["missing_filled"])]) + " |")
    leaky = [f"- **{r['model']}** {r['variant']}: {r['why']}" for r in df.iter_rows(named=True) if r["leaky"] != "no"]
    leaky_section = "\n\nLeaky or possibly leaky rows (not ranked):\n\n" + "\n".join(leaky) if leaky else ""
    out.write_text(f"""# SL model battery ({E.BENCH.name}, {split} split)

Generated by `slbench battery --split {split}` from `battery.yaml`. Do not edit by hand.

Published SL predictors we could obtain and run, plus our dev-selected ensemble, re-trained on SLB train where the code allows
(leakage rules in the SLB README), scored on the {split} split with
the same evaluator. SLB = mean over headline species; italic columns are auxiliary species (n/a: fewer
than 20 positives in this split). Rows a model cannot score (other species, genes outside its
vocabulary) are filled with its median, i.e. tied. "Native coverage" comes from the adapter's
`.coverage.json` when present; otherwise a complete prediction file is marked 100%. "Filled" counts
missing scores imputed by the scorer. Constant predictions remain legitimate ties.
Adapters: `scripts/models/<model>/run.sh`.

{chr(10).join(lines)}{leaky_section}
""")
