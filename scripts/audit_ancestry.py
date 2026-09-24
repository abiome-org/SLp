"""Audit frozen SLB-1.3 test predictions by human cell-line ancestry.

Run from the repository root with the private benchmark:
    SLB_BENCH=data/bench/slb1.3 uv run python scripts/audit_ancestry.py

This is descriptive analysis after the test readout, not a model-selection gate.
The ancestry estimates are from Dutil et al. 2019 (PMID 30894373) via the
checksum-pinned Cellosaurus file. No ancestry labels are inferred from outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from slpbench import evaluate as E


GROUPS = ("AFR", "EAS", "EUR")
REPS = 400
SEED = 20260924
OUT = Path("reference/slb1.3_ancestry_audit.json")
REPORT = Path("ANCESTRY_AUDIT.md")
CELLOSAURUS = Path("data/raw/cellosaurus/cellosaurus.txt")
FEATURED = (
    "SLP Fusion (loss)", "SL-Predict 2026 (MAE branch)",
    "Ryan 2026 (full clean refit)", "Ontotype",
)


def support_table(h: pl.DataFrame, contexts: pl.DataFrame) -> dict:
    out = {}
    for group in (*GROUPS, "unknown"):
        d = h.filter(pl.col("ancestry_group") == group)
        lines = contexts.filter((pl.col("species") == "human")
                                & (pl.col("ancestry_group") == group))
        out[group] = {
            "pairs": d.height,
            "sl": int(d["label"].sum()),
            "cell_lines": d["context_id"].n_unique(),
            "genotype_lines": lines.filter(pl.col("ancestry_basis") == "genotype").height,
            "self_reported_lines": lines.filter(pl.col("ancestry_basis") == "self_reported").height,
        }
    return out


def verify_ancestry_source(contexts: pl.DataFrame) -> None:
    """Every genotype-annotated benchmark line must cite Dutil in the raw file."""
    accessions = set(contexts.filter((pl.col("species") == "human")
                                     & (pl.col("ancestry_basis") == "genotype"))["cellosaurus_ac"])
    cited = set()
    accession = None
    with CELLOSAURUS.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.startswith("AC   "):
                accession = line[5:].strip()
            elif line.startswith("CC   Genome ancestry:") and accession in accessions:
                if "PubMed=30894373" in line:
                    cited.add(accession)
            elif line.startswith("//"):
                accession = None
    if cited != accessions:
        raise ValueError(f"genotype ancestry lacks Dutil provenance: {sorted(accessions - cited)}")


def audit() -> dict:
    gold = E.load_gold("test")
    human_gold = gold.filter(pl.col("species") == "human")
    contexts = pl.read_parquet(E.BENCH / "contexts.parquet")
    verify_ancestry_source(contexts)
    support = support_table(human_gold, contexts)
    context_support = {
        row["context_id"]: {"ancestry_group": row["ancestry_group"],
                            "pairs": row["pairs"], "sl": row["sl"]}
        for row in (human_gold.filter(pl.col("ancestry_group") == "AFR")
                    .group_by("context_id", "ancestry_group").agg(
                        pl.len().alias("pairs"), pl.col("label").sum().alias("sl")
                    ).iter_rows(named=True))
    }
    masks = {g: (human_gold["ancestry_group"] == g).to_numpy() for g in GROUPS}
    pair_key = ["sources", "gene_a", "gene_b"]
    afr_gold = human_gold.filter(pl.col("ancestry_group") == "AFR")
    eur_gold = human_gold.filter(pl.col("ancestry_group") == "EUR")
    # Restrict both groups to screen/pair keys observed in the other group.
    # This removes panel differences, while leaving cell line and cancer-type
    # differences visible as an explicit limit of the comparison.
    matched_gold = {
        "AFR": afr_gold.join(eur_gold.select(pair_key).unique(), on=pair_key, how="semi"),
        "EUR": eur_gold.join(afr_gold.select(pair_key).unique(), on=pair_key, how="semi"),
    }
    matched_masks = {g: human_gold["example_id"].is_in(d["example_id"].implode()).to_numpy()
                     for g, d in matched_gold.items()}
    matched_support = {g: {"pairs": d.height, "sl": int(d["label"].sum()),
                           "cell_lines": d["context_id"].n_unique()}
                       for g, d in matched_gold.items()}
    ia, ib, family_count = E._family_index(human_gold)
    resamples = np.random.default_rng(SEED).poisson(1.0, size=(REPS, family_count))
    entries = yaml.safe_load(Path("leaderboard.yaml").read_text())
    models = []
    for entry in entries:
        if entry.get("leaky") or entry.get("ranked") is False:
            continue
        result_path = Path(entry["result"])
        pred_path = Path(entry["predictions"])
        if E.file_sha256(result_path) != entry["result_sha256"]:
            raise ValueError(f"result pin mismatch: {result_path}")
        result = json.loads(result_path.read_text())
        if (result["benchmark"] != E.BENCH.name or result["split"] != "test"
                or result["scorer_version"] != E.SCORER_VERSION
                or result["manifest_sha256"] != E.file_sha256(E.BENCH / "manifest.json")
                or result["predictions_sha256"] != E.file_sha256(pred_path)
                or result["missing_filled"] != 0):
            raise ValueError(f"stale test result: {result_path}")
        scored, missing = E.validated_join(gold, E.read_predictions(pred_path))
        if missing:
            raise ValueError(f"incomplete predictions: {pred_path}")
        h = scored.filter(pl.col("species") == "human")
        if not h["example_id"].equals(human_gold["example_id"]):
            raise ValueError(f"human row order changed: {pred_path}")
        parts = {g: h.filter(pl.Series(mask)) for g, mask in masks.items()}
        point = {g: E._auc(parts[g])[0] for g in GROUPS}
        recorded = {r["stratum"].split("=", 1)[1]: r["slb_auroc"]
                    for r in result["strata"] if r["stratum"].startswith("human ancestry=")}
        if any(abs(point[g] - recorded[g]) > 1e-10 for g in GROUPS):
            raise ValueError(f"ancestry score differs from pinned result: {entry['name']}")
        if abs(np.mean(list(point.values())) - result["species_scores"]["human"]) > 1e-10:
            raise ValueError(f"human mean differs from pinned result: {entry['name']}")

        draws = {g: [] for g in GROUPS}
        for counts in resamples:
            weights = E._family_weights(counts, ia, ib)
            for g in GROUPS:
                draws[g].append(E._auc(parts[g], weights[masks[g]])[0])
        if any(not np.isfinite(draws[g]).all() for g in GROUPS):
            raise ValueError(f"non-finite bootstrap draw: {entry['name']}")
        by_group = {g: {"score": point[g],
                        "ci95": np.percentile(draws[g], [2.5, 97.5]).tolist()}
                    for g in GROUPS}
        gap = np.array(draws["AFR"]) - np.array(draws["EUR"])
        afr = parts["AFR"]
        afr_lines = sorted(afr["context_id"].unique().to_list())
        row = {
            "name": entry["name"],
            "predictions_sha256": result["predictions_sha256"],
            "human_score": result["species_scores"]["human"],
            "by_group": by_group,
            "afr_minus_eur": {"delta": point["AFR"] - point["EUR"],
                              "ci95": np.percentile(gap, [2.5, 97.5]).tolist()},
            "matched_screen_pair_scores": {
                g: E._auc(h.filter(pl.Series(mask)))[0]
                for g, mask in matched_masks.items()
            },
            "afr_context_scores": {
                context: E._auc(afr.filter(pl.col("context_id") == context))[0]
                for context in afr_lines
            },
            "afr_leave_one_context_out": {
                context: E._auc(afr.filter(pl.col("context_id") != context))[0]
                for context in afr_lines
            },
        }
        models.append(row)
        print(f"{entry['name']}: AFR {point['AFR']:.3f}, EAS {point['EAS']:.3f}, "
              f"EUR {point['EUR']:.3f}", flush=True)

    return {
        "benchmark": E.BENCH.name,
        "manifest_sha256": E.file_sha256(E.BENCH / "manifest.json"),
        "scorer_version": E.SCORER_VERSION,
        "ancestry_source": "Dutil et al. 2019, PMID 30894373, via Cellosaurus",
        "cellosaurus_raw_sha256": E.file_sha256(CELLOSAURUS),
        "split": "test",
        "metric": "fitness-balanced AUROC within cell line x screen",
        "bootstrap": {"unit": "gene family", "reps": REPS, "seed": SEED},
        "support": support,
        "matched_screen_pair_support": matched_support,
        "context_support": context_support,
        "models": models,
    }


def fmt_ci(record: dict) -> str:
    a, b = record["ci95"]
    return f"{record['score']:.3f} ({a:.3f}–{b:.3f})"


def render(audit_result: dict) -> str:
    models = {m["name"]: m for m in audit_result["models"]}
    support = audit_result["support"]
    fusion_gap = models["SLP Fusion (loss)"]["afr_minus_eur"]
    ryan = models["Ryan 2026 (full clean refit)"]
    lines = [
        "# Held-out human SL performance by cell-line ancestry", "",
        "The [Estimated Cell Line Ancestry study by Dutil et al. (2019)](https://pubmed.ncbi.nlm.nih.gov/30894373/)"
        " estimated ancestry from genome-wide genotypes. All 47 genotype-annotated"
        " SLB cell lines cite that paper in the checksum-pinned Cellosaurus file."
        " This audit uses those **existing**"
        " annotations to evaluate frozen SLB-1.3 test predictions; the paper supplies no SL labels.", "",
        "The measure is SLB's fitness-balanced AUROC **within each cell line and screen**."
        " CIs use 400 shared gene-family bootstrap resamples; they describe pair/family"
        " sampling uncertainty, not variation across future donors. This audit follows the"
        " frozen test readout and does not select new model weights.", "",
        "## Test support", "",
        "| Cell-line genetic ancestry | Cell lines | Measured test pairs | SL pairs | Assignment |",
        "|---|---:|---:|---:|---|",
    ]
    for g in (*GROUPS, "unknown"):
        s = support[g]
        assignment = f"{s['genotype_lines']} genotype"
        if s["self_reported_lines"]:
            assignment += f", {s['self_reported_lines']} self-reported"
        lines.append(f"| {g} | {s['cell_lines']} | {s['pairs']:,} | {s['sl']} | {assignment} |")
    lines += ["", "The SLB human headline averages AFR, EAS and EUR AUROCs equally,"
              " so the 38 EUR lines cannot dominate it by row count. `unknown` is"
              " reported for coverage but excluded. One EAS cell line (PC-9) has a"
              " reported population but no genotype estimate.", "",
              "## Representative frozen models", "",
              "| Model | AFR (95% CI) | EAS (95% CI) | EUR (95% CI) | AFR − EUR (95% CI) |",
              "|---|---:|---:|---:|---:|"]
    for name in FEATURED:
        m = models[name]
        ci = m["afr_minus_eur"]["ci95"]
        lines.append(f"| {name} | " + " | ".join(fmt_ci(m["by_group"][g]) for g in GROUPS)
                     + f" | {m['afr_minus_eur']['delta']:+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f}) |")
    lines += ["", "[Figure 4](figures/04_human_ancestry.svg) plots these and two additional"
              " reference methods with the same uncertainty intervals. The point estimates"
              " do **not** show worse performance on AFR-annotated lines. For Fusion,"
              f" AFR − EUR is {fusion_gap['delta']:+.3f}, with an interval from"
              f" {fusion_gap['ci95'][0]:+.3f} to {fusion_gap['ci95'][1]:+.3f}."
              " The groups contain different cell lines, cancer types, screens and pair sets,"
              " so the contrast is not an isolated effect of donor ancestry.", "",
              "### African-ancestry cell-line sensitivity", "",
              "| Cell line omitted | Remaining SL pairs | Fusion | SL-Predict MAE | Ryan full refit |",
              "|---|---:|---:|---:|---:|"]
    afr_contexts = [c for c, s in audit_result["context_support"].items()
                    if s["ancestry_group"] == "AFR"]
    for context in sorted(afr_contexts):
        pos = support["AFR"]["sl"] - audit_result["context_support"][context]["sl"]
        vals = [models[name]["afr_leave_one_context_out"][context] for name in FEATURED[:3]]
        lines.append(f"| {context.removeprefix('human:')} | {pos} | "
                     + " | ".join(f"{x:.3f}" for x in vals) + " |")
    lines += ["", f"Ryan's AFR score falls from {ryan['by_group']['AFR']['score']:.3f}"
              f" to {ryan['afr_leave_one_context_out']['human:RKO']:.3f} when RKO is removed."
              f" The {support['AFR']['cell_lines']} AFR lines contribute only"
              f" {support['AFR']['sl']} held-out SL pairs, so even an"
              " apparently separated family-bootstrap interval cannot establish a"
              " population-level advantage or disadvantage.", "",
              "### Same-screen, same-pair panel check", "",
              "Only pairs measured in both an AFR-annotated and a EUR-annotated cell line"
              " under the same screen are retained. This leaves"
              f" {audit_result['matched_screen_pair_support']['AFR']['pairs']:,} AFR"
              f" examples ({audit_result['matched_screen_pair_support']['AFR']['sl']} SL)"
              f" and {audit_result['matched_screen_pair_support']['EUR']['pairs']:,} EUR"
              f" examples ({audit_result['matched_screen_pair_support']['EUR']['sl']} SL)."
              " It equalizes the gene-pair panel and screen name, but different cell"
              " lines and cancer types still determine the labels.", "",
              "| Model | AFR matched AUROC | EUR matched AUROC |",
              "|---|---:|---:|"]
    for name in FEATURED:
        m = models[name]["matched_screen_pair_scores"]
        lines.append(f"| {name} | {m['AFR']:.3f} | {m['EUR']:.3f} |")
    lines += ["", "This is a panel-matching diagnostic, not a donor-matched"
              " estimate of an ancestry effect.", "",
              "## All ranked models", "",
              "Every clean, ranked model is included below. Full precision, intervals,"
              " input hashes and leave-one-line-out values are in"
              " [`reference/slb1.3_ancestry_audit.json`](reference/slb1.3_ancestry_audit.json).", "",
              "| Model | AFR | EAS | EUR | Human mean |",
              "|---|---:|---:|---:|---:|"]
    for m in sorted(audit_result["models"], key=lambda x: -x["human_score"]):
        lines.append(f"| {m['name']} | "
                     + " | ".join(f"{m['by_group'][g]['score']:.3f}" for g in GROUPS)
                     + f" | {m['human_score']:.3f} |")
    lines += ["", "## What the next equity test needs", "",
              "A credible disparity estimate needs more independent AFR and other"
              " underrepresented-ancestry donors, with overlapping cancer lineages,"
              " perturbation libraries and gene-pair panels across groups. The next"
              " study should hold out donors, report within-line ranking and calibration"
              " or precision at a fixed experimental budget, and quantify the worst-group"
              " gap with donor-level uncertainty. It should also audit guide-target"
              " germline variants and verify hits with independent guides or assays:"
              " [Misek et al. (2024)](https://doi.org/10.1038/s41467-024-48957-z)"
              " found ancestry-associated CRISPR false negatives from guide mismatches."
              " Cell-line genetic ancestry is not a patient's race or a clinical outcome;"
              " these SLB results are a screen-level coverage audit.", ""]
    return "\n".join(lines)


def main() -> None:
    result = audit()
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    REPORT.write_text(render(result))
    print(f"wrote {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
