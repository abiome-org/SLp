"""Rebuild the retrospective SLB-ANC-1.0 model battery and report.

    SLB_BENCH=data/slb uv run python scripts/benchmark_ancestry.py

The gate deliberately bars a population-level ancestry verdict on these
legacy, donor-seen screens. Every ranked, clean frozen test prediction is
scored on matched panels and recorded with its checksum.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import openpyxl
import polars as pl
import yaml

from slbench import ancestry as A
from slbench import evaluate as E
from slbench import ids


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference/ancestry_benchmark.json"
REPORT = ROOT / "results/reports/ancestry_benchmark.md"
HARLE = ROOT / "data/raw/harle2025_calls/MOESM1_additional_file1.xlsx"
FLISTER = ROOT / "data/raw/flister2025/mmc6.xlsx"
HGNC = ROOT / "data/raw/ids/hgnc_complete_set.txt"
FEATURED = ("Dev Rank Ensemble (loss)", "SL-Predict 2026 (MAE branch)",
            "Ryan 2026 (full clean refit)", "SynLeaF (human)")


def harle_guide_design_coverage(bench: A.AncestryBenchmark, contexts: pl.DataFrame) -> dict:
    """Link Harle per-line design counts; this does not inspect donor variants."""
    wb = openpyxl.load_workbook(HARLE, read_only=True, data_only=True)
    resolve = ids.human()
    it = wb["Table S4"].iter_rows(min_row=5, values_only=True)
    ix = {name: i for i, name in enumerate(next(it))}
    records = []
    for row in it:
        if not row[0]:
            continue
        gene_a, gene_b = sorted(resolve(g) or g for g in str(row[ix["sorted_gene_pair"]]).split("|"))
        records.append((row[ix["depMapID"]], gene_a, gene_b,
                        row[ix["n_guide_pairs"]], row[ix["n_replicates"]]))
    designs = pl.DataFrame(records, schema=["depmap_id", "gene_a", "gene_b", "guide_pairs", "replicates"],
                           orient="row", infer_schema_length=None)
    if designs.select("depmap_id", "gene_a", "gene_b").is_duplicated().any():
        raise ValueError("Harle guide design keys are not unique")
    context_ids = contexts.select("context_id", "depmap_id")
    out = {}
    for target, panel in bench.panels.items():
        h = panel.filter(pl.col("sources").str.contains("harle2025")).join(context_ids, on="context_id")
        linked = h.join(designs, on=["depmap_id", "gene_a", "gene_b"], how="left")
        if linked.height != h.height:
            raise ValueError("Harle guide join changed the example count")
        out[target] = {}
        for group in (target, "EUR"):
            d = linked.filter(pl.col("ancestry_group") == group)
            out[target][group] = {"test_pairs": d.height,
                                  "linked_design_records": d.height - d["guide_pairs"].null_count(),
                                  "at_least_16_guide_pairs_and_2_replicates": int(
                                      ((d["guide_pairs"] >= 16) & (d["replicates"] >= 2)).sum()),
                                  "donor_variant_overlap_checked": False}
    return {"source": "Harle et al. 2025 Table S4", "raw_sha256": E.file_sha256(HARLE),
            "meaning": "Gene-pair design/replicate coverage only; no per-donor guide-target genotype check",
            "matched_panels": out}


def verify_entry(entry: dict, gold: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    result_path, pred_path = ROOT / entry["result"], ROOT / entry["predictions"]
    if E.file_sha256(result_path) != entry["result_sha256"]:
        raise ValueError(f"result pin mismatch: {result_path}")
    result = json.loads(result_path.read_text())
    if (result["benchmark"] != E.bench_id() or result["split"] != "test"
            or result["scorer_version"] != E.SCORER_VERSION
            or result["manifest_sha256"] != E.file_sha256(E.BENCH / "manifest.json")
            or result["predictions_sha256"] != E.file_sha256(pred_path)
            or result["missing_filled"] != 0):
        raise ValueError(f"stale or incomplete test result: {result_path}")
    scored, missing = E.validated_join(gold, E.read_predictions(pred_path))
    if missing:
        raise ValueError(f"incomplete predictions: {pred_path}")
    return scored, result


def verify_raw_pins() -> dict:
    pins = {name: digest for digest, name in
            (line.split("  ./", 1) for line in
             (ROOT / "reference/raw_sha256sums.txt").read_text().splitlines())}
    out = {}
    for path in (A.CELLOSAURUS, HARLE, FLISTER, HGNC):
        key = path.relative_to(ROOT / "data/raw").as_posix()
        digest = E.file_sha256(path)
        if digest != pins.get(key):
            raise ValueError(f"raw source pin mismatch: {path}")
        out[key] = digest
    return out


def audit() -> dict:
    cfg = A.protocol()
    if cfg["split"] != "test":
        raise ValueError("this report must use frozen test predictions")
    gold = E.load_gold("test")
    contexts = pl.read_parquet(E.BENCH / "contexts.parquet")
    raw_pins_verified = verify_raw_pins()
    dutil_lines_verified = A.verify_dutil_annotations(contexts)
    bench = A.AncestryBenchmark(gold, contexts, cfg)
    train = pl.read_parquet(E.BENCH / "train.parquet").filter(pl.col("species") == "human")
    test_contexts = sorted({r["context_id"] for support in bench.support.values()
                            for block in support["blocks"] if block["scorable"]
                            for r in block["lines"]})
    train_context_exposure = {c: {"train_pairs": d.height, "train_sl": int(d["label"].sum())}
                              for c in test_contexts
                              if (d := train.filter(pl.col("context_id") == c)).height}
    entries = yaml.safe_load((ROOT / "leaderboard.yaml").read_text())
    models = []
    for entry in entries:
        if entry.get("leaky") or entry.get("ranked") is False:
            continue
        scored, result = verify_entry(entry, gold)
        ancestry_result = bench.evaluate(scored)
        complete_case_result = bench.evaluate(scored, complete_case=True)
        models.append({"name": entry["name"], "human_score": result["species_scores"]["human"],
                       "slb_score": result["slb_score"],
                       "predictions_sha256": result["predictions_sha256"],
                       "result_sha256": entry["result_sha256"],
                       "comparisons": ancestry_result,
                       "complete_case_comparisons": complete_case_result})
        print(f"{entry['name']}: " + ", ".join(
            f"{target}-EUR {ancestry_result[target]['gap_auroc']:+.3f}"
            for target in cfg["target_groups"]), flush=True)
    if cfg["primary_model"] not in {m["name"] for m in models}:
        raise ValueError("primary model absent from clean ranked entries")
    primary = next(m for m in models if m["name"] == cfg["primary_model"])
    eur_line_sd = {}
    for block in primary["comparisons"]["AFR"]["blocks"]:
        if block["scorable"]:
            values = [line["auroc"] for line in block["lines"]
                      if line["group"] == "EUR" and line["auroc"] is not None]
            eur_line_sd[f"{block['cancer_site']}:{block['sources']}"] = float(np.std(values, ddof=1))
    pooled_sd = float(np.sqrt(np.mean(np.square(list(eur_line_sd.values())))))
    # Normal approximation for two equally weighted blocks and equal donor
    # counts per group/block: SE(gap) ≈ SD(line AUROC) / sqrt(n per block).
    z = 2.241402727604947  # N(0,1) quantile for the familywise 97.5% interval.
    planning_pilot = {"EUR_line_auroc_sd_by_block": eur_line_sd,
                      "root_mean_square_sd": pooled_sd,
                      "normal_approx_lines_per_group_per_block_for_halfwidth_0_05": int(
                          np.ceil((z * pooled_sd / cfg["adequacy_gate"]["max_gap_interval_halfwidth"]) ** 2)),
                      "meaning": "rough planning only; 13 EUR comparator lines do not estimate AFR donor variability"}
    ancestry_prior = gold.select("example_id", "ancestry_group").with_columns(
        pl.when(pl.col("ancestry_group") == "AFR").then(2.0)
        .when(pl.col("ancestry_group") == "EAS").then(1.0).otherwise(0.0).alias("score"))
    controls = {"ancestry_only_prior": bench.evaluate(ancestry_prior)}
    for target in cfg["target_groups"]:
        scores = controls["ancestry_only_prior"][target]["auroc"]
        if any(abs(value - 0.5) > 1e-12 for value in scores.values()):
            raise ValueError("ancestry-only control defeated the within-line scorer")
    return {
        "protocol": cfg,
        "provenance": {
            "benchmark": E.bench_id(), "split": "test", "scorer_version": E.SCORER_VERSION,
            "manifest_sha256": E.file_sha256(E.BENCH / "manifest.json"),
            "test_inputs_sha256": E.file_sha256(E.BENCH / "test_inputs.parquet"),
            "test_labels_sha256": E.file_sha256(E.BENCH / "hidden/test_labels.parquet"),
            "test_propensity_sha256": E.file_sha256(E.BENCH / "hidden/test_propensity.parquet"),
            "contexts_sha256": E.file_sha256(E.BENCH / "contexts.parquet"),
            "cellosaurus_raw_sha256": E.file_sha256(A.CELLOSAURUS),
            "raw_pins_verified": raw_pins_verified,
            "dutil_genotype_lines_verified": dutil_lines_verified,
            "protocol_sha256": E.file_sha256(A.PROTOCOL_PATH),
            "ancestry_code_sha256": E.file_sha256(Path(A.__file__)),
            "report_code_sha256": E.file_sha256(Path(__file__)),
            "figures_code_sha256": E.file_sha256(ROOT / "figures/make_figures.py"),
            "leaderboard_sha256": E.file_sha256(ROOT / "leaderboard.yaml"),
            "python": platform.python_version(), "numpy": np.__version__, "polars": pl.__version__,
            "rebuild_command": "SLB_BENCH=data/slb uv run python scripts/benchmark_ancestry.py",
        },
        "genotype_only_support": {
            g: {"cell_lines": bench.human.filter(pl.col("ancestry_group") == g)["context_id"].n_unique(),
                "test_pairs": bench.human.filter(pl.col("ancestry_group") == g).height,
                "sl": int(bench.human.filter(pl.col("ancestry_group") == g)["label"].sum())}
            for g in ("AFR", "EAS", "EUR")
        },
        "unrepresented_genotype_groups": [g for g in ("AMR", "SAS", "admixed")
                                            if g not in set(bench.human["ancestry_group"].unique().to_list())],
        "majority_fraction_sensitivity": bench.ancestry_sensitivity(),
        "panel_support": bench.support,
        "complete_case_panel_support": bench.complete_case_support,
        "train_context_exposure": train_context_exposure,
        "guide_design_audit": harle_guide_design_coverage(bench, contexts),
        "controls": controls,
        "planning_pilot": planning_pilot,
        "models": models,
    }


def _n(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def _block_table(result: dict, target: str) -> list[str]:
    lines = ["| Cancer site × screen | Group-shared pairs | In every line | " + target + " lines / pairs / SL | EUR lines / pairs / SL |",
             "|---|---:|---:|---:|---:|"]
    for b in result["panel_support"][target]["blocks"]:
        items = {}
        for group in (target, "EUR"):
            rows = [r for r in b["lines"] if r["group"] == group]
            items[group] = f"{len(rows)} / {sum(r['pairs'] for r in rows):,} / {sum(r['sl'] for r in rows)}"
        lines.append(f"| {b['cancer_site']} × `{b['sources']}` | {b['shared_pairs']} | "
                     f"{b['complete_case_pairs']} | "
                     f"{items[target]} | {items['EUR']} |")
    return lines


def render(result: dict) -> str:
    models = {m["name"]: m for m in result["models"]}
    cfg = result["protocol"]
    ensemble = models[cfg["primary_model"]]
    fafr = ensemble["comparisons"]["AFR"]
    feas = ensemble["comparisons"]["EAS"]
    support = result["panel_support"]
    lines = [
        "# SLB-ANC-1.0: matched cell-line ancestry benchmark", "",
        "**Question.** Do frozen synthetic-lethality predictors rank *measured* gene pairs"
        " as well in cancer cell lines of African or East Asian genetic ancestry as in"
        " European-ancestry lines, on the same cancer-site, screen and pair panel?", "",
        "**Result.** The matched African–European comparison contains"
        f" {support['AFR']['by_group']['AFR']['cell_lines']} AFR lines"
        f" ({support['AFR']['by_group']['AFR']['sl']} SL pairs) and"
        f" {support['AFR']['by_group']['EUR']['cell_lines']} EUR lines"
        f" ({support['AFR']['by_group']['EUR']['sl']} SL pairs) across two scorable"
        " site × screen blocks. Each AFR block has **one** cell line."
        f" The frozen Dev Rank Ensemble's equal-cell-line, equal-block AUROC is {_n(fafr['auroc']['AFR'])}"
        f" versus {_n(fafr['auroc']['EUR'])} (gap {fafr['gap_auroc']:+.3f})."
        " The predeclared adequacy gate fails, so the benchmark issues **no population-level"
        " disparity or parity verdict**. The East Asian comparison also fails the gate;"
        f" the ensemble scores {_n(feas['auroc']['EAS'])} versus {_n(feas['auroc']['EUR'])}"
        " on its own matched blocks. These are retrospective diagnostic scores.", "",
        "The ancestry annotations are genome-wide genotype estimates from"
        " [Dutil et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30894373/) through the"
        " checksum-pinned Cellosaurus file. Dutil's resource supplies ancestry metadata,"
        " **not** SL labels. The SL labels come from the dual-perturbation screens described"
        " in the SLB README; all included sources passed"
        " the replication audit. Cell-line ancestry is not patient race or"
        " clinical treatment response.", "",
        "The scorable matched rows come from"
        " [Flister et al. 2025](https://doi.org/10.1016/j.celrep.2025.116512)"
        " (Cas12a paralog screens) and"
        " [Harle et al. 2025](https://doi.org/10.1186/s13059-025-03737-w)"
        " (dual-guide knockout screens); exact source files and label thresholds"
        " are pinned in the base benchmark.", "",
        "## Fixed evaluation protocol", "",
        "- Include only genotype-inferred majority groups (fraction >0.50); exclude the"
        " self-reported PC-9 annotation. The exact genotype fractions remain visible below.",
        "- Match **cancer site + exact merged source set + exact ordered gene pair**"
        " across each target group and EUR: each retained pair occurs in at least one"
        " line of each group. A complete-case sensitivity uses only pairs with usable"
        " labels in every line of a block. Cancer site is a coarse mapping of the"
        " pinned Cellosaurus disease; histology and molecular subtype are not matched.",
        "- Recompute the existing SLB fitness-overlap weights after panel restriction."
        " Compute AUROC separately per cell line × screen, average cell lines equally"
        " in each site × screen block, then average scorable blocks equally. The"
        " secondary metric is tie-aware precision among the top 10 pairs per line.",
        "- Treat a cell line as the independent sampling unit. The only primary model"
        " for a future disparity decision is the previously dev-selected rank ensemble;"
        " every other clean ranked model is descriptive. The matching rule uses"
        " assay availability rather than model scores or SL calls.",
        "- The adequacy gate requires at least 20 distinct lines per group, two shared"
        " blocks, five qualified lines per group in every block (each with ≥100 pairs"
        " and ≥10 SL pairs), ≥100 complete-case pairs per block, donor-disjoint"
        " training, and variant-aware guide QC."
        " It then requires a donor-bootstrap gap interval half-width ≤0.05."
        " Two ancestry comparisons use Bonferroni-adjusted 97.5% intervals; a gap"
        " below −0.05 is the practical-disadvantage margin."
        " The exact machine-readable protocol is"
        " [`reference/ancestry_protocol.json`](reference/ancestry_protocol.json).", "",
        "This protocol was written **after** the SLB test readout; it is a"
        " retrospective diagnostic and a fixed template for newly collected donors,"
        " not a claim of prospective preregistration. Original SLB test genes are"
        " family-held-out; cell lines were present in SLB training and therefore do"
        " not meet donor-disjoint training for supervised entries.", "",
        "## Panel support", "",
        "The full genotype-only test has " + ", ".join(
            f"{g} {result['genotype_only_support'][g]['cell_lines']} lines / "
            f"{result['genotype_only_support'][g]['test_pairs']:,} pairs / "
            f"{result['genotype_only_support'][g]['sl']} SL"
            for g in ("AFR", "EAS", "EUR")) + ". The matched panel is selected"
        " solely from measured pair availability; no model score determines inclusion."
        " No test line is assigned to " + ", ".join(result["unrepresented_genotype_groups"])
        + ", so those groups have no scorecard.", "",
        "### AFR versus EUR", "",
    ]
    lines += _block_table(result, "AFR")
    lines += ["", "HeLa has no European-ancestry cervical line on the same source/pair"
              " panel, so it contributes no strict matched score. RKO supplies the"
              " colorectal block; NCI-H23 supplies the lung block. The unequal"
              " number of measured rows cannot turn either single line into more donors."
              " Group-shared pairs can be absent or ambiguously labelled in individual"
              " lines; the `In every line` column exposes that attrition.", "",
              "### EAS versus EUR", ""]
    lines += _block_table(result, "EAS")
    lines += ["", "The three-row `flister2025,harle2025` lung block has no EAS"
              " positive and is recorded as unscorable. EAS gastric lines have no"
              " European gastric comparator in these sources. The two target comparisons"
              " use **different** matched blocks, so their score gaps are not directly"
              " interchangeable.", "",
              "## Frozen model battery on the matched panels", "",
              "AUROC values are equal-cell-line and equal-block means. `Δ` is target"
              " minus EUR within that target's matched panel. A positive Δ is a point"
              " estimate, not a population-level advantage. Every row is blocked from"
              " a fairness verdict by the same data-adequacy gate. Full line-level"
              " scores, top-10 precision, hashes, and gate reasons are in"
              " [`reference/ancestry_benchmark.json`](reference/ancestry_benchmark.json).", "",
              "| Clean ranked model | AFR | EUR matched to AFR | Δ AFR−EUR | EAS | EUR matched to EAS | Δ EAS−EUR |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for m in sorted(result["models"], key=lambda r: -r["human_score"]):
        af, ea = m["comparisons"]["AFR"], m["comparisons"]["EAS"]
        lines.append(f"| {m['name']} | {_n(af['auroc']['AFR'])} | {_n(af['auroc']['EUR'])} | "
                     f"{af['gap_auroc']:+.3f} | {_n(ea['auroc']['EAS'])} | {_n(ea['auroc']['EUR'])} | "
                     f"{ea['gap_auroc']:+.3f} |")
    lines += ["", "The table is ordered by each model's original human SLB score,"
              " not by the ancestry gap. The complete-case sensitivity below is"
              " restricted to pairs labelled in **every** line of a block. It leaves"
              " only colorectal × Flister for AFR–EUR and head/neck × Flister plus"
              " lung × Flister for EAS–EUR; Harle's all-line overlap is zero and"
              " pancreas has one pair with no scorable class contrast.", "",
              "| Model | AFR complete-case | EUR | Δ | EAS complete-case | EUR | Δ |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in FEATURED:
        m = models[name]
        af, ea = m["complete_case_comparisons"]["AFR"], m["complete_case_comparisons"]["EAS"]
        lines.append(f"| {name} | {_n(af['auroc']['AFR'])} | {_n(af['auroc']['EUR'])} | "
                     f"{af['gap_auroc']:+.3f} | {_n(ea['auroc']['EAS'])} | "
                     f"{_n(ea['auroc']['EUR'])} | {ea['gap_auroc']:+.3f} |")
    lines += ["", "The AFR complete-case score is RKO on 165 pairs with only **2 SL labels**."
              " The EAS lung complete-case block has one SL label in LK-2. These"
              " large score swings demonstrate how little stable information remains"
              " after enforcing identical panels; they do not provide stronger"
              " evidence of an ancestry effect.", "",
              "### Fixed experimental budget", "",
              "Tie-aware precision@10 counts hits among up to ten proposed measurements"
              " per line in each shared panel; if fewer than ten pairs remain, all"
              " are used. A random ranker achieves the local SL"
              " prevalence. It is a useful lab-budget diagnostic but these panels"
              " have very different base rates.", "",
              "| Model | AFR P@10 | EUR P@10 (AFR panel) | EAS P@10 | EUR P@10 (EAS panel) |",
              "|---|---:|---:|---:|---:|"]
    for name in FEATURED:
        m = models[name]
        af, ea = m["comparisons"]["AFR"], m["comparisons"]["EAS"]
        lines.append(f"| {name} | {_n(af['precision_at_10']['AFR'])} | "
                     f"{_n(af['precision_at_10']['EUR'])} | {_n(ea['precision_at_10']['EAS'])} | "
                     f"{_n(ea['precision_at_10']['EUR'])} |")
    random_afr = models["random"]["comparisons"]["AFR"]
    lines += ["", "[Figure 5](figures/05_ancestry_support.svg) displays the matched"
              " support and the donor count. [Figure 6](figures/06_ancestry_line_scores.svg)"
              " displays individual-line scores and block means for the primary model."
              " Both plots read the pinned JSON above.", "",
              "The ancestry-only prior scores exactly 0.500 in both groups because"
              " each cell line is scored internally. A single frozen random-score"
              f" control yields an AFR–EUR point gap of {random_afr['gap_auroc']:+.3f}"
              " on this tiny donor panel. That large accidental contrast is a direct"
              " warning against interpreting a model's point gap as evidence of"
              " ancestry-related performance.", "",
              "## Adequacy and assay-quality checks", "",
              "The actual gate results for the primary model are:", "",
              "| Comparison | Scorable blocks | Target / EUR lines | Gate |",
              "|---|---:|---:|---|"]
    for target in cfg["target_groups"]:
        gate = ensemble["comparisons"][target]["gate"]
        panel = result["panel_support"][target]
        lines.append(f"| {target}–EUR | {gate['scorable_blocks']} | "
                     f"{panel['by_group'][target]['cell_lines']} / {panel['by_group']['EUR']['cell_lines']} | "
                     f"{'pass' if gate['passed'] else 'fail'} |")
    lines += ["", "Both comparisons fail the donor-count, within-block pair/positive-count,"
              " complete-case pair, donor-disjoint training and variant-aware guide-QC"
              " requirements. The JSON lists every exact gate failure. No"
              " population interval is emitted from one AFR donor per block.", "",
              "For collection planning, the two EUR comparator blocks have donor-level"
              f" Dev Rank Ensemble AUROC SDs of {result['planning_pilot']['EUR_line_auroc_sd_by_block']['colorectal:flister2025']:.3f}"
              " (colorectal) and"
              f" {result['planning_pilot']['EUR_line_auroc_sd_by_block']['lung:harle2025']:.3f}"
              " (lung). A normal approximation for two balanced blocks and a"
              " familywise 97.5% half-width of 0.05 suggests roughly"
              f" {result['planning_pilot']['normal_approx_lines_per_group_per_block_for_halfwidth_0_05']}"
              " independent lines **per ancestry, per block** if the SD is similar."
              " The observed donor-bootstrap precision gate determines the actual"
              " requirement; this 13-line EUR pilot is only a planning estimate.", "",
              "The SLB training split contains labelled rows from every AFR"
              " test cell line. For example, HeLa has 804 train pairs, NCI-H23 99,"
              " and RKO 2,441. This does not prove that every method used those labels,"
              " but the current frozen battery is not certified donor-disjoint."
              " A new evaluation must hold donor lines out *before* fitting or feature"
              " selection and publish per-model training-context provenance.", "",
              "The [Harle et al. 2025](https://doi.org/10.1186/s13059-025-03737-w)"
              " supplementary Table S4 includes guide-pair and replicate counts."
              " With the same pinned HGNC symbol resolver used in SLB, all 27"
              " NCI-H23 pair records on the AFR strict panel link to ≥16 guide"
              " pairs and ≥2 replicates. Those counts check design coverage, **not** whether"
              " a guide matches that donor's genome. The pinned Flister RKO label"
              " table is gene-level. No per-donor guide-target variant overlap or"
              " independent confirmation is available in this benchmark, so the"
              " variant-aware QC gate fails."
              " [Misek et al. 2024](https://doi.org/10.1038/s41467-024-48957-z)"
              " measured ancestry-dependent false negatives from germline variants"
              " in CRISPR guide targets; this is a concrete reason to require that QC.", "",
              "### Ancestry-definition sensitivity", "",
              "| Minimum genotype fraction | AFR lines / test SL | EAS lines / test SL | EUR lines / test SL |",
              "|---|---:|---:|---:|"]
    for threshold, data in result["majority_fraction_sensitivity"].items():
        cells = [f"{data[g]['cell_lines']} / {data[g]['sl']}" for g in ("AFR", "EAS", "EUR")]
        lines.append(f"| >{threshold} | " + " | ".join(cells) + " |")
    lines += ["", "AFR fractions are HeLa 0.6474, NCI-H23 0.6786 and RKO"
              " 0.8143. At a >0.75 threshold, the strict matched comparison retains"
              " only RKO; at >0.90 there is no AFR line. These sensitivity checks"
              " make the group definition visible rather than silently treating"
              " admixed lines as homogeneous.", "",
              "## Next data collection and decision rule", "",
              "Run the *same* dual-perturbation library with matched cancer-site"
              " blocks across independently derived, genotype-annotated donor lines."
              " Recruit enough underrepresented-ancestry lines to clear the counts"
              " in the fixed protocol, and include variants in every guide target"
              " plus orthogonal guide or assay confirmation. Freeze donor-disjoint"
              " training, scoring code and the assay panel before new labels are"
              " read. When the gate passes, use the donor-bootstrap interval:"
              " upper bound below −0.05 means a material target-group disadvantage;"
              " lower bound above −0.05 supports performance within the margin;"
              " otherwise the comparison is inconclusive. The unit remains the"
              " donor line, not the number of gene pairs."
              " This experiment would evaluate screen generalization, not clinical"
              " effectiveness in patients.", "",
              "A [West African patient-derived breast organoid screen](https://doi.org/10.1158/0008-5472.CAN-24-0775)"
              " shows a route to more representative models, but it perturbs single"
              " kinases and drug treatments rather than the same dual-gene panel."
              " Its outcomes are therefore not relabelled as SLB pair truth.", "",
              "## Rebuild", "",
              "```bash", result["provenance"]["rebuild_command"],
              "uv run python figures/make_figures.py", "```", "",
              "All 24 clean ranked model predictions and result JSON are checked"
              " against `leaderboard.yaml`; benchmark, annotation, protocol, guide"
              " supplement and scoring-code SHA-256 hashes are in the result JSON.", ""]
    return "\n".join(lines)


def main() -> None:
    result = audit()
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(result))
    print(f"wrote {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
