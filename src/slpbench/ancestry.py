"""Cell-line ancestry comparison on identical measured screen/pair panels.

SLB-ANC-1.0 is a *retrospective diagnostic*: its inherited test split holds
gene families out, but does not hold cell-line donors out of model training.
The adequacy gate is deliberately evaluated before a disparity verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from slpbench import evaluate as E
from slpbench.metrics import stratified_auc


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "reference/slb_ancestry_protocol_v1.json"
CELLOSAURUS = ROOT / "data/raw/cellosaurus/cellosaurus.txt"


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def verify_dutil_annotations(contexts: pl.DataFrame) -> int:
    """Check that every genotype-assigned human line cites Dutil in pinned raw metadata."""
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
    return len(cited)


def cancer_site(context_id: str, disease: str | None) -> str:
    """Coarse, outcome-independent site mapping; histology remains a limitation."""
    if context_id == "human:CFPAC-1":
        return "pancreas"  # Cellosaurus' first disease field is cystic fibrosis.
    s = (disease or "").lower()
    if "lung" in s:
        return "lung"
    if any(t in s for t in ("colon", "rectal", "cecum")):
        return "colorectal"
    if "pancrea" in s:
        return "pancreas"
    if "melanoma" in s:
        return "melanoma"
    if "gastric" in s:
        return "gastric"
    if any(t in s for t in ("oral cavity", "pharyngeal")):
        return "head_neck"
    if any(t in s for t in ("cervical", "endocervical")):
        return "cervix"
    if "breast" in s:
        return "breast"
    return "other"


def precision_at_k_ties(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Expected hit fraction under random ordering of scores tied at rank k."""
    if len(y) == 0 or k <= 0:
        return float("nan")
    k = min(k, len(y))
    cutoff = np.partition(score, len(score) - k)[len(score) - k]
    above, tied = score > cutoff, score == cutoff
    return float((y[above].sum() + (k - above.sum()) * y[tied].mean()) / k)


def _rebalance(d: pl.DataFrame) -> pl.DataFrame:
    """Rebalance classes after panel restriction, within line and screen."""
    e = pl.col("propensity")
    by = ["context_id", "sources", "label"]
    return d.with_columns(pl.when(pl.col("label") == 1).then(1 - e).otherwise(e).alias("_panel_bw")) \
        .with_columns((pl.col("_panel_bw") * pl.len().over(by) / pl.col("_panel_bw").sum().over(by))
                      .alias("_panel_bw"))


def _line_metrics(d: pl.DataFrame, top_k: int) -> dict:
    y = d["label"].to_numpy()
    out = {"pairs": d.height, "sl": int(y.sum()), "non_sl": int(len(y) - y.sum()),
           "auroc": None, "precision_at_10": None, "prevalence": float(y.mean())}
    if 0 < y.sum() < len(y):
        auc, _ = stratified_auc(np.zeros(len(y), dtype=int), y, d["score"].to_numpy(),
                                d["_panel_bw"].to_numpy())
        out["auroc"] = float(auc)
        out["precision_at_10"] = precision_at_k_ties(y, d["score"].to_numpy(), top_k)
    return out


def matched_panel(human: pl.DataFrame, target: str, reference: str = "EUR") -> pl.DataFrame:
    """Keep site/source/pairs observed in at least one line of each group."""
    key = ["cancer_site", "sources", "gene_a", "gene_b"]
    a = human.filter(pl.col("ancestry_group") == target)
    b = human.filter(pl.col("ancestry_group") == reference)
    ak, bk = a.select(key).unique(), b.select(key).unique()
    return pl.concat([a.join(bk, on=key, how="semi"),
                      b.join(ak, on=key, how="semi")]).sort("example_id")


def complete_case_panel(panel: pl.DataFrame) -> pl.DataFrame:
    """Retain pairs with a usable label in every line of each site/screen block."""
    blocks = []
    for _, d in sorted(panel.partition_by(["cancer_site", "sources"], as_dict=True).items()):
        n = d["context_id"].n_unique()
        key = d.group_by("gene_a", "gene_b").agg(pl.col("context_id").n_unique().alias("lines")) \
            .filter(pl.col("lines") == n).select("gene_a", "gene_b")
        blocks.append(d.join(key, on=["gene_a", "gene_b"], how="semi"))
    return pl.concat(blocks).sort("example_id") if blocks else panel.head(0)


def _support(panel: pl.DataFrame, target: str, reference: str) -> dict:
    block_cols = ["cancer_site", "sources"]
    out = []
    for (site, source), d in sorted(panel.partition_by(block_cols, as_dict=True).items()):
        lines = []
        for (group, context), line in sorted(d.partition_by(["ancestry_group", "context_id"], as_dict=True).items()):
            lines.append({"group": group, "context_id": context, "pairs": line.height,
                          "sl": int(line["label"].sum())})
        keys = d.select("gene_a", "gene_b").unique().height
        total_lines = d["context_id"].n_unique()
        complete = d.group_by("gene_a", "gene_b").agg(pl.col("context_id").n_unique().alias("n")) \
            .filter(pl.col("n") == total_lines).height
        evaluable = {g: sum(r["group"] == g and 0 < r["sl"] < r["pairs"] for r in lines)
                     for g in (target, reference)}
        out.append({"cancer_site": site, "sources": source, "shared_pairs": keys,
                    "complete_case_pairs": complete,
                    "lines": lines, "evaluable_lines": evaluable,
                    "scorable": all(evaluable.values())})
    return {"pairs": panel.height, "sl": int(panel["label"].sum()), "blocks": out,
            "by_group": {g: {"pairs": panel.filter(pl.col("ancestry_group") == g).height,
                             "sl": int(panel.filter(pl.col("ancestry_group") == g)["label"].sum()),
                             "cell_lines": panel.filter(pl.col("ancestry_group") == g)["context_id"].n_unique()}
                         for g in (target, reference)}}


def adequacy(support: dict, cfg: dict, target: str, reference: str,
             *, donor_disjoint_training: bool, variant_aware_guide_qc: bool) -> dict:
    """A predeclared gate for a population-level disparity or parity decision."""
    g = cfg["adequacy_gate"]
    blocks = [b for b in support["blocks"] if b["scorable"]]
    failures = []
    if len(blocks) < g["min_shared_blocks"]:
        failures.append("too_few_scorable_site_screen_blocks")
    for group in (target, reference):
        distinct = {r["context_id"] for b in blocks for r in b["lines"] if r["group"] == group
                    and 0 < r["sl"] < r["pairs"]}
        if len(distinct) < g["min_distinct_lines_per_group"]:
            failures.append(f"{group}_too_few_independent_cell_lines")
    for b in blocks:
        if b["complete_case_pairs"] < g["min_complete_case_pairs_per_block"]:
            failures.append(f"{b['cancer_site']}:{b['sources']}_too_few_complete_case_pairs")
        for group in (target, reference):
            qualified = sum(r["group"] == group
                            and r["pairs"] >= g["min_measured_pairs_per_line_block"]
                            and r["sl"] >= g["min_sl_pairs_per_line_block"]
                            and r["pairs"] > r["sl"] for r in b["lines"])
            if qualified < g["min_qualified_lines_per_group_per_block"]:
                failures.append(f"{b['cancer_site']}:{b['sources']}:{group}_too_few_qualified_lines")
    if g["require_donor_disjoint_training"] and not donor_disjoint_training:
        failures.append("test_cell_lines_not_proven_disjoint_from_model_training")
    if g["require_variant_aware_guide_qc"] and not variant_aware_guide_qc:
        failures.append("guide_target_variants_not_audited_per_cell_line")
    return {"passed": not failures, "failures": failures,
            "scorable_blocks": len(blocks)}


def _group_scores(blocks: list[dict], groups: tuple[str, str], metric: str) -> dict:
    vals = {g: [] for g in groups}
    for b in blocks:
        if not b["scorable"]:
            continue
        for group in groups:
            line_values = [r[metric] for r in b["lines"] if r["group"] == group and r[metric] is not None]
            vals[group].append(float(np.mean(line_values)))
    return {g: float(np.mean(v)) if v else None for g, v in vals.items()}


def _bootstrap_gap(blocks: list[dict], target: str, reference: str, cfg: dict) -> list[float]:
    """Resample cell lines within group, reusing a line's draw across blocks."""
    reps, rng = cfg["inference"]["bootstrap_reps"], np.random.default_rng(cfg["inference"]["seed"])
    group_data = {}
    for group in (target, reference):
        contexts = sorted({r["context_id"] for b in blocks if b["scorable"] for r in b["lines"]
                           if r["group"] == group and r["auroc"] is not None})
        n = len(contexts)
        block_data = []
        for b in blocks:
            if not b["scorable"]:
                continue
            scores = {r["context_id"]: r["auroc"] for r in b["lines"]
                      if r["group"] == group and r["auroc"] is not None}
            use = np.array([i for i, c in enumerate(contexts) if c in scores], dtype=int)
            block_data.append((use, np.array([scores[contexts[i]] for i in use])))
        group_data[group] = (n, block_data)
    # Reject bootstrap draws that omit an entire block. Averaging the remaining
    # blocks would silently change the estimand from draw to draw.
    accepted = []
    attempted = 0
    while len(accepted) < reps and attempted < reps * 20:
        batch = min(reps, (reps - len(accepted)) * 2)
        draws = {}
        for group in (target, reference):
            n, block_data = group_data[group]
            counts = rng.multinomial(n, [1 / n] * n, size=batch)
            vals = []
            for use, scores in block_data:
                weight = counts[:, use]
                num = weight @ scores
                den = weight.sum(axis=1)
                vals.append(np.divide(num, den, out=np.full(batch, np.nan), where=den > 0))
            draws[group] = np.mean(np.vstack(vals), axis=0)
        delta = draws[target] - draws[reference]
        accepted.extend(delta[np.isfinite(delta)].tolist())
        attempted += batch
    if len(accepted) < reps:
        raise ValueError("too few donor bootstrap draws retained every matched block")
    gap = np.array(accepted[:reps])
    alpha = cfg["inference"]["familywise_alpha"] / cfg["inference"]["comparisons"]
    return [float(x) for x in np.quantile(gap, [alpha / 2, 1 - alpha / 2])]


class AncestryBenchmark:
    def __init__(self, gold: pl.DataFrame, contexts: pl.DataFrame, cfg: dict | None = None):
        self.cfg = cfg or protocol()
        assert self.cfg["match_key"] == ["cancer_site", "sources", "gene_a", "gene_b"]
        if gold["example_id"].n_unique() != gold.height:
            raise ValueError("duplicate benchmark example IDs")
        meta = contexts.filter(pl.col("species") == "human").select(
            "context_id", "disease", "ancestry_basis", "cellosaurus_ac", "anc_AFR", "anc_EAS", "anc_EUR")
        h = gold.filter(pl.col("species") == "human").join(meta, on="context_id", how="left", maintain_order="left")
        if h["ancestry_basis"].null_count():
            raise ValueError("human context missing ancestry metadata")
        h = h.filter(pl.col("ancestry_basis") == self.cfg["ancestry_basis"])
        h = h.with_columns(pl.struct("context_id", "disease")
                           .map_elements(lambda r: cancer_site(r["context_id"], r["disease"]),
                                         return_dtype=pl.String).alias("cancer_site"))
        self.human = h
        self.panels = {}
        self.support = {}
        self.complete_case_panels = {}
        self.complete_case_support = {}
        for target in self.cfg["target_groups"]:
            panel = matched_panel(h, target, self.cfg["reference_group"])
            self.panels[target] = panel
            self.support[target] = _support(panel, target, self.cfg["reference_group"])
            complete = complete_case_panel(panel)
            self.complete_case_panels[target] = complete
            self.complete_case_support[target] = _support(complete, target, self.cfg["reference_group"])

    def ancestry_sensitivity(self) -> dict:
        out = {}
        for cutoff in [self.cfg["majority_fraction"], *self.cfg["sensitivity_fractions"]]:
            key = f"{cutoff:g}"
            out[key] = {}
            for group in [*self.cfg["target_groups"], self.cfg["reference_group"]]:
                contexts = self.human.filter(pl.col(f"anc_{group}") > cutoff)
                out[key][group] = {"cell_lines": contexts["context_id"].n_unique(),
                                   "test_pairs": contexts.height, "sl": int(contexts["label"].sum())}
        return out

    def evaluate(self, scored: pl.DataFrame, *, donor_disjoint_training: bool = False,
                 variant_aware_guide_qc: bool = False, complete_case: bool = False) -> dict:
        if scored["score"].null_count() or not np.isfinite(scored["score"].to_numpy()).all():
            raise ValueError("ancestry benchmark requires finite, complete predictions")
        out = {}
        reference = self.cfg["reference_group"]
        panels = self.complete_case_panels if complete_case else self.panels
        supports = self.complete_case_support if complete_case else self.support
        for target, base in panels.items():
            d = _rebalance(base.join(scored.select("example_id", "score"), on="example_id", how="left",
                                     maintain_order="left", validate="1:1"))
            if d["score"].null_count():
                raise ValueError("ancestry panel predictions incomplete")
            blocks = []
            for support_block in supports[target]["blocks"]:
                site, source = support_block["cancer_site"], support_block["sources"]
                part = d.filter((pl.col("cancer_site") == site) & (pl.col("sources") == source))
                lines = []
                for (group, context), line in sorted(part.partition_by(["ancestry_group", "context_id"], as_dict=True).items()):
                    lines.append({"group": group, "context_id": context,
                                  **_line_metrics(line, self.cfg["top_k"])})
                blocks.append({**{k: v for k, v in support_block.items() if k != "lines"}, "lines": lines})
            groups = (target, reference)
            auc = _group_scores(blocks, groups, "auroc")
            p10 = _group_scores(blocks, groups, "precision_at_10")
            gate = adequacy(supports[target], self.cfg, target, reference,
                            donor_disjoint_training=donor_disjoint_training,
                            variant_aware_guide_qc=variant_aware_guide_qc)
            interval = _bootstrap_gap(blocks, target, reference, self.cfg) if gate["passed"] else None
            if interval is not None:
                width = (interval[1] - interval[0]) / 2
                if width > self.cfg["adequacy_gate"]["max_gap_interval_halfwidth"]:
                    gate = {**gate, "passed": False,
                            "failures": [*gate["failures"], "gap_interval_too_wide"]}
            gap = auc[target] - auc[reference] if all(v is not None for v in auc.values()) else None
            verdict = "insufficient_support"
            if gate["passed"] and interval is not None:
                margin = self.cfg["reference_margin_auroc"]
                verdict = ("disadvantage" if interval[1] < -margin else
                           "within_margin" if interval[0] > -margin else "inconclusive")
            out[target] = {"auroc": auc, "precision_at_10": p10, "gap_auroc": gap,
                           "gap_ci_familywise": interval, "gate": gate, "verdict": verdict,
                           "blocks": blocks}
        return out
