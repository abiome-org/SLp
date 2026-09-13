"""Prepare an immutable matched GO benchmark packet within the existing allocation."""

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("feature_mlp", "no_pretraining", "mixed_pretraining")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    state = json.loads((args.state / "campaign.json").read_text())
    plan = json.loads(args.original.read_text())
    features = json.loads(args.features.read_text())
    dimension = features["functional_annotations"]["annotation_dim"]
    if features["fitted_human_intervention_genes"] != []:
        raise ValueError("Features contain intervention fitting")
    now = time.time()
    hourly = state["gpu_and_disk_hourly_estimate_usd"]
    spent = (now - state["created_at"]) / 3600 * hourly
    pretrain_updates, adapt_updates = 30 * 2 * 8000, 30 * 2 * len(FAMILIES) * 1000
    hours = (pretrain_updates * 0.18 + adapt_updates * 0.12) / 3600 + 8
    reserved_hours = hours * 1.20
    total = (
        spent
        + reserved_hours * hourly
        + state["r2_first_month_reserve_usd"]
        + state["shutdown_reserve_usd"]
    )
    if (
        total > state["total_campaign_ceiling_usd"]
        or now + reserved_hours * 3600 > state["terminate_at"] - 600
    ):
        raise ValueError("Matched suite does not fit the remaining allocation")
    journal = json.loads((args.state / "campaign-journal.json").read_text())
    prior_archive = state.get("auxiliary_archive_raw_bytes", 0) + sum(
        journal.get("published_jobs", {}).values()
    )
    ceiling_gb = math.floor((600e9 - prior_archive * 1.002) / 1e9)
    if ceiling_gb < 450:
        raise ValueError("Insufficient retained-artifact allowance")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocols").mkdir()
    plan["variants"] = {name: plan["variants"][name] for name in FAMILIES}
    for name, entry in plan["variants"].items():
        recipe = json.loads((args.original.parent / entry["recipe"]).read_text())
        recipe["model"]["annotation_dim"] = dimension
        for phase in ("pretrain", "adapt"):
            if phase in recipe:
                recipe[phase]["numeric_loss"] = "mse"
                recipe[phase]["center_rna_queries"] = False
        recipe["adapt"]["sl_fraction"] = 1.0
        if name == "mixed_pretraining":
            recipe["pretrain"]["retain_updates"] = [8000]
        path = output / entry["recipe"]
        path.write_text(json.dumps(recipe, indent=2) + "\n")
        entry["sha256"] = digest(path)
    jobs = []
    for original in plan["inner_jobs"]:
        if original["candidate"] not in FAMILIES:
            continue
        job = copy.deepcopy(original)
        if job["pretraining_updates"]:
            job["pretraining_probe_checkpoints"] = [8000]
            job["probes"] = [p for p in job["probes"] if p["checkpoint_update"] == 8000]
        if len(job["probes"]) != 1:
            raise ValueError("Require one fixed training schedule per family")
        jobs.append(job)
    plan["inner_jobs"] = jobs
    plan["source"] = {relative: digest(ROOT / relative) for relative in plan["source"]}
    plan["features_sha256"] = digest(args.features)
    plan["outer_evaluation"] = "all_families"
    plan["created_at"] = now
    plan["launch_status"] = "prepared-continuation-not-launched"
    plan["research_training_authorized"] = True
    plan["budget_requires_fresh_launch_authorization"] = False
    plan["outer_refits"] = (
        "Fresh outer refits for all three inner-selected families before test access; report the overall inner winner and each matched family."
    )
    plan["checkpoint_policy"] = (
        "Keep rolling optimizer checkpoints every 2,000 pretraining/500 adaptation updates; retain only pretraining update 8,000 as an inner transfer probe. Publish required checkpoints before cache cleanup."
    )
    plan["selection"]["test_access"] = (
        "Fetch each outer test only after all three families have completed fresh outer refitting; do not alter recipes using test results."
    )
    plan["development_reuse"] = (
        "GO inputs and the fixed MSE/SL-only recipe were developed on MuSL seed42 fold0 inner data. Cross-fold suite results are retrospective development evidence, not an untouched SOTA evaluation."
    )
    plan["continuation"] = {
        "parent_plan_sha256": digest(args.original),
        "allocation_pod_id": state["pod_id"],
        "allocation_deadline": state["terminate_at"],
        "new_allocation": False,
        "basis": "Functional inputs improved matched inner AP; test pretraining contribution with matched outer models.",
    }
    plan["cost_estimate"] = {
        "gpu_disk_hourly_usd": hourly,
        "conservative_pretrain_seconds_per_update": 0.18,
        "conservative_adapt_seconds_per_update": 0.12,
        "pretraining_updates_total_with_outer_refits_upper": pretrain_updates,
        "adaptation_updates_total_with_probes_and_outer_refits": adapt_updates,
        "io_startup_allowance_hours": 8,
        "remaining_hours_before_reserve": hours,
        "remaining_hours_with_20_percent_reserve": reserved_hours,
        "elapsed_gpu_disk_estimate_usd": spent,
        "total_campaign_estimate_including_reserve_usd": total,
        "r2_retained_compressed_gb_ceiling": ceiling_gb,
        "prior_archive_accounted_bytes": prior_archive,
        "r2_first_month_reserve_usd": state["r2_first_month_reserve_usd"],
        "not_a_quote": "Recheck credit and elapsed time at handoff; all original financial guards remain binding.",
    }
    for fold in plan["protocols"]:
        shutil.copyfile(
            args.original.parent / "protocols" / fold["file"],
            output / "protocols" / fold["file"],
        )
    path = output / "plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": digest(path),
                "inner_jobs": len(jobs),
                "outer_refits": len(jobs),
                "cost": plan["cost_estimate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args())
