"""Produce a reviewable finite r2 training packet; never provision or run training."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output, *, budget_usd=50, pretrain_updates=8000, adapt_updates=1000):
    if not 0 < budget_usd <= 50 or pretrain_updates < 2000 or adapt_updates < 1:
        raise ValueError("Require a finite proposed campaign allocation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evidence = json.loads(
        (ROOT / "results/readiness-r2-20260912-v2/complete.json").read_text()
    )
    if evidence["state"] != "complete" or evidence["research_training_launched"]:
        raise ValueError("Require disposable full-corpus readiness evidence")
    final_path = ROOT / "results/optimizer-readiness-r2-20260912-v1/complete.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else {}
    numerical = (
        "model.py",
        "records.py",
        "data.py",
        "packed.py",
        "population.py",
        "fit.py",
        "train.py",
        "adaptation.py",
        "benchmarks.py",
        "score.py",
        "baselines.py",
        "artifact.py",
        "inference.py",
    )
    engineering_ready = (
        final.get("state") == "complete"
        and final.get("optimizer_r2_roundtrip_exact") is True
        and final.get("continued_full_state_bitwise_exact") is True
        and final.get("research_training_launched") is False
        and all(
            final.get("source", {}).get(name)
            == digest(ROOT / "modules/slp-1-2-r2" / name)
            for name in numerical
        )
    )
    protocols = []
    (output / "protocols").mkdir()
    for job in ["protocol-r2-20260912-v4", "feng-folds-r2-20260912-v1"]:
        for path in sorted((ROOT / "results" / job).glob("*-fold.json")):
            fold = json.loads(path.read_text())
            shutil.copyfile(path, output / "protocols" / path.name)
            protocols.append(
                {
                    "job": job,
                    "file": path.name,
                    "name": fold["name"],
                    "benchmark": fold["benchmark"],
                    "metadata_sha256": digest(path),
                    "outer_gene_count": len(fold["outer_held"]),
                    "inner_gene_count": len(fold["inner_held"]),
                    "partition_manifests": {
                        role: {
                            "name": name,
                            "manifest": json.loads((path.parent / name).read_text()),
                        }
                        for role, name in fold["partitions"].items()
                    },
                }
            )
    if len(protocols) != 30:
        raise ValueError("The declared 30-fold suite is incomplete")
    phase = dict(
        quantitative_weights={"rna": 0.5, "fitness": 0.25, "interaction": 0.25},
        temperature=0.7,
        updates=pretrain_updates,
        batch_size=16,
        max_seconds=3600,
        checkpoint_every=2000,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup=400,
        max_queries=128,
        context_dropout=0.1,
        retain_updates=sorted({2000, pretrain_updates}),
    )
    base = {
        "purpose": "research",
        "seed": 731,
        "cpu_threads": 4,
        "model": {},
        "pretrain": phase,
        "adapt": {
            **phase,
            "updates": adapt_updates,
            "checkpoint_every": min(500, adapt_updates),
            "warmup": 100,
            "sl_fraction": 0.75,
        },
    }
    variants = {}
    for name in (
        "sequence_similarity",
        "feature_mlp",
        "no_pretraining",
        "human_pretraining",
        "mixed_pretraining",
    ):
        recipe = json.loads(json.dumps(base))
        if name in ("sequence_similarity", "feature_mlp"):
            recipe["baseline"] = name
            recipe["adapt"].update(sl_fraction=1.0, context_dropout=0.0)
        if name == "human_pretraining":
            recipe["pretrain"]["taxa"] = [9606]
        path = output / (name + ".json")
        path.write_text(json.dumps(recipe, indent=2) + "\n")
        variants[name] = {
            "recipe": path.name,
            "sha256": digest(path),
            "pretraining": name.endswith("_pretraining") and name != "no_pretraining",
        }
    points = sorted({2000, pretrain_updates})
    # These are conservative extrapolations, not measured end-to-end campaign costs.
    n = len(protocols)
    pretrain_total = 3 * n * pretrain_updates
    adaptation_total = n * (3 + 2 * len(points) + 1) * adapt_updates
    compute_hours = (pretrain_total * 0.15 + adaptation_total * 0.10) / 3600
    io_hours = n * 8 * 90 / 3600
    hourly = 0.74 + 80 * 0.20 / 730
    estimate = (compute_hours + io_hours) * hourly
    storage_reserve = 9.25
    if estimate * 1.25 + storage_reserve > budget_usd:
        raise ValueError(
            "The declared comparison packet exceeds its proposed total budget"
        )
    stages = []
    for fold in protocols:
        for name, spec in variants.items():
            stages.append(
                {
                    "fold": fold["name"],
                    "candidate": name,
                    "scope": "inner",
                    "pretraining_updates": pretrain_updates
                    if spec["pretraining"]
                    else 0,
                    "adaptation_probe_updates": adapt_updates,
                    "pretraining_probe_checkpoints": points
                    if spec["pretraining"]
                    else [],
                    "fitting_command": [
                        "python",
                        "fit.py",
                        "--features",
                        "/workspace/slp-r2/features",
                        "--data",
                        "/workspace/slp-r2/mixed-corpus.json",
                        "--fold",
                        "/workspace/slp-r2/protocols/" + fold["file"],
                        "--recipe",
                        "/workspace/slp-r2/campaign/" + spec["recipe"],
                        "--scope",
                        "inner",
                        "--stage",
                        "adapt" if not spec["pretraining"] else "pretrain",
                        "--deadline",
                        "<guarded-unix-deadline>",
                        "--output",
                        "/workspace/slp-r2/runs/" + fold["name"] + "/" + name,
                    ],
                    "status": "planned-not-launched",
                }
            )
            entry = stages[-1]
            run_root = entry["fitting_command"][-1]
            entry["fitting_command"][-1] = run_root + (
                "/pretrain" if spec["pretraining"] else "/adapt"
            )
            entry["fitting_command"] += ["--cache", "/workspace/slp-r2/packed-cache"]
            entry["probes"] = []
            for point in points if spec["pretraining"] else [None]:
                adapted = run_root + (
                    "/adapt-u" + f"{point:06d}" if point is not None else "/adapt"
                )
                command = list(entry["fitting_command"])
                command[command.index("--stage") + 1] = "adapt"
                command[command.index("--output") + 1] = adapted
                if point is not None:
                    command += [
                        "--initialize",
                        run_root + "/pretrain/" + f"checkpoint-u{point:06d}.pt",
                    ]
                candidate = name + (f"-u{point:06d}" if point is not None else "")
                entry["probes"].append(
                    {
                        "checkpoint_update": point,
                        "adaptation_command": command if point is not None else None,
                        "scoring_command": [
                            "python",
                            "score.py",
                            "--checkpoint",
                            adapted + "/checkpoint.pt",
                            "--features",
                            "/workspace/slp-r2/features",
                            "--fold",
                            "/workspace/slp-r2/protocols/" + fold["file"],
                            "--basal",
                            adapted + "/basal.json",
                            "--output",
                            adapted + "/inner-scores",
                            "--candidate",
                            candidate,
                            "--scope",
                            "inner",
                        ],
                    }
                )
    plan = {
        "schema": "slp.r2-training-packet/v1",
        "created_at": time.time(),
        "launch_status": "HOLD",
        "engineering_status": "READY" if engineering_ready else "PENDING",
        "engineering_receipt": {
            "path": str(final_path.relative_to(ROOT)),
            "sha256": digest(final_path),
        }
        if engineering_ready
        else None,
        "research_training_authorized": False,
        "purpose": "human strict-CV3 SL prediction through perturbation modeling",
        "proposed_spending_ceiling_usd": budget_usd,
        "budget_requires_fresh_launch_authorization": True,
        "gpu": {
            "type": "RTX 4090",
            "measured_hourly_usd": 0.74,
            "disposable_disk_gb": 80,
        },
        "cost_estimate": {
            "gpu_disk_hourly_usd": hourly,
            "conservative_pretrain_seconds_per_update": 0.15,
            "conservative_adapt_seconds_per_update": 0.10,
            "pretraining_updates_total_with_outer_refits_upper": pretrain_total,
            "adaptation_updates_total_with_probes_and_outer_refits": adaptation_total,
            "compute_hours": compute_hours,
            "io_startup_allowance_hours": io_hours,
            "estimate_usd": estimate,
            "with_25_percent_reserve_usd": estimate * 1.25,
            "r2_first_month_reserve_usd": storage_reserve,
            "r2_first_month_storage_usd": 9.0,
            "r2_operation_reserve_usd": 0.25,
            "r2_retained_compressed_gb_ceiling": 600,
            "r2_standard_usd_per_gb_month": 0.015,
            "r2_price_source": "https://developers.cloudflare.com/r2/pricing/",
            "total_with_compute_reserve_and_first_month_storage_usd": estimate * 1.25
            + storage_reserve,
            "not_a_quote": "Derived from short numerical checks. Fresh pod pricing, credit, transfers, storage retention and the actual selected outer paths must be reconciled at launch.",
        },
        "source": {
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted((ROOT / "modules/slp-1-2-r2").glob("*"))
            if p.is_file()
        },
        "corpus": evidence["corpus"],
        "corpus_sha256": hashlib.sha256(
            json.dumps(evidence["corpus"], sort_keys=True).encode()
        ).hexdigest(),
        "variants": variants,
        "protocols": protocols,
        "inner_jobs": stages,
        "selection": {
            "per_outer_fold": "Select its pretraining point and comparison recipe using only that outer fold's own inner-CV3 AP. Refit from scratch on permitted outer-training genes.",
            "suite_summary": "Equal benchmark weight after within-benchmark fold means; report AP, trapezoidal PR-AUC, prevalence and AUROC separately.",
            "no_cross_outer_selection": "The suite summary does not choose a nested outer model: other folds can have fitted its held genes.",
            "test_access": "Official test labels are fetched only for the final matching outer refit, never for readiness or selection.",
        },
        "outer_refits": "One selected recipe per outer fold, then the explicit score.py --scope outer --allow-outer-test path; preserve all original rows.",
        "checkpoint_policy": "Optimizer/RNG/sampler checkpoint every 2000 pretraining updates and every 500 adaptation updates; retain the 2000 and final pretraining points for matched probes. Publish each required checkpoint to a new exact R2 object prefix before deleting the pod.",
        "storage_prefix": "s3://abiome-artifacts/slp/runs/slp-1.2-r2/campaign-r2-<launch-id>-<stage>/",
        "execution": {
            "runner": "modules/slp-1-2-r2/campaign.py",
            "ticket_broker": "scripts/slp12_r2_broker.py",
            "default": "Validate only; research execution requires --execute-research and final scoring requires --allow-outer-test",
            "local_controller": "Run the broker against the owned pod allocation while the campaign is active. Keep the Mac awake for ticket delivery; no corpus or model payload is routed through the Mac.",
            "outer_refit": "Preserve the full pretraining LR schedule and stop at the inner-selected update, then adapt from that fresh outer checkpoint.",
            "publication": "Archive each stage under an immutable prefix. Verify R2 manifests before removing a completed fold's local cache; preserve partial-stage checkpoints on a bounded stage stop.",
        },
        "shutdown_policy": "Both scoped in-pod and local owner guards; reserve 600 seconds for finalization, refuse stages that cannot fit, stop at the lesser of authorized budget or account credit with a $5 reserve.",
        "quarantine": [
            "Costanzo 2016: pending rights confirmation; no fitting",
            "SLKB raw counts: study-specific count normalization/admission not complete",
            "Unresolved human guide/alias rows and yeast identities remain excluded",
        ],
        "launch_requirements": [
            "Final engineering receipt complete",
            "Costanzo is excluded from this campaign; no source-permission decision is required to launch the admitted corpus",
            "Fresh budget/credit and explicit user launch instruction",
            "Freeze source/data packet, issue exact expiring transfer scopes for that launch, arm and verify both guards",
        ],
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    summary = {k: v for k, v in plan.items() if k != "inner_jobs"}
    summary["protocols"] = [
        {
            **fold,
            "partition_manifests": {
                role: {
                    "name": spec["name"],
                    "rows": spec["manifest"]["rows"],
                    "sha256": hashlib.sha256(
                        json.dumps(spec["manifest"], sort_keys=True).encode()
                    ).hexdigest(),
                }
                for role, spec in fold["partition_manifests"].items()
            },
        }
        for fold in protocols
    ]
    summary["inner_comparison_count"] = len(stages)
    summary["expanded_commands"] = (
        "Generate plan.json with scripts/slp12_r2_campaign.py; it includes each fitting, adaptation-probe and scoring argv."
    )
    (output / "campaign.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "packet": str(output),
                "status": "HOLD",
                "engineering_status": "READY" if engineering_ready else "PENDING",
                "folds": n,
                "inner_comparisons": len(stages),
                "estimated_gpu_disk_usd_with_reserve": estimate * 1.25,
                "estimated_total_usd_with_first_month_storage": estimate * 1.25
                + storage_reserve,
                "training_launched": False,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--budget-usd", type=float, default=50)
    p.add_argument("--pretrain-updates", type=int, default=8000)
    p.add_argument("--adapt-updates", type=int, default=1000)
    a = p.parse_args()
    build(
        a.output,
        budget_usd=a.budget_usd,
        pretrain_updates=a.pretrain_updates,
        adapt_updates=a.adapt_updates,
    )
