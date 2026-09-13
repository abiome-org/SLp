"""Read-only fitting-pool diagnostics for saved r2 pretraining checkpoints.

No SL labels are used. Perturbation identities are swapped within each source;
context, measurement queries, mechanisms and targets remain fixed. These are
training-pool diagnostics, not independent generalization scores.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def wrong_identity(batch):
    if not batch["action_mask"].all() or batch["action_mask"].shape[1] != 1:
        raise ValueError("Diagnostic requires single-action units")
    changed = dict(batch)
    for key in ("action_sequence", "action_annotation", "action_known"):
        changed[key] = batch[key].roll(1, 0)
    return changed


def query_mean(view):
    if not hasattr(view, "targets"):
        return None
    selected = np.concatenate([f["indices"] for f in view.families])
    total = np.zeros(len(view.query), np.float64)
    count = np.zeros(len(view.query), np.int64)
    for lo in range(0, len(selected), 256):
        ix = selected[lo : lo + 256]
        mask = np.asarray(view.observed[ix], bool)
        values = np.asarray(view.targets[ix], np.float64)
        total += np.where(mask, values, 0).sum(0)
        count += mask.sum(0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return {int(g): float(v) for g, v in zip(view.query, mean)}


def original_coordinates(view, template, query):
    """Undo fitted query centering into the original assay-standardized space."""
    if getattr(view, "query_offsets", None) is None:
        return np.ones(len(query)), np.zeros(len(query))
    position = {int(g): i for i, g in enumerate(view.query)}
    offset = view.query_offsets[template, [position[int(g)] for g in query]]
    native_mean, native_std = view.native_scales[template]
    mean, std = view.scales[template]
    return (
        np.full(len(query), std / native_std),
        (offset + mean - native_mean) / native_std,
    )


def main(args):
    sys.path.insert(0, str(args.root))
    from benchmarks import Benchmark
    from data import Features, digest
    from diagnostics import numeric_metrics
    from fit import views
    from model import Config, WorldModel
    import packed
    import population

    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(90217)
    data_root = args.data_root or args.root
    features = Features(args.root / "features")
    genes = list(features.index)
    fold = Benchmark(data_root / "protocols/musl-s42-f0-fold.json", features).fold(
        "inner"
    )
    recipe_path = args.recipe or args.root / "campaign/mixed_pretraining.json"
    recipe = json.loads(recipe_path.read_text())
    data_views, _, basal = views(
        data_root / "mixed-corpus.json",
        features,
        fold,
        "pretrain",
        data_root / "packed-cache",
        center_rna_queries=recipe["pretrain"].get("center_rna_queries", False),
    )
    mixture = population.Mixture(
        data_views, recipe["pretrain"]["quantitative_weights"], seed=90217
    )
    wanted = {"depmap-24q2", "molecular-k562", "molecular-gwps", "molecular-yeast"}
    panels = []
    for i, j in mixture.candidates:
        view, family = data_views[i], data_views[i].families[j]
        if family["source"] not in wanted:
            continue
        mean = query_mean(view)
        units, baselines, transforms = [], [], []
        for _ in range(8):
            group, baseline, transform, seen = [], [], [], set()
            for attempt in range(20000):
                rows = mixture.children[i].draw_family(j, 128)
                ids = rows["targets"][0]
                ids = ids[ids >= 0]
                if len(ids) != 1 or int(ids[0]) in seen:
                    continue
                seen.add(int(ids[0]))
                tid = int(rows["template"][0])
                if view.templates[tid]["taxon"] == 9606:
                    assert genes[int(ids[0])] not in fold.forbidden
                values = np.zeros(len(rows), np.float32)
                if mean is not None:
                    scale = getattr(view, "native_scales", view.scales)[tid]
                    values = np.array(
                        [
                            (mean[int(g)] - scale[0]) / scale[1]
                            for g in rows["query"]
                        ],
                        np.float32,
                    )
                transform.append(original_coordinates(view, tid, rows["query"]))
                rows["template"] += mixture.offsets[i]
                group.append(rows)
                baseline.append(values)
                if len(group) == 16:
                    break
            if len(group) != 16:
                raise ValueError("Insufficient distinct single-action units")
            units.append(group)
            baselines.append(baseline)
            transforms.append(transform)
        panels.append((family["source"], units, baselines, mean is not None, transforms))
    if {x[0] for x in panels} != wanted:
        raise ValueError("Diagnostic source coverage incomplete")
    report = {
        "schema": "slp.r2-specificity-diagnostic/v1",
        "scope": "inner fitting pool only",
        "fold": fold.name,
        "source_sha256": digest(__file__),
        "seed": 90217,
        "interventions_per_source": 128,
        "forbidden_human_exposures": 0,
        "models": {},
        "numeric_loss": recipe["pretrain"].get("numeric_loss", "student_t"),
        "center_rna_queries": recipe["pretrain"].get("center_rna_queries", False),
        "metric_units": "original fitting-assay standardization, centering undone",
    }
    run_root = args.run_root or args.root / "runs/musl-s42-f0"
    for variant in args.variants:
        for update in args.updates:
            if time.time() > args.deadline - 30:
                raise TimeoutError("Diagnostic time allowance exhausted")
            directory = run_root / variant / "pretrain"
            file = directory / f"checkpoint-u{update:06d}.pt"
            saved = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
            assert saved["update"] == update
            model = WorldModel(Config(**saved["config"])).cuda().eval()
            model.load_state_dict(saved["model"], strict=True)
            builder = packed.Builder(
                features,
                mixture.templates,
                mixture.scales,
                saved["identity"]["vocabulary"],
                model.config,
                basal,
            )
            result = {"checkpoint_sha256": digest(file), "sources": {}}
            for source, groups, baselines, per_query, transforms in panels:
                truth, correct, wrong, reference, scales = [], [], [], [], []
                conditional_means = []
                for units, baseline, transform in zip(groups, baselines, transforms):
                    batch = {k: v.cuda() for k, v in builder(units).items()}
                    assert batch["action_mask"].sum(1).eq(1).all()
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        output = model(batch)
                        swapped = model(wrong_identity(batch))
                    mask = batch["query_mask"]
                    gain = np.concatenate([t[0] for t in transform])
                    shift = np.concatenate([t[1] for t in transform])
                    truth.append(batch["target"][mask].float().cpu().numpy() * gain + shift)
                    correct.append(output["location"][mask].float().cpu().numpy() * gain + shift)
                    wrong.append(swapped["location"][mask].float().cpu().numpy() * gain + shift)
                    conditional_means.append(shift)
                    scales.append(output["log_scale"][mask].float().cpu().numpy())
                    reference.append(np.concatenate(baseline))
                y, p, w, b, scale = (
                    np.concatenate(v)
                    for v in (truth, correct, wrong, reference, scales)
                )
                metrics = numeric_metrics(y, p, fitting_mean=b, wrong_gene_prediction=w)
                if per_query and report["center_rna_queries"]:
                    mean_mse = float(np.square(y - np.concatenate(conditional_means)).mean())
                    metrics["conditional_fitting_mean_mse"] = mean_mse
                    metrics["skill_over_conditional_fitting_mean"] = (
                        1 - metrics["mse"] / mean_mse if mean_mse else None
                    )
                metrics.update(
                    baseline="fitting per-query mean"
                    if per_query
                    else "fitting assay mean",
                    log_scale_median=float(np.median(scale)),
                    scale_floor_fraction=float((scale <= -5.99).mean()),
                )
                result["sources"][source] = metrics
            report["models"][f"{variant}-u{update:06d}"] = result
            del model, saved
            torch.cuda.empty_cache()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
    report["elapsed_seconds"] = time.time() - started
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/workspace/slp-r2"))
    p.add_argument("--data-root", type=Path)
    p.add_argument("--run-root", type=Path)
    p.add_argument("--recipe", type=Path)
    p.add_argument("--variants", nargs="+", choices=("human_pretraining", "mixed_pretraining"),
                   default=["human_pretraining", "mixed_pretraining"])
    p.add_argument("--updates", type=int, nargs="+", default=[2000, 8000])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline", type=float, required=True)
    main(p.parse_args())
