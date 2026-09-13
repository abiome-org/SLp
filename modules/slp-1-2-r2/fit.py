"""Finite ordinary-script fitting entry point. This is never called by prep jobs."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import adaptation
from baselines import PairBaseline
from benchmarks import Benchmark, verify_training_labels
from data import Features, Scales, ConditionSampler, BatchBuilder, digest
from model import Config, WorldModel
import packed
import population
from train import optimize, resume


def check_file(root, spec):
    path = (root / spec["path"]).resolve()
    if digest(path) != spec["sha256"]:
        raise ValueError("Declared data manifest checksum mismatch")
    return path


def views(data_path, features, fold, stage, cache, taxa=None, center_rna_queries=False):
    data_path = Path(data_path)
    spec = json.loads(data_path.read_text())
    root = data_path.parent
    genes = list(features.index)
    result = []
    excluded = []
    basal = {}
    manifests = [check_file(root, item) for item in spec.get("packed", [])]
    if manifests:
        materialized = packed.materialize(manifests, cache, genes)
        result.append(packed.View(materialized, genes, fold, stage, taxa=taxa))
    for item in spec.get("populations", []):
        path = check_file(root, item)
        m = json.loads(path.read_text())
        if (stage == "adapt" and all(t["taxon"] != 9606 for t in m["templates"])) or (
            taxa is not None and all(t["taxon"] not in taxa for t in m["templates"])
        ):
            excluded.append(
                {
                    "path": str(path),
                    "reason": "human-only adaptation"
                    if stage == "adapt"
                    else "declared source-species comparison",
                }
            )
            continue
        try:
            result.append(
                population.View(
                    path, genes, fold, stage, taxa=taxa,
                    center_queries=center_rna_queries,
                )
            )
        except ValueError as exc:
            if str(exc) != "No admitted molecular fitting populations":
                raise
            excluded.append(
                {"path": str(path), "reason": "no permitted populations in this fold"}
            )
            continue
        control = path.parent / "basal.json"
        if digest(control) != m["basal_sha256"]:
            raise ValueError("Changed molecular controls")
        document = json.loads(control.read_text())
        if document.get("fitted_human_intervention_genes") != []:
            raise ValueError("Basal data have human intervention fitting")
        for context, value in document["contexts"].items():
            if set(value.get("exposure_genes", [])) & fold.forbidden:
                continue
            if context in basal and basal[context] != value:
                raise ValueError("Conflicting basal context identity")
            basal[context] = value
    for item in spec.get("basal", []):
        document = json.loads(check_file(root, item).read_text())
        if document.get("fitted_human_intervention_genes") != []:
            raise ValueError("Observational context has intervention fitting")
        for context, value in document["contexts"].items():
            if set(value.get("exposure_genes", [])) & fold.forbidden:
                continue
            if context in basal and basal[context] != value:
                raise ValueError("Conflicting basal context identity")
            basal[context] = value
        for alias, target in document.get("aliases", {}).items():
            if target not in document["contexts"]:
                raise ValueError("Unresolved observational context alias")
            if target not in basal:
                continue
            if alias in basal and basal[alias] != basal[target]:
                # Source-matched controls retain their own assay semantics.
                continue
            basal[alias] = basal[target]
    for alias, target in spec.get("basal_aliases", {}).items():
        if target not in basal or (alias in basal and basal[alias] != basal[target]):
            raise ValueError("Unresolved or conflicting basal alias")
        basal[alias] = basal[target]
    return result, excluded, basal


def fitted_human_genes(views, genes):
    indices = set()
    for view in views:
        human = np.array([t["taxon"] == 9606 for t in view.templates])
        for family in view.families:
            for lo in range(0, len(family["indices"]), 262144):
                rows = view.rows[family["indices"][lo : lo + 262144]]
                indices.update(
                    np.unique(rows["targets"][human[rows["template"]]]).tolist()
                )
    return sorted(genes[i] for i in indices if i >= 0)


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists() and not args.resume:
        raise ValueError("Existing run requires explicit exact resume")
    if args.resume and args.initialize:
        raise ValueError("Resume and initialization are different operations")
    recipe = json.loads(Path(args.recipe).read_text())
    phase = recipe[args.stage]
    if recipe.get("purpose") not in ("research", "disposable-readiness"):
        raise ValueError("The run purpose must be explicit")
    if args.deadline - time.time() < 600:
        raise ValueError("Insufficient guarded time for fitting and finalization")
    torch.manual_seed(recipe["seed"])
    torch.set_num_threads(recipe.get("cpu_threads", 4))
    features = Features(args.features)
    genes = list(features.index)
    benchmark = Benchmark(args.fold, features)
    fold = benchmark.fold(args.scope)
    data_views, excluded, basal = views(
        args.data,
        features,
        fold,
        args.stage,
        Path(args.cache) if args.cache else output.parent / "packed-cache",
        taxa=phase.get("taxa"),
        center_rna_queries=phase.get("center_rna_queries", False),
    )
    config = Config(**recipe["model"])
    baseline = recipe.get("baseline")
    if baseline and (
        args.stage != "adapt" or args.initialize or phase["sl_fraction"] != 1
    ):
        raise ValueError("Direct baseline uses SL-only adaptation from scratch")
    model = (PairBaseline(config, baseline) if baseline else WorldModel(config)).to(
        args.device
    )
    quantitative = (
        population.Mixture(
            data_views,
            phase["quantitative_weights"],
            seed=recipe["seed"],
            temperature=phase.get("temperature", 0.7),
            max_cycles=phase.get("max_cycles"),
        )
        if data_views
        else None
    )
    templates = quantitative.templates if quantitative else []
    # Metadata vocabulary is fixed over the declared corpus, so an excluded
    # species/assay cannot renumber the human tokens during adaptation.
    declared = json.loads(Path(args.data).read_text())
    all_templates = []
    for item in declared.get("packed", []) + declared.get("populations", []):
        all_templates.extend(
            json.loads(check_file(Path(args.data).parent, item).read_text())[
                "templates"
            ]
        )
    vocab = packed.vocabulary(
        [
            *all_templates,
            {"assay": "human-SL", "taxon": 9606, "scope": "pan-cancer"},
            {"assay": "human-SL", "taxon": 9606, "scope": "cell-line"},
        ]
    )
    vocab.update(
        {
            "mechanism": {
                "unknown": 0,
                "knockout": 1,
                "crispri": 2,
                "crispra": 3,
                "deletion": 4,
                "temperature_sensitive": 5,
                "damP": 6,
                "ligand_addition": 7,
            },
            "method": {
                "unknown": 0,
                "Cas9": 1,
                "Cas12a": 2,
                "dCas9-KRAB": 3,
                "SGA": 4,
                "protein-addition": 5,
            },
        }
    )
    for key in ("assay", "taxon", "scope", "mechanism", "method"):
        if len(vocab[key]) > getattr(config, key + "_count"):
            raise ValueError("Model categorical capacity is too small")
    quantum_builder = (
        packed.Builder(features, templates, quantitative.scales, vocab, config, basal)
        if quantitative
        else None
    )
    exposures = set(fitted_human_genes(data_views, genes))
    if args.stage == "adapt":
        labels = benchmark.records(args.scope, "fit")
        verify_training_labels(labels, fold)
        exposures.update(g for r in labels for g in r.targets)
        sampler = adaptation.Sampler(
            ConditionSampler(labels, {"sl": 1.0}, seed=recipe["seed"]),
            quantitative,
            sl_fraction=phase["sl_fraction"],
            seed=recipe["seed"],
        )
        builder = adaptation.Builder(
            BatchBuilder(features, Scales([]), vocab, config, basal), quantum_builder
        )
    else:
        if quantitative is None:
            raise ValueError("No quantitative pretraining data")
        sampler, builder = quantitative, quantum_builder
    inherited = None
    resume_identity = None
    if args.resume:
        saved = torch.load(
            args.resume, map_location="cpu", weights_only=True, mmap=True
        )
        resume_identity = saved["identity"]
        del saved
        exposures.update(resume_identity["human_fitted_intervention_genes"])
    if args.initialize:
        state = torch.load(args.initialize, map_location=args.device, weights_only=True)
        inherited = state["identity"]
        if (
            state.get("schema") != "slp.r2-checkpoint/v1"
            or state["config"] != model.configuration()
        ):
            raise ValueError("Initializer model contract mismatch")
        if inherited.get("features_sha256") != digest(
            Path(args.features) / "manifest.json"
        ):
            raise ValueError("Initializer used different static features")
        if (
            set(inherited.get("human_fitted_intervention_genes", [])) & fold.forbidden
            or "human_fitted_intervention_genes" not in inherited
        ):
            raise ValueError(
                "Initializer violates or lacks the current human exposure contract"
            )
        if inherited.get("vocabulary") != vocab:
            raise ValueError("Initializer categorical vocabulary changed")
        model.load_state_dict(state["model"], strict=True)
        exposures.update(inherited["human_fitted_intervention_genes"])
        del state
    if exposures & fold.forbidden:
        raise ValueError("A forbidden human intervention entered fitting")
    (output / "basal.json").write_text(json.dumps(basal, sort_keys=True))
    identity = {
        "schema": "slp.r2-fitting-identity/v1",
        "purpose": recipe["purpose"],
        "phase": args.stage,
        "scope": args.scope,
        "fold": benchmark.identity,
        "forbidden_genes": sorted(fold.forbidden),
        "human_fitted_intervention_genes": sorted(exposures),
        "features_sha256": digest(Path(args.features) / "manifest.json"),
        "corpus_sha256": digest(args.data),
        "recipe_sha256": digest(args.recipe),
        "source": {
            p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "vocabulary": vocab,
        "initializer_sha256": digest(args.initialize)
        if args.initialize
        else resume_identity["initializer_sha256"]
        if resume_identity
        else None,
        "sampler_signature": sampler.signature,
        "basal_sha256": digest(output / "basal.json"),
        "numeric_scales": [
            {
                "source": t["source"],
                "assay": t["assay"],
                "kind": t["kind"],
                "units": t["units"],
                "mean": float(s[0]),
                "std": float(s[1]),
            }
            for t, s in zip(templates, quantitative.scales)
            if np.isfinite(s).all()
        ]
        if quantitative
        else [],
    }
    output_transforms = [
        v.output_transform for v in data_views if hasattr(v, "output_transform")
    ]
    if output_transforms:
        identity["numeric_output_transforms"] = output_transforms
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase["learning_rate"],
        weight_decay=phase["weight_decay"],
    )
    at = (
        resume(
            args.resume,
            model,
            optimizer,
            sampler,
            identity=identity,
            device=args.device,
        )
        if args.resume
        else 0
    )
    (output / "identity.json").write_text(json.dumps(identity, sort_keys=True))
    (output / "exposure.json").write_text(
        json.dumps(
            {"views": [v.receipt for v in data_views], "excluded_sources": excluded},
            sort_keys=True,
        )
    )
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = optimize(
        model,
        sampler,
        builder,
        optimizer,
        updates=phase["updates"],
        batch_size=phase["batch_size"],
        max_seconds=min(phase["max_seconds"], args.deadline - time.time() - 600),
        device=args.device,
        identity=identity,
        checkpoint_path=output / "checkpoint.pt",
        start_update=at,
        checkpoint_every=phase["checkpoint_every"],
        learning_rate=phase["learning_rate"],
        warmup=phase["warmup"],
        max_queries=phase["max_queries"],
        context_dropout=phase.get("context_dropout", 0.0),
        retain_updates=phase.get("retain_updates", []),
        stop_at_update=phase.get("stop_at_update"),
        numeric_loss=phase.get("numeric_loss", "student_t"),
    )
    report = {
        "stage": args.stage,
        "scope": args.scope,
        "result": result,
        "identity_sha256": digest(output / "identity.json"),
        "checkpoint_sha256": digest(output / "checkpoint.pt"),
        "exposure_file": "exposure.json",
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if torch.device(args.device).type == "cuda"
        else None,
        "quantitative_condition_draws": [
            {
                **child.view.families[j],
                "indices": None,
                "boundaries": None,
                "draws": child.cycles[j] * child.counts[j] + child.positions[j],
                "cycles": child.cycles[j],
            }
            for child in quantitative.children
            for j in range(len(child.counts))
        ]
        if quantitative
        else [],
        "sl_condition_draws": list(sampler.labels.draws)
        if args.stage == "adapt"
        else [],
    }
    (output / "complete.json").write_text(json.dumps(report, sort_keys=True))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "features", "data", "fold", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stage", choices=("pretrain", "adapt"), required=True)
    parser.add_argument("--scope", choices=("inner", "outer"), required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initialize")
    parser.add_argument("--resume")
    parser.add_argument(
        "--cache",
        help="Shared immutable raw-array cache; fitting indices and scales remain fold-specific",
    )
    main(parser.parse_args())
