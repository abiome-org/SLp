"""Run fail-closed molecular generalization validation on a canonical NPZ pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.evaluation import (  # noqa: E402
    PROTOCOLS,
    EvidenceRequirements,
    GeneralizationTable,
    additive_single_baseline,
    cardinality_mean_baseline,
    make_suite,
    regression_metrics,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(values: np.ndarray, row_count: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), row_count)
    if array.shape != (row_count,):
        raise ValueError(f"{name} must be scalar or have shape ({row_count},)")
    return array


def _first(pack, names: tuple[str, ...]):
    for name in names:
        if name in pack:
            return name, pack[name]
    return None, None


def _source(pack, row_count: int):
    name, values = _first(pack, ("source_id", "source"))
    if values is None:
        return None, None
    values = _rows(values, row_count, name)
    if name == "source" and np.issubdtype(values.dtype, np.integer) and "sources" in pack:
        labels = np.asarray(pack["sources"]).astype(str)
        if np.any(values < 0) or np.any(values >= len(labels)):
            raise ValueError("source contains an index outside sources")
        values = labels[values.astype("int64")]
    return name, values.astype(str)


def _condition(pack, row_count: int):
    name, values = _first(
        pack,
        ("experimental_condition_id", "perturbation_condition_id"),
    )
    if values is not None:
        return name, _rows(values, row_count, name).astype(str)
    mode_name, mode = _first(pack, ("mode", "modality"))
    time_name, time = _first(pack, ("duration_hours", "time_hours"))
    if mode is None or time is None:
        return None, None
    mode = _rows(mode, row_count, mode_name).astype(str)
    time = _rows(time, row_count, time_name).astype("float64")
    if not np.all(np.isfinite(time)):
        return None, None
    dose_name, dose = _first(pack, ("dose", "dose_value"))
    if dose is None:
        dose_text = np.repeat("not_applicable_or_unreported", row_count)
    else:
        dose = _rows(dose, row_count, dose_name)
        dose_text = dose.astype(str)
    values = np.asarray(
        [f"mode={m}|hours={t:.9g}|dose={d}" for m, t, d in zip(mode, time, dose_text)]
    )
    return f"{mode_name}+{time_name}" + ("" if dose is None else f"+{dose_name}"), values


def load_table(path: Path) -> tuple[GeneralizationTable, dict[str, object]]:
    with np.load(path, allow_pickle=False) as pack:
        action_name, actions = _first(pack, ("actions", "pairs"))
        if actions is None:
            raise ValueError("pack must contain actions or pairs")
        actions = np.asarray(actions)
        if actions.ndim != 2:
            raise ValueError("actions/pairs must have shape [rows, slots]")
        row_count = len(actions)
        target_name, target = _first(pack, ("target_delta", "target"))
        context_name, context = _first(pack, ("context_id", "context"))
        if context is not None:
            context = _rows(context, row_count, context_name).astype(str)
        source_name, source = _source(pack, row_count)
        condition_name, condition = _condition(pack, row_count)
        semantics_name, semantics = _first(pack, ("target_semantics",))
        if semantics is not None:
            semantics_array = np.asarray(semantics).astype(str)
            if semantics_array.size != 1:
                raise ValueError("target_semantics must be a scalar")
            semantics = str(semantics_array.reshape(-1)[0])
        table = GeneralizationTable(
            actions=actions,
            target=target,
            context=context,
            source=source,
            condition=condition,
            target_semantics=semantics,
        )
        fields = {
            "actions": action_name,
            "target": target_name,
            "target_semantics": semantics_name,
            "context": context_name,
            "source": source_name,
            "condition": condition_name,
        }
    return table, fields


def _finite_metrics(values: dict[str, float]) -> dict[str, float | None]:
    return {key: value if np.isfinite(value) else None for key, value in values.items()}


def run(args: argparse.Namespace) -> tuple[dict[str, object], bool]:
    table, fields = load_table(args.input)
    requirements = EvidenceRequirements(
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
        min_test_action_sets=args.min_test_action_sets,
        min_test_genes=args.min_test_genes,
    )
    splits = make_suite(
        table,
        protocols=args.protocols,
        folds=args.folds,
        seed=args.seed,
        requirements=requirements,
    )
    split_reports = []
    for split in splits:
        report = split.as_dict()
        if split.eligible and table.target is not None:
            observed = table.target[split.test]
            mean = cardinality_mean_baseline(table, split)
            baselines: dict[str, object] = {
                "cardinality_mean": _finite_metrics(regression_metrics(observed, mean))
            }
            if table.target_semantics == "perturbation_delta":
                additive, additive_audit = additive_single_baseline(table, split)
                baselines["additive_single"] = {
                    "metrics": _finite_metrics(regression_metrics(observed, additive)),
                    "audit": additive_audit,
                }
            report["baselines"] = baselines
        split_reports.append(report)

    schema_failures = []
    if table.target is None:
        schema_failures.append("missing target")
    if table.target_semantics != "perturbation_delta":
        schema_failures.append("target_semantics must be perturbation_delta")
    required = set(args.require)
    required_failures = [
        f"{split.protocol}/fold-{split.fold}: " + "; ".join(split.reasons)
        for split in splits
        if split.protocol in required and not split.eligible
    ]
    missing_protocols = required - set(args.protocols)
    required_failures.extend(f"required protocol was not run: {name}" for name in missing_protocols)
    passed = not schema_failures and not required_failures
    report = {
        "schema": "slp-molecular-generalization-audit-v1",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "rows": len(table.actions),
        "action_slots": table.actions.shape[1],
        "target_dimensions": 0 if table.target is None else table.target.shape[1],
        "resolved_fields": fields,
        "folds": args.folds,
        "seed": args.seed,
        "required_protocols": sorted(required),
        "schema_failures": schema_failures,
        "required_failures": required_failures,
        "passed": passed,
        "splits": split_reports,
    }
    return report, passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit hard molecular generalization splits and baselines."
    )
    parser.add_argument("input", type=Path, help="canonical perturbation-outcome NPZ")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument(
        "--require",
        nargs="+",
        choices=PROTOCOLS,
        default=list(PROTOCOLS),
        help="protocols that must meet evidence thresholds in every fold",
    )
    parser.add_argument("--min-train-rows", type=int, default=128)
    parser.add_argument("--min-test-rows", type=int, default=32)
    parser.add_argument("--min-test-action-sets", type=int, default=16)
    parser.add_argument("--min-test-genes", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, passed = run(args)
    output = args.output or ROOT / "results/generalization" / f"{args.input.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "audit": str(output), "failures": report["required_failures"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
