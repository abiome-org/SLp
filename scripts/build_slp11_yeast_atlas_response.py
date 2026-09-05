#!/usr/bin/env python3
"""Build a source-specific train/validation yeast RNA development snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules/slp-1-1-yeast-atlas-response-v1/atlas_response.py"
RDATA_SITE = ROOT / "data/tools/rdata-1.1.0/site-packages"
FC_SHA256 = "c210fe541b0b91bc6eead28aa2265065afceec763ade1abd682c58896299a240"
PTB_SHA256 = "01c2d54ac838179be29694ed300cb17edac47dd4db23a4018407546e0651b165"
WHEEL_SHA256 = "12efb7597725d6db6cc78d84eb522a9634008f8fce1c2733b7fd42b9013bc41f"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module():
    sys.path.insert(0, str(RDATA_SITE))
    spec = importlib.util.spec_from_file_location("slp11_yeast_atlas_response", MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def intervention_map(evidence_path: Path) -> dict[str, str]:
    result = {}
    for raw in evidence_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(raw)
        if (
            item.get("schema") == "slp.atlas-genotype-identity-evidence/v1"
            and item.get("mappingClass") == "current-exact"
            and item.get("presentInConditions") == ["control", "nacl"]
            and len(item.get("currentSgdCuries", [])) == 1
        ):
            result[item["sourceAssignment"]] = item["currentSgdCuries"][0]
    if len(result) != 2_941 or len(set(result.values())) != len(result):
        raise ValueError("pinned exact atlas intervention mapping drift")
    return result


def parse_frame_name(module, value: str) -> tuple[str, str]:
    matched = module.FILE_PATTERN.fullmatch(value)
    if matched is None:
        raise ValueError(f"unexpected fcs frame name {value}")
    condition, genotype = matched.groups()
    return condition, "bc-" + genotype


def cell_counts(ptbs: dict[str, object]) -> dict[tuple[str, str], int]:
    result = {}
    for condition, source_name in (("Control", "control"), ("NaCl", "nacl")):
        frame = ptbs[source_name]
        if list(frame.columns) != [
            "assignment_consensus2", "cell_number", "avg_lvscoreFU2", "var_lvscoreFU2",
            "std_lvscoreFU2", "avg_lvscore_scaledFU2", "var_lvscore_scaledFU2",
            "sd_lvscore_scaledFU2", "Stucked",
        ]:
            raise ValueError("ptb_summary frame contract drift")
        for assignment, count in zip(frame["assignment_consensus2"], frame["cell_number"], strict=True):
            key = (condition, str(assignment))
            if key in result or int(count) != count or int(count) <= 5:
                raise ValueError("invalid or duplicate ptb_summary cell count")
            result[key] = int(count)
    return result


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("immutable adapter output exists")
    expected = {
        args.fc: FC_SHA256,
        args.ptb: PTB_SHA256,
        args.wheel: WHEEL_SHA256,
        args.current_orfs: args.current_orfs_sha256,
        args.identity_evidence: args.identity_evidence_sha256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"input digest mismatch: {path}")
    module = load_module()
    import rdata

    if rdata.__version__ != "1.1.0":
        raise ValueError("rdata runtime drift")
    args.output.mkdir(parents=True)
    source_dir = args.output / "source"
    source_dir.mkdir()
    shutil.copyfile(MODULE, source_dir / MODULE.name)
    shutil.copyfile(Path(__file__), source_dir / Path(__file__).name)
    started = time.monotonic()
    values_path = args.output / ".aligned-values.npy"
    observed_path = args.output / ".aligned-observed.npy"
    inventory = module.extract_fcs(args.fc, values_path, observed_path)
    values = np.load(values_path, mmap_mode="r")
    observed = np.load(observed_path, mmap_mode="r")
    converted = rdata.read_rda(args.ptb)
    if list(converted) != ["ptbs"] or set(converted["ptbs"]) != {"control", "nacl"}:
        raise ValueError("ptb_summary top-level contract drift")
    counts = cell_counts(converted["ptbs"])
    action_mapping = intervention_map(args.identity_evidence)
    query_mapping, ambiguous_query_names = module.exact_current_maps(args.current_orfs)

    source_query_names = np.asarray(inventory["query_names"], dtype=str)
    query_candidates: dict[str, list[int]] = {}
    unmapped_query_names = []
    for index, name in enumerate(source_query_names):
        stable = query_mapping.get(str(name))
        if stable is None:
            unmapped_query_names.append(str(name))
        else:
            query_candidates.setdefault(stable, []).append(index)
    duplicate_query_ids = {key for key, rows in query_candidates.items() if len(rows) != 1}
    query_rows = np.asarray(
        [rows[0] for key, rows in sorted(query_candidates.items()) if key not in duplicate_query_ids],
        dtype=np.int64,
    )
    query_ids = np.asarray(
        [key for key, rows in sorted(query_candidates.items()) if key not in duplicate_query_ids],
        dtype=str,
    )
    if not len(query_ids):
        raise ValueError("no exact stable query mappings")

    selected = []
    excluded = {
        "unmapped_intervention": 0,
        "protected_validation_or_final": 0,
        "development_test": 0,
        "missing_cell_count": 0,
    }
    seen = set()
    for frame, name in enumerate(inventory["frame_names"]):
        condition, assignment = parse_frame_name(module, name)
        stable = action_mapping.get(assignment)
        if stable is None:
            excluded["unmapped_intervention"] += 1
            continue
        if module.protected_role(stable) != "pretrain":
            excluded["protected_validation_or_final"] += 1
            continue
        role = module.development_role(stable)
        if role == "test":
            excluded["development_test"] += 1
            continue
        count = counts.get((condition, assignment))
        if count is None:
            excluded["missing_cell_count"] += 1
            continue
        key = (condition, stable)
        if key in seen:
            raise ValueError("duplicate condition/stable intervention frame")
        seen.add(key)
        selected.append((frame, condition, assignment, stable, role, count, name))
    selected.sort(key=lambda item: (item[1], item[3]))
    frame_rows = np.asarray([item[0] for item in selected], dtype=np.int64)
    target = np.asarray(values[np.ix_(frame_rows, query_rows)], dtype=np.float32)
    mask = np.asarray(observed[np.ix_(frame_rows, query_rows)], dtype=np.bool_)
    if not mask.any(1).all() or not np.isfinite(target[mask]).all():
        raise ValueError("selected source response support is invalid")
    target[~mask] = 0
    roles = np.asarray([item[4] for item in selected])
    context = np.asarray([0 if item[1] == "Control" else 1 for item in selected], dtype=np.int64)
    output = args.output / "development.npz"
    np.savez_compressed(
        output,
        targets=target,
        observed=mask,
        action_ids=np.asarray([item[3] for item in selected]),
        action_source_ids=np.asarray([item[2] for item in selected]),
        record_ids=np.asarray([item[6] for item in selected]),
        context_index=context,
        context_ids=np.asarray(["control", "nacl-0.4M-15min"]),
        num_cells=np.asarray([item[5] for item in selected], dtype=np.int32),
        query_ids=query_ids,
        query_source_names=source_query_names[query_rows],
        split_train=np.flatnonzero(roles == "train").astype(np.int64),
        split_validation=np.flatnonzero(roles == "validation").astype(np.int64),
        split_test=np.asarray([], dtype=np.int64),
        target_value_space=np.asarray("author-logfoldchanges-unknown-upstream-transform"),
        ncbi_taxon=np.asarray(4932, dtype=np.int64),
    )
    values._mmap.close()
    observed._mmap.close()
    values_path.unlink()
    observed_path.unlink()
    manifest = {
        "schema": "slp.nadal-ribelles-yeast-response-development/v1",
        "status": "source-specific development candidate; no model fit or OMF admission",
        "output": {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size},
        "inputs": {str(path): sha256(path) for path in expected},
        "source": {
            "organism": "Saccharomyces cerevisiae", "ncbiTaxon": 4932,
            "conditions": ["control", "0.4 M NaCl for 15 min"],
            "endpoint": "authors' per-genotype logfoldchanges versus wild type in the corresponding condition",
            "endpointLimit": "The deposited generating code copies names/logfoldchanges from upstream DEG CSV files, but does not contain the upstream differential-expression call. Exact normalization, test, and log-fold-change estimator are therefore not established from these summaries.",
        },
        "counts": {
            **{key: int(value) for key, value in inventory.items() if isinstance(value, int)},
            "records": len(selected), "queriesExactStable": len(query_ids),
            "observedTargets": int(mask.sum()),
            "trainRecords": int(np.count_nonzero(roles == "train")),
            "validationRecords": int(np.count_nonzero(roles == "validation")),
            "contexts": {name: int(np.count_nonzero(context == index)) for index, name in enumerate(("control", "nacl"))},
            "unmappedQueryNames": len(unmapped_query_names),
            "ambiguousCurrentMappingNames": len(ambiguous_query_names),
            "duplicateStableQueryMappingsExcluded": len(duplicate_query_ids),
            "exclusions": excluded,
        },
        "split": {
            "protected": "retain only slp-1.1-yeast-global-held-v1 pretrain bucket30-99",
            "development": "sha256 slp11-development-v1|731|4932|SGD_CURIE; train<70, validation70-84; development-test>=85 omitted",
            "testOutcomesEmitted": False,
        },
        "contextInput": "No raw wild-type expression or measured basal context is present in the two summary objects; no basal vector is fabricated.",
        "queryMapping": "exact case-sensitive unique current SGD systematic or standard name; ambiguous, unmapped, and duplicate stable mappings excluded",
        "rdata": {"version": rdata.__version__, "wheelSha256": WHEEL_SHA256, "license": "MIT"},
        "runtimeSeconds": time.monotonic() - started,
        "sourceHashes": {path.name: sha256(path) for path in source_dir.iterdir()},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fc", type=Path, required=True)
    parser.add_argument("--ptb", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--current-orfs", type=Path, required=True)
    parser.add_argument("--current-orfs-sha256", required=True)
    parser.add_argument("--identity-evidence", type=Path, required=True)
    parser.add_argument("--identity-evidence-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
