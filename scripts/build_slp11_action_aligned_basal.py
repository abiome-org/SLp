"""Build control-only action-aligned basal abundance feature sidecars.

Only identity, split, control-expression, and support arrays are opened.  No
perturbed target member is accessed.  Missing action genes remain zero-filled
storage with an explicit false observation mask.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HUMAN = ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz"
HUMAN_REFERENCE = ROOT / "results/slp11-transition/human-source3-vs-four-context-mean-objective-seed731-v2/arm-source3/reference.npz"
YEAST_MANIFEST = ROOT / "data/derived/slp11-yeast-atlas-counts/nadal-ribelles-raw-rna-development-v1/moments-manifest.json"
YEAST_WT = ROOT / "results/slp11-transition/yeast-wildtype-batch-diagnostic-v1/wildtype-reference.npz"
OUTPUT = ROOT / "data/derived/slp11-action-aligned-basal-v1"
REPORT = ROOT / "results/slp11-transition/action-aligned-basal-audit-v1"
PINS = {
    "humanDevelopment": "55def8f73e026b453a7250c82a2c3478db0290e2cca4f26e02ba1100c3f3384c",
    "humanReference": "54cac4bc2e2ee02a6d78f812d5646cf3988154d5ae4f371265b24751f03c99b1",
    "yeastMomentsManifest": "70a49ecaeb271fc72ecc93ede207c59a816e74d1ae3133bbf3a2803cce5d8eba",
    "yeastWildType": "190dc64dd9ee8809f56f82b690265827376c72b36286e46e04b8aebee64fa1b5",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def align_action_values(
    action_ids: np.ndarray,
    query_ids: np.ndarray,
    values: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact stable-ID alignment with explicit missingness."""
    action_ids = np.asarray(action_ids).astype(str)
    query_ids = np.asarray(query_ids).astype(str)
    values = np.asarray(values)
    observed = np.asarray(observed, dtype=bool)
    if (
        action_ids.ndim != 1
        or query_ids.ndim != 1
        or values.ndim != 2
        or values.shape != observed.shape
        or values.shape[1] != len(query_ids)
        or len(set(action_ids)) != len(action_ids)
        or len(set(query_ids)) != len(query_ids)
    ):
        raise ValueError("invalid action/query alignment inputs")
    lookup = {gene: index for index, gene in enumerate(query_ids)}
    result = np.zeros((values.shape[0], len(action_ids)), dtype=values.dtype)
    mask = np.zeros(result.shape, dtype=bool)
    for action_index, action in enumerate(action_ids):
        if action in lookup:
            query_index = lookup[action]
            mask[:, action_index] = observed[:, query_index]
            result[:, action_index] = np.where(mask[:, action_index], values[:, query_index], 0)
    if not np.isfinite(result).all() or np.any(result[~mask] != 0):
        raise ValueError("aligned values violate finite masked storage")
    return result, mask


def weighted_normalizer(values: np.ndarray, observed: np.ndarray, context: np.ndarray, weights: np.ndarray, contexts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    context = np.asarray(context, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (values.shape == observed.shape == context.shape == weights.shape):
        raise ValueError("normalizer inputs must have identical one-dimensional shapes")
    means, scales, counts = np.zeros(contexts), np.ones(contexts), np.zeros(contexts, dtype=np.int64)
    for index in range(contexts):
        selected = (context == index) & observed & (weights > 0)
        if not selected.any() or not np.isfinite(values[selected]).all():
            raise ValueError("each context needs observed fitting control values")
        local_weight = weights[selected]
        local_weight = local_weight / local_weight.sum()
        means[index] = local_weight @ values[selected]
        scales[index] = np.sqrt(local_weight @ np.square(values[selected] - means[index]))
        if scales[index] <= 1e-12:
            scales[index] = 1.0
        counts[index] = selected.sum()
    return means, scales, counts


def _roles(size: int, train: np.ndarray, validation: np.ndarray) -> np.ndarray:
    result = np.full(size, "excluded", dtype="<U10")
    result[np.asarray(train, dtype=np.int64)] = "train"
    result[np.asarray(validation, dtype=np.int64)] = "validation"
    if np.any(result == "excluded"):
        raise ValueError("development row lacks an explicit role")
    return result


def build_human() -> tuple[Path, dict[str, object]]:
    if sha(HUMAN) != PINS["humanDevelopment"] or sha(HUMAN_REFERENCE) != PINS["humanReference"]:
        raise ValueError("human input hash drift")
    # np.load reads individual ZIP members lazily. Targets/observed/control_targets
    # are deliberately absent from this access list.
    with np.load(HUMAN, allow_pickle=False) as z:
        action_by_row = np.asarray(z["action_ids"]).astype(str)
        query_ids = np.asarray(z["query_ids"]).astype(str)
        contexts = np.asarray(z["context_ids"]).astype(str)
        context_by_row = np.asarray(z["context_index"], dtype=np.int64)
        basal = np.asarray(z["context_basal_expression"], dtype=np.float32)
        basal_observed = np.asarray(z["context_basal_observed"], dtype=bool)
        value_space = str(z["context_value_space"].item())
        train = np.asarray(z["split_train"], dtype=np.int64)
        validation = np.asarray(z["split_validation"], dtype=np.int64)
        if len(z["split_test"]):
            raise ValueError("source3 development unexpectedly contains test rows")
    roles = _roles(len(action_by_row), train, validation)
    action_ids = np.unique(action_by_row)
    action_index = {gene: index for index, gene in enumerate(action_ids)}
    row_action_index = np.asarray([action_index[gene] for gene in action_by_row], dtype=np.int64)
    action_values, action_observed = align_action_values(action_ids, query_ids, basal, basal_observed)
    row_values = action_values[context_by_row, row_action_index]
    row_observed = action_observed[context_by_row, row_action_index]
    normalization_weight = np.zeros(len(action_by_row), dtype=np.float64)
    # Equal unique-gene mass within context; duplicates divide that mass.
    for context_index in range(len(contexts)):
        fitting = np.flatnonzero((context_by_row == context_index) & (roles == "train") & row_observed)
        local_actions = action_by_row[fitting]
        for gene in np.unique(local_actions):
            positions = fitting[local_actions == gene]
            normalization_weight[positions] = 1.0 / len(positions)
    means, scales, counts = weighted_normalizer(
        row_values, row_observed, context_by_row, normalization_weight, len(contexts)
    )
    with np.load(HUMAN_REFERENCE, allow_pickle=False) as reference:
        selected = np.asarray(reference["context_query_indices"], dtype=np.int64)
        reference_values = np.asarray(reference["context_values"][:3], dtype=np.float32)
        reference_mask = np.asarray(reference["context_mask"][:3], dtype=bool)
    common = basal_observed.all(0)
    basal_mean = np.asarray([basal[i, common].mean() for i in range(len(contexts))])[:, None]
    basal_std = np.asarray([basal[i, common].std() for i in range(len(contexts))])[:, None]
    reconstructed = np.where(basal_observed, (basal - basal_mean) / np.maximum(basal_std, 1e-5), 0)
    if not np.array_equal(reference_mask, basal_observed[:, selected]) or not np.array_equal(reference_values, reconstructed[:, selected]):
        raise ValueError("saved human context-token values do not reproduce from controls")
    path = OUTPUT / "human-source3-action-basal.npz"
    np.savez_compressed(
        path,
        schema=np.asarray("slp.action-aligned-basal/human-source3-v1"),
        action_taxon=np.full(len(action_ids), 9606, dtype=np.int64),
        action_ids=action_ids,
        context_ids=contexts,
        action_basal_value=action_values,
        action_basal_observed=action_observed,
        population_action_index=row_action_index,
        population_context_index=context_by_row,
        population_role=roles,
        population_basal_value=row_values,
        population_basal_observed=row_observed,
        population_normalization_weight=normalization_weight,
        fitting_value_mean=means,
        fitting_value_scale=scales,
        fitting_observed_population_count=counts,
        value_space=np.asarray(value_space),
        missing_storage=np.asarray("zero only where action_basal_observed is false"),
    )
    context_report = {}
    for index, name in enumerate(contexts):
        context_report[name] = {}
        for role in ("train", "validation"):
            rows = (context_by_row == index) & (roles == role)
            genes = np.unique(action_by_row[rows])
            positions = np.asarray([action_index[g] for g in genes])
            context_report[name][role] = {
                "rows": int(rows.sum()), "genes": len(genes),
                "observedActionBasalGenes": int(action_observed[index, positions].sum()),
            }
    selected_genes = set(query_ids[selected].tolist())
    return path, {
        "rows": len(action_by_row), "actions": len(action_ids),
        "queryMappedActions": int(np.isin(action_ids, query_ids).sum()),
        "fixedPanelObservedActionsAllContexts": int(action_observed.all(0).sum()),
        "contextTokenCount": len(selected),
        "actionRosterGenesAmong64ContextTokens": len(selected_genes & set(action_ids)),
        "contexts": context_report,
        "fittingNormalizerMean": means.tolist(),
        "fittingNormalizerScale": scales.tolist(),
        "targetMembersRead": [],
    }


def build_yeast() -> tuple[Path, dict[str, object]]:
    if sha(YEAST_MANIFEST) != PINS["yeastMomentsManifest"] or sha(YEAST_WT) != PINS["yeastWildType"]:
        raise ValueError("yeast input hash drift")
    manifest = json.loads(YEAST_MANIFEST.read_text(encoding="utf-8"))
    with np.load(YEAST_WT, allow_pickle=False) as z:
        query_ids = np.asarray(z["query_ids"]).astype(str)
        context_ids = np.asarray(["Control", "NaCl"])
        batch_ids = np.concatenate((np.asarray(z["control_batch_ids"]).astype(str), np.asarray(z["nacl_batch_ids"]).astype(str)))
        batch_context = np.concatenate((np.zeros(len(z["control_batch_ids"]), dtype=np.int64), np.ones(len(z["nacl_batch_ids"]), dtype=np.int64)))
        control_values = np.concatenate((np.asarray(z["control_mean"], dtype=np.float64), np.asarray(z["nacl_mean"], dtype=np.float64)))
        control_cells = np.concatenate((np.asarray(z["control_num_cells"], dtype=np.int64), np.asarray(z["nacl_num_cells"], dtype=np.int64)))
    populations = []
    action_union = set()
    for entry in manifest["shards"]:
        with np.load(Path(entry["path"]), allow_pickle=False) as z:
            # Only metadata members are accessed; sum/sum_squares stay unopened.
            context, batch = str(z["context"].item()), str(z["batch_id"].item())
            actions = np.asarray(z["group_action_id"]).astype(str)
            roles = np.asarray(z["development_role"]).astype(str)
            cells = np.asarray(z["num_cells"], dtype=np.int64)
        selected = roles != "control"
        for action, role, count in zip(actions[selected], roles[selected], cells[selected], strict=True):
            populations.append((context, batch, action, role, int(count)))
            action_union.add(action)
    action_ids = np.asarray(sorted(action_union))
    if len(action_ids) != 2013:
        raise ValueError("yeast development action roster drift")
    action_values, action_observed = align_action_values(
        action_ids, query_ids, control_values, np.ones_like(control_values, dtype=bool)
    )
    batch_lookup = {(context_ids[c], batch): i for i, (c, batch) in enumerate(zip(batch_context, batch_ids, strict=True))}
    action_lookup = {gene: i for i, gene in enumerate(action_ids)}
    pop_context = np.asarray([0 if row[0] == "Control" else 1 for row in populations], dtype=np.int64)
    pop_batch = np.asarray([batch_lookup[(row[0], row[1])] for row in populations], dtype=np.int64)
    pop_action = np.asarray([action_lookup[row[2]] for row in populations], dtype=np.int64)
    pop_role = np.asarray([row[3] for row in populations])
    pop_cells = np.asarray([row[4] for row in populations], dtype=np.int64)
    pop_values = action_values[pop_batch, pop_action]
    pop_observed = action_observed[pop_batch, pop_action]
    normalization_weight = np.zeros(len(populations), dtype=np.float64)
    for context in range(2):
        fitting = np.flatnonzero((pop_context == context) & (pop_role == "train") & pop_observed)
        local_actions = pop_action[fitting]
        for action in np.unique(local_actions):
            positions = fitting[local_actions == action]
            normalization_weight[positions] = pop_cells[positions] / pop_cells[positions].sum()
    means, scales, counts = weighted_normalizer(
        pop_values, pop_observed, pop_context, normalization_weight, 2
    )
    path = OUTPUT / "yeast-2013-action-wt-batch-basal.npz"
    np.savez_compressed(
        path,
        schema=np.asarray("slp.action-aligned-basal/yeast-batch-v1"),
        action_taxon=np.full(len(action_ids), 4932, dtype=np.int64),
        action_ids=action_ids,
        context_ids=context_ids,
        batch_ids=batch_ids,
        batch_context_index=batch_context,
        control_num_cells=control_cells,
        action_basal_value=action_values,
        action_basal_observed=action_observed,
        population_action_index=pop_action,
        population_context_index=pop_context,
        population_batch_index=pop_batch,
        population_role=pop_role,
        population_num_cells=pop_cells,
        population_basal_value=pop_values,
        population_basal_observed=pop_observed,
        population_normalization_weight=normalization_weight,
        fitting_value_mean=means,
        fitting_value_scale=scales,
        fitting_observed_population_count=counts,
        value_space=np.asarray("per-WT-batch mean of per-cell ln(1+10000*count/sum_all_6951_RNA_rows)"),
        missing_storage=np.asarray("zero only where action_basal_observed is false"),
    )
    return path, {
        "populations": len(populations), "actions": len(action_ids),
        "strictQueryObservedActions": int(action_observed.all(0).sum()),
        "unmeasuredActions": int((~action_observed.all(0)).sum()),
        "batches": {"Control": int((batch_context == 0).sum()), "NaCl": int((batch_context == 1).sum())},
        "controlCells": {"Control": int(control_cells[batch_context == 0].sum()), "NaCl": int(control_cells[batch_context == 1].sum())},
        "fittingNormalizerMean": means.tolist(), "fittingNormalizerScale": scales.tolist(),
        "perturbedMomentMembersRead": [],
    }


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite immutable action-basal audit")
    OUTPUT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    protocol = {
        "schema": "slp.action-aligned-basal-audit-protocol/v1",
        "question": "Does the current action encoder directly receive measured matched-control abundance for the intervention gene?",
        "pins": PINS,
        "access": "identity, split, and control-only expression members; no perturbed target members",
        "proposedMatchedStaticTest": {
            "arms": ["base features plus zero basal value and measured-presence", "same base features plus fold-local normalized measured basal value and the identical presence"],
            "normalization": "refit per context and inner fitting fold only; equal-gene/cell-within-gene weights for yeast batch populations and equal unique fitting genes for human source3",
            "rule": "at least 1% gene-profile MSE reduction with no independently query-centered correlation regression in every represented context",
            "identity": "stable IDs and taxonomy only for alignment; no learned gene-ID embedding",
            "scope": "point ridge first; a neural arm requires a separate frozen protocol only if the point test is useful",
        },
    }
    (REPORT / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    human_path, human = build_human()
    yeast_path, yeast = build_yeast()
    report = {
        "schema": "slp.action-aligned-basal-audit/v1",
        "finding": "No current response-query action encoder input directly contains measured matched-control abundance of the intervention gene.",
        "humanCurrentAccess": "The action encoder receives static descriptors only. A separate context encoder mean-pools 64 globally selected control-query tokens, identical for every action in a context; it performs no action-ID lookup or target-aligned scalar join.",
        "yeastCurrentAccess": "The completed static batch-ridge uses only 577 static descriptors and batch intercepts. The prepared neural core likewise requires an explicit appended scalar to expose action-aligned WT abundance.",
        "human": human,
        "yeast": yeast,
        "observationSpacesNotComparable": "Human is log2 fixed-6789-panel CP10k from core-control pseudobulk means; yeast is natural-log full-6951-source-row CP10k from WT cells. Context-local normalization is mandatory and does not make the assays biologically equivalent.",
        "artifacts": {
            "human": {"path": str(human_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(human_path)},
            "yeast": {"path": str(yeast_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(yeast_path)},
        },
        "protocolSha256": sha(REPORT / "protocol.json"),
        "sourceSha256": sha(Path(__file__)),
        "heldOrPerturbedOutcomeMembersRead": [],
    }
    (REPORT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
