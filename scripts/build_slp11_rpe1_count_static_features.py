#!/usr/bin/env python3
"""Build outcome-free raw static577 features for the RPE1 raw-cell roster."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TAXON = 9606
ENTITY = re.compile(r"ENSG[0-9]+")
ROUTING_SHA256 = "10f3d313a5671122bde10a9bd586e3a2808d6f9b554f737ddcbbc28becc5e2f2"
SOURCE_SHA256 = "9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1"
QUERY_ROSTER_SHA256 = "20f22e3f4c58981d6805911e4dc1f2069a387b2b2be695c8eabd155d62432e79"
STATIC_SHA256 = "20313e37d70d52253fa7b4b9b569b0fd504686a35be46b0607db1ab1c7484e54"
STATIC_MANIFEST_SHA256 = "857e34d73c45f94ef078cc1e2e271f91ec223d51e217f87d5411869457b3fe3c"
BASIS_SHA256 = "718764f4ebb6ab9ac31dba65d7d6453525e04a98b999aa7dcfeb4c3a1ab62abd"
MAPPING_SHA256 = "0d6fe982ce7023b2901171fd0e1419a2e9fbd7fbb9b3473a82ac6dee454f6e56"
GAF_SHA256 = "8b97980a895cb74255615f7cdbdd818f72a3999867b7d2a14f867874480693e1"
K562_STATIC_SHA256 = "6706f8867adedef8822897bc275ea90680584f84afd24771e4beb3c8ecf07659"
K562_ROSTER_SHA256 = "f2ee702a0714ca7f11f4fd2aa96f4c1825617c0e4f2bcdac42135cd0ba938d7b"
K562_BUILDER_SHA256 = "26eff902ff3582b2b2a6dbc684c8207e8331408f1574fd529360993ca06679c0"
FEATURE_NAME = "rpe1-essential-count-static577.npz"
ROSTER_NAME = "roster-index.npz"
NORMALIZER_NAME = "fitting-action-normalizers.npz"


class RpeStaticError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RpeStaticError(f"SHA-256 mismatch for {path}: {actual}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_k_builder(path: Path):
    spec = importlib.util.spec_from_file_location("rpe1_static_k562_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def action_rosters(routing: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, str]]:
    action = routing["action_ids"].astype(str)
    role = routing["intervention_role"].astype(str)
    control = np.asarray(routing["is_control"], dtype=np.bool_)
    unresolved = np.asarray(routing["unresolved_action"], dtype=np.bool_)
    if any(len(item) != len(action) for item in (role, control, unresolved)):
        raise RpeStaticError("routing cell arrays differ")
    if np.any(control & (action != "")) or np.any(unresolved & (role != "unresolved-excluded")):
        raise RpeStaticError("control or unresolved identity contract drift")
    selected = (~control) & (~unresolved)
    if np.any([ENTITY.fullmatch(item) is None for item in action[selected]]):
        raise RpeStaticError("resolved action is not a stable ENSG")
    by_action: dict[str, str] = {}
    allowed = {"train", "validation", "test-excluded"}
    for gene, item_role in zip(action[selected], role[selected], strict=True):
        if item_role not in allowed:
            raise RpeStaticError("resolved action has an invalid role")
        prior = by_action.setdefault(gene, item_role)
        if prior != item_role:
            raise RpeStaticError("one action occurs in multiple global roles")
    return np.asarray(sorted(by_action)), by_action


def normalizer(values: np.ndarray, rows: np.ndarray, floor: float = 1e-5):
    """Return exact float64 fitting statistics; do not round before transform."""
    array = np.asarray(values)
    index = np.asarray(rows, dtype=np.int64)
    if array.ndim != 2 or array.dtype != np.float32 or not np.isfinite(array).all():
        raise RpeStaticError("raw static features must be finite float32 [E,F]")
    if index.ndim != 1 or len(index) != len(set(index.tolist())) or not len(index):
        raise RpeStaticError("normalizer rows must be unique and nonempty")
    selected = array[index].astype(np.float64)
    mean = selected.mean(0)
    sd = selected.std(0, ddof=0)
    scale = np.where(sd <= floor, 1.0, sd)
    return mean, sd, scale


def assemble(
    routing: dict[str, np.ndarray],
    source: dict[str, np.ndarray],
    exact_mapping_ids: set[str],
    missing_go: dict[str, np.ndarray],
    missing_go_direct: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    query = routing["query_ids"].astype(str)
    if (
        query.shape != (8749,)
        or len(set(query.tolist())) != len(query)
        or any(ENTITY.fullmatch(item) is None for item in query)
        or not np.all(routing["query_taxon"] == TAXON)
        or not np.asarray(routing["query_in_matrix"]).all()
    ):
        raise RpeStaticError("RPE1 query identity contract mismatch")
    actions, role_by_action = action_rosters(routing)
    role_count = {
        role: sum(value == role for value in role_by_action.values())
        for role in ("train", "validation", "test-excluded")
    }
    if len(actions) != 2390 or role_count != {"train": 1666, "validation": 360, "test-excluded": 364}:
        raise RpeStaticError(f"RPE1 action role counts drifted: {role_count}")

    source_ids = source["entity_id"].astype(str)
    source_values = source["feature_values"]
    if (
        source_values.shape != (24031, 577)
        or source_values.dtype != np.float32
        or not np.all(source["entity_taxon"] == TAXON)
        or list(source_ids) != sorted(set(source_ids.tolist()))
        or not np.isfinite(source_values).all()
    ):
        raise RpeStaticError("shared human static source contract mismatch")
    source_index = {gene: row for row, gene in enumerate(source_ids)}
    entity_ids = np.asarray(sorted(set(query.tolist()) | set(actions.tolist())))
    entity_index = {gene: row for row, gene in enumerate(entity_ids)}
    values = np.zeros((len(entity_ids), 577), dtype=np.float32)
    source_present = np.zeros(len(entity_ids), dtype=np.bool_)
    esm_present = np.zeros(len(entity_ids), dtype=np.bool_)
    go_direct = np.zeros(len(entity_ids), dtype=np.bool_)
    translated = np.zeros(len(entity_ids), dtype=np.bool_)
    for row, gene in enumerate(entity_ids):
        old = source_index.get(gene)
        if old is None:
            values[row, 321:] = missing_go[gene]
            go_direct[row] = gene in missing_go_direct
            continue
        values[row] = source_values[old]
        source_present[row] = True
        esm_present[row] = bool(source["esm_present"][old])
        go_direct[row] = bool(source["go_direct_annotation_present"][old])
        translated[row] = bool(source["is_ensembl116_translated_gene"][old])

    fitting = np.asarray(sorted(gene for gene, role in role_by_action.items() if role == "train"))
    validation = np.asarray(sorted(gene for gene, role in role_by_action.items() if role == "validation"))
    test = np.asarray(sorted(gene for gene, role in role_by_action.items() if role == "test-excluded"))
    query_flag = np.isin(entity_ids, query)
    action_flag = np.isin(entity_ids, actions)
    arrays = {
        "feature_values": values,
        "entity_taxon": np.full(len(entity_ids), TAXON, dtype=np.int64),
        "entity_id": entity_ids,
        "source_static_row_present": source_present,
        "esm_present": esm_present,
        "protein_presence": values[:, 320].astype(np.bool_),
        "go_direct_annotation_present": go_direct,
        "go_exact_uniprot_mapping_present": np.isin(entity_ids, np.asarray(sorted(exact_mapping_ids))),
        "is_ensembl116_translated_gene": translated,
        "is_query": query_flag,
        "is_action": action_flag,
        "is_fitting_action": np.isin(entity_ids, fitting),
        "is_validation_action": np.isin(entity_ids, validation),
        "is_test_excluded_action": np.isin(entity_ids, test),
    }
    roster = {
        "query_ids": query,
        "query_entity_index": np.asarray([entity_index[gene] for gene in query], dtype=np.int64),
        "action_ids": actions,
        "action_entity_index": np.asarray([entity_index[gene] for gene in actions], dtype=np.int64),
        "action_role": np.asarray([role_by_action[gene] for gene in actions]),
        "fitting_action_ids": fitting,
        "fitting_action_entity_index": np.asarray([entity_index[gene] for gene in fitting], dtype=np.int64),
    }
    all_zero = np.all(values == 0, axis=1)
    row_bytes = [row.tobytes() for row in np.ascontiguousarray(values)]
    groups: dict[bytes, int] = {}
    for item in row_bytes:
        groups[item] = groups.get(item, 0) + 1
    role_masks = {
        "query": query_flag,
        "action": action_flag,
        "fittingAction": arrays["is_fitting_action"],
        "validationAction": arrays["is_validation_action"],
        "testExcludedAction": arrays["is_test_excluded_action"],
    }
    audit = {
        "entities": len(entity_ids),
        "queries": len(query),
        "actions": len(actions),
        "roleCounts": role_count,
        "sourceRowsCopied": int(source_present.sum()),
        "sourceRowsMissing": int((~source_present).sum()),
        "sourceOverlapBitExact": all(
            np.array_equal(values[row], source_values[source_index[gene]])
            for row, gene in enumerate(entity_ids) if gene in source_index
        ),
        "esmPresent": int(esm_present.sum()),
        "translated": int(translated.sum()),
        "goDirectAnnotationPresent": int(go_direct.sum()),
        "goExactUniprotMappingPresent": int(arrays["go_exact_uniprot_mapping_present"].sum()),
        "distinctFeatureRows": len(groups),
        "duplicateEquivalenceGroups": sum(count > 1 for count in groups.values()),
        "largestEquivalenceGroup": max(groups.values()),
        "allZeroRows": int(all_zero.sum()),
        "coverageByRole": {
            label: {
                "rows": int(mask.sum()),
                "sourceStaticRowPresent": int((mask & source_present).sum()),
                "esmPresent": int((mask & esm_present).sum()),
                "goDirectAnnotationPresent": int((mask & go_direct).sum()),
                "allZero": int((mask & all_zero).sum()),
            }
            for label, mask in role_masks.items()
        },
    }
    return arrays, roster, audit


def build(args: argparse.Namespace) -> dict[str, object]:
    pins = {
        "routing": (args.routing, ROUTING_SHA256),
        "static": (args.static, STATIC_SHA256),
        "staticManifest": (args.static_manifest, STATIC_MANIFEST_SHA256),
        "sharedGoBasis": (args.shared_go_basis, BASIS_SHA256),
        "ensemblMapping": (args.mapping, MAPPING_SHA256),
        "humanGoGaf": (args.gaf, GAF_SHA256),
        "k562Static": (args.k562_static, K562_STATIC_SHA256),
        "k562Roster": (args.k562_roster, K562_ROSTER_SHA256),
        "k562Builder": (args.k562_builder, K562_BUILDER_SHA256),
    }
    for path, expected in pins.values():
        require_hash(path, expected)
    routing, source = load_npz(args.routing), load_npz(args.static)
    if str(routing["schema"].item()) != "slp.replogle-rpe1-essential-cell-routing/v1":
        raise RpeStaticError("RPE1 routing schema mismatch")
    if str(routing["source_sha256"].item()) != SOURCE_SHA256:
        raise RpeStaticError("RPE1 source identity mismatch")
    query_bytes = "".join(gene + "\n" for gene in routing["query_ids"].astype(str)).encode("ascii")
    if hashlib.sha256(query_bytes).hexdigest() != QUERY_ROSTER_SHA256:
        raise RpeStaticError("ordered RPE1 query roster drift")

    helper = load_k_builder(args.k562_builder)
    requested = set(routing["query_ids"].astype(str).tolist())
    requested.update(
        routing["action_ids"][~routing["unresolved_action"] & ~routing["is_control"]].astype(str).tolist()
    )
    source_ids = set(source["entity_id"].astype(str).tolist())
    missing = frozenset(requested - source_ids)
    xrefs, exact_mapping_ids = helper.exact_mappings(args.mapping.read_bytes(), frozenset(requested))
    missing_xrefs = {
        xref: frozenset(genes & missing) for xref, genes in xrefs.items() if genes & missing
    }
    basis = load_npz(args.shared_go_basis)
    missing_go, missing_direct, projection = helper.project_missing_go(
        sorted(missing), missing_xrefs, args.gaf.read_bytes(), basis
    )
    arrays, roster, audit = assemble(
        routing, source, exact_mapping_ids, missing_go, missing_direct
    )
    if audit["sourceRowsMissing"] != len(missing) or not audit["sourceOverlapBitExact"]:
        raise RpeStaticError("source overlap audit failed")

    rpe_mean, rpe_sd, rpe_scale = normalizer(
        arrays["feature_values"], roster["fitting_action_entity_index"]
    )
    k_static, k_roster = load_npz(args.k562_static), load_npz(args.k562_roster)
    k_mean, k_sd, k_scale = normalizer(
        k_static["feature_values"], k_roster["fitting_action_entity_index"]
    )
    normalizers = {
        "schema": np.asarray("slp.rpe1-k562-count-static577-normalizers/v1"),
        "feature_dimension": np.asarray(577, dtype=np.int64),
        "normalization_floor": np.asarray(1e-5, dtype=np.float64),
        "rpe1_fitting_action_ids": roster["fitting_action_ids"],
        "rpe1_feature_mean": rpe_mean,
        "rpe1_feature_sd": rpe_sd,
        "rpe1_feature_scale": rpe_scale,
        "k562_fitting_action_ids": k_roster["fitting_action_ids"],
        "k562_feature_mean": k_mean,
        "k562_feature_sd": k_sd,
        "k562_feature_scale": k_scale,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    feature_path, roster_path = args.output_dir / FEATURE_NAME, args.output_dir / ROSTER_NAME
    normalizer_path = args.output_dir / NORMALIZER_NAME
    feature_path.write_bytes(helper.deterministic_npz(arrays))
    roster_path.write_bytes(helper.deterministic_npz(roster))
    normalizer_path.write_bytes(helper.deterministic_npz(normalizers))
    missing_path = args.output_dir / "source-static-missing-ids.txt"
    missing_bytes = "".join(gene + "\n" for gene in arrays["entity_id"][~arrays["source_static_row_present"]]).encode("ascii")
    missing_path.write_bytes(missing_bytes)

    manifest = {
        "schema": "slp.rpe1-essential-count-static577/v1",
        "artifacts": {
            "features": {"path": FEATURE_NAME, "sha256": sha256_file(feature_path), "shape": [audit["entities"], 577]},
            "rosterIndex": {"path": ROSTER_NAME, "sha256": sha256_file(roster_path)},
            "normalizers": {"path": NORMALIZER_NAME, "sha256": sha256_file(normalizer_path)},
            "sourceStaticMissingRoster": {"path": missing_path.name, "sha256": sha256_file(missing_path), "rows": len(missing)},
        },
        "identity": {
            "key": ["ncbiTaxon", "entityId"], "taxon": TAXON,
            "namespace": "stable unversioned Ensembl gene",
            "entityOrdering": "ascending codepoint stable ENSG",
            "queryOrdering": "exact source var/gene_id order from routing sidecar",
            "actionOrdering": "ascending codepoint stable ENSG",
        },
        "featureDefinition": {
            "dimension": 577,
            "columns": {"0:320": "frozen ESM-2 t6 8M full-protein vector", "320": "exact Ensembl-116 translation presence", "321:577": "frozen shared human/yeast direct MF/CC GO coordinates"},
            "rawArray": "feature_values",
            "rawCoordinates": "bit-exact copies from the frozen full-human pack where present; source-missing rows retain explicit zeros except eligible GO projection through the same frozen basis",
            "normalizers": "Separate exact float64 mean/SD/scale for 1,666 RPE1 fitting actions and 1,443 K562 fitting actions; normalize raw float32 through float64 arithmetic, scale=1 when SD<=1e-5, then cast to float32; no clipping",
            "learnedGeneIdentity": False,
            "sharedGoBasisSha256": BASIS_SHA256,
        },
        "coverage": audit,
        "missingProjection": {
            "rowsOutsideFrozenHumanStaticPack": len(missing),
            "exactEnsembl108UniProtMappingRows": len(exact_mapping_ids & missing),
            "eligibleFrozenBasisProjectionRows": projection["rowsWithEligibleTerms"],
            "retainedAssociations": projection["retainedAssociations"],
        },
        "inputs": {label: {"path": path.as_posix(), "sha256": expected} for label, (path, expected) in pins.items()},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "source": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": sha256_file(Path(__file__).resolve())}},
        "accessBoundary": {"routingMetadataRead": True, "staticSourcesRead": True, "countMatrixValuesRead": False, "developmentOutcomeRead": False, "testOutcomeRead": False},
        "limitations": ["Source-missing genes remain explicit and may have no protein or GO representation.", "Exact duplicate static rows cannot be distinguished by a feature-only model.", "RPE1 and K562 normalizers are alternatives with different fitting-roster provenance; no context-transfer experiment is performed here."],
    }
    (args.output_dir / "manifest.json").write_bytes(helper.canonical_json(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routing", type=Path, default=ROOT / "data/derived/slp11-human-rpe1-essential-singlecell-metadata-v1/cell-routing-metadata.npz")
    p.add_argument("--static", type=Path, default=ROOT / "data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2/human-static-esm8m-shared-go-mf-cc-features.npz")
    p.add_argument("--static-manifest", type=Path, default=ROOT / "data/derived/slp11-human-shared-static/ensembl116-source3-esm8m-shared-go-complete-v2/manifest.json")
    p.add_argument("--shared-go-basis", type=Path, default=ROOT / "data/derived/slp11-shared-human-yeast-go/goa-2022-09-19-mf-cc-svd256-v1/human-yeast-shared-go-mf-cc-svd256-basis.npz")
    p.add_argument("--mapping", type=Path, default=ROOT / "data/derived/slp11-human-go/source/Homo_sapiens.GRCh38.108.uniprot.tsv.gz")
    p.add_argument("--gaf", type=Path, default=ROOT / "data/derived/slp11-human-go/source/goa_human_2022-09-19.gaf.gz")
    p.add_argument("--k562-static", type=Path, default=ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/k562-essential-count-static577.npz")
    p.add_argument("--k562-roster", type=Path, default=ROOT / "data/derived/slp11-human-k562-essential-count-static/ensembl116-esm8m-shared-go-v1/roster-index.npz")
    p.add_argument("--k562-builder", type=Path, default=ROOT / "scripts/build_slp11_k562_count_static_features.py")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/derived/slp11-human-rpe1-essential-count-static/ensembl116-esm8m-shared-go-v1")
    return p


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
