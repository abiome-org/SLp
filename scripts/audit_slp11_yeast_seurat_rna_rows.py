"""Audit Seurat RNA row identities against the pinned current SGD mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

INVENTORY_SHA256 = "c51fdbd303a9cce3253efa4a6ce78631bdb8f5097bac4c68def4d3c72a38808d"
ROW_COUNT = 6_951
ROW_ROSTER_SHA256 = "bf0103a1419d17075c8f12d5b7cf139b7ec747c0e3c553e960642b3dd37bc7e5"
CURRENT_ORFS_SHA256 = "df7b717cad88dc3672f72f8148f6a9132d12abe6ba020b220b091a8da8f7004d"
FEATURES_SHA256 = "636b4fc0407dd9f4fe74dceb5f5cd056194623d36a25c620a3c1ec2394af3dcc"
RETIRED_SHA256 = "c9143bbd08fcb75a3789cc4c041790aa5a291ca90f1b1b2e6f6a85a171a70df2"
BIOSTUDIES_SHA256 = "5c034919f23cc720ce5692f39f9d389ee72e7ce759366cfc8d61512eda4e325f"
MAPPING_ID = "slp-sgd-map:2026-08-28-object-set-v1"
MAPPING_SHA256 = "6fd789df6099b78a8842baa8f1d20ab0a3fe77f27ce512ee783444eb2627ef2a"
SGD_RE = re.compile(r"^S[0-9]{9}$")
ARTIFICIAL_PATTERNS = {
    "barcode-prefix": re.compile(r"(?i)^bc[-_]"),
    "barcode-word": re.compile(r"(?i)barcode"),
    "reporter-word": re.compile(r"(?i)reporter"),
    "fluorescent": re.compile(r"(?i)(?:gfp|rfp|mcherry|fluorescen)"),
}


class RowAuditError(ValueError):
    """Raised when an identity-only source contract is violated."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(path: Path, digest: str, label: str) -> None:
    if not path.is_file() or sha256(path) != digest:
        raise RowAuditError(f"{label} does not match its pinned SHA-256")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def unique_target(index: Mapping[str, set[str]], name: str) -> tuple[str | None, bool]:
    targets = index.get(name, set())
    return (next(iter(targets)), False) if len(targets) == 1 else (None, len(targets) > 1)


def load_rows(inventory_path: Path) -> tuple[str, ...]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidates = [
        item
        for item in inventory.get("sparseCandidates", [])
        if item.get("admissibleRawRnaCountsPath") is True
    ]
    if len(candidates) != 2:
        raise RowAuditError("expected exactly Control and NaCl raw RNA/counts matrices")
    rosters = []
    for candidate in candidates:
        audit = candidate.get("rawRnaRowIdentifierAudit")
        if (
            not isinstance(audit, dict)
            or audit.get("count") != ROW_COUNT
            or audit.get("orderedSha256") != ROW_ROSTER_SHA256
        ):
            raise RowAuditError("RNA row roster contract drift")
        roster = tuple(audit.get("allIdentifiers", []))
        payload = ("\n".join(roster) + "\n").encode("utf-8")
        if len(roster) != ROW_COUNT or hashlib.sha256(payload).hexdigest() != ROW_ROSTER_SHA256:
            raise RowAuditError("RNA identifier payload hash mismatch")
        rosters.append(roster)
    if rosters[0] != rosters[1] or len(set(rosters[0])) != ROW_COUNT:
        raise RowAuditError("Control/NaCl RNA rows differ or contain duplicate source names")
    return rosters[0]


def load_indices(current_orfs: Path, features: Path) -> dict[str, object]:
    indices: dict[str, defaultdict[str, set[str]]] = {
        name: defaultdict(set)
        for name in (
            "orf_systematic",
            "orf_standard",
            "orf_alias",
            "feature_systematic",
            "feature_standard",
            "feature_alias",
        )
    }
    feature_type_by_curie: defaultdict[str, set[str]] = defaultdict(set)
    for raw in current_orfs.read_text(encoding="utf-8").splitlines():
        item = json.loads(raw)
        if item.get("schema") != "slp.sgd-current-orf/v1" or item.get("ncbiTaxon") != 4932:
            raise RowAuditError("current ORF schema drift")
        curie = item["canonicalSgdCurie"]
        indices["orf_systematic"][item["systematicName"]].add(curie)
        standard = item.get("displayMetadata", {}).get("standardGeneName")
        if standard:
            indices["orf_standard"][standard].add(curie)
        for alias in item.get("displayMetadata", {}).get("aliases", []):
            indices["orf_alias"][alias].add(curie)
        feature_type_by_curie[curie].add("ORF")

    physical = 0
    blanks = 0
    for line_number, line in enumerate(features.read_text(encoding="utf-8").splitlines(), 1):
        physical += 1
        if not line:
            blanks += 1
            continue
        fields = line.split("\t")
        if len(fields) != 16 or SGD_RE.fullmatch(fields[0]) is None:
            raise RowAuditError(f"invalid SGD feature row {line_number}")
        curie = "SGD:" + fields[0]
        feature_type_by_curie[curie].add(fields[1])
        if fields[3]:
            indices["feature_systematic"][fields[3]].add(curie)
        if fields[4]:
            indices["feature_standard"][fields[4]].add(curie)
        for alias in fields[5].split("|"):
            if alias:
                indices["feature_alias"][alias].add(curie)
    if physical != 16_461 or blanks != 2:
        raise RowAuditError("pinned SGD feature-table row counts drift")
    return {**indices, "feature_types": feature_type_by_curie}


def classify_row(name: str, indices: Mapping[str, object]) -> dict[str, object]:
    exact_levels = (
        ("orf_systematic", "current-orf-systematic", "strict-current-orf"),
        ("orf_standard", "current-orf-standard", "dataset-exact-current-orf"),
        ("feature_systematic", "current-feature-systematic", "strict-current-feature"),
        ("feature_standard", "current-feature-standard", "dataset-exact-current-feature"),
    )
    for index_name, mapping_class, evidence_level in exact_levels:
        target, ambiguous = unique_target(indices[index_name], name)  # type: ignore[arg-type]
        if ambiguous:
            return {
                "mappingClass": "ambiguous-" + mapping_class,
                "mappingEvidence": evidence_level,
                "canonicalSgdCurie": None,
                "biologicalRnaEvidence": True,
            }
        if target is not None:
            return {
                "mappingClass": mapping_class,
                "mappingEvidence": evidence_level,
                "canonicalSgdCurie": target,
                "biologicalRnaEvidence": True,
            }

    alias_levels = (
        ("orf_alias", "current-orf-alias-only"),
        ("feature_alias", "current-feature-alias-only"),
    )
    for index_name, mapping_class in alias_levels:
        target, ambiguous = unique_target(indices[index_name], name)  # type: ignore[arg-type]
        if ambiguous:
            return {
                "mappingClass": "ambiguous-" + mapping_class,
                "mappingEvidence": "display-alias-does-not-resolve-global-identity",
                "canonicalSgdCurie": None,
                "biologicalRnaEvidence": True,
            }
        if target is not None:
            return {
                "mappingClass": mapping_class,
                "mappingEvidence": "exact-pinned-SGD-alias; candidate-only-for-stable-identity",
                "canonicalSgdCurie": target,
                "biologicalRnaEvidence": True,
            }

    if "-" in name:
        underscore_name = name.replace("-", "_")
        transformed = set()
        for index_name in ("feature_systematic", "feature_standard"):
            transformed.update(indices[index_name].get(underscore_name, set()))  # type: ignore[union-attr]
        if len(transformed) == 1:
            return {
                "mappingClass": "seurat-dash-normalized-current-feature-candidate",
                "mappingEvidence": "unique exact SGD feature after dash-to-underscore reversal; candidate-only",
                "canonicalSgdCurie": next(iter(transformed)),
                "normalizedSourceNameCandidate": underscore_name,
                "biologicalRnaEvidence": True,
            }
        if len(transformed) > 1:
            return {
                "mappingClass": "ambiguous-seurat-dash-normalized-candidate",
                "mappingEvidence": "dash-to-underscore reversal has multiple current SGD targets",
                "canonicalSgdCurie": None,
                "normalizedSourceNameCandidate": underscore_name,
                "biologicalRnaEvidence": True,
            }

    artificial = [label for label, pattern in ARTIFICIAL_PATTERNS.items() if pattern.search(name)]
    return {
        "mappingClass": "unresolved-artificial-candidate" if artificial else "unresolved-native-candidate",
        "mappingEvidence": "no exact current SGD name or alias",
        "canonicalSgdCurie": None,
        "biologicalRnaEvidence": not artificial,
        "artificialLexicalFlags": artificial,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    expected = (
        (args.inventory, INVENTORY_SHA256, "full Seurat inventory"),
        (args.current_orfs, CURRENT_ORFS_SHA256, "current ORFs"),
        (args.features, FEATURES_SHA256, "SGD features"),
        (args.retired, RETIRED_SHA256, "retired quarantine"),
        (args.biostudies, BIOSTUDIES_SHA256, "BioStudies metadata"),
    )
    for path, digest, label in expected:
        require(path, digest, label)
    if args.output.exists():
        raise FileExistsError("immutable RNA identity audit output exists")
    rows = load_rows(args.inventory)
    indices = load_indices(args.current_orfs, args.features)
    retired_names = {
        item.get("systematicName")
        for item in map(
            json.loads, args.retired.read_text(encoding="utf-8").splitlines()
        )
        if item.get("systematicName")
    }

    records = []
    for index, name in enumerate(rows):
        classification = classify_row(name, indices)
        curie = classification.get("canonicalSgdCurie")
        types = sorted(indices["feature_types"].get(curie, set())) if curie else []  # type: ignore[union-attr]
        record = {
            "schema": "slp.yeast-seurat-rna-row-identity/v1",
            "rowIndex": index,
            "sourceIdentifier": name,
            **classification,
            "currentFeatureTypes": types,
            "retiredExactNameMatch": name in retired_names,
            "includeInLibrarySizeDenominator": bool(classification["biologicalRnaEvidence"]),
        }
        records.append(record)

    mapping_counts = Counter(record["mappingClass"] for record in records)
    feature_type_counts = Counter(
        feature_type
        for record in records
        for feature_type in record["currentFeatureTypes"]
    )
    strict = [
        record
        for record in records
        if record["mappingClass"]
        in {
            "current-orf-systematic",
            "current-orf-standard",
            "current-feature-systematic",
            "current-feature-standard",
        }
    ]
    alias_only = [record for record in records if "alias-only" in record["mappingClass"]]
    candidate_mapped = [record for record in records if record["canonicalSgdCurie"]]
    by_target: defaultdict[str, list[str]] = defaultdict(list)
    for record in candidate_mapped:
        by_target[record["canonicalSgdCurie"]].append(record["sourceIdentifier"])
    duplicates = {key: value for key, value in by_target.items() if len(value) > 1}
    denominator = [
        record["sourceIdentifier"]
        for record in records
        if record["includeInLibrarySizeDenominator"]
    ]
    if len(denominator) != ROW_COUNT:
        raise RowAuditError("one or more source rows lacks biological denominator evidence")

    args.output.mkdir(parents=True)
    mapping_path = args.output / "row-mapping.jsonl"
    denominator_path = args.output / "denominator-row-identifiers.txt"
    mapping_path.write_text(
        "".join(canonical_json(record) + "\n" for record in records), encoding="utf-8"
    )
    denominator_path.write_text("\n".join(denominator) + "\n", encoding="utf-8")
    report = {
        "schema": "slp.yeast-seurat-rna-row-identity-audit/v1",
        "status": "identity-only-audit-no-count-values-read",
        "sourceRows": {
            "count": len(rows),
            "orderedSha256": ROW_ROSTER_SHA256,
            "controlAndNaclRostersExact": True,
            "uniqueSourceIdentifiers": len(set(rows)),
        },
        "mapping": {
            "releaseId": MAPPING_ID,
            "releaseSha256": MAPPING_SHA256,
            "classCounts": dict(sorted(mapping_counts.items())),
            "strictExactCurrentRows": len(strict),
            "exactAliasOnlyRows": len(alias_only),
            "candidateMappedRowsIncludingAliasesAndDashNormalization": len(candidate_mapped),
            "candidateUniqueStableIds": len(by_target),
            "duplicateStableIdsAfterCandidateMapping": len(duplicates),
            "duplicateSourceRowsAfterCandidateMapping": sum(len(value) for value in duplicates.values()),
            "featureTypeCounts": dict(sorted(feature_type_counts.items())),
            "unresolvedAfterCandidateMapping": sum(
                record["canonicalSgdCurie"] is None for record in records
            ),
            "aliasPolicy": "Alias-only rows retain candidate targets but do not silently become globally admitted identities.",
            "dashNormalizationPolicy": "Four unique dash-to-underscore candidates retain candidate targets only; no general symbol normalization is admitted.",
        },
        "artificialConstructAudit": {
            "lexicallyFlaggedRows": sum(
                bool(record.get("artificialLexicalFlags")) for record in records
            ),
            "nativeUra3Rows": sum(record["sourceIdentifier"] == "URA3" for record in records),
            "sourceEvidence": {
                "statement": "BioStudies says artificial bc-<systematic-name> chromosomes were added for genotype barcodes and genotype barcodes were removed during standard Seurat preparation.",
                "sha256": BIOSTUDIES_SHA256,
            },
            "conclusion": "No bc-/barcode/reporter/fluorescent row occurs in the actual 6,951-row RNA roster. The single ordinary URA3 row is retained as native RNA.",
        },
        "recommendedNormalization": {
            "denominatorRows": len(denominator),
            "denominatorRosterSha256": sha256(denominator_path),
            "formula": "For each cell, library_size=sum(raw RNA counts over all 6,951 audited source rows); value[q]=ln(1+10000*count[q]/library_size).",
            "reason": "All source rows have pinned evidence as native biological RNA features; no artificial genotype-barcode row remains. Alias-only or stable-ID-unresolved status must not remove native counts from the library-size denominator.",
            "stableMappingTiming": "Compute the source-row library size before any stable-ID projection or aggregation.",
            "zeroLibraryPolicy": "Reject or explicitly mask a zero-library cell; never invent a denominator.",
        },
        "inputs": {
            "inventory": {"path": args.inventory.as_posix(), "sha256": INVENTORY_SHA256},
            "currentOrfs": {"path": args.current_orfs.as_posix(), "sha256": CURRENT_ORFS_SHA256},
            "sgdFeatures": {"path": args.features.as_posix(), "sha256": FEATURES_SHA256},
            "retiredQuarantine": {"path": args.retired.as_posix(), "sha256": RETIRED_SHA256},
            "biostudiesMetadata": {"path": args.biostudies.as_posix(), "sha256": BIOSTUDIES_SHA256},
        },
        "outputs": {
            "rowMapping": {
                "path": mapping_path.name,
                "rows": len(records),
                "sha256": sha256(mapping_path),
            },
            "denominatorRoster": {
                "path": denominator_path.name,
                "rows": len(denominator),
                "sha256": sha256(denominator_path),
            },
        },
        "accessBoundary": {
            "countValuesRead": False,
            "cellMetadataValuesRead": False,
            "quantitativeOutcomesRead": False,
            "rowIdentifiersOnly": True,
        },
        "limitations": [
            "Exact aliases are biological source evidence but are not globally resolving identity under the pinned SGD mapping policy.",
            "Dash-to-underscore candidates are consistent with Seurat name normalization but are not treated as exact native identifiers.",
            "This audit establishes row identity and denominator membership, not raw-count integer or cell-level library validation.",
        ],
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--inventory", required=True, type=Path)
    result.add_argument("--current-orfs", required=True, type=Path)
    result.add_argument("--features", required=True, type=Path)
    result.add_argument("--retired", required=True, type=Path)
    result.add_argument("--biostudies", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
