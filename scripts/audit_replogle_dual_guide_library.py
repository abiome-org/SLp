"""Build a metadata-only guide-pair sidecar for Replogle et al. 2022.

This adapter joins the exact ``obs.sgID_AB`` strings preserved in the K562 and
RPE1 routing snapshots to Table S1 from the source paper.  It never opens an
expression matrix.  Commas in the paper workbook's multi-transcript guide IDs
are normalized to hyphens solely for matching the serialization used by the
published H5AD metadata; both original identifiers remain in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data/sources/replogle-2022-dual-guide-library-v1/1-s2.0-S0092867422005979-mmc1.xlsx"
K562 = ROOT / "data/derived/slp11-human-k562-essential-singlecell-metadata-v1/cell-routing-metadata.npz"
RPE1 = ROOT / "data/derived/slp11-human-rpe1-essential-singlecell-metadata-v1/cell-routing-metadata.npz"
ELIFE_COMMON_ESSENTIAL = ROOT / "data/sources/replogle-2022-dual-guide-library-v1/elife-81856-supp12-v2.csv"
OUT_DIR = ROOT / "data/derived/slp11-human-replogle-guide-library-v2"
RESULT_DIR = ROOT / "results/slp11-transition/replogle-dual-guide-library-audit-v2"

EXPECTED = {
    "table": "1e8d67490d562a48c03a641fc6170119548b19bc6dfe64a7cca1f6c3b6d506ae",
    "k562": "47c89c5082c0a9d4008c6b567407c530933a36fb7603621c37cbe913143f15ad",
    "rpe1": "10f3d313a5671122bde10a9bd586e3a2808d6f9b554f737ddcbbc28becc5e2f2",
    "elife_common_essential": "514e0e3351a85fc7cd54a6043e8c2bbd3c91d2761e0cbf87c7203d1150178191",
}

GUIDE_RE = re.compile(
    r"^(?P<gene>.+)_(?P<strand>[+-])_(?P<coordinate>[0-9]+)"
    r"(?P<design_token>\.[0-9]+)-(?P<transcript_target>.+)$"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_guide_id(value: str) -> str:
    """Match the H5AD's comma-to-hyphen multi-transcript serialization."""
    return str(value).replace(",", "-")


def canonical_pair(sgid_a: str, sgid_b: str) -> str:
    return f"{canonical_guide_id(sgid_a)}|{canonical_guide_id(sgid_b)}"


def parse_guide_id(value: str) -> dict[str, object]:
    """Parse literal author sgID fields without inferring a genome build."""
    value = str(value)
    if re.fullmatch(r"non-targeting_[0-9]+", value):
        return {
            "is_control": True,
            "gene_symbol": "",
            "strand": "",
            "coordinate": -1,
            "coordinate_observed": False,
            "design_token": "",
            "transcript_target": "",
        }
    match = GUIDE_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"unrecognized guide identifier: {value}")
    return {
        "is_control": False,
        "gene_symbol": match.group("gene"),
        "strand": match.group("strand"),
        "coordinate": int(match.group("coordinate")),
        "coordinate_observed": True,
        "design_token": match.group("design_token"),
        "transcript_target": canonical_guide_id(match.group("transcript_target")),
    }


def _deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def _load_table(path: Path) -> dict[str, pd.DataFrame]:
    sheets = {
        "k562": "TabB_K562_day6_library",
        "rpe1": "TabC_RPE1_day7_library",
    }
    required = {
        "unique sgRNA pair ID", "gene", "transcript", "ensembl gene id",
        "sgID_A", "targeting sequence A", "sgID_B", "targeting sequence B",
        "duplicated guide pair?", "either guide duplicated?",
    }
    output = {}
    for context, sheet in sheets.items():
        frame = pd.read_excel(path, sheet_name=sheet, dtype=str)
        if not required.issubset(frame.columns):
            raise ValueError(f"Table S1 schema drift in {sheet}")
        frame = frame.copy()
        frame["original_pair"] = frame["sgID_A"] + "|" + frame["sgID_B"]
        frame["canonical_pair"] = [
            canonical_pair(a, b) for a, b in zip(frame["sgID_A"], frame["sgID_B"])
        ]
        if frame["canonical_pair"].duplicated().any():
            raise ValueError(f"ambiguous canonical guide-pair identifiers in {sheet}")
        for column in ("targeting sequence A", "targeting sequence B"):
            seq = frame[column].astype(str).str.upper()
            if not ((seq.str.len() == 20) & seq.str.fullmatch("[ACGT]{20}")).all():
                raise ValueError(f"invalid 20-nt targeting sequence in {sheet}")
            frame[column] = seq
        output[context] = frame.set_index("canonical_pair", drop=False)
    return output


def _routing_rows(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as source:
        keys = ("guide_pair_ids", "action_ids", "gene_transcript", "is_control")
        if not all(key in source.files for key in keys):
            raise ValueError("routing sidecar schema drift")
        frame = pd.DataFrame({key: source[key].astype(str) for key in keys[:-1]})
        frame["is_control"] = source["is_control"].astype(bool)
    counts = (
        frame.groupby(list(keys), dropna=False)
        .size()
        .rename("cell_count")
        .reset_index()
    )
    per_pair = counts.groupby("guide_pair_ids", dropna=False)
    if int(per_pair.size().max()) != 1:
        raise ValueError("one observed guide pair maps to multiple routing identities")
    return counts.sort_values("guide_pair_ids").reset_index(drop=True)


def build_arrays(table: dict[str, pd.DataFrame], routing: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    pair_ids = np.asarray(sorted(set(routing["k562"].guide_pair_ids) | set(routing["rpe1"].guide_pair_ids)))
    n = len(pair_ids)
    table_rows: dict[str, pd.Series] = {}
    exact = {"k562": 0, "rpe1": 0}
    normalized = {"k562": 0, "rpe1": 0}
    for context in ("k562", "rpe1"):
        original_pairs = set(table[context]["original_pair"].astype(str))
        for pair in routing[context].guide_pair_ids:
            a, b = pair.split("|", 1)
            if f"{a}|{b}" in original_pairs:
                exact[context] += 1
            key = canonical_pair(a, b)
            if key not in table[context].index:
                raise ValueError(f"unmatched {context} guide pair: {pair}")
            normalized[context] += 1
            row = table[context].loc[key]
            prior = table_rows.get(pair)
            if prior is not None:
                fields = ("targeting sequence A", "targeting sequence B", "ensembl gene id")
                if any(str(prior[f]) != str(row[f]) for f in fields):
                    raise ValueError(f"cross-context library conflict for {pair}")
            table_rows[pair] = row

    route_lookup = {}
    for context, frame in routing.items():
        for row in frame.itertuples(index=False):
            route_lookup[(context, row.guide_pair_ids)] = row

    for context, frame in routing.items():
        for routed in frame.itertuples(index=False):
            row = table[context].loc[canonical_guide_id(routed.guide_pair_ids)]
            if canonical_guide_id(str(row["unique sgRNA pair ID"])) != canonical_guide_id(
                str(routed.gene_transcript)
            ):
                raise ValueError(
                    f"gene_transcript/library pair identity mismatch: {routed.guide_pair_ids}"
                )

    action_ids, action_observed, unresolved = [], [], []
    records = []
    for pair in pair_ids:
        row = table_rows[pair]
        ids = {
            str(route_lookup[(ctx, pair)].action_ids)
            for ctx in ("k562", "rpe1") if (ctx, pair) in route_lookup
            if str(route_lookup[(ctx, pair)].action_ids) not in {"", "nan"}
        }
        if len(ids) > 1:
            raise ValueError(f"guide pair maps to different stable genes: {pair}")
        action = next(iter(ids), "")
        is_control = all(
            bool(route_lookup[(ctx, pair)].is_control)
            for ctx in ("k562", "rpe1") if (ctx, pair) in route_lookup
        )
        library_action = str(row["ensembl gene id"])
        library_action = "" if library_action == "nan" else library_action.split(".")[0]
        is_unresolved = (not is_control) and not action
        if action and action != library_action:
            raise ValueError(f"stable gene mismatch for {pair}: {action} != {library_action}")
        sg_a, sg_b = str(row["sgID_A"]), str(row["sgID_B"])
        parsed_a, parsed_b = parse_guide_id(sg_a), parse_guide_id(sg_b)
        if bool(parsed_a["is_control"]) != is_control or bool(parsed_b["is_control"]) != is_control:
            raise ValueError(f"control classification mismatch for {pair}")
        action_ids.append(action)
        action_observed.append(bool(action))
        unresolved.append(is_unresolved)
        records.append((row, sg_a, sg_b, parsed_a, parsed_b, is_control))

    def values(name: str) -> np.ndarray:
        return np.asarray([str(record[0][name]) for record in records])

    duplicated_k = np.asarray([
        ("k562", pair) in route_lookup
        and str(table["k562"].loc[canonical_guide_id(pair)]["duplicated guide pair?"]) == "True"
        for pair in pair_ids
    ], bool)
    duplicated_r = np.asarray([
        ("rpe1", pair) in route_lookup
        and str(table["rpe1"].loc[canonical_guide_id(pair)]["duplicated guide pair?"]) == "True"
        for pair in pair_ids
    ], bool)
    either_k = np.asarray([
        ("k562", pair) in route_lookup
        and str(table["k562"].loc[canonical_guide_id(pair)]["either guide duplicated?"]) == "True"
        for pair in pair_ids
    ], bool)
    either_r = np.asarray([
        ("rpe1", pair) in route_lookup
        and str(table["rpe1"].loc[canonical_guide_id(pair)]["either guide duplicated?"]) == "True"
        for pair in pair_ids
    ], bool)
    arrays = {
        "schema": np.asarray("slp.replogle-dual-guide-metadata/v2"),
        "guide_pair_ids": pair_ids,
        "library_pair_ids": values("unique sgRNA pair ID"),
        "action_ids": np.asarray(action_ids),
        "action_observed": np.asarray(action_observed, bool),
        "unresolved_action": np.asarray(unresolved, bool),
        "entity_taxon": np.full(n, 9606, np.int64),
        "gene_symbols": values("gene"),
        "transcript_labels": values("transcript"),
        "sgid_a": np.asarray([r[1] for r in records]),
        "sgid_b": np.asarray([r[2] for r in records]),
        "targeting_sequence_a": values("targeting sequence A"),
        "targeting_sequence_b": values("targeting sequence B"),
        "strand_a": np.asarray([r[3]["strand"] for r in records]),
        "strand_b": np.asarray([r[4]["strand"] for r in records]),
        "coordinate_a": np.asarray([r[3]["coordinate"] for r in records], np.int64),
        "coordinate_b": np.asarray([r[4]["coordinate"] for r in records], np.int64),
        "coordinate_observed_a": np.asarray([r[3]["coordinate_observed"] for r in records], bool),
        "coordinate_observed_b": np.asarray([r[4]["coordinate_observed"] for r in records], bool),
        "design_token_a": np.asarray([r[3]["design_token"] for r in records]),
        "design_token_b": np.asarray([r[4]["design_token"] for r in records]),
        "transcript_target_a": np.asarray([r[3]["transcript_target"] for r in records]),
        "transcript_target_b": np.asarray([r[4]["transcript_target"] for r in records]),
        "is_control": np.asarray([r[5] for r in records], bool),
        "author_duplicated_pair": duplicated_k | duplicated_r,
        "author_either_guide_duplicated": either_k | either_r,
        "author_duplicated_pair_k562": duplicated_k,
        "author_duplicated_pair_rpe1": duplicated_r,
        "author_either_guide_duplicated_k562": either_k,
        "author_either_guide_duplicated_rpe1": either_r,
        "present_k562": np.asarray([("k562", pair) in route_lookup for pair in pair_ids], bool),
        "present_rpe1": np.asarray([("rpe1", pair) in route_lookup for pair in pair_ids], bool),
        "cell_count_k562": np.asarray([getattr(route_lookup.get(("k562", pair)), "cell_count", 0) for pair in pair_ids], np.int64),
        "cell_count_rpe1": np.asarray([getattr(route_lookup.get(("rpe1", pair)), "cell_count", 0) for pair in pair_ids], np.int64),
        "context_ids": np.asarray(["replogle-2022-k562-essential-day-6", "replogle-2022-rpe1-essential-day-7"]),
        "assay_day_post_transduction": np.asarray([6, 7], np.int64),
        "coordinate_reference": np.asarray("literal author sgID coordinate; genome assembly not specified in Table S1"),
        "sequence_convention": np.asarray("Cell 2022 Table S1 targeting sequence, 20 nt, position A/B"),
    }
    report = {
        "schema": "slp.replogle-dual-guide-library-audit/v2",
        "expressionMatrixAccessed": False,
        "molecularOutcomesAccessed": False,
        "pairUnion": n,
        "sharedPairs": int(np.sum(arrays["present_k562"] & arrays["present_rpe1"])),
        "sequencesObserved": int(n),
        "coordinateObservedPairs": int(np.sum(arrays["coordinate_observed_a"] & arrays["coordinate_observed_b"])),
        "controlsWithoutCoordinates": int(np.sum(arrays["is_control"])),
        "unresolvedStableActions": int(np.sum(arrays["unresolved_action"])),
        "authorDuplicatedPairFlag": int(np.sum(arrays["author_duplicated_pair"])),
        "authorEitherGuideDuplicatedFlag": int(np.sum(arrays["author_either_guide_duplicated"])),
        "contexts": {
            "k562": {"observedPairs": len(routing["k562"]), "exactStringJoins": exact["k562"], "canonicalJoins": normalized["k562"], "punctuationNormalizedJoins": normalized["k562"] - exact["k562"], "assayDay": 6},
            "rpe1": {"observedPairs": len(routing["rpe1"]), "exactStringJoins": exact["rpe1"], "canonicalJoins": normalized["rpe1"], "punctuationNormalizedJoins": normalized["rpe1"] - exact["rpe1"], "assayDay": 7},
        },
        "joinNormalization": "comma-separated multi-transcript suffixes in Table S1 are serialized with hyphens in H5AD sgID_AB; 9 pairs per context require this punctuation-only normalization",
        "coordinateLimitation": "The sgID carries strand and an integer coordinate, but Table S1 does not state the coordinate assembly. GRCh38 2020-A in the paper describes RNA alignment and is not assumed to define sgID coordinates.",
        "modelingConclusion": "Gene-only action features omit available 20-nt sequences, guide-pair composition, strand, author coordinate, transcript target, and duplication flags. Pair IDs remain provenance and must not become learned identity embeddings.",
    }
    return arrays, report


def audit_later_library_convention(cell_table: pd.DataFrame, path: Path) -> dict[str, int]:
    """Measure, without reconciling, the later eLife sequence convention."""
    later = pd.read_csv(path, dtype=str)
    required = {"sgID_A", "sgID_B", "protospacer_A", "protospacer_B"}
    if not required.issubset(later.columns):
        raise ValueError("eLife Supplementary file 12 schema drift")
    later = later.copy()
    later["canonical_pair"] = [
        canonical_pair(a, b) for a, b in zip(later.sgID_A, later.sgID_B)
    ]
    source = cell_table.reset_index(drop=True)
    joined = source.merge(later, on="canonical_pair", suffixes=("_cell", "_elife"))
    complement = str.maketrans("ACGT", "TGCA")
    reverse_complement = lambda value: str(value).translate(complement)[::-1]
    a_cell = joined["targeting sequence A"].astype(str).str.upper()
    b_cell = joined["targeting sequence B"].astype(str).str.upper()
    a_later = joined["protospacer_A"].astype(str).str.upper()
    b_later = joined["protospacer_B"].astype(str).str.upper()
    return {
        "overlappingPairs": len(joined),
        "positionAExact": int((a_cell == a_later).sum()),
        "positionAReverseComplement": int((a_cell.map(reverse_complement) == a_later).sum()),
        "positionBExact": int((b_cell == b_later).sum()),
        "positionBReverseComplement": int((b_cell.map(reverse_complement) == b_later).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()
    for name, path in (("table", TABLE), ("k562", K562), ("rpe1", RPE1), ("elife_common_essential", ELIFE_COMMON_ESSENTIAL)):
        if sha256(path) != EXPECTED[name]:
            raise ValueError(f"checksum mismatch: {name}")
    if args.output_dir.exists() or (args.result_dir / "report.json").exists():
        raise FileExistsError("immutable output already exists")
    table = _load_table(TABLE)
    routing = {"k562": _routing_rows(K562), "rpe1": _routing_rows(RPE1)}
    arrays, report = build_arrays(table, routing)
    report["laterLibraryConventionAudit"] = audit_later_library_convention(
        table["rpe1"], ELIFE_COMMON_ESSENTIAL
    )
    report["laterLibraryConventionNote"] = (
        "The later eLife common-essential library uses the same 20-nt string at "
        "position A and the reverse-complement convention at position B for every "
        "overlapping pair. The sidecar therefore preserves Cell 2022 Table S1 "
        "targeting sequences and does not merge the conventions."
    )
    sidecar = args.output_dir / "guide-pair-metadata.npz"
    _deterministic_npz(sidecar, arrays)
    report["inputs"] = {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": EXPECTED[name]} for name, path in (("table", TABLE), ("k562", K562), ("rpe1", RPE1), ("elife_common_essential", ELIFE_COMMON_ESSENTIAL))}
    report["sidecar"] = {"path": str(sidecar.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(sidecar)}
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"schema": "slp.replogle-dual-guide-metadata-manifest/v2", "source": report["inputs"], "artifact": report["sidecar"], "counts": {k: report[k] for k in ("pairUnion", "sharedPairs", "sequencesObserved", "coordinateObservedPairs", "controlsWithoutCoordinates", "unresolvedStableActions")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
