"""Create a metadata-only census for the public Frangieh Perturb-CITE-seq files.

The audit reads AnnData axes and annotations. It samples a bounded prefix of
the sparse value arrays solely to verify the documented UMI-count value space;
it never groups expression values by perturbation or condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

SPLIT_PREFIX = "slp11-development-v1|731"
ENSG_RE = re.compile(r"ENSG[0-9]{11}$")


def gene_split(ensembl_id: str) -> str:
    """Return the repository-wide deterministic intervention-gene split."""
    digest = hashlib.sha256(
        f"{SPLIT_PREFIX}|9606|{ensembl_id}".encode()
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _decode(values: object) -> np.ndarray:
    out = np.asarray(values)
    if out.dtype.kind == "S":
        return np.char.decode(out, "utf-8")
    if out.dtype.kind == "O":
        return np.asarray(
            [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in out],
            dtype=object,
        )
    return out


def read_column(group: h5py.Group, key: str) -> np.ndarray:
    node = group[key]
    if isinstance(node, h5py.Group) and {"codes", "categories"} <= set(node):
        categories = _decode(node["categories"])
        codes = np.asarray(node["codes"])
        out = np.empty(codes.shape, dtype=object)
        out[codes < 0] = ""
        out[codes >= 0] = categories[codes[codes >= 0]]
        return out
    return _decode(node)


def _counts(values: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _matrix_metadata(handle: h5py.File, sample_limit: int) -> dict[str, object]:
    x = handle["X"]
    data = x["data"]
    sample = np.asarray(data[: min(sample_limit, data.shape[0])])
    return {
        "encoding": str(x.attrs["encoding-type"]),
        "shape": [int(v) for v in x.attrs["shape"]],
        "stored_values": int(data.shape[0]),
        "dtype": str(data.dtype),
        "bounded_value_audit": {
            "sample_location": "prefix of sparse X/data; no outcome grouping",
            "sample_size": int(sample.size),
            "finite": bool(np.isfinite(sample).all()),
            "nonnegative": bool((sample >= 0).all()),
            "integer_valued": bool(np.equal(sample, np.round(sample)).all()),
            "minimum": float(np.min(sample)),
            "maximum": float(np.max(sample)),
        },
    }


def audit(rna_path: Path, protein_path: Path, sample_limit: int = 1_000_000) -> dict:
    with h5py.File(rna_path, "r") as rna, h5py.File(protein_path, "r") as protein:
        rna_cells = read_column(rna["obs"], "cell_name")
        protein_cells = read_column(protein["obs"], "cell_name")
        if not np.array_equal(rna_cells, protein_cells):
            raise ValueError("RNA and protein cell axes are not exactly aligned")

        ensembl = read_column(rna["var"], "ensembl_id").astype(str)
        symbols = read_column(rna["var"], "gene_symbol").astype(str)
        if len(set(ensembl)) != len(ensembl):
            raise ValueError("RNA source query identifiers are not unique")
        if len(set(symbols)) != len(symbols):
            raise ValueError("RNA source gene symbols are not unique")
        source_symbol_to_ensembl = dict(zip(symbols, ensembl, strict=True))

        perturbation = read_column(rna["obs"], "perturbation").astype(str)
        environments = read_column(rna["obs"], "perturbation_2").astype(str)
        guides = read_column(rna["obs"], "guide_id").astype(str)
        sgrna = read_column(rna["obs"], "sgRNA").astype(str)
        action_symbols = sorted(set(perturbation) - {"control"})
        resolved = {
            s: source_symbol_to_ensembl[s]
            for s in action_symbols
            if ENSG_RE.fullmatch(source_symbol_to_ensembl.get(s, ""))
        }
        unresolved = sorted(set(action_symbols) - set(resolved))
        split_counts = Counter(gene_split(g) for g in resolved.values())

        protein_roster = []
        for i, name in enumerate(read_column(protein["var"], "protein").astype(str)):
            protein_roster.append(
                {
                    "protein": name,
                    "target": str(read_column(protein["var"], "Target")[i]),
                    "barcode": str(read_column(protein["var"], "Barcode")[i]),
                    "clone": str(read_column(protein["var"], "Clone")[i]).strip(),
                    "isotype_control": str(read_column(protein["var"], "Isotype_control")[i]),
                }
            )

        common_obs = sorted(set(rna["obs"]) & set(protein["obs"]))
        invariant = [
            key
            for key in common_obs
            if np.array_equal(read_column(rna["obs"], key), read_column(protein["obs"], key))
        ]
        return {
            "schema": "slp.frangieh-perturb-cite-metadata-audit/v1",
            "organism": {"name": "Homo sapiens", "ncbi_taxon": 9606},
            "paired_cells": {
                "count": len(rna_cells),
                "unique_cell_ids": len(set(rna_cells)),
                "exact_same_order": True,
                "shared_identical_obs_fields": invariant,
                "patient_field_present": False,
                "replicate_or_plate_field_present": False,
            },
            "conditions": _counts(environments),
            "interventions": {
                "mode": "CRISPR-Cas9 loss-of-function",
                "target_gene_symbols": len(action_symbols),
                "source_local_exact_symbol_to_ensembl_resolved": len(resolved),
                "unresolved_symbols": unresolved,
                "stable_action_ensembl_ids": sorted(set(resolved.values())),
                "control_cells": int(np.sum(perturbation == "control")),
                "targeting_cells": int(np.sum(perturbation != "control")),
                "unique_guide_id_strings": len(set(guides)),
                "unique_primary_sgrna_labels": len(set(sgrna)),
                "guide_id_note": "May contain semicolon-delimited multi-guide provenance; excluded from model inputs.",
                "split_contract": "sha256('slp11-development-v1|731|9606|ENSG'); first 8 bytes big-endian modulo 100; <70 train, 70-84 validation, >=85 test",
                "gene_split_counts": {k: int(split_counts.get(k, 0)) for k in ("train", "validation", "test")},
                "held_gene_rule": "All cells for test-split target genes stay out of future fitting across every condition and modality.",
            },
            "rna": {
                "queries": len(ensembl),
                "query_namespace": "mixed source field: unversioned Ensembl IDs where available, otherwise source symbols",
                "unique_query_ids": True,
                "stable_unversioned_ensembl_queries": int(sum(bool(ENSG_RE.fullmatch(x)) for x in ensembl)),
                "non_ensembl_source_query_ids": int(sum(not ENSG_RE.fullmatch(x) for x in ensembl)),
                "identity_constraint": "Only ENSG-form entries are immediately admissible to a stable-ID model query axis; non-ENSG entries require an explicit versioned mapping or quarantine.",
                "matrix": _matrix_metadata(rna, sample_limit),
                "library_umi_metadata": _numeric_summary(read_column(rna["obs"], "ncounts")),
                "preprocessing": "Cumulus 0.14.0 CellRanger v3, GRCh38 3.0.0; UMI counts; cells >=200 detected genes and <=18% mitochondrial genes; genes detected in >=200 cells.",
            },
            "protein": {
                "features": len(protein_roster),
                "feature_roster": protein_roster,
                "matrix": _matrix_metadata(protein, sample_limit),
                "library_umi_metadata": _numeric_summary(read_column(protein["obs"], "ncounts")),
                "value_space": "integer UMI counts for antibody-derived tags; no normalization applied in this snapshot",
            },
            "limitations": [
                "The harmonized AnnData files contain no patient identifier or plate/replicate assignment.",
                "Study-level plate replication described by the paper cannot be reconstructed per cell from these files.",
                "RNA and protein are paired but have distinct count depths and require separate observation heads and normalization contracts.",
                "The source-local symbol mapping is provenance for stable action IDs; symbols are never cross-source identity keys.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=1_000_000)
    args = parser.parse_args()
    result = audit(args.rna, args.protein, args.sample_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
