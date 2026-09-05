#!/usr/bin/env python3
"""Profile a deterministic sample of the proposed Ensembl116 ESM2-8M extension."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_slp11_human_sequence_features as sequence


def run(args: argparse.Namespace) -> dict[str, object]:
    fasta = sequence.verify_source_dir(args.ensembl_source)
    translations, _ = sequence.parse_longest_translations(fasta)
    with np.load(args.old_features, allow_pickle=False) as archive:
        present = {
            str(gene)
            for gene, flag in zip(
                archive["entity_id"], archive["feature_values"][:, 320], strict=True
            )
            if flag == 1
        }
    missing = sorted(set(translations) - present, key=lambda gene: (len(translations[gene].peptide), gene))
    indices = np.linspace(0, len(missing) - 1, args.sample, dtype=np.int64)
    genes = [missing[index] for index in indices]
    peptides = [sequence.normalize_for_esm(translations[gene].peptide) for gene in genes]
    started = time.monotonic()
    _, stats = sequence.extract_esm(
        peptides,
        args.model,
        device_name="cuda",
        batch_size=16,
        max_residues=sequence.ESM_MAX_RESIDUES,
        overlap=sequence.ESM_DEFAULT_OVERLAP,
    )
    elapsed = time.monotonic() - started
    all_missing_windows = sum(
        len(sequence.chunk_windows(len(translations[gene].peptide))) for gene in missing
    )
    report = {
        "schema": "slp.frangieh-esm2-8m-extension-profile/v1",
        "sampleRule": "even positions after sorting missing Ensembl116 genes by selected-peptide length then stable ENSG",
        "samplePeptides": len(peptides),
        "sampleWindows": stats["windowCount"],
        "sampleElapsedSeconds": elapsed,
        "secondsPerWindow": elapsed / stats["windowCount"],
        "allMissingPeptides": len(missing),
        "allMissingWindows": all_missing_windows,
        "projectedAllMissingSeconds": elapsed / stats["windowCount"] * all_missing_windows,
        "configuration": {
            "model": sequence.ESM_REPOSITORY,
            "revision": sequence.ESM_REVISION,
            "hiddenSize": sequence.ESM_HIDDEN_SIZE,
            "maxResidues": sequence.ESM_MAX_RESIDUES,
            "overlap": sequence.ESM_DEFAULT_OVERLAP,
            "batchSize": 16,
            "pooling": "inverse-overlap-weighted full-residue mean; no truncation",
        },
        "runtime": stats,
        "molecularOutcomesRead": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensembl-source", type=Path, required=True)
    parser.add_argument("--old-features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
