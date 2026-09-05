from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/audit_slp11_frangieh_static_features.py"
SPEC = importlib.util.spec_from_file_location("frangieh_static_audit_test", PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_roster_coverage_separates_feature_peptide_biotype_and_release() -> None:
    coverage = AUDIT.roster_coverage(
        {"ENSG00000000001", "ENSG00000000002", "ENSG00000000003"},
        {"ENSG00000000001"},
        {"ENSG00000000001"},
        {
            "ENSG00000000001": "protein_coding",
            "ENSG00000000002": "lncRNA",
        },
    )
    assert coverage["existingPhysicalFeatureRows"] == 1
    assert coverage["missingExistingPhysicalFeatureRows"] == 2
    assert coverage["withoutEnsembl116SelectedPeptide"] == 2
    assert coverage["lncRnaWithoutPeptide"] == 1
    assert coverage["withoutPeptideAndNotPresentInEnsembl116Gtf"] == 1
    assert coverage["proteinCodingWithoutPeptide"] == 0
