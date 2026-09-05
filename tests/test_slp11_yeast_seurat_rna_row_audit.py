from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_slp11_yeast_seurat_rna_rows as audit


class RowAuditTests(unittest.TestCase):
    def indices(self):
        result = {
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
        result["orf_systematic"]["YAA001W"].add("SGD:S000000001")
        result["orf_standard"]["GENE1"].add("SGD:S000000001")
        result["orf_alias"]["OLD1"].add("SGD:S000000001")
        result["feature_standard"]["15S_RRNA"].add("SGD:S000000002")
        result["feature_alias"]["tA(AGC)A"].add("SGD:S000000003")
        return result

    def test_evidence_levels_are_not_collapsed(self) -> None:
        indices = self.indices()
        exact = audit.classify_row("YAA001W", indices)
        alias = audit.classify_row("OLD1", indices)
        normalized = audit.classify_row("15S-RRNA", indices)
        self.assertEqual(exact["mappingClass"], "current-orf-systematic")
        self.assertEqual(alias["mappingClass"], "current-orf-alias-only")
        self.assertIn("candidate-only", alias["mappingEvidence"])
        self.assertEqual(
            normalized["mappingClass"],
            "seurat-dash-normalized-current-feature-candidate",
        )
        self.assertTrue(normalized["biologicalRnaEvidence"])

    def test_artificial_unresolved_row_is_not_denominator_evidence(self) -> None:
        result = audit.classify_row("bc-YAA001W", self.indices())
        self.assertEqual(result["mappingClass"], "unresolved-artificial-candidate")
        self.assertFalse(result["biologicalRnaEvidence"])


if __name__ == "__main__":
    unittest.main()
