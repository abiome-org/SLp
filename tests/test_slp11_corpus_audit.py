"""Leakage and identity checks for OMF corpus snapshots."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-corpus-audit"
sys.path.insert(0, str(MODULE))

from audit import CorpusAuditError, audit_corpora  # noqa: E402


class CorpusAuditTest(unittest.TestCase):
    def _corpus(self, root: Path, name: str, role: str, genes: list[str]) -> Path:
        directory = root / name
        directory.mkdir()
        (directory / "trajectory-genes.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
        shard = directory / "shard-000.bin"
        shard.write_bytes((name + "\n").encode())
        manifest = {
            "schema": "slp.corpus/v1",
            "datasetId": name,
            "version": "fixture-v1",
            "role": role,
            "labelClass": "molecular",
            "benchmarkLabelsPresent": False,
            "speciesTaxa": [4932 if "yeast" in name else 9606],
            "modalities": ["synthetic-fixture"],
            "trajectoryGenes": "trajectory-genes.txt",
            "shards": [
                {
                    "path": shard.name,
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                    "records": 1,
                }
            ],
        }
        (directory / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    def test_strict_audit_accepts_species_native_disjoint_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = audit_corpora(
                {
                    "pretrain": self._corpus(
                        root, "yeast-pretrain", "pretrain", ["SGD:S000000001"]
                    ),
                    "molecularValidation": self._corpus(
                        root, "human-validation", "molecular-validation", ["ENSEMBL:ENSG000001"]
                    ),
                    "molecularReward": self._corpus(
                        root, "human-reward", "molecular-reward", ["ENSEMBL:ENSG000002"]
                    ),
                }
            )
        self.assertTrue(result["auditPassed"])
        self.assertEqual(result["leakageViolations"], 0)
        self.assertEqual(result["speciesTaxa"], [4932, 9606])

    def test_strict_audit_rejects_any_held_intervention_gene(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "pretrain": self._corpus(
                    root, "human-pretrain", "pretrain", ["ENSEMBL:ENSG000001"]
                ),
                "molecularValidation": self._corpus(
                    root, "human-validation", "molecular-validation", ["ENSEMBL:ENSG000001"]
                ),
                "molecularReward": self._corpus(
                    root, "human-reward", "molecular-reward", ["ENSEMBL:ENSG000002"]
                ),
            }
            result = audit_corpora(paths)
        self.assertFalse(result["auditPassed"])
        self.assertEqual(result["leakedTrajectoryGenes"], ["ENSEMBL:ENSG000001"])

    def test_digest_drift_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain = self._corpus(root, "human-pretrain", "pretrain", ["ENSEMBL:ENSG1"])
            (pretrain / "shard-000.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(CorpusAuditError, "digest mismatch"):
                audit_corpora(
                    {
                        "pretrain": pretrain,
                        "molecularValidation": self._corpus(
                            root, "human-validation", "molecular-validation", ["ENSEMBL:ENSG2"]
                        ),
                        "molecularReward": self._corpus(
                            root, "human-reward", "molecular-reward", ["ENSEMBL:ENSG3"]
                        ),
                    }
                )


if __name__ == "__main__":
    unittest.main()
