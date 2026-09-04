"""Committed-fixture integration smoke for the OMF audit-to-world path."""

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORLD_MODULE = ROOT / "modules" / "slp-1-1-world"
AUDIT_MODULE = ROOT / "modules" / "slp-1-1-corpus-audit"
FIXTURE = ROOT / "data" / "fixtures" / "slp11-world-smoke"
sys.path.insert(0, str(WORLD_MODULE))
sys.path.insert(0, str(AUDIT_MODULE))

from audit import audit_corpora  # noqa: E402
from trainer import train_world  # noqa: E402


class WorldOmfSmokeTest(unittest.TestCase):
    def test_committed_two_species_snapshots_audit_and_train(self) -> None:
        roots = {
            "pretrain": FIXTURE / "pretrain",
            "molecularValidation": FIXTURE / "validation",
            "molecularReward": FIXTURE / "reward",
        }
        audit = audit_corpora(roots)
        self.assertTrue(audit["auditPassed"])
        self.assertEqual(audit["leakageViolations"], 0)
        self.assertEqual(audit["benchmarkLabelRecords"], 0)
        self.assertEqual(audit["speciesTaxa"], [4932, 9606])

        model, report, _baselines = train_world(
            roots,
            {
                "seed": 17,
                "epochs": 1,
                "reinforcementEpochs": 0,
                "batchSize": 4,
                "learningRate": 0.001,
                "dModel": 8,
                "nhead": 2,
                "encoderLayers": 1,
                "decoderLayers": 1,
                "ffnMultiplier": 2,
                "dropout": 0.0,
            },
            audit,
        )

        self.assertGreater(model.count_parameters(), 0)
        self.assertEqual(report["schema"], "slp.training-report/v1.1")
        self.assertEqual(
            report["corpora"],
            {
                name: {
                    key: value
                    for key, value in audit["datasets"][name].items()
                    if key != "modalities"
                }
                for name in roots
            },
        )
        self.assertTrue(np.isfinite(report["selected"]["nll"]))
        self.assertEqual(set(report["selected"]["species"]), {"4932", "9606"})


if __name__ == "__main__":
    unittest.main()
