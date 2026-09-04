"""Guard claims at the OMF 1.0 integrity/confidentiality boundary."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class FactoryIsolationBoundaryTest(unittest.TestCase):
    def test_world_training_template_cannot_receive_raw_or_held_truth(self) -> None:
        workload = yaml.safe_load(
            (ROOT / "workloads/slp-1-1-world-sparse.yaml.tmpl").read_text(
                encoding="utf-8"
            )
        )
        stages = workload["spec"]["graph"]["stages"]
        self.assertEqual(len(stages), 1)
        stage = stages[0]
        self.assertEqual(stage["name"], "train-world")
        self.assertEqual(
            set(stage["inputs"]),
            {
                "pretrain",
                "molecularPredictionQuery",
                "corpusAuditEvidence",
                "heldRosterEvidence",
            },
        )
        serialized = str(stage).casefold()
        for forbidden in (
            "rawproteome",
            "moleculartruth",
            "molecularvalidationtruth",
            "molecularfinaltruth",
            "proteome-observation-molecular-validation",
            "proteome-observation-molecular-final",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_policy_does_not_pretend_to_offer_dataset_acl_keys(self) -> None:
        policy = yaml.safe_load(
            (ROOT / "policies/default.yaml").read_text(encoding="utf-8")
        )
        supported_match_keys = {
            "actor",
            "action",
            "resource",
            "purpose",
            "classification",
            "residency",
            "evidence",
        }
        forbidden_pseudo_acl_keys = {
            "dataset",
            "workload",
            "stage",
            "module",
            "allowedActors",
            "allowedPurposes",
        }
        for rule in policy["spec"]["rules"]:
            keys = set(rule.get("match", {}))
            self.assertLessEqual(keys, supported_match_keys)
            self.assertFalse(keys & forbidden_pseudo_acl_keys)


if __name__ == "__main__":
    unittest.main()
