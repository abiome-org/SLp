"""Exact rights boundaries for derived quantitative proteome snapshots."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ProteomeDerivedRightsTest(unittest.TestCase):
    PRIMARY_RUN = (
        "omf://abiome/slp/runresult/"
        "result-01a06d42-8493-7cd5-8c57-779dc8512436@"
        "sha256:6ec04e07cd917b66e16274a04f565844fa7acc1b538fb460c408f9276b5f694c"
    )
    REPRODUCTION_RUN = (
        "omf://abiome/slp/runresult/"
        "result-01a06d43-4260-707f-8fa5-4beee63c7856@"
        "sha256:d81d5c4f8a790542f3b2c16ed6d0954b27907784aa946e157cee76406c3b22a6"
    )
    VALIDATION_PRIMARY_RUN = (
        "omf://abiome/slp/runresult/"
        "result-01a06d60-6e5c-7552-b9ce-01cf6d046c31@"
        "sha256:3efaf6bf8db78894f478badc4b46b54432b54008556aefb9cbe3844760424dde"
    )
    VALIDATION_REPRODUCTION_RUN = (
        "omf://abiome/slp/runresult/"
        "result-01a06d60-e6d5-7326-a0ed-2c2dda05a18b@"
        "sha256:2eff9f50be60e4dd80be3dcd2915c08cc1d33ec272777d544563f25ad0c1be96"
    )
    DATASETS = [
        "omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-raw-v2@sha256:5392d4df7e962c9f59798b83fbdf8e71cd568b30c78a498702d50fabc059397e",
        "omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-intervention-inventory-v1@sha256:bd688dffdf4d96c01d4147580b1a8705c2149acadbc843a719537817a74505d9",
        "omf://abiome/slp/datasetsnapshot/slp-1-1-proteome-protein-relations-v1@sha256:acad3427907644f8ab8af38ed36066a6e1148ef92557b727351b0a4fba2b446c",
        "omf://abiome/slp/datasetsnapshot/slp-1-1-held-roster-v1@sha256:1b9a4800370a5398bf83e0a636007f466bf6ca5a6232e2ebb8fc64c5beb63450",
    ]
    INPUT_ARTIFACTS = {
        "sgdCurrentOrfs": "sha256:e67f0e8773feae108ecdb687139885e01ca972ff4aec95cd1358b33db1ea1192",
        "sgdMappingManifest": "sha256:c74ea81ce604357b998e5f09130dff85bf8a7a26504b9b2426f8038608c52d9c",
    }

    def _load(self, name: str) -> dict:
        value = yaml.safe_load((ROOT / "rights" / name).read_text(encoding="utf-8"))
        self.assertIsInstance(value, dict)
        return value

    def _assert_common(self, rights: dict) -> None:
        self.assertEqual(rights["license"], "CC-BY-4.0")
        self.assertIs(rights["trainingAllowed"], True)
        self.assertIs(rights["redistributionAllowed"], True)
        self.assertIs(rights["attributionRequired"], True)
        self.assertEqual(rights["derivedFrom"]["datasets"], self.DATASETS)
        self.assertEqual(rights["derivedFrom"]["inputArtifacts"], self.INPUT_ARTIFACTS)
        self.assertEqual(rights["derivedFrom"]["runResult"], self.PRIMARY_RUN)
        self.assertEqual(rights["reproducedBy"]["runResult"], self.REPRODUCTION_RUN)
        self.assertEqual(
            rights["derivedFrom"]["contentSha256"],
            rights["reproducedBy"]["contentSha256"],
        )
        self.assertEqual(len(rights["derivedFrom"]["artifacts"]), 1)
        for forbidden in (
            "preparation-audit.json",
            "analytical-QC",
            "reward",
            "synthetic-lethality",
            "future source",
        ):
            self.assertTrue(
                any(forbidden in item for item in rights["exclusions"]),
                forbidden,
            )

    def test_pretraining_observation_rights_are_one_file_and_fitting_only(self) -> None:
        rights = self._load(
            "slp-1-1-proteome-observation-pretrain-cc-by-4.0.yaml"
        )
        self._assert_common(rights)
        self.assertIn("Exactly one file named observation-corpus.tar", rights["scope"])
        self.assertIn("fitting-only", rights["scope"])
        self.assertIn("no molecular-validation", rights["scope"])
        self.assertEqual(
            rights["derivedFrom"]["artifacts"],
            {
                "observationCorpus":
                    "sha256:da147a203b93a89e0807e624a43b51cd4037d6e80169fd1601f40a5b3a4250ab"
            },
        )
        self.assertEqual(
            rights["derivedFrom"]["contentSha256"],
            "1f533d7dfb5bd76489b5b4576268e5d5b58fc6200416362876b5a2301c611f0b",
        )
        self.assertEqual(
            rights["reproducedBy"]["artifact"],
            "sha256:102c6b48e0e48f1fcdde2fe56790d6f3a0edde23e092b307110e1c75d8a8949d",
        )
        self.assertTrue(
            any("basal-control.tar" in item for item in rights["exclusions"])
        )

    def test_basal_rights_are_one_file_and_exclude_every_knockout(self) -> None:
        rights = self._load("slp-1-1-proteome-basal-control-cc-by-4.0.yaml")
        self._assert_common(rights)
        self.assertIn("Exactly one file named basal-control.tar", rights["scope"])
        self.assertIn("exactly 388 HIS3", rights["scope"])
        self.assertIn("no knockout trajectory", rights["scope"])
        self.assertEqual(
            rights["derivedFrom"]["artifacts"],
            {
                "basalControl":
                    "sha256:e3e011fcb5543c714a1b7d2032e6493f4026da54dc865ef4de848b6bde53380a"
            },
        )
        self.assertEqual(
            rights["derivedFrom"]["contentSha256"],
            "9be4596a59f3730e7b16995ba562e6561b8f424f46f99aecaeaee78ffe536a71",
        )
        self.assertEqual(
            rights["reproducedBy"]["artifact"],
            "sha256:0bf29c0b15dd8909496fa2ff3f704c9a89b32f24f4767410ec23a2ce77e33a51",
        )
        self.assertTrue(
            any("every knockout trajectory" in item for item in rights["exclusions"])
        )
        self.assertTrue(
            any("observation-corpus.tar" in item for item in rights["exclusions"])
        )

    def test_validation_rights_are_one_file_and_operationally_protected(self) -> None:
        rights = self._load(
            "slp-1-1-proteome-observation-molecular-validation-cc-by-4.0.yaml"
        )
        self.assertEqual(
            set(rights),
            {
                "license",
                "trainingAllowed",
                "redistributionAllowed",
                "attributionRequired",
                "source",
                "evidence",
                "licenseTerms",
                "scope",
                "operationalUse",
                "derivedFrom",
                "reproducedBy",
                "rightsBasis",
                "attribution",
                "exclusions",
            },
        )
        self.assertEqual(rights["license"], "CC-BY-4.0")
        self.assertIs(rights["trainingAllowed"], True)
        self.assertIs(rights["redistributionAllowed"], True)
        self.assertIs(rights["attributionRequired"], True)
        self.assertEqual(rights["derivedFrom"]["datasets"], self.DATASETS)
        self.assertEqual(rights["derivedFrom"]["inputArtifacts"], self.INPUT_ARTIFACTS)
        self.assertEqual(
            rights["derivedFrom"]["runResult"], self.VALIDATION_PRIMARY_RUN
        )
        self.assertEqual(
            rights["reproducedBy"]["runResult"],
            self.VALIDATION_REPRODUCTION_RUN,
        )
        self.assertEqual(
            rights["derivedFrom"]["artifacts"],
            {
                "molecularValidationObservation":
                    "sha256:c2ae4c787504f39f01e78e88e46c3009e32542002adb00b802b6c98e2611d87c"
            },
        )
        self.assertEqual(
            rights["reproducedBy"]["artifact"],
            "sha256:36f6edcdf585319553093e172b05412bfe5b80377bdd6964fd411f5a4ce08d52",
        )
        self.assertEqual(
            rights["derivedFrom"]["contentSha256"],
            "f8263d4813282799625182e8286a0af42311d5e76d58c84071ae9071e8a4bc69",
        )
        self.assertEqual(
            rights["derivedFrom"]["contentSha256"],
            rights["reproducedBy"]["contentSha256"],
        )
        self.assertIn("Exactly one file named observation-corpus.tar", rights["scope"])
        self.assertIn("537 protected molecular-validation", rights["scope"])
        self.assertIn("529 stable SGD", rights["scope"])
        self.assertEqual(rights["operationalUse"]["currentFactoryRole"], "custodian")
        self.assertEqual(
            set(rights["operationalUse"]),
            {
                "currentFactoryRole",
                "currentFactoryAdmissionAllowed",
                "currentFactoryEvaluationAllowed",
                "intendedUse",
                "fittingAllowed",
                "rewardAllowed",
                "finalHoldoutAllowed",
                "confidentialityEnforcedByCurrentOMF",
                "operationalRestrictionsEnforcedByCurrentOMF",
                "requiredEvaluationBoundary",
                "note",
            },
        )
        self.assertIs(
            rights["operationalUse"]["currentFactoryAdmissionAllowed"], True
        )
        self.assertIs(
            rights["operationalUse"]["currentFactoryEvaluationAllowed"], False
        )
        self.assertEqual(
            rights["operationalUse"]["intendedUse"],
            "protected-molecular-validation-evaluation-truth",
        )
        self.assertIs(rights["operationalUse"]["fittingAllowed"], False)
        self.assertIs(rights["operationalUse"]["rewardAllowed"], False)
        self.assertIs(rights["operationalUse"]["finalHoldoutAllowed"], False)
        self.assertIs(
            rights["operationalUse"]["confidentialityEnforcedByCurrentOMF"],
            False,
        )
        self.assertIs(
            rights["operationalUse"]["operationalRestrictionsEnforcedByCurrentOMF"],
            False,
        )
        for forbidden in (
            "preparation-audit.json",
            "molecular-final",
            "model fitting",
            "molecular reward",
            "synthetic-lethality",
            "clean training factory",
            "future source",
        ):
            self.assertTrue(
                any(forbidden in item for item in rights["exclusions"]),
                forbidden,
            )


if __name__ == "__main__":
    unittest.main()
