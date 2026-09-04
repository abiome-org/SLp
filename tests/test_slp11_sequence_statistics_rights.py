"""Exact rights boundary for the derived sequence-statistics feature block."""

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RIGHTS_PATH = (
    ROOT
    / "rights"
    / "slp-1-1-sequence-statistics-feature-block-v1-cc-by-4.0.yaml"
)

DATASETS = [
    (
        "omf://abiome/slp/datasetsnapshot/slp-1-1-static-entity-universe-v1@"
        "sha256:de3efddf5a9e4f66496a1edda14b04de774e972bc7b9efd30964644de2a56cac"
    ),
    (
        "omf://abiome/slp/datasetsnapshot/"
        "slp-1-1-sgd-protein-sequences-r64-5-1@"
        "sha256:3b76017f5ac74d8d96efb1db52d14af91c9fb15995062110558ce4651cf3ba0c"
    ),
]
DATASET_MANIFESTS = {
    "staticEntityUniverse": (
        "sha256:a65f94081c0b60a8b486ed968b58fc4d021ba3ea7f5f11425d3a1635cbb10684"
    ),
    "sgdProteinSequences": (
        "sha256:8f88480196b5cd8f3c15d65dbdbc09f83305c371fb476c70a38825dad2be4283"
    ),
}
INPUT_ARTIFACTS = {
    "sgdCurrentOrfs": (
        "artifact:sha256:"
        "e67f0e8773feae108ecdb687139885e01ca972ff4aec95cd1358b33db1ea1192"
    ),
    "sgdMappingManifest": (
        "artifact:sha256:"
        "c74ea81ce604357b998e5f09130dff85bf8a7a26504b9b2426f8038608c52d9c"
    ),
}
INPUT_PAYLOADS = {
    "sgdCurrentOrfs": {
        "bytes": 2_135_394,
        "sha256": (
            "df7b717cad88dc3672f72f8148f6a9132d12abe6ba020b220b091a8da8f7004d"
        ),
    },
    "sgdMappingManifest": {
        "bytes": 3_818,
        "sha256": (
            "570557ab1201913a18de9790f8adc5ee2e3cb56c6bb0e8d588fe43660c0214e1"
        ),
    },
}
RUN_RESULTS = [
    (
        "omf://abiome/slp/runresult/"
        "result-01a06df0-2427-7737-9321-1615583dedd8@"
        "sha256:72a9b2509069c05ed8aae82734fc31402f02229a64d8b39cd2f7afd06496a53b"
    ),
    (
        "omf://abiome/slp/runresult/"
        "result-01a06df0-6255-7e42-bf6d-e87425e8a19c@"
        "sha256:bf66d15da02370b2bfb0b1989cfd473876b2c55f482a7c9ed272c96886084eb2"
    ),
]
OMF_ARTIFACTS = {
    "firstFeatureBlock": (
        "sha256:8d7eba8e435ad4ff5a020bec68d452969b923b0722cd13227ed058f05236d878"
    ),
    "firstAudit": (
        "sha256:559129602ac30a4903a0823665941e583e8dbd79bc61402d3318a565c607ac52"
    ),
    "replicationFeatureBlock": (
        "sha256:d6f76d8c09cd68c9e461a249e64f25db14a9d97d9f52588b2f20a4b9bca0e30c"
    ),
    "replicationAudit": (
        "sha256:1001f228cffa5da235ca079af845722b5f450aef2fe32d898b518867ecd54809"
    ),
}
PAYLOADS = {
    "sequenceFeatureBlockTar": {
        "bytes": 4_392_960,
        "sha256": (
            "1b0aaec738b10ad3baa082d907d0c962c35c9b159b89fffca893fa1ecf5a7bed"
        ),
    },
    "sequenceFeatureBlockAudit": {
        "bytes": 7_851,
        "sha256": (
            "5d3a9fba29e9c31979fbda5a07951f244b66b35cf6c45de53c27fd231586a5e7"
        ),
    },
}


class SequenceStatisticsRightsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.rights = yaml.safe_load(RIGHTS_PATH.read_text(encoding="utf-8"))

    def test_declaration_is_closed_and_requires_attribution(self) -> None:
        self.assertEqual(
            set(self.rights),
            {
                "license",
                "trainingAllowed",
                "redistributionAllowed",
                "attributionRequired",
                "source",
                "evidence",
                "licenseTerms",
                "scope",
                "derivedFrom",
                "rightsBasis",
                "attribution",
                "purposeRestrictions",
                "exclusions",
            },
        )
        self.assertEqual(self.rights["license"], "CC-BY-4.0")
        self.assertIs(self.rights["trainingAllowed"], True)
        self.assertIs(self.rights["redistributionAllowed"], True)
        self.assertIs(self.rights["attributionRequired"], True)
        self.assertEqual(
            self.rights["licenseTerms"],
            "https://creativecommons.org/licenses/by/4.0/",
        )
        self.assertIn("Saccharomyces Genome Database", self.rights["attribution"])
        self.assertIn("10.17632/w8jtmnszd9.2", self.rights["attribution"])

    def test_exact_inputs_runs_artifacts_and_payloads_are_pinned(self) -> None:
        provenance = self.rights["derivedFrom"]
        self.assertEqual(
            set(provenance),
            {
                "datasets",
                "datasetOuterManifests",
                "inputArtifacts",
                "inputArtifactPayloads",
                "runResults",
                "omfArtifacts",
                "payloads",
            },
        )
        self.assertEqual(provenance["datasets"], DATASETS)
        self.assertEqual(provenance["datasetOuterManifests"], DATASET_MANIFESTS)
        self.assertEqual(provenance["inputArtifacts"], INPUT_ARTIFACTS)
        self.assertEqual(provenance["inputArtifactPayloads"], INPUT_PAYLOADS)
        self.assertEqual(provenance["runResults"], RUN_RESULTS)
        self.assertEqual(provenance["omfArtifacts"], OMF_ARTIFACTS)
        self.assertEqual(provenance["payloads"], PAYLOADS)

    def test_scope_is_static_only_and_contains_no_outcome_authorization(self) -> None:
        self.assertEqual(
            self.rights["purposeRestrictions"],
            {
                "staticFeatureInput": True,
                "quantitativeOutcomeTrainingCorpus": False,
                "quantitativeOutcomeTarget": False,
                "molecularReward": False,
                "syntheticLethalityLabels": False,
                "benchmarkDevelopment": False,
            },
        )
        exclusions = " ".join(self.rights["exclusions"]).casefold()
        for required in (
            "quantitative molecular",
            "held-roster",
            "molecular reward",
            "synthetic-lethality benchmark",
            "learned sequence embeddings",
            "free-text",
            "future parent datasetsnapshot",
        ):
            self.assertIn(required, exclusions)


if __name__ == "__main__":
    unittest.main()
