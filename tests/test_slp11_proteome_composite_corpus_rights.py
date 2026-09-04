"""Exact rights boundary for the fitting-only composite proteome corpus."""

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RIGHTS_PATH = (
    ROOT
    / "rights"
    / "slp-1-1-proteome-composite-corpus-v1-cc-by-4.0.yaml"
)

DATASETS = {
    "observations": {
        "resource": (
            "omf://abiome/slp/datasetsnapshot/"
            "slp-1-1-proteome-observation-pretrain-v1@"
            "sha256:631f66e32a218e167af9edb60115a04514d0bcf675a13bcb244c465ffab2f751"
        ),
        "outerManifestDigest": (
            "sha256:0bc00463f8641fc91d6fcb82266b6f41d4c55cc78275b737eaad257dd2053130"
        ),
        "treeDigest": (
            "sha256:fc1f812308af999c601bee9b53ce21035bdd6fd9952cead11451c72b612a833f"
        ),
    },
    "staticFeatures": {
        "resource": (
            "omf://abiome/slp/datasetsnapshot/"
            "slp-1-1-sequence-statistics-feature-block-v1@"
            "sha256:e9733974c551bca3af93c4cb488972f5167da5e7e3cf48ef5803348cd20d91e5"
        ),
        "outerManifestDigest": (
            "sha256:6b4b32c794d7787b9b9076d78726ea0ad7706d64fd82b5f918f0c6da20da0d2a"
        ),
        "treeDigest": (
            "sha256:3f4549114a181c162596d60ef1b94d222ec494282d23ece8da7e19142135cb8d"
        ),
    },
    "heldInterventionRoster": {
        "resource": (
            "omf://abiome/slp/datasetsnapshot/slp-1-1-held-roster-v1@"
            "sha256:1b9a4800370a5398bf83e0a636007f466bf6ca5a6232e2ebb8fc64c5beb63450"
        ),
        "outerManifestDigest": (
            "sha256:f8aac504a2d56fdc9e13cc9b1c9fa87a08ebc7ff2d7036c0b6b135c26d187425"
        ),
        "treeDigest": (
            "sha256:ba62f5855f46e693f2a27f4ed06efeec046ccd99c9993c145f9983807dfed0b1"
        ),
    },
}

RUN_RESULTS = {
    "primary": (
        "omf://abiome/slp/runresult/"
        "result-01a06e27-c6b8-7eee-ba08-54c27d8ada57@"
        "sha256:8d4362b2d77d855abea6357b6e34abc0b2052fdc71675355548ded16390d9281"
    ),
    "replication": (
        "omf://abiome/slp/runresult/"
        "result-01a06e28-2f3f-7ccb-b71d-8b7654fc26ca@"
        "sha256:58116fb6b3a075ff188d47141d326cd55d895d43cf2f18666b35ae371501348d"
    ),
}

OMF_ARTIFACT_MANIFESTS = {
    "primaryCorpus": (
        "sha256:9c208900fc871b2a60ceddb1a5d72ea5670a327feea4b2bdf39f92132376a0c6"
    ),
    "primaryAudit": (
        "sha256:ae9e3fa7ee9207dae17fef63b906f218904fdf5a18d534534801dd96a7bac286"
    ),
    "replicationCorpus": (
        "sha256:fb4fa62ccd06b921947bccc01f4ebb4155d4239df74a644e5105962c5cf7198f"
    ),
    "replicationAudit": (
        "sha256:bf3f020466c284f2814f6cee7ed2569417ed9b7ba49258ef78e03936114c0d2e"
    ),
}

PAYLOADS = {
    "compositeCorpusTar": {
        "path": "corpus-v1-2.tar",
        "bytes": 89_149_440,
        "sha256": (
            "0a5322c46e15e8a15d17000e8993c0ad642fcc70bc8fff00cbba8fb2905708bf"
        ),
    },
    "compositionAudit": {
        "path": "corpus-compose-audit.json",
        "bytes": 5_277,
        "sha256": (
            "898e4069b2bd9575bd7380b57ed6214bf3d75043feb401c0fa50371972623c52"
        ),
    },
}

COUNTS = {
    "entities": 7_038,
    "featureRows": 7_037,
    "contexts": 1,
    "queries": 1_850,
    "panels": 1,
    "trajectoryInterventions": 3_679,
    "records": 3_811,
    "targetValues": 6_865_493,
    "shards": 8,
}

DIGESTS = {
    "corpusManifestSha256": (
        "d91cbbc0b98ea05ccbf56201f50143f57c2b71ffe53211a5a5128ce706c60ad7"
    ),
    "entityKeySetSha256": (
        "9ca16d4f44ca97b4940bd389ca8bbdafe0c6fd711d557a98743218a83caeb87d"
    ),
    "featureEntityKeySetSha256": (
        "4a7e15d5aca02862a80acbd182f5a52c86c35e4dbadf8d95297c2ba47a95dce5"
    ),
    "featurePackSha256": (
        "016753a94bacd6e2b8dd299abc7906fa874c3d5926ff73605a1f9c913a12d66b"
    ),
    "featurePresentBytesSha256": (
        "08de1975edffb1a14cbea7d27d7fde8abedf8e2cc1899f70838d68a9b5b287af"
    ),
    "featureValueBytesSha256": (
        "3f51a98266c855800917ca6c7b87e205d9df7268260c44b4ba0341605840b7a0"
    ),
    "targetValueBytesSha256": (
        "7eda2fee14728865518c1133e4a7c122a2180ac29b9c5977946518a0c3d46af6"
    ),
}


class ProteomeCompositeCorpusRightsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.rights = yaml.safe_load(RIGHTS_PATH.read_text(encoding="utf-8"))

    def test_declaration_is_closed_cc_by_and_attributed(self) -> None:
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
                "corpusContract",
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
        attribution = self.rights["attribution"]
        self.assertIn("10.17632/w8jtmnszd9.2", attribution)
        self.assertIn("Saccharomyces Genome Database", attribution)
        self.assertIn("10.1093/genetics/iyae185", attribution)
        self.assertIn("10.5281/zenodo.14062629", attribution)

    def test_exact_parent_snapshots_runs_artifacts_and_payloads_are_pinned(self) -> None:
        provenance = self.rights["derivedFrom"]
        self.assertEqual(
            set(provenance),
            {
                "datasets",
                "runResults",
                "omfArtifactManifests",
                "payloads",
            },
        )
        self.assertEqual(provenance["datasets"], DATASETS)
        for snapshot in provenance["datasets"].values():
            self.assertEqual(
                set(snapshot),
                {"resource", "outerManifestDigest", "treeDigest"},
            )
        self.assertEqual(provenance["runResults"], RUN_RESULTS)
        self.assertEqual(
            provenance["omfArtifactManifests"], OMF_ARTIFACT_MANIFESTS
        )
        self.assertEqual(provenance["payloads"], PAYLOADS)
        self.assertNotEqual(
            provenance["omfArtifactManifests"]["primaryCorpus"],
            provenance["omfArtifactManifests"]["replicationCorpus"],
        )
        self.assertNotEqual(
            provenance["omfArtifactManifests"]["primaryAudit"],
            provenance["omfArtifactManifests"]["replicationAudit"],
        )

    def test_composite_contract_counts_and_digests_are_exact(self) -> None:
        contract = self.rights["corpusContract"]
        self.assertEqual(
            set(contract),
            {
                "schema",
                "rightsRevision",
                "role",
                "identityKey",
                "rewardEnabled",
                "counts",
                "digests",
            },
        )
        self.assertEqual(contract["schema"], "slp.corpus/v1.2")
        self.assertEqual(
            contract["rightsRevision"],
            "slp-rights:proteome-composite-corpus-v1-cc-by-4.0",
        )
        self.assertEqual(contract["role"], "pretrain")
        self.assertEqual(contract["identityKey"], ["ncbiTaxon", "entityId"])
        self.assertIs(contract["rewardEnabled"], False)
        self.assertEqual(contract["counts"], COUNTS)
        self.assertEqual(contract["digests"], DIGESTS)

    def test_authorization_is_fitting_only_and_fail_closed(self) -> None:
        self.assertEqual(
            self.rights["purposeRestrictions"],
            {
                "worldModelPretrainingFitting": True,
                "molecularValidation": False,
                "molecularReward": False,
                "molecularFinalHoldout": False,
                "worldModelArchitectureSelection": False,
                "calibration": False,
                "syntheticLethalityLabels": False,
                "benchmarkDevelopment": False,
            },
        )
        scope = self.rights["scope"]
        self.assertIn("corpus-v1-2.tar and corpus-compose-audit.json", scope)
        self.assertIn("fitting-only slp.corpus/v1.2", scope)
        self.assertIn("no protected quantitative outcome", scope)
        self.assertIn("reward record", scope)
        self.assertIn("benchmark record", scope)
        exclusions = " ".join(self.rights["exclusions"]).casefold()
        for required in (
            "molecular-validation",
            "outcome-blind roster",
            "role assignments",
            "reward",
            "benchmark labels",
            "calibration",
            "static features other than",
            "cross-taxon identity merging",
            "performance claims",
            "future parent datasetsnapshot",
        ):
            self.assertIn(required, exclusions)


if __name__ == "__main__":
    unittest.main()
