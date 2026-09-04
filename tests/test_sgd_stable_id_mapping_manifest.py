"""Fail-closed contract tests for the pinned SGD identity mapping source."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "sources" / "sgd-stable-id-mapping-2026-08-28.yaml"
WORKLOAD_PATH = ROOT / "workloads" / "slp-1-1-sgd-map.yaml"


class SgdStableIdMappingManifestTest(unittest.TestCase):
    @staticmethod
    def _load(path: Path) -> dict:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise AssertionError(f"expected YAML object: {path}")
        return value

    def setUp(self) -> None:
        self.source = self._load(SOURCE_PATH)
        self.rights = self._load(ROOT / self.source["rights"])

    def test_rights_are_training_permitted_and_separate_from_access(self) -> None:
        self.assertEqual(self.rights["license"], "CC-BY-4.0")
        self.assertIs(self.rights["trainingAllowed"], True)
        self.assertIs(self.rights["redistributionAllowed"], True)
        self.assertIs(self.rights["attributionRequired"], True)
        self.assertEqual(
            self.rights["evidence"],
            "https://doi.org/10.1093/genetics/iyae185",
        )
        self.assertIs(self.rights["technicalAccessIsRightsEvidence"], False)
        self.assertEqual(
            self.rights["technicalAccessEvidence"]["verifiedAt"], "2026-09-04"
        )
        self.assertIn("SGD_features.tab", self.rights["scope"])
        self.assertIn("dbxref.tab", self.rights["scope"])
        self.assertIn("deleted_merged_features.tab", self.rights["scope"])

    def test_snapshot_is_exactly_object_version_pinned_but_not_admitted(self) -> None:
        self.assertEqual(self.source["schema"], "slp.source-acquisition/v1")
        self.assertEqual(self.source["organism"]["ncbiTaxon"], 4932)
        self.assertEqual(
            self.source["mappingRelease"]["id"],
            "slp-sgd-map:2026-08-28-object-set-v1",
        )
        self.assertIs(
            self.source["mappingRelease"]["mappingReleaseIsGenomeAnnotationRelease"],
            False,
        )
        self.assertEqual(
            self.source["mappingRelease"]["upstreamGenomeAnnotationRelease"],
            "R64.5.1",
        )
        normalized = self.source["normalizedMappingArtifact"]
        self.assertEqual(
            normalized["identityMappingId"], self.source["mappingRelease"]["id"]
        )
        self.assertIsNone(normalized["identityMappingSha256"])
        self.assertEqual(normalized["status"], "not-admitted")
        self.assertIs(self.source["admission"]["allowed"], False)
        self.assertIs(
            self.source["admission"]["rawSnapshot"]["contentVerified"], True
        )
        self.assertIs(
            self.source["admission"]["rawSnapshot"]["readyForAdmission"], True
        )
        self.assertIs(self.source["admission"]["rawSnapshot"]["admitted"], False)
        self.assertIs(
            self.source["admission"]["normalizedMapping"]["admitted"], False
        )
        self.assertIs(
            self.source["admission"]["normalizedMapping"]
            ["disposableSmokeRunCountsAsAdmission"],
            False,
        )
        self.assertIs(
            self.source["admission"]["exactObjectsDownloadedToUntrackedTemporaryStorage"],
            True,
        )
        self.assertIs(self.source["admission"]["biologicalBytesStoredInGit"], False)

        expected = {
            "SGD_features.tab": (3382715, "44vkGKd1ao1YRMz6fRDrG754T10OWpYa", "636b4fc0407dd9f4fe74dceb5f5cd056194623d36a25c620a3c1ec2394af3dcc"),
            "SGD_features.README": (1586, "1tTdGAYrXGoMlKtlkxV5OowXh6JxgBoO", "befbd275783dda3772a19a20fadc4b1d62916dfe04e436540d4d935693a49328"),
            "dbxref.tab": (15365158, "vXD5MzDg0MZo06dF_8ETYCszXgJoxt3v", "2ff198c4127c226fa965cf514e10d10e2e2e581be228a22f4da69773fd7df5b7"),
            "dbxref.README": (3338, "ony2MZCEglI3DPYUTBOW0eku2IumqQwf", "96deaf79c0ae0bc05fdcc7cd42bf31ee34261970ad422476038e7b737250cb2c"),
            "deleted_merged_features.tab": (50549, "s23EsjjnQG_YwR5KWED7YkWqAiE_6u6G", "df979ff33732eb90220ef93c7ec38f817578dffbd31ebe3eb7978c98dd3f35b2"),
            "deleted_merged_features.README": (1024, "dx5YqH2HULlldRUhERyu8.8q8xNrQD9B", "b68857febe13cb07f6127e50069d80e784f01d4f3e7a48c5119848869eb4ade3"),
        }
        files = {item["name"]: item for item in self.source["allowlist"]}
        self.assertEqual(set(files), set(expected))
        for name, (size, version_id, sha256) in expected.items():
            with self.subTest(name=name):
                item = files[name]
                self.assertEqual(item["bytes"], size)
                self.assertEqual(item["s3VersionId"], version_id)
                self.assertTrue(item["versionedUrl"].endswith("?versionId=" + version_id))
                self.assertNotIn("latest", item["versionedUrl"].casefold())
                self.assertIs(item["etagIsCryptographicChecksum"], False)
                self.assertEqual(item["sha256"], sha256)

        checksum_policy = self.source["checksumPolicy"]
        self.assertIs(checksum_policy["upstreamEtagAcceptedAsChecksum"], False)
        self.assertIs(checksum_policy["admissionRequiresLocallyComputedSha256"], True)
        self.assertIs(checksum_policy["everyRecordedSha256Verified"], True)
        self.assertIs(checksum_policy["rawSnapshotReadyForAdmission"], True)
        self.assertEqual(checksum_policy["onByteOrVersionDrift"], "fail")
        probe = self.source["rawProbe"]
        self.assertEqual(probe["featureTable"]["currentOrfRows"], 6613)
        self.assertEqual(probe["featureTable"]["ambiguousExactSystematicNames"], 0)
        self.assertEqual(probe["crossReferenceTable"]["rows"], 267097)
        self.assertEqual(probe["crossReferenceTable"]["columns"], 6)

    def test_normalization_never_promotes_symbols_or_ambiguity_to_identity(self) -> None:
        identity = self.source["identity"]
        self.assertEqual(identity["canonicalNamespace"], "SGD")
        self.assertIs(identity["displaySymbolsAreIdentity"], False)
        self.assertIs(identity["standardGeneNamesAreIdentity"], False)
        self.assertIs(identity["freeTextAliasesAreIdentity"], False)
        self.assertIs(identity["simpleSystematicOrfRegexRequired"], False)

        contract = self.source["normalizationContract"]
        self.assertEqual(contract["releaseIdentityField"], "identityMappingId")
        self.assertEqual(
            contract["canonicalGeneRecord"]["identityField"], "canonicalSgdCurie"
        )
        self.assertNotEqual(
            contract["releaseIdentityField"],
            contract["canonicalGeneRecord"]["identityField"],
        )
        self.assertEqual(contract["canonicalGeneRecord"]["allowedFeatureType"], "ORF")
        self.assertEqual(
            contract["systematicOrfResolution"]["target"],
            "canonical SGD CURIE from column 1 on the same current ORF row",
        )
        self.assertEqual(
            contract["systematicOrfResolution"]["onDuplicateOrMissingTarget"],
            "quarantine",
        )
        self.assertEqual(contract["systematicOrfResolution"]["caseNormalization"], "none")
        external = contract["externalAccessionResolution"]
        self.assertIs(external["typedKeyRequired"], True)
        self.assertIs(external["bareAccessionAccepted"], False)
        self.assertEqual(external["caseNormalization"], "none")
        self.assertEqual(external["namespaceInferenceFromLexicalShape"], "forbidden")
        self.assertIs(external["normalizationPreservesAllTypedRelations"], True)
        self.assertIs(external["targetMustExistAsCurrentPrimaryOrf"], False)
        self.assertEqual(
            external["onMultipleCurrentTargets"], "emit-all-exact-relations"
        )
        self.assertEqual(
            external["onMissingCurrentTarget"],
            "preserve-relation-with-target-status",
        )
        self.assertEqual(external["collapseOrDiscardRelations"], "forbidden")
        self.assertEqual(
            external["emittedTargetStatus"],
            [
                "current-orf",
                "current-non-orf",
                "retired-or-merged",
                "not-in-current-feature-table",
            ],
        )
        downstream = contract["downstreamUntypedAccessionPolicy"]
        self.assertIs(downstream["assumeUniProtFromSixCharacterShape"], False)
        self.assertIn("explicit typed relation filter", downstream["requirement"])
        proteome = contract["downstreamQueryPolicies"]["proteomeUniProt"]
        self.assertIs(proteome["preserveInputAccessionKey"], True)
        self.assertEqual(
            proteome["typedRelationFilter"],
            {
                "accessionSources": ["EBI", "UniProtKB"],
                "accessionType": "UniProtKB ID",
            },
        )
        self.assertEqual(proteome["eligibleTargetStatus"], "current-orf")
        self.assertIs(proteome["preserveAllExactCurrentOrfRelations"], True)
        self.assertEqual(
            proteome["onMultipleCurrentOrfRelations"],
            "retain-all-and-defer-query-specific-admissibility",
        )
        self.assertIs(proteome["chooseFirst"], False)
        self.assertEqual(contract["displayMetadata"]["resolutionUse"], "forbidden")
        retired = contract["retiredIdentityPolicy"]
        self.assertIs(retired["deletedAndMergedIdsAreCanonicalInputs"], False)
        self.assertIs(retired["automaticRedirectToNewPrimaryId"], False)
        self.assertTrue(retired["action"].startswith("quarantine"))

    def test_workload_pins_the_raw_identity_snapshot_and_all_bounds(self) -> None:
        workload = self._load(WORKLOAD_PATH)
        self.assertEqual(workload["kind"], "WorkloadSpec")
        stages = workload["spec"]["graph"]["stages"]
        self.assertEqual(len(stages), 1)
        stage = stages[0]
        self.assertEqual(stage["module"], "modules/slp-1-1-sgd-map/module.yaml")
        self.assertEqual(stage["operation"], "run")
        self.assertEqual(
            stage["inputs"],
            {"rawSgdMapping": "dataset/slp-1-1-sgd-map-raw-2026-08-28"},
        )
        self.assertEqual(
            set(stage["config"]),
            {
                "maxFeatureRecords",
                "maxExternalRecords",
                "maxRetiredPhysicalLines",
                "maxLineBytes",
                "maxTargetsPerExternalKey",
                "maxAssertionsPerExternalTarget",
                "maxDisplayAliasesPerOrf",
            },
        )
        self.assertEqual(
            set(stage["outputs"]),
            {
                "mappingSummary",
                "identityMappingId",
                "identityMappingSha256",
                "mappingManifestSha256",
                "currentOrfCount",
                "typedExternalRelationCount",
                "oneToManyExternalRelationCount",
                "retiredQuarantineCount",
                "retiredIrregularCount",
            },
        )


if __name__ == "__main__":
    unittest.main()
