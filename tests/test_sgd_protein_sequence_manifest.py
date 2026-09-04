"""Fail-closed contract checks for the exact SGD R64.5.1 protein source."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources" / "sgd-protein-sequences-r64-5-1.yaml"


class SgdProteinSequenceManifestTest(unittest.TestCase):
    @staticmethod
    def _load(path: Path) -> dict:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise AssertionError(f"expected YAML object: {path}")
        return value

    def setUp(self) -> None:
        self.source = self._load(SOURCE)
        self.rights = self._load(ROOT / self.source["rights"])

    def test_exact_release_objects_are_immutable_and_checksum_pinned(self) -> None:
        self.assertEqual(self.source["schema"], "slp.source-acquisition/v1")
        self.assertEqual(
            self.source["status"], "raw-snapshot-admitted-feature-generation-pending"
        )
        release = self.source["release"]
        self.assertEqual(release["upstreamGenomeAnnotationRelease"], "R64.5.1")
        self.assertEqual(release["upstreamHeaderRelease"], "64-5-1")
        self.assertEqual(str(release["releaseDate"]), "2024-05-29")

        expected = {
            "orf_trans_all_R64-5-1_20240529.fasta.gz": (
                2_689_634,
                "GRiDuJlE44rFsMHE63VZUFxVcA4GBun6",
                "17e8b47e1ae23178c6000fbc4ab548f102d1b250ef9dff5d811feb3f03dd2c5b",
            ),
            "orf_protein.README": (
                930,
                "qrmr4JzXcqmddqAYm5cA4i4ywgAp2xkp",
                "b53064bef6424f0e9b5c5a6af88602bb15949e68bedb397b6731b094ebca5be9",
            ),
            "dates_of_genome_releases.tab": (
                2_050,
                "7Ze3kUv.yxtt8cDceDVCDbYH8pdhu43S",
                "cc5d40722442a605d1d6dcf9a36442d87076829a04f776f87b3de0020f92f9e7",
            ),
        }
        objects = {item["name"]: item for item in self.source["allowlist"]}
        self.assertEqual(set(objects), set(expected))
        for name, (size, version_id, sha256) in expected.items():
            with self.subTest(name=name):
                item = objects[name]
                self.assertEqual(item["bytes"], size)
                self.assertEqual(item["s3VersionId"], version_id)
                self.assertTrue(
                    item["versionedUrl"].endswith("?versionId=" + version_id)
                )
                self.assertNotIn("latest", item["versionedUrl"].casefold())
                self.assertIs(item["etagIsCryptographicChecksum"], False)
                self.assertEqual(item["sha256"], sha256)

        sequence = objects["orf_trans_all_R64-5-1_20240529.fasta.gz"]
        self.assertEqual(sequence["decompressedBytes"], 5_511_467)
        self.assertEqual(
            sequence["decompressedSha256"],
            "e01f9e1ef7e5a01ff7cd0ee7a843e6d1c1da8c3777fdfac3a5293711d4c56518",
        )
        mirror = self.source["mirror"]
        self.assertIs(mirror["admittedPayload"], False)
        self.assertEqual(mirror["sha256"], sequence["sha256"])
        self.assertNotIn(mirror["path"], {item["path"] for item in objects.values()})

    def test_taxonomy_identity_and_content_assertions_prevent_symbol_joins(self) -> None:
        organism = self.source["organism"]
        self.assertEqual(organism["speciesTaxon"], 4932)
        self.assertEqual(organism["strainTaxon"], 559292)
        self.assertIn("strain-of", organism["taxonomyRelation"])

        content = self.source["expectedContent"]
        self.assertEqual(content["fastaCompression"], "gzip")
        self.assertEqual(content["fastaRecords"], 6_722)
        self.assertEqual(content["uniqueSgdIds"], 6_722)
        self.assertEqual(content["releaseMarkerRecords"], 6_722)
        self.assertEqual(content["currentOrfCoverage"]["currentOrfs"], 6_613)
        self.assertEqual(content["currentOrfCoverage"]["coveredCurrentOrfs"], 6_613)
        self.assertEqual(content["currentOrfCoverage"]["missingCurrentOrfs"], 0)
        self.assertEqual(sum(content["sourceClasses"].values()), 6_722)
        self.assertEqual(content["nonCurrentOrfRecords"], 109)

        identity = self.source["identity"]
        self.assertIn("SGDID:S#########", identity["fastaHeaderPrefix"])
        self.assertIn("exact SGDID token", identity["canonicalJoin"])
        self.assertIs(identity["displayNameResolvesIdentity"], False)
        self.assertIs(identity["simpleSystematicOrfRegexAllowed"], False)
        self.assertIs(identity["duplicateCanonicalIdsAllowed"], False)

    def test_source_is_static_only_and_rights_scope_is_narrow(self) -> None:
        policy = self.source["featurePolicy"]
        self.assertIs(policy["staticFeatureSourceAllowed"], True)
        self.assertIs(policy["quantitativeTrainingCorpusAllowed"], False)
        self.assertIs(policy["includeHeldEntitySequences"], True)
        self.assertIs(policy["freeTextHeaderDescriptionAsFeature"], False)
        self.assertIs(policy["displayNameAsFeature"], False)
        self.assertIs(policy["identifierDerivedNumericFeatures"], False)
        self.assertIs(policy["outcomeFieldsPresent"], False)
        self.assertIs(policy["benchmarkFieldsPresent"], False)

        self.assertEqual(self.rights["license"], "CC-BY-4.0")
        self.assertIs(self.rights["trainingAllowed"], True)
        self.assertIs(self.rights["redistributionAllowed"], True)
        self.assertIs(self.rights["attributionRequired"], True)
        self.assertIs(self.rights["technicalAccessIsRightsEvidence"], False)
        self.assertTrue(self.rights["purposeRestrictions"]["staticFeatureGeneration"])
        self.assertFalse(
            self.rights["purposeRestrictions"]["molecularOutcomeTrainingCorpus"]
        )
        self.assertFalse(self.rights["purposeRestrictions"]["syntheticLethalityLabels"])
        self.assertIn("exact", self.rights["scope"])
        self.assertIn("future SGD object version", self.rights["exclusions"][-1])

        admission = self.source["admission"]
        self.assertIs(admission["biologicalBytesStoredInGit"], False)
        raw = admission["rawSnapshot"]
        self.assertIs(raw["admitted"], True)
        self.assertIs(raw["contentVerified"], True)
        self.assertEqual(raw["actor"], "slp-researcher")
        self.assertEqual(
            raw["resource"],
            "omf://abiome/slp/datasetsnapshot/"
            "slp-1-1-sgd-protein-sequences-r64-5-1@"
            "sha256:3b76017f5ac74d8d96efb1db52d14af91c9fb15995062110558ce4651cf3ba0c",
        )
        self.assertEqual(
            raw["manifestDigest"],
            "sha256:8f88480196b5cd8f3c15d65dbdbc09f83305c371fb476c70a38825dad2be4283",
        )
        self.assertEqual(
            raw["treeDigest"],
            "sha256:823a18ed8039ee44ee44b860551fea749b9012c941e6b9cd5163938da19b168a",
        )
        self.assertIs(admission["staticFeatureBlock"]["produced"], False)
        self.assertIs(admission["trainingCorpusAllowed"], False)


if __name__ == "__main__":
    unittest.main()
