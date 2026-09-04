"""Contract checks for the first biological SLp-1.1 source candidates.

These tests deliberately stop at acquisition metadata.  They prevent a future
run from silently changing source versions, admitting derived targets, or
treating a rights review as if data had already been admitted to OMF.
"""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class YeastSourceManifestTest(unittest.TestCase):
    def _load(self, relative: str) -> dict:
        value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        self.assertIsInstance(value, dict)
        return value

    def _assert_rights(self, relative: str, version_doi: str) -> None:
        rights = self._load(relative)
        self.assertEqual(rights["license"], "CC-BY-4.0")
        self.assertIs(rights["trainingAllowed"], True)
        self.assertIs(rights["redistributionAllowed"], True)
        self.assertIs(rights["attributionRequired"], True)
        self.assertIn(version_doi, rights["source"])
        self.assertTrue(rights["evidence"].startswith("https://"))
        self.assertIn(version_doi, rights["attribution"])

    def test_single_cell_atlas_is_pinned_but_not_admitted(self) -> None:
        manifest = self._load("sources/yeast-single-cell-atlas-v1.yaml")
        self.assertEqual(manifest["schema"], "slp.source-acquisition/v1")
        self.assertEqual(manifest["status"], "contract-blocked")
        self.assertEqual(manifest["organism"]["ncbiTaxon"], 4932)
        self.assertEqual(manifest["identity"]["primaryNamespace"], "SGD")
        self.assertIs(manifest["identity"]["displaySymbolsAreIdentity"], False)
        self.assertEqual(
            manifest["source"]["versionDoi"], "10.5281/zenodo.14062629"
        )
        self.assertEqual(
            manifest["source"]["rawSequencingAccession"],
            "ArrayExpress:E-MTAB-14004",
        )
        self.assertEqual(manifest["admission"]["firstOperation"],
                         "metadata-and-schema-probe-only")
        self.assertEqual(
            {(item["name"], item["bytes"], item["upstreamChecksum"])
             for item in manifest["allowlist"]},
            {
                (
                    "seus_split.RData",
                    5_907_877_873,
                    "md5:65bb56efd8120f32f65c044de5f040aa",
                ),
                (
                    "README.txt",
                    2_050,
                    "md5:1fa718ad98d6eacbf6134372299f2b83",
                ),
            },
        )
        self.assertIn("seu.RData", manifest["excludedFiles"])
        self.assertIn("DEG.Rdata", manifest["excludedFiles"])
        self.assertIs(manifest["modeling"]["useSignificanceCallsAsTargets"], False)
        probe = manifest["metadataProbe"]
        self.assertEqual(
            probe["files"][0]["localSha256"],
            "268533b10c59d3f4ca941ff31ac8b9c108b61f55f00d85792d44b3a90b3b9da8",
        )
        self.assertTrue(probe["unresolved"])
        script_audit = probe["analysisScriptAudit"]
        self.assertEqual(
            script_audit["status"],
            "schema-expectations-only-not-an-admitted-input",
        )
        self.assertEqual(
            {item["name"]: item["upstreamChecksum"]
             for item in script_audit["scripts"]},
            {
                "summary.genotypes_Rev.R":
                    "md5:1b8cb9069f56101dc1d972a7300a3ac7",
                "Figures_Rev.R":
                    "md5:3f097ed4ce3055970b385c932b94bd91",
                "clone.comparison_Rev.R":
                    "md5:2fa7a992e277c6b7990c13acd2324625",
            },
        )
        expected = script_audit["expectedObjectContract"]
        self.assertEqual(expected["rootObject"], "seus")
        self.assertEqual(expected["conditionObjects"], ["Control", "NaCl"])
        self.assertEqual(expected["wildTypeAssignment"], "WT")
        self.assertEqual(expected["authorsCountAccess"],
                         "GetAssayData with assay RNA and slot counts")
        self.assertEqual(
            set(expected["metadataFieldsObservedInScripts"]),
            {
                "assignment_consensus2",
                "kogene",
                "kosym",
                "condition",
                "batch2",
                "clone",
            },
        )
        self._assert_rights(
            manifest["rights"], "10.5281/zenodo.14062629"
        )

    def test_proteome_is_non_imputed_and_content_pinned(self) -> None:
        manifest = self._load("sources/yeast-proteome-v2.yaml")
        self.assertEqual(manifest["schema"], "slp.source-acquisition/v1")
        self.assertEqual(manifest["status"], "contract-blocked")
        self.assertEqual(manifest["organism"]["ncbiTaxon"], 4932)
        self.assertEqual(manifest["source"]["versionDoi"],
                         "10.17632/w8jtmnszd9.2")
        value_space = manifest["upstreamValueSpace"]
        self.assertEqual(
            value_space["quantity"],
            "positive batch-corrected MaxLFQ relative protein intensity",
        )
        self.assertIs(value_space["tableLogTransformed"], False)
        self.assertIn("literal NA", value_space["missing"])
        self.assertEqual(value_space["processing"]["software"],
                         "DIA-NN 1.7.12 and DIA-NN R maxLFQ")
        controls = manifest["controlSemantics"]
        self.assertEqual(controls["HIS3"]["role"], "biological-WT-control")
        self.assertEqual(controls["HIS3"]["count"], 388)
        self.assertEqual(controls["qc"]["role"],
                         "pooled-digest-analytical-QC-only")
        self.assertEqual(controls["qc"]["count"], 389)
        files = {item["name"]: item for item in manifest["allowlist"]}
        self.assertEqual(set(files), {
            "yeast5k_noimpute_wide.csv",
            "yeast5k_metadata.csv",
            "Detection_of_KO_proteins.csv",
            "summary_fileupload.pdf",
        })
        self.assertEqual(
            files["yeast5k_noimpute_wide.csv"]["sha256"],
            "69a9df05b6db011f595a4e0b3ce25c1cc247f22cbdd066c79e6da9a706aa1df9",
        )
        self.assertEqual(files["yeast5k_noimpute_wide.csv"]["bytes"],
                         167_754_298)
        self.assertIn("yeast5k_impute_wide.csv", manifest["excludedFiles"])
        self.assertIn("yeast5k_stat_DE.csv", manifest["excludedFiles"])
        self.assertIs(manifest["admission"]["requireNonImputedValues"], True)
        self.assertIs(manifest["modeling"]["useGrowthPhenotypesAsMolecularTargets"],
                      False)
        probe = manifest["metadataProbe"]
        self.assertEqual(probe["observations"]["sampleTypes"], {
            "ko": 4699,
            "qc": 389,
            "HIS3": 388,
        })
        self.assertEqual(
            probe["observations"]["exactCaseUniqueKnockoutOrfStrings"], 4550
        )
        self.assertEqual(
            probe["observations"]["caseFoldedUniqueKnockoutOrfStrings"], 4549
        )
        self.assertEqual(probe["observations"]["exactDuplicateKnockoutRows"], 149)
        self.assertEqual(
            probe["observations"]["uniqueExactUppercaseSuffixedSystematicIds"],
            236,
        )
        self.assertEqual(
            probe["observations"]["extendedSystematicSuffixRows"],
            {"A": 214, "B": 29, "C": 3},
        )
        self.assertEqual(
            probe["observations"]["nonCanonicalCaseSystematicIds"],
            ["YAL043C-a", "YML009c"],
        )
        self.assertEqual(
            probe["observations"]["caseFoldIdentityCollisions"],
            [{"exactValues": ["YML009C", "YML009c"]}],
        )
        identity_probe = manifest["rawIdentityJoinProbe"]
        self.assertEqual(identity_probe["interventionIdentities"]["exactCurrentRows"], 4623)
        self.assertEqual(
            identity_probe["interventionIdentities"]["exactRetiredOrMergedRows"], 35
        )
        self.assertEqual(identity_probe["interventionIdentities"]["unmatchedExactRows"], 41)
        self.assertEqual(
            identity_probe["interventionIdentities"]["malformedSgdRowsNotTreatedAsIdentity"],
            [
                {
                    "rawFirstField": "YKR099C-A",
                    "sourceLine": 35,
                    "observedColumns": 12,
                    "requiredColumns": 13,
                    "downstreamClassification": "unmatched-exact-systematic-name",
                }
            ],
        )
        self.assertEqual(probe["observations"]["knockoutDetectionRows"], 959)
        self.assertEqual(
            probe["observations"]["knockoutDetectionStatusRows"],
            {"Missing": 838, "Not missing": 121},
        )
        self.assertEqual(
            probe["observations"]["knockoutDifferentialStatusRows"],
            {"missing": 838, "not DE": 82, "DE": 39},
        )
        self.assertEqual(
            probe["observations"]["knockoutDetectionDuplicatedOrfIds"], 23
        )
        matrix = probe["observations"]["wideMatrix"]
        self.assertEqual((matrix["rows"], matrix["columns"]), (1850, 5477))
        self.assertEqual(matrix["uniqueReadoutIds"], 1850)
        self.assertIn("candidate UniProt", matrix["readoutIdObservedFormat"])
        self.assertEqual(
            matrix["readoutIdPrefixCounts"],
            {"A": 1, "D": 1, "O": 7, "P": 1422, "Q": 419},
        )
        self.assertIs(matrix["sampleHeaderOrderExactlyMatchesMetadata"], True)
        self.assertEqual(matrix["missingToken"], "NA")
        self.assertEqual(matrix["missingCells"], 255715)
        self.assertEqual(matrix["quotedSampleHeadersContainingComma"], 1)
        self.assertEqual(
            {item["localSha256"] for item in probe["files"]},
            {
                "48864282c82d516ae929dc87aff7fae9e05e9b922e316c001f3d29dce0ff878b",
                "ca7c8f2ac33272df3763807add7b8982b8a8b52d4276bd929a61ecf19e0ae405",
                "69a9df05b6db011f595a4e0b3ce25c1cc247f22cbdd066c79e6da9a706aa1df9",
                "4078289dc86dd6b526d9b0c963e6df61d53acdfdf6260abdeae307588623f828",
            },
        )
        self.assertTrue(probe["unresolved"])
        identity_probe = manifest["rawIdentityJoinProbe"]
        self.assertEqual(
            identity_probe["mappingReleaseId"],
            "slp-sgd-map:2026-08-28-object-set-v1",
        )
        interventions = identity_probe["interventionIdentities"]
        self.assertEqual(interventions["exactCaseUniqueInputIds"], 4550)
        self.assertEqual(interventions["exactCurrentRows"], 4623)
        self.assertEqual(interventions["exactCurrentOneToOne"], 4476)
        self.assertEqual(interventions["strictQuarantineRows"], 76)
        self.assertEqual(interventions["strictQuarantineBeforeSourceIntersection"], 74)
        self.assertIn("including exact YAL043C-a", interventions["policy"])
        self.assertIn("including YML009c", interventions["policy"])
        readouts = identity_probe["readoutIdentities"]
        self.assertEqual(readouts["exactTypedAccessionsCovered"], 1850)
        self.assertEqual(readouts["oneToOneCurrentOrfRelations"], 1845)
        self.assertEqual(readouts["oneToManyCurrentOrfRelations"], 5)
        self.assertEqual(
            [item["accession"] for item in readouts["oneToManyAccessions"]],
            [
                "UniProtKB:P02309",
                "UniProtKB:P02994",
                "UniProtKB:P10081",
                "UniProtKB:P32324",
                "UniProtKB:P61830",
            ],
        )
        target = manifest["molecularTargetProtocol"]
        self.assertEqual(
            target["outputValue"]["transform"],
            "log2 of each observed input intensity with no pseudocount",
        )
        self.assertEqual(target["outputValue"]["additionalCentering"], "none")
        self.assertEqual(target["sampleRoles"]["qc"],
                         "analytical-QC-only-excluded-from-world-records")
        self.assertIn("WT controls only", target["basalProfile"]["fittingDataUsed"])
        self.assertIn(
            "all knockout and WT",
            target["prohibited"][0],
        )
        self._assert_rights(
            manifest["rights"], "10.17632/w8jtmnszd9.2"
        )

    def test_top_level_registry_keeps_candidates_contract_blocked(self) -> None:
        registry = self._load("sources/yeast-v1.yaml")
        sources = {source["id"]: source for source in registry["sources"]}
        expected = {
            "zenodo-14062629-yeast-transcriptome":
                "sources/yeast-single-cell-atlas-v1.yaml",
            "mendeley-w8jtmnszd9-v2-yeast-proteome":
                "sources/yeast-proteome-v2.yaml",
        }
        for source_id, manifest_path in expected.items():
            source = sources[source_id]
            self.assertEqual(source["status"], "contract-blocked")
            self.assertEqual(source["manifest"], manifest_path)
            self.assertNotIn("snapshot", source)
            self.assertNotIn("revision", source)
        mapping = sources["sgd-stable-id-mapping-2026-08-28"]
        self.assertEqual(
            mapping["status"],
            "raw-snapshot-admitted-normalized-map-produced",
        )
        self.assertEqual(
            mapping["manifest"],
            "sources/sgd-stable-id-mapping-2026-08-28.yaml",
        )


if __name__ == "__main__":
    unittest.main()
