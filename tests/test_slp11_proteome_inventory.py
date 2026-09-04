"""Synthetic, outcome-blind tests for the yeast proteome identity adapter."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "modules" / "slp-1-1-proteome-inventory"
SPEC = importlib.util.spec_from_file_location(
    "slp11_proteome_inventory", MODULE_ROOT / "inventory.py"
)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)

HELD_ROOT = ROOT / "modules" / "slp-1-1-held-roster"
HELD_SPEC = importlib.util.spec_from_file_location(
    "slp11_held_roster_for_proteome_test", HELD_ROOT / "roster.py"
)
assert HELD_SPEC is not None and HELD_SPEC.loader is not None
held_roster = importlib.util.module_from_spec(HELD_SPEC)
sys.modules[HELD_SPEC.name] = held_roster
HELD_SPEC.loader.exec_module(held_roster)


class ProteomeInventoryTest(unittest.TestCase):
    MAPPING_ID = "sgd:fixture-2026-08-28"
    ARTIFACT_DIGESTS = {
        "sgdCurrentOrfs": "sha256:" + "1" * 64,
        "sgdExternalRelations": "sha256:" + "2" * 64,
        "sgdRetiredQuarantine": "sha256:" + "3" * 64,
        "sgdMappingManifest": "sha256:" + "4" * 64,
    }
    PRODUCTION_ARTIFACT_DIGESTS = {
        "sgdCurrentOrfs": "sha256:e67f0e8773feae108ecdb687139885e01ca972ff4aec95cd1358b33db1ea1192",
        "sgdExternalRelations": "sha256:75e0fef99bbae3bb4e4dc3e2f24cfd0ab62919c0e6e3e321e8d82f3bd557f4da",
        "sgdRetiredQuarantine": "sha256:07ea82f877224496c24effc2aa2a2b684c01b85017e616a70f003a5363f6925f",
        "sgdMappingManifest": "sha256:c74ea81ce604357b998e5f09130dff85bf8a7a26504b9b2426f8038608c52d9c",
    }

    @staticmethod
    def _canonical(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, object]]) -> dict[str, object]:
        path.write_text(
            "".join(ProteomeInventoryTest._canonical(record) + "\n" for record in records),
            encoding="utf-8",
            newline="\n",
        )
        return {
            "name": path.name,
            "records": len(records),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    @staticmethod
    def _current(curie: str, systematic: str) -> dict[str, object]:
        return {
            "schema": "slp.sgd-current-orf/v1",
            "ncbiTaxon": 4932,
            "canonicalSgdCurie": curie,
            "systematicName": systematic,
            "featureQualifier": None,
            "secondaryIdentifiers": [],
            "secondaryIdentifiersResolve": False,
            "displayMetadata": {
                "standardGeneName": "display-only",
                "aliases": [],
                "resolvesIdentity": False,
            },
        }

    @staticmethod
    def _relation(accession: str, targets: list[str]) -> dict[str, object]:
        return {
            "schema": "slp.sgd-external-accession-relation/v1",
            "ncbiTaxon": 4932,
            "typedAccession": {
                "value": accession,
                "source": "UniProtKB",
                "type": "UniProtKB ID",
                "caseNormalization": "none",
                "namespaceInferred": False,
            },
            "relationOnly": True,
            "targetCount": len(targets),
            "targets": [
                {
                    "canonicalSgdCurie": curie,
                    "targetStatus": "current-orf",
                    "assertions": [],
                }
                for curie in targets
            ],
        }

    @staticmethod
    def _raw_specs(raw: Path) -> tuple[inventory.FileSpec, ...]:
        return tuple(
            inventory.FileSpec(
                name,
                (raw / name).stat().st_size,
                hashlib.sha256((raw / name).read_bytes()).hexdigest(),
            )
            for name in (
                "yeast5k_noimpute_wide.csv",
                "yeast5k_metadata.csv",
                "Detection_of_KO_proteins.csv",
                "summary_fileupload.pdf",
            )
        )

    def _fixture(self, root: Path, *, matrix_tail: bytes = b"\xff") -> dict[str, object]:
        raw = root / "raw"
        mapping = root / "mapping"
        raw.mkdir(parents=True)
        mapping.mkdir()

        metadata_rows = [
            ["sample,one", "1", "1", "1", "ko", "YAL043C-a"],
            ["sample2", "2", "2", "1", "ko", "YAL043C-a"],
            ["sample3", "3", "3", "1", "ko", "YRET"],
            ["sample4", "4", "4", "1", "ko", "YML009c"],
            ["sample5", "5", "5", "1", "ko", "YUNKNOWN"],
            ["sample6", "6", "6", "1", "ko", "YAMB"],
            ["sample7", "7", "7", "1", "HIS3", "YOR202W"],
            ["sample8", "8", "8", "1", "qc", ""],
        ]
        with (raw / "yeast5k_metadata.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(inventory.METADATA_COLUMNS)
            writer.writerows(metadata_rows)

        header_buffer = io.StringIO(newline="")
        csv.writer(header_buffer, lineterminator="\n").writerow(
            ["Protein.Group", *(row[0] for row in metadata_rows)]
        )
        matrix = bytearray(header_buffer.getvalue().encode("utf-8"))
        opaque_values = b",".join([matrix_tail] * len(metadata_rows))
        matrix.extend(b"PONE," + opaque_values + b"\n")
        matrix.extend(b"PAMB," + opaque_values + b"\n")
        (raw / "yeast5k_noimpute_wide.csv").write_bytes(bytes(matrix))
        (raw / "Detection_of_KO_proteins.csv").write_bytes(b"\xffopaque detection values\n")
        (raw / "summary_fileupload.pdf").write_bytes(b"%PDF synthetic identity fixture\n")

        current = [
            self._current("SGD:S000000001", "YAL043C-a"),
            self._current("SGD:S000000002", "YAMB"),
            self._current("SGD:S000000003", "YAMB"),
        ]
        external = [
            self._relation("PONE", ["SGD:S000000001"]),
            self._relation("PAMB", ["SGD:S000000002", "SGD:S000000003"]),
        ]
        retired = [
            {
                "schema": "slp.sgd-retired-quarantine/v1",
                "recordKind": "retired-or-merged",
                "sourceLine": 1,
                "ncbiTaxon": 4932,
                "systematicName": "YRET",
                "status": "Deleted",
                "automaticRedirectAllowed": False,
            }
        ]
        mapping_paths = {
            "sgdCurrentOrfs": mapping / "current-orfs.jsonl",
            "sgdExternalRelations": mapping / "external-accessions.jsonl",
            "sgdRetiredQuarantine": mapping / "retired-merged-quarantine.jsonl",
            "sgdMappingManifest": mapping / "mapping-manifest.json",
        }
        outputs = [
            self._write_jsonl(mapping_paths["sgdCurrentOrfs"], current),
            self._write_jsonl(mapping_paths["sgdExternalRelations"], external),
            self._write_jsonl(mapping_paths["sgdRetiredQuarantine"], retired),
        ]
        digest_basis = {
            "schema": "slp.sgd-stable-id-mapping-digest/v1",
            "identityMappingId": self.MAPPING_ID,
            "ncbiTaxon": 4932,
            "inputFiles": [],
            "outputFiles": outputs,
            "normalizationPolicy": {
                "systematicNameMatch": "exact-case-sensitive",
                "displayMetadataResolvesIdentity": False,
                "externalKey": ["value", "source", "type"],
                "externalRelationsChooseFirst": False,
                "retiredIdentifiersAutoRedirect": False,
                "benchmarkFieldsAllowed": False,
            },
        }
        mapping_sha = hashlib.sha256(
            (self._canonical(digest_basis) + "\n").encode("utf-8")
        ).hexdigest()
        mapping_manifest = {
            "schema": "slp.sgd-stable-id-mapping/v1",
            "identityMappingId": self.MAPPING_ID,
            "identityMappingSha256": mapping_sha,
            "ncbiTaxon": 4932,
            "digestBasis": digest_basis,
        }
        mapping_paths["sgdMappingManifest"].write_text(
            json.dumps(mapping_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest_sha = hashlib.sha256(
            mapping_paths["sgdMappingManifest"].read_bytes()
        ).hexdigest()
        expected_counts = {
            "metadataRows": 8,
            "knockoutRows": 6,
            "controlRows": 1,
            "analyticalQcRows": 1,
            "eligibleKnockoutRows": 2,
            "eligibleInterventions": 1,
            "quarantineRows": 4,
            "quarantineUniqueRawIds": 4,
            "retiredOrMergedRows": 1,
            "unmatchedRows": 3,
            "proteinRecords": 2,
            "oneToOneProteinRelations": 1,
            "oneToManyProteinRelations": 1,
        }
        return {
            "raw": raw,
            "mapping": mapping_paths,
            "raw_specs": self._raw_specs(raw),
            "mapping_sha": mapping_sha,
            "manifest_sha": manifest_sha,
            "expected_counts": expected_counts,
        }

    def _build(self, fixture: dict[str, object], destination: Path) -> dict[str, object]:
        return inventory.build_inventory(
            fixture["raw"],
            fixture["mapping"],
            destination,
            inventory.Bounds(),
            raw_specs=fixture["raw_specs"],
            mapping_artifact_digests=self.ARTIFACT_DIGESTS,
            expected_mapping_manifest_sha256=fixture["manifest_sha"],
            expected_mapping_id=self.MAPPING_ID,
            expected_mapping_sha256=fixture["mapping_sha"],
            expected_counts=fixture["expected_counts"],
            expected_one_to_many={
                "UniProtKB:PAMB": ("SGD:S000000002", "SGD:S000000003")
            },
            provenance=inventory.SourceProvenance(
                resource="omf://fixture/datasetsnapshot/proteome@sha256:" + "a" * 64,
                revision="sha256:" + "a" * 64,
                manifest_digest="sha256:" + "b" * 64,
                mapping_artifacts=self.ARTIFACT_DIGESTS,
            ),
        )

    def test_deterministic_held_compatible_and_preserves_relations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_fixture = self._fixture(root / "first")
            second_fixture = self._fixture(root / "second")
            first = root / "output-one"
            second = root / "output-two"
            report = self._build(first_fixture, first)
            self._build(second_fixture, second)

            for relative in (
                "intervention-inventory/inventory.json",
                "intervention-inventory/interventions.jsonl",
                "protein-relations/manifest.json",
                "protein-relations/relations.jsonl",
                "audit.json",
            ):
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())

            loaded = held_roster.load_inventory(
                first / "intervention-inventory",
                held_roster.RosterBounds(minimum_intersection_size=1),
            )
            self.assertEqual(loaded.records, 2)
            self.assertEqual(loaded.duplicate_records, 1)
            self.assertEqual(loaded.qc_passing, frozenset({"SGD:S000000001"}))

            relations = [
                json.loads(line)
                for line in (first / "protein-relations/relations.jsonl").read_text().splitlines()
            ]
            shared = next(item for item in relations if item["proteinId"] == "UniProtKB:PAMB")
            self.assertEqual(
                shared["currentOrfRelations"],
                ["SGD:S000000002", "SGD:S000000003"],
            )
            self.assertIs(shared["chooseFirstAllowed"], False)

            audit = json.loads((first / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(
                audit["identityMapping"]["artifactManifestDigests"], self.ARTIFACT_DIGESTS
            )
            self.assertEqual(
                audit["identityMapping"]["mappingManifestContentSha256"],
                first_fixture["manifest_sha"],
            )
            quarantined = {
                row["rawInterventionId"]: row["reason"] for row in audit["quarantineRows"]
            }
            self.assertNotIn("YAL043C-a", quarantined)
            self.assertEqual(quarantined["YRET"], "retired-or-merged-exact-systematic-name")
            self.assertEqual(quarantined["YML009c"], "mixed-case-not-current-exact")
            self.assertEqual(quarantined["YAMB"], "ambiguous-exact-current-systematic-name")
            self.assertEqual(report["eligibleKnockoutRows"], 2)

    def test_quantitative_bytes_are_never_decoded_or_reflected_in_identity_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_fixture = self._fixture(root / "first", matrix_tail=b"\xff")
            second_fixture = self._fixture(root / "second", matrix_tail=b"\xfe")
            first = root / "out-one"
            second = root / "out-two"
            self._build(first_fixture, first)
            self._build(second_fixture, second)
            for relative in (
                "intervention-inventory/inventory.json",
                "intervention-inventory/interventions.jsonl",
                "protein-relations/manifest.json",
                "protein-relations/relations.jsonl",
            ):
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
            audit = json.loads((first / "audit.json").read_text())
            self.assertIs(audit["accessBoundary"]["matrixQuantitativeFieldsDecoded"], False)
            self.assertIs(audit["accessBoundary"]["matrixQuantitativeValuesParsed"], False)

    def test_digest_file_set_and_mapping_overlap_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root / "digest")
            (fixture["raw"] / "yeast5k_metadata.csv").write_bytes(b"drift")
            with self.assertRaisesRegex(inventory.ProteomeInventoryError, "byte drift|digest drift"):
                self._build(fixture, root / "out-drift")

            extra = self._fixture(root / "extra")
            (extra["raw"] / "unexpected.txt").write_text("no")
            with self.assertRaisesRegex(inventory.ProteomeInventoryError, "file set"):
                self._build(extra, root / "out-extra")

            overlap = self._fixture(root / "overlap")
            retired_path = overlap["mapping"]["sgdRetiredQuarantine"]
            retired_record = json.loads(retired_path.read_text().splitlines()[0])
            retired_record["systematicName"] = "YAL043C-a"
            retired_spec = self._write_jsonl(retired_path, [retired_record])
            mapping_manifest_path = overlap["mapping"]["sgdMappingManifest"]
            manifest = json.loads(mapping_manifest_path.read_text())
            manifest["digestBasis"]["outputFiles"][2] = retired_spec
            overlap["mapping_sha"] = hashlib.sha256(
                (self._canonical(manifest["digestBasis"]) + "\n").encode()
            ).hexdigest()
            manifest["identityMappingSha256"] = overlap["mapping_sha"]
            mapping_manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            overlap["manifest_sha"] = hashlib.sha256(mapping_manifest_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(inventory.ProteomeInventoryError, "both current and retired"):
                self._build(overlap, root / "out-overlap")

    def test_artifact_provenance_mismatch_and_mapping_content_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root / "fixture")
            bad = dict(self.ARTIFACT_DIGESTS)
            bad["sgdRetiredQuarantine"] = "sha256:" + "9" * 64
            with self.assertRaisesRegex(inventory.ProteomeInventoryError, "resolved SGD artifacts"):
                inventory.build_inventory(
                    fixture["raw"],
                    fixture["mapping"],
                    root / "out-provenance",
                    inventory.Bounds(),
                    raw_specs=fixture["raw_specs"],
                    mapping_artifact_digests=bad,
                    expected_mapping_manifest_sha256=fixture["manifest_sha"],
                    expected_mapping_id=self.MAPPING_ID,
                    expected_mapping_sha256=fixture["mapping_sha"],
                    provenance=inventory.SourceProvenance(
                        resource="omf://fixture/datasetsnapshot/proteome@sha256:" + "a" * 64,
                        revision="sha256:" + "a" * 64,
                        manifest_digest="sha256:" + "b" * 64,
                        mapping_artifacts=self.ARTIFACT_DIGESTS,
                    ),
                )

            (fixture["mapping"]["sgdCurrentOrfs"]).write_bytes(
                fixture["mapping"]["sgdCurrentOrfs"].read_bytes() + b"\n"
            )
            with self.assertRaisesRegex(inventory.ProteomeInventoryError, "does not match"):
                self._build(fixture, root / "out-map-drift")

    def test_literal_input_resolvers_accept_only_omf_materialized_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_resource_name = "mendeley-w8jtmnszd9-v2-fixture"
            raw = root / "inputs" / "rawProteome" / raw_resource_name
            raw.mkdir(parents=True)
            dataset = {
                "resource": (
                    "omf://fixture/datasetsnapshot/"
                    + raw_resource_name
                    + "@sha256:"
                    + "a" * 64
                ),
                "mode": "copy",
                "path": str(raw),
                "manifestDigest": "sha256:" + "b" * 64,
            }
            resolved = inventory.resolve_pinned_raw_dataset(dataset)
            self.assertEqual(resolved.path, raw.resolve())
            with self.assertRaisesRegex(inventory.ProteomeInventoryError, "spoofed"):
                inventory.resolve_pinned_raw_dataset({**dataset, "extra": True})
            with self.assertRaisesRegex(inventory.ProteomeInventoryError, "copied"):
                inventory.resolve_pinned_raw_dataset({**dataset, "mode": "reference"})

            artifact_path = root / "inputs" / "sgdCurrentOrfs" / "payload" / "payload"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_text("payload")
            digest = self.ARTIFACT_DIGESTS["sgdCurrentOrfs"]
            artifact = {
                "resource": f"artifact:{digest}",
                "kind": "artifact",
                "artifacts": {"payload": digest},
                "paths": {"payload": str(artifact_path)},
                "path": str(artifact_path),
            }
            resolved_artifact = inventory.resolve_literal_artifact(
                artifact, "sgdCurrentOrfs", digest
            )
            self.assertEqual(resolved_artifact.path, artifact_path.resolve())
            legacy_path = root / "inputs" / "sgdCurrentOrfs" / "payload-legacy"
            legacy_path.write_text("payload")
            with self.assertRaisesRegex(
                inventory.ProteomeInventoryError, "OMF materialization"
            ):
                inventory.resolve_literal_artifact(
                    {
                        **artifact,
                        "paths": {"payload": str(legacy_path)},
                        "path": str(legacy_path),
                    },
                    "sgdCurrentOrfs",
                    digest,
                )
            with self.assertRaisesRegex(inventory.ProteomeInventoryError, "does not match"):
                inventory.resolve_literal_artifact(
                    {**artifact, "resource": "artifact:sha256:" + "f" * 64},
                    "sgdCurrentOrfs",
                    digest,
                )

    def test_module_and_workload_pin_four_artifacts_and_no_benchmark(self) -> None:
        self.assertEqual(inventory.MAPPING_ARTIFACT_DIGESTS, self.PRODUCTION_ARTIFACT_DIGESTS)
        self.assertEqual(
            inventory.MAPPING_MANIFEST_SHA256,
            "570557ab1201913a18de9790f8adc5ee2e3cb56c6bb0e8d588fe43660c0214e1",
        )
        self.assertEqual(inventory.EXPECTED_COUNTS["eligibleKnockoutRows"], 4623)
        self.assertEqual(inventory.EXPECTED_COUNTS["eligibleInterventions"], 4476)
        self.assertEqual(inventory.EXPECTED_COUNTS["quarantineRows"], 76)
        self.assertEqual(inventory.EXPECTED_COUNTS["retiredOrMergedRows"], 35)
        self.assertEqual(inventory.EXPECTED_COUNTS["unmatchedRows"], 41)
        module = yaml.safe_load((MODULE_ROOT / "module.yaml").read_text(encoding="utf-8"))
        required = set(module["spec"]["contracts"]["input"]["required"])
        self.assertEqual(required, {"rawProteome", *self.ARTIFACT_DIGESTS})
        self.assertEqual(
            module["spec"]["environment"]["dependencyDigest"],
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        self.assertEqual((MODULE_ROOT / "requirements.lock").read_bytes(), b"")

        workload_text = (
            ROOT / "workloads" / "slp-1-1-proteome-inventory.yaml.tmpl"
        ).read_text(encoding="utf-8")
        self.assertIn("@@RAW_PROTEOME_DATASET@@", workload_text)
        for digest in self.PRODUCTION_ARTIFACT_DIGESTS.values():
            self.assertIn(f"artifact:{digest}", workload_text)
        self.assertNotIn("benchmark", workload_text.casefold())
        for artifact_name in (
            "proteomeInterventionInventory",
            "proteomeInterventionRecords",
            "proteomeProteinRelations",
            "proteomeProteinRelationRecords",
            "proteomeIdentityAudit",
        ):
            self.assertIn(f"- {artifact_name}", workload_text)
        main_text = (MODULE_ROOT / "main.py").read_text(encoding="utf-8")
        for relative_file in (
            "intervention-inventory/inventory.json",
            "intervention-inventory/interventions.jsonl",
            "protein-relations/manifest.json",
            "protein-relations/relations.jsonl",
            "proteome-inventory/audit.json",
        ):
            self.assertIn(relative_file, main_text)
        self.assertNotIn('"path": "proteome-inventory/intervention-inventory",', main_text)
        self.assertNotIn('"path": "proteome-inventory/protein-relations",', main_text)


if __name__ == "__main__":
    unittest.main()
