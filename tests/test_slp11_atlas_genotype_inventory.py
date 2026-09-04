"""Synthetic tests for the outcome-blind atlas genotype identity adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "modules" / "slp-1-1-atlas-genotype-inventory"
SPEC = importlib.util.spec_from_file_location(
    "slp11_atlas_genotype_inventory", MODULE_ROOT / "inventory.py"
)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)

HELD_ROOT = ROOT / "modules" / "slp-1-1-held-roster"
HELD_SPEC = importlib.util.spec_from_file_location(
    "slp11_held_roster_for_atlas_test", HELD_ROOT / "roster.py"
)
assert HELD_SPEC is not None and HELD_SPEC.loader is not None
held_roster = importlib.util.module_from_spec(HELD_SPEC)
sys.modules[HELD_SPEC.name] = held_roster
HELD_SPEC.loader.exec_module(held_roster)


class FakeSeries:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def tolist(self) -> list[object]:
        return list(self.values)


class FakeFrame:
    def __init__(
        self,
        assignments: list[object],
        cell_numbers: list[object],
        phenotype_token: object,
        *,
        columns: tuple[str, ...] = inventory.FRAME_COLUMNS,
    ) -> None:
        self.columns = columns
        self.shape = (len(assignments), len(columns))
        self._selected = {
            "assignment_consensus2": FakeSeries(assignments),
            "cell_number": FakeSeries(cell_numbers),
        }
        self.phenotype_token = phenotype_token

    def __getitem__(self, name: str) -> FakeSeries:
        if name not in self._selected:
            raise AssertionError(f"phenotype column was accessed: {name}")
        return self._selected[name]


class AtlasGenotypeInventoryTest(unittest.TestCase):
    MAPPING_ID = "sgd:atlas-fixture-2026-08-28"
    ARTIFACT_DIGESTS = {
        "sgdCurrentOrfs": "sha256:" + "1" * 64,
        "sgdExternalRelations": "sha256:" + "2" * 64,
        "sgdRetiredQuarantine": "sha256:" + "3" * 64,
        "sgdMappingManifest": "sha256:" + "4" * 64,
    }
    EXPECTED_COUNTS = {
        "controlRows": 7,
        "controlNonWildType": 6,
        "naclRows": 7,
        "naclNonWildType": 6,
        "allAssignmentIntersectionIncludingWildType": 6,
        "allAssignmentUnionIncludingWildType": 8,
        "candidateNonWildTypeIntersection": 5,
        "candidateNonWildTypeUnion": 7,
        "exactCurrentCandidateAssignments": 2,
        "uniqueCurrentInterventions": 2,
        "retiredOrMergedCandidateAssignments": 1,
        "unmatchedCandidateAssignments": 1,
        "ambiguousCurrentCandidateAssignments": 1,
    }

    @staticmethod
    def _canonical(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, object]]) -> dict[str, object]:
        path.write_text(
            "".join(AtlasGenotypeInventoryTest._canonical(record) + "\n" for record in records),
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
            "secondaryIdentifiersResolve": False,
            "displayMetadata": {"resolvesIdentity": False},
        }

    @staticmethod
    def _parsed(phenotype_token: object = 0) -> dict[str, object]:
        shared = ["WT", "bc-YA", "bc-YB", "bc-YRET", "bc-YUNK", "bc-YAMB"]
        return {
            "ptbs": {
                "control": FakeFrame(
                    [*shared, "bc-YONLYCONTROL"],
                    [6, 7, 8, 9, 10, 11, 12],
                    phenotype_token,
                ),
                "nacl": FakeFrame(
                    [*shared, "bc-YONLYNACL"],
                    [13, 14, 15, 16, 17, 18, 19],
                    phenotype_token,
                ),
            }
        }

    def _fixture(self, root: Path, phenotype_token: object = 0) -> dict[str, object]:
        raw = root / "raw"
        mapping = root / "mapping"
        raw.mkdir(parents=True)
        mapping.mkdir()
        raw_file = raw / inventory.RAW_FILE_NAME
        raw_file.write_bytes(b"synthetic RData bytes\n")
        raw_spec = inventory.FileSpec(
            inventory.RAW_FILE_NAME,
            raw_file.stat().st_size,
            hashlib.sha256(raw_file.read_bytes()).hexdigest(),
            hashlib.md5(raw_file.read_bytes(), usedforsecurity=False).hexdigest(),
        )

        current = [
            self._current("SGD:S000000001", "YA"),
            self._current("SGD:S000000002", "YB"),
            self._current("SGD:S000000003", "YAMB"),
            self._current("SGD:S000000004", "YAMB"),
        ]
        retired = [
            {
                "schema": "slp.sgd-retired-quarantine/v1",
                "recordKind": "retired-or-merged",
                "ncbiTaxon": 4932,
                "systematicName": "YRET",
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
            self._write_jsonl(mapping_paths["sgdExternalRelations"], []),
            self._write_jsonl(mapping_paths["sgdRetiredQuarantine"], retired),
        ]
        digest_basis = {
            "schema": "slp.sgd-stable-id-mapping-digest/v1",
            "identityMappingId": self.MAPPING_ID,
            "ncbiTaxon": 4932,
            "inputFiles": [],
            "outputFiles": outputs,
            "normalizationPolicy": {},
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
        return {
            "raw": raw,
            "raw_spec": raw_spec,
            "mapping": mapping_paths,
            "mapping_sha": mapping_sha,
            "manifest_sha": manifest_sha,
            "parsed": self._parsed(phenotype_token),
        }

    def _build(self, fixture: dict[str, object], destination: Path) -> dict[str, object]:
        return inventory.build_inventory(
            fixture["raw"],
            fixture["mapping"],
            destination,
            inventory.Bounds(),
            raw_spec=fixture["raw_spec"],
            mapping_artifact_digests=self.ARTIFACT_DIGESTS,
            expected_mapping_manifest_sha256=fixture["manifest_sha"],
            expected_mapping_id=self.MAPPING_ID,
            expected_mapping_sha256=fixture["mapping_sha"],
            expected_counts=self.EXPECTED_COUNTS,
            provenance=inventory.SourceProvenance(
                resource="omf://fixture/datasetsnapshot/atlas-summary@sha256:" + "a" * 64,
                revision="sha256:" + "a" * 64,
                manifest_digest="sha256:" + "b" * 64,
                mapping_artifacts=self.ARTIFACT_DIGESTS,
            ),
            rdata_loader=lambda _path: (fixture["parsed"], "1.1.0"),
        )

    def test_deterministic_held_compatible_and_quarantine_evidence(self) -> None:
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
                "identity-evidence/manifest.json",
                "identity-evidence/evidence.jsonl",
                "identity-evidence/quarantine.jsonl",
                "audit.json",
            ):
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())

            loaded = held_roster.load_inventory(
                first / "intervention-inventory",
                held_roster.RosterBounds(minimum_intersection_size=1),
            )
            self.assertEqual(
                loaded.qc_passing,
                frozenset({"SGD:S000000001", "SGD:S000000002"}),
            )
            self.assertEqual(loaded.duplicate_records, 0)
            quarantine = [
                json.loads(line)
                for line in (first / "identity-evidence/quarantine.jsonl").read_text().splitlines()
            ]
            reasons = {record["exactSystematicName"]: record["reason"] for record in quarantine}
            self.assertEqual(reasons["YRET"], "retired-or-merged-exact")
            self.assertEqual(reasons["YUNK"], "unmatched-exact-current")
            self.assertEqual(reasons["YAMB"], "ambiguous-exact-current")
            self.assertEqual(report["candidateNonWildTypeIntersection"], 5)
            self.assertEqual(report["uniqueCurrentInterventions"], 2)

            audit = json.loads((first / "audit.json").read_text())
            self.assertEqual(
                audit["identityMapping"]["artifactManifestDigests"], self.ARTIFACT_DIGESTS
            )
            self.assertIs(audit["phenotypeBoundary"]["phenotypeValuesUsed"], False)
            self.assertIs(
                audit["phenotypeBoundary"]["phenotypeValuesReadByAdapter"], False
            )
            self.assertNotIn("phenotypeValuesInterpreted", audit["phenotypeBoundary"])
            self.assertEqual(audit["counts"]["allAssignmentUnionIncludingWildType"], 8)
            self.assertEqual(audit["counts"]["candidateNonWildTypeUnion"], 7)

    def test_phenotype_changes_do_not_reach_any_identity_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_fixture = self._fixture(root / "first", phenotype_token={"values": [1, 2]})
            second_fixture = self._fixture(root / "second", phenotype_token={"values": [999]})
            first = root / "output-one"
            second = root / "output-two"
            self._build(first_fixture, first)
            self._build(second_fixture, second)
            for relative in (
                "intervention-inventory/inventory.json",
                "intervention-inventory/interventions.jsonl",
                "identity-evidence/manifest.json",
                "identity-evidence/evidence.jsonl",
                "identity-evidence/quarantine.jsonl",
                "audit.json",
            ):
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())

    def test_frame_contract_cell_threshold_and_literal_prefix_fail_closed(self) -> None:
        valid = self._parsed()
        inventory.extract_condition_identities(valid, inventory.Bounds())

        bad_columns = self._parsed()
        bad_columns["ptbs"]["control"] = FakeFrame(
            ["WT", "bc-YA"],
            [6, 7],
            0,
            columns=inventory.FRAME_COLUMNS[:-1],
        )
        with self.assertRaisesRegex(inventory.AtlasInventoryError, "nine-column"):
            inventory.extract_condition_identities(bad_columns, inventory.Bounds())

        for value in (5, None, 6.0, True):
            bad_cell = self._parsed()
            bad_cell["ptbs"]["nacl"]._selected["cell_number"].values[1] = value
            with self.assertRaisesRegex(inventory.AtlasInventoryError, "cell_number"):
                inventory.extract_condition_identities(bad_cell, inventory.Bounds())

        bad_prefix = self._parsed()
        bad_prefix["ptbs"]["control"]._selected["assignment_consensus2"].values[1] = "YA"
        with self.assertRaisesRegex(inventory.AtlasInventoryError, "literal bc-"):
            inventory.extract_condition_identities(bad_prefix, inventory.Bounds())

        with self.assertRaisesRegex(inventory.AtlasInventoryError, "exactly the ptbs"):
            inventory.extract_condition_identities({**self._parsed(), "extra": {}}, inventory.Bounds())

    def test_raw_mapping_and_provenance_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root / "raw-drift")
            (fixture["raw"] / inventory.RAW_FILE_NAME).write_bytes(b"drift")
            with self.assertRaisesRegex(inventory.AtlasInventoryError, "byte-count|SHA-256"):
                self._build(fixture, root / "out-raw")

            extra = self._fixture(root / "extra")
            (extra["raw"] / "extra").write_bytes(b"no")
            with self.assertRaisesRegex(inventory.AtlasInventoryError, "exactly"):
                self._build(extra, root / "out-extra")

            mapping = self._fixture(root / "mapping-drift")
            mapping["mapping"]["sgdCurrentOrfs"].write_bytes(
                mapping["mapping"]["sgdCurrentOrfs"].read_bytes() + b"\n"
            )
            with self.assertRaisesRegex(inventory.AtlasInventoryError, "does not match"):
                self._build(mapping, root / "out-mapping")

            provenance = self._fixture(root / "provenance")
            wrong = dict(self.ARTIFACT_DIGESTS)
            wrong["sgdRetiredQuarantine"] = "sha256:" + "9" * 64
            with self.assertRaisesRegex(inventory.AtlasInventoryError, "resolved SGD artifacts"):
                inventory.build_inventory(
                    provenance["raw"],
                    provenance["mapping"],
                    root / "out-provenance",
                    inventory.Bounds(),
                    raw_spec=provenance["raw_spec"],
                    mapping_artifact_digests=wrong,
                    provenance=inventory.SourceProvenance(
                        resource="omf://fixture/datasetsnapshot/atlas@sha256:" + "a" * 64,
                        revision="sha256:" + "a" * 64,
                        manifest_digest="sha256:" + "b" * 64,
                        mapping_artifacts=self.ARTIFACT_DIGESTS,
                    ),
                    rdata_loader=lambda _path: (provenance["parsed"], "1.1.0"),
                )

    def test_literal_materialized_input_shapes_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = (
                root
                / "inputs"
                / "rawAtlasSummary"
                / "slp-1-1-atlas-genotype-summary-raw-v1"
            )
            raw.mkdir(parents=True)
            dataset = {
                "resource": inventory.RAW_DATASET_RESOURCE,
                "mode": "copy",
                "path": str(raw),
                "manifestDigest": inventory.RAW_DATASET_MANIFEST_DIGEST,
            }
            self.assertEqual(inventory.resolve_pinned_raw_dataset(dataset).path, raw.resolve())
            with self.assertRaisesRegex(inventory.AtlasInventoryError, "spoofed"):
                inventory.resolve_pinned_raw_dataset({**dataset, "kind": "DatasetSnapshot"})
            with self.assertRaisesRegex(inventory.AtlasInventoryError, "copied"):
                inventory.resolve_pinned_raw_dataset({**dataset, "mode": "reference"})
            with self.assertRaisesRegex(inventory.AtlasInventoryError, "admitted raw atlas"):
                inventory.resolve_pinned_raw_dataset(
                    {**dataset, "resource": dataset["resource"].replace("c7cad889", "f7cad889")}
                )
            with self.assertRaisesRegex(inventory.AtlasInventoryError, "outer manifest"):
                inventory.resolve_pinned_raw_dataset(
                    {**dataset, "manifestDigest": "sha256:" + "f" * 64}
                )

            artifact_path = root / "inputs" / "sgdCurrentOrfs" / "payload"
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
            self.assertEqual(
                inventory.resolve_literal_artifact(artifact, "sgdCurrentOrfs", digest).path,
                artifact_path.resolve(),
            )
            with self.assertRaisesRegex(inventory.AtlasInventoryError, "does not match"):
                inventory.resolve_literal_artifact(
                    {**artifact, "resource": "artifact:sha256:" + "f" * 64},
                    "sgdCurrentOrfs",
                    digest,
                )

    def test_runtime_lock_workload_and_source_contract_are_frozen(self) -> None:
        lock = (MODULE_ROOT / "requirements.lock").read_bytes()
        self.assertEqual(
            hashlib.sha256(lock).hexdigest(),
            "fddb8df4097a73fa51e0a00835971f23ec060780f28af74a628a6da1ff445c20",
        )
        lock_text = lock.decode("ascii")
        lines = lock_text.splitlines()
        self.assertEqual(len(lines), 20)
        self.assertTrue(all(line.endswith(" \\") for line in lines[0::2]))
        packages = [tuple(line[:-2].split("==", 1)) for line in lines[0::2]]
        self.assertEqual(
            packages,
            [
                ("numpy", "2.2.6"),
                ("packaging", "26.3"),
                ("pandas", "2.3.2"),
                ("python-dateutil", "2.9.0.post0"),
                ("pytz", "2026.3.post1"),
                ("rdata", "1.1.0"),
                ("six", "1.17.0"),
                ("typing-extensions", "4.16.0"),
                ("tzdata", "2026.3"),
                ("xarray", "2026.7.0"),
            ],
        )
        self.assertEqual(lock_text.count("--hash=sha256:"), 10)
        self.assertNotIn("file:", lock_text)
        self.assertNotIn("C:\\", lock_text)

        main_text = (MODULE_ROOT / "main.py").read_text()
        self.assertIn("from omf_protocol import", main_text)
        self.assertNotIn("from omf.sdk import", main_text)
        protocol_spec = importlib.util.spec_from_file_location(
            "atlas_omf_protocol", MODULE_ROOT / "omf_protocol.py"
        )
        assert protocol_spec is not None and protocol_spec.loader is not None
        protocol_module = importlib.util.module_from_spec(protocol_spec)
        sys.modules[protocol_spec.name] = protocol_module
        protocol_spec.loader.exec_module(protocol_module)
        request = protocol_module.ProtocolRequest.from_bytes(
            b'{"operation":"validate","protocol":"omf.module/v1"}'
        )
        self.assertEqual(request.operation, "validate")

        module = yaml.safe_load((MODULE_ROOT / "module.yaml").read_text())
        self.assertEqual(
            module["spec"]["environment"]["dependencyDigest"],
            "sha256:" + hashlib.sha256(lock).hexdigest(),
        )
        required = set(module["spec"]["contracts"]["input"]["required"])
        self.assertEqual(required, {"rawAtlasSummary", *inventory.MAPPING_ARTIFACT_DIGESTS})

        workload = (
            ROOT / "workloads" / "slp-1-1-atlas-genotype-inventory.yaml"
        ).read_text()
        self.assertIn(inventory.RAW_DATASET_RESOURCE, workload)
        for digest in inventory.MAPPING_ARTIFACT_DIGESTS.values():
            self.assertIn(f"artifact:{digest}", workload)
        self.assertNotIn("seus_split.RData", workload)
        self.assertNotIn("benchmark", workload.casefold())

        source = yaml.safe_load(
            (ROOT / "sources" / "yeast-single-cell-atlas-v1.yaml").read_text()
        )
        summary = source["identityOnlyAllowlist"][0]
        self.assertEqual(summary["name"], "ptb_summary.Rdata")
        self.assertEqual(summary["bytes"], 345032)
        self.assertEqual(
            summary["localSha256"],
            "01c2d54ac838179be29694ed300cb17edac47dd4db23a4018407546e0651b165",
        )
        quantitative = source["allowlist"][0]
        self.assertEqual(quantitative["name"], "seus_split.RData")
        self.assertEqual(quantitative["bytes"], 5907877873)
        self.assertEqual(
            quantitative["localSha256"],
            "da99869c11d1a6c034454568098aa50bc3313cd4508dbd506d43241b0fb4695d",
        )
        quantitative_probe = next(
            item
            for item in source["metadataProbe"]["files"]
            if item["name"] == "seus_split.RData"
        )
        self.assertIs(quantitative_probe["gzipIntegrityVerified"], True)
        self.assertEqual(quantitative_probe["decompressedSerializationBytes"], 21596869016)
        self.assertEqual(quantitative_probe["localRamEnvelopeBytes"], 16106127360)
        probe = source["metadataProbe"]["genotypeIdentityProbe"]
        self.assertEqual(probe["counts"]["allAssignmentUnionIncludingWildType"], 3259)
        self.assertEqual(probe["counts"]["candidateNonWildTypeUnion"], 3258)
        self.assertIs(probe["phenotypeValuesUsed"], False)
        self.assertEqual(
            source["admission"]["quantitativeAtlasSnapshot"]["status"], "contract-blocked"
        )
        self.assertEqual(
            source["admission"]["genotypeIdentitySnapshot"]["exactFileSet"],
            ["ptb_summary.Rdata"],
        )
        self.assertEqual(
            source["admission"]["genotypeIdentitySnapshot"]["resource"],
            inventory.RAW_DATASET_RESOURCE,
        )
        admission = source["admission"]["genotypeIdentitySnapshot"]
        self.assertEqual(admission["manifestDigest"], inventory.RAW_DATASET_MANIFEST_DIGEST)
        self.assertIs(admission["containsPhenotypeColumns"], True)
        self.assertIs(admission["adapterUsesPhenotypeValues"], False)
        self.assertIs(admission["protectedMolecularTruthSnapshot"], False)
        self.assertEqual(
            admission["prohibitedUses"], ["fitting", "reward", "molecular-evaluation"]
        )


if __name__ == "__main__":
    unittest.main()
