"""Adversarial tests for the fail-closed corpus audit v1.1."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-corpus-audit"
SPEC = importlib.util.spec_from_file_location("slp11_corpus_audit", MODULE / "audit.py")
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CorpusAuditTest(unittest.TestCase):
    MAPPING_ID = "TESTMAP:sgd-fixture-v1"
    MAPPING_SHA = hashlib.sha256(b"fixture mapping").hexdigest()

    @classmethod
    def setUpClass(cls) -> None:
        cls.by_role: dict[str, str] = {}
        for number in range(1, 10_000):
            identifier = f"SGD:S{number:09d}"
            role, _, _ = audit.assign_intervention(identifier)
            cls.by_role.setdefault(role, identifier)
            if len(cls.by_role) == 3:
                break
        assert len(cls.by_role) == 3

    @staticmethod
    def _snapshot_root(root: Path, input_name: str, resource_name: str) -> Path:
        directory = root / "stages" / "audit" / "inputs" / input_name / resource_name
        directory.mkdir(parents=True)
        return directory

    @staticmethod
    def _dataset_input(path: Path, input_name: str, resource_name: str) -> dict[str, object]:
        revision = hashlib.sha256((resource_name + " revision").encode()).hexdigest()
        manifest = hashlib.sha256((resource_name + " manifest").encode()).hexdigest()
        return {
            "resource": (
                f"omf://abiome/slp/datasetsnapshot/{resource_name}@sha256:{revision}"
            ),
            "mode": "copy",
            "path": str(path),
            "manifestDigest": f"sha256:{manifest}",
        }

    def _corpus(
        self,
        root: Path,
        input_name: str,
        role: str,
        genes: list[str],
        *,
        benchmark: bool = False,
    ) -> dict[str, object]:
        resource_name = f"fixture-{input_name}"
        directory = self._snapshot_root(root, input_name, resource_name)
        genes = sorted(set(genes))
        records = len(genes)
        entity_path = directory / "entities.npz"
        np.savez(
            entity_path,
            entity_id=np.asarray(genes, dtype="<U32"),
            entity_type=np.zeros(records, dtype=np.int64),
            entity_species_taxon=np.asarray([4932] * records, dtype=np.int64),
            entity_feature_value=np.ones((records, 1), dtype=np.float32),
            entity_feature_present=np.ones((records, 1), dtype=np.bool_),
        )
        query_path = directory / "queries.npz"
        np.savez(
            query_path,
            query_id=np.asarray(["TEST:query"], dtype="<U32"),
            query_entity_index=np.zeros(1, dtype=np.int64),
            query_readout_index=np.zeros(1, dtype=np.int64),
        )
        panel_path = directory / "panels.npz"
        np.savez(
            panel_path,
            panel_id=np.asarray(["TEST:panel"], dtype="<U32"),
            panel_indptr=np.asarray([0, 1], dtype=np.int64),
            panel_query_index=np.zeros(1, dtype=np.int64),
        )
        gene_path = directory / "trajectory-genes.txt"
        gene_path.write_bytes("".join(f"{gene}\n" for gene in genes).encode("ascii"))
        shard_path = directory / "shard-000.npz"
        np.savez(
            shard_path,
            record_id=np.asarray(
                [f"TEST:{input_name}-{index}" for index in range(records)], dtype="<U64"
            ),
            observation_unit_id=np.asarray(
                [f"TEST:observation-{index}" for index in range(records)], dtype="<U64"
            ),
            source_index=np.zeros(records, dtype=np.int64),
            replicate_id=np.asarray(
                [f"TEST:replicate-{index}" for index in range(records)], dtype="<U64"
            ),
            perturbation_id=np.asarray(
                [f"TEST:perturbation-{index}" for index in range(records)], dtype="<U64"
            ),
            species_taxon=np.full(records, 4932, dtype=np.int64),
            species_feature_value=np.ones((records, 1), dtype=np.float32),
            species_feature_present=np.ones((records, 1), dtype=np.bool_),
            context_entity_index=np.full((records, 1), -1, dtype=np.int64),
            context_type=np.full((records, 1), -1, dtype=np.int64),
            context_mask=np.zeros((records, 1), dtype=np.bool_),
            context_covariate_value=np.empty((records, 1, 0), dtype=np.float32),
            context_covariate_present=np.empty((records, 1, 0), dtype=np.bool_),
            record_covariate_value=np.empty((records, 0), dtype=np.float32),
            record_covariate_present=np.empty((records, 0), dtype=np.bool_),
            action_entity_index=np.arange(records, dtype=np.int64).reshape(records, 1),
            action_type=np.zeros((records, 1), dtype=np.int64),
            action_mask=np.ones((records, 1), dtype=np.bool_),
            action_covariate_value=np.empty((records, 1, 0), dtype=np.float32),
            action_covariate_present=np.empty((records, 1, 0), dtype=np.bool_),
            observation_covariate_value=np.empty((records, 0), dtype=np.float32),
            observation_covariate_present=np.empty((records, 0), dtype=np.bool_),
            query_panel_index=np.zeros(records, dtype=np.int64),
            target_indptr=np.arange(records + 1, dtype=np.int64),
            target_query_index=np.zeros(records, dtype=np.int64),
            target_value=np.ones(records, dtype=np.float32),
        )
        manifest = {
            "schema": "slp.corpus/v1.1",
            "datasetId": f"TEST:{input_name}",
            "version": "fixture-v1",
            "role": role,
            "labelClass": "molecular",
            "benchmarkLabelsPresent": benchmark,
            "rights": {
                "revision": "TESTRIGHTS:fixture-v1",
                "trainingAllowed": True,
                "redistributionAllowed": True,
            },
            "modalities": ["EFO:0002691"],
            "sources": [{"id": "TESTSOURCE:fixture"}],
            "sampling": {
                "scheme": "slp.source-intervention-replicate-record/v1",
                "sourceWeights": [1.0],
            },
            "species": [
                {"taxon": 4932, "featureValue": [1.0], "featurePresent": [True]}
            ],
            "featurePack": {
                "revision": "TESTFEATURE:fixture-v1",
                "sha256": "0" * 64,
                "entityFeatureDim": 1,
                "speciesFeatureDim": 1,
            },
            "entityTypes": ["SLPET:gene"],
            "contextTypes": ["SLPCTX:basal"],
            "actionTypes": ["SLPACT:deletion"],
            "covariates": {"record": [], "context": [], "action": [], "observation": []},
            "readoutTypes": [
                {
                    "id": "SLPRO:continuous-effect",
                    "likelihood": "gaussian",
                    "unit": "UCUM:1",
                    "implicitZero": False,
                }
            ],
            "entityDictionary": {
                "path": entity_path.name,
                "sha256": _sha256(entity_path),
                "count": records,
            },
            "queryDictionary": {
                "path": query_path.name,
                "sha256": _sha256(query_path),
                "count": 1,
            },
            "queryPanels": {
                "path": panel_path.name,
                "sha256": _sha256(panel_path),
                "count": 1,
            },
            "trajectoryGenes": {
                "path": gene_path.name,
                "sha256": _sha256(gene_path),
                "count": records,
            },
            "normalization": {"id": "SLPNORM:none", "valueSpace": "SLPVS:fixture"},
            "bounds": {
                "maxRecordsPerShard": max(1, records),
                "maxContextTokens": 1,
                "maxActionTokens": 1,
                "maxPanelQueries": 1,
                "maxTargetsPerRecord": 1,
            },
            "shards": [
                {
                    "path": shard_path.name,
                    "sha256": _sha256(shard_path),
                    "records": records,
                    "targetValues": records,
                }
            ],
        }
        (directory / "corpus.json").write_bytes(audit.canonical_json_bytes(manifest, newline=True))
        return self._dataset_input(directory, input_name, resource_name)

    def _protected_inventory(
        self,
        root: Path,
        input_name: str,
        source_id: str,
        identifiers: list[str],
    ) -> dict[str, object]:
        resource_name = f"fixture-{input_name}"
        directory = self._snapshot_root(root, input_name, resource_name)
        records_path = directory / "interventions.jsonl"
        records_path.write_bytes(
            b"".join(
                audit.canonical_json_bytes(
                    {
                        "schema": audit.INVENTORY_RECORD_SCHEMA,
                        "interventionId": identifier,
                        "ncbiTaxon": 4932,
                        "qcPassing": True,
                    },
                    newline=True,
                )
                for identifier in sorted(identifiers)
            )
        )
        manifest = {
            "schema": audit.INVENTORY_SCHEMA,
            "sourceId": source_id,
            "sourceRelease": source_id.split(":", 1)[1] + "-2026-09-04",
            "ncbiTaxon": 4932,
            "stableIdNamespace": "SGD",
            "identityMappingId": self.MAPPING_ID,
            "identityMappingSha256": self.MAPPING_SHA,
            "inventoryFormat": audit.INVENTORY_RECORD_SCHEMA,
            "files": [
                {
                    "path": records_path.name,
                    "sha256": _sha256(records_path),
                    "records": len(identifiers),
                }
            ],
        }
        (directory / "inventory.json").write_bytes(
            audit.canonical_json_bytes(manifest, newline=True)
        )
        return self._dataset_input(directory, input_name, resource_name)

    def _held_roster(
        self,
        root: Path,
        identifiers: list[str],
        inventory_inputs: dict[str, object],
    ) -> dict[str, object]:
        input_name = "heldRoster"
        resource_name = "fixture-held-roster"
        directory = self._snapshot_root(root, input_name, resource_name)
        assignments = [
            (identifier, *audit.assign_intervention(identifier)[:2])
            for identifier in sorted(identifiers)
        ]
        roster_path = directory / "held-intervention-roster.tsv"
        roster_path.write_bytes(
            "".join(
                f"{identifier}\t{role}\t{digest}\n"
                for identifier, role, digest in assignments
            ).encode("ascii")
        )
        counts = {
            role: sum(assignment_role == role for _, assignment_role, _ in assignments)
            for role in ("pretrain", "molecular-validation", "molecular-final")
        }
        source_values = []
        for inventory_input in inventory_inputs.values():
            inventory_root = Path(inventory_input["path"])
            inventory = json.loads((inventory_root / "inventory.json").read_bytes())
            source_id = inventory["sourceId"]
            source_values.append(
                {
                    "sourceId": source_id,
                    "sourceRelease": inventory["sourceRelease"],
                    "identityMappingId": self.MAPPING_ID,
                    "identityMappingSha256": self.MAPPING_SHA,
                    "manifestSha256": _sha256(inventory_root / "inventory.json"),
                    "records": len(assignments),
                    "duplicateRecords": 0,
                    "uniqueInterventions": len(assignments),
                    "qcPassing": len(assignments),
                    "qcFailed": 0,
                    "intersectionCoverage": len(assignments),
                    "exclusions": [],
                }
            )
        source_values.sort(key=lambda item: item["sourceId"])
        coverage = {
            "schema": "slp.held-intervention-roster-report/v1",
            "assignment": {
                "domainHex": audit.ASSIGNMENT_DOMAIN_HEX,
                "digest": "sha256",
                "bucketRule": audit.BUCKET_RULE,
                "roles": {
                    "0-9": "molecular-final",
                    "10-29": "molecular-validation",
                    "30-99": "pretrain",
                },
            },
            "sourceCount": len(source_values),
            "identityMapping": {"id": self.MAPPING_ID, "sha256": self.MAPPING_SHA},
            "minimumIntersectionSize": len(assignments),
            "intersectionSize": len(assignments),
            "roleCounts": counts,
            "rejectionCounts": {
                "qcFailed": 0,
                "notPassingAllProtectedSources": 0,
                "identicalDuplicatesCollapsed": 0,
            },
            "rosterPath": roster_path.name,
            "rosterSha256": _sha256(roster_path),
            "sources": source_values,
        }
        (directory / "coverage.json").write_bytes(
            audit.canonical_json_bytes(coverage, newline=True)
        )
        return self._dataset_input(directory, input_name, resource_name)

    def _fixture(
        self,
        root: Path,
        overrides: dict[str, list[str]] | None = None,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        genes = {
            "pretrain": [self.by_role["pretrain"]],
            "molecularReward": [self.by_role["pretrain"]],
            "molecularValidation": [self.by_role["molecular-validation"]],
            "molecularFinal": [self.by_role["molecular-final"]],
        }
        genes.update(overrides or {})
        inputs = {
            name: self._corpus(root, name, role, genes[name])
            for name, role in audit.EXPECTED_ROLES.items()
        }
        roster_ids = sorted(
            {
                self.by_role["pretrain"],
                self.by_role["molecular-validation"],
                self.by_role["molecular-final"],
            }
        )
        inventory_inputs = {
            "protectedInventoryAtlas": self._protected_inventory(
                root,
                "protectedInventoryAtlas",
                "TESTSOURCE:atlas",
                roster_ids,
            ),
            "protectedInventoryProteome": self._protected_inventory(
                root,
                "protectedInventoryProteome",
                "TESTSOURCE:proteome",
                roster_ids,
            ),
        }
        return inputs, self._held_roster(root, roster_ids, inventory_inputs), inventory_inputs

    def test_passing_audit_is_deterministic_and_schema_valid(self) -> None:
        outputs: list[bytes] = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs, held, inventories = self._fixture(root)
                report, digest = audit.write_audit_artifact(
                    inputs, held, inventories, root / "output" / "corpus-audit"
                )
                payload = root / "output" / "corpus-audit" / "corpus-audit.json"
                self.assertEqual(hashlib.sha256(payload.read_bytes()).hexdigest(), digest)
                self.assertEqual(set(path.name for path in payload.parent.iterdir()), {payload.name})
                schema = json.loads((MODULE / "corpus-audit.schema.json").read_text())
                self.assertEqual(set(schema["required"]), set(report))
                self.assertEqual(
                    schema["properties"]["schema"]["const"],
                    "slp.corpus-audit/v1.1",
                )
                outputs.append(payload.read_bytes())
        self.assertEqual(outputs[0], outputs[1])
        document = json.loads(outputs[0])
        self.assertEqual(document["schema"], "slp.corpus-audit/v1.1")
        self.assertEqual(set(document["datasets"]), set(audit.EXPECTED_ROLES))
        self.assertTrue(document["omfPriorAdmissionRequired"])
        self.assertEqual(document["leakedTrajectoryGenes"], [])

    def test_spoofed_dataset_shape_path_revision_and_digest_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, held, inventories = self._fixture(root)
            pretrain = inputs["pretrain"]
            cases = (
                ({"path": pretrain["path"]}, "fields mismatch"),
                ({**pretrain, "mode": "mount"}, "immutable copied"),
                (
                    {**pretrain, "resource": pretrain["resource"].rsplit("@", 1)[0] + "@latest"},
                    "sha256",
                ),
                ({**pretrain, "manifestDigest": "latest"}, "sha256"),
                (
                    {**pretrain, "resource": pretrain["resource"].replace("fixture-pretrain@", "spoof@")},
                    "path is inconsistent",
                ),
            )
            for value, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    audit.CorpusAuditError, message
                ):
                    audit.audit_corpora(
                        {**inputs, "pretrain": value}, held, inventories
                    )

    def test_internal_digest_and_exact_file_set_drift_fail(self) -> None:
        for mutation, message in (
            ("digest", "content digest mismatch"),
            ("extra", "directory entry count exceeds"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs, held, inventories = self._fixture(root)
                directory = Path(inputs["pretrain"]["path"])
                if mutation == "digest":
                    with (directory / "shard-000.npz").open("ab") as stream:
                        stream.write(b"drift")
                else:
                    (directory / "extra.txt").write_text("not declared")
                with self.assertRaisesRegex(audit.CorpusAuditError, message):
                    audit.audit_corpora(inputs, held, inventories)

    def test_hidden_or_missing_shard_members_are_fatal(self) -> None:
        for mutation in ("hidden", "missing-target"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs, held, inventories = self._fixture(root)
                directory = Path(inputs["pretrain"]["path"])
                shard = directory / "shard-000.npz"
                with np.load(shard, allow_pickle=False) as source:
                    arrays = {name: source[name] for name in source.files}
                if mutation == "hidden":
                    arrays["benchmark_label"] = np.ones(1, dtype=np.int64)
                else:
                    arrays.pop("target_value")
                np.savez(shard, **arrays)
                manifest_path = directory / "corpus.json"
                manifest = json.loads(manifest_path.read_bytes())
                manifest["shards"][0]["sha256"] = _sha256(shard)
                manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
                with self.assertRaisesRegex(audit.CorpusAuditError, "NPZ arrays mismatch"):
                    audit.audit_corpora(inputs, held, inventories)

    def test_sgd_taxon_zero_and_cross_species_active_actions_are_fatal(self) -> None:
        for entity_taxon, add_declared_species, message in (
            (0, False, "SGD entities must retain"),
            (9606, True, "active action entity taxon"),
        ):
            with self.subTest(entity_taxon=entity_taxon), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs, held, inventories = self._fixture(root)
                directory = Path(inputs["pretrain"]["path"])
                entity_path = directory / "entities.npz"
                with np.load(entity_path, allow_pickle=False) as source:
                    arrays = {name: source[name] for name in source.files}
                arrays["entity_species_taxon"] = np.asarray([entity_taxon], dtype=np.int64)
                if add_declared_species:
                    arrays["entity_id"] = np.asarray(["HGNC:5"], dtype="<U32")
                np.savez(entity_path, **arrays)
                manifest_path = directory / "corpus.json"
                manifest = json.loads(manifest_path.read_bytes())
                if add_declared_species:
                    manifest["species"].append(
                        {"taxon": 9606, "featureValue": [0.0], "featurePresent": [True]}
                    )
                    trajectory_path = directory / "trajectory-genes.txt"
                    trajectory_path.write_bytes(b"HGNC:5\n")
                    manifest["trajectoryGenes"]["sha256"] = _sha256(trajectory_path)
                manifest["entityDictionary"]["sha256"] = _sha256(entity_path)
                manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
                with self.assertRaisesRegex(audit.CorpusAuditError, message):
                    audit.audit_corpora(inputs, held, inventories)

    def test_duplicate_record_ids_are_fatal(self) -> None:
        second_pretrain = next(
            identifier
            for number in range(10_000, 20_000)
            if (
                (identifier := f"SGD:S{number:09d}") != self.by_role["pretrain"]
                and audit.assign_intervention(identifier)[0] == "pretrain"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, held, inventories = self._fixture(
                root, {"pretrain": [self.by_role["pretrain"], second_pretrain]}
            )
            directory = Path(inputs["pretrain"]["path"])
            shard = directory / "shard-000.npz"
            with np.load(shard, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["record_id"] = np.asarray(
                ["TEST:duplicate", "TEST:duplicate"], dtype="<U64"
            )
            np.savez(shard, **arrays)
            manifest_path = directory / "corpus.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["shards"][0]["sha256"] = _sha256(shard)
            manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
            with self.assertRaisesRegex(audit.CorpusAuditError, "record_id values must be unique"):
                audit.audit_corpora(inputs, held, inventories)

    def test_roster_intersection_is_recomputed_from_protected_inventories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, held, inventories = self._fixture(root)
            omitted = "SGD:S999999999"
            for inventory_input in inventories.values():
                directory = Path(inventory_input["path"])
                records_path = directory / "interventions.jsonl"
                with records_path.open("ab") as stream:
                    stream.write(
                        audit.canonical_json_bytes(
                            {
                                "schema": audit.INVENTORY_RECORD_SCHEMA,
                                "interventionId": omitted,
                                "ncbiTaxon": 4932,
                                "qcPassing": True,
                            },
                            newline=True,
                        )
                    )
                manifest_path = directory / "inventory.json"
                manifest = json.loads(manifest_path.read_bytes())
                manifest["files"][0]["sha256"] = _sha256(records_path)
                manifest["files"][0]["records"] += 1
                manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
            with self.assertRaisesRegex(audit.CorpusAuditError, "exact QC-passing"):
                audit.audit_corpora(inputs, held, inventories)
            with self.assertRaisesRegex(audit.CorpusAuditError, "at least two"):
                audit.audit_corpora(inputs, held, dict(list(inventories.items())[:1]))

    def test_final_only_and_reward_leakage_are_fatal(self) -> None:
        for role in ("pretrain", "molecularReward"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs, held, inventories = self._fixture(
                    root, {role: [self.by_role["molecular-final"]]}
                )
                with self.assertRaisesRegex(
                    audit.CorpusAuditError, "quantitative fitting trajectories"
                ):
                    audit.audit_corpora(inputs, held, inventories)

    def test_forged_roster_role_is_fatal_even_when_file_hash_is_updated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, held, inventories = self._fixture(root)
            directory = Path(held["path"])
            roster = directory / "held-intervention-roster.tsv"
            lines = roster.read_text().splitlines()
            fields = lines[0].split("\t")
            fields[1] = "molecular-final" if fields[1] != "molecular-final" else "pretrain"
            lines[0] = "\t".join(fields)
            roster.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
            coverage_path = directory / "coverage.json"
            coverage = json.loads(coverage_path.read_text())
            coverage["rosterSha256"] = _sha256(roster)
            coverage_path.write_bytes(audit.canonical_json_bytes(coverage, newline=True))
            with self.assertRaisesRegex(audit.CorpusAuditError, "forged or drifted"):
                audit.audit_corpora(inputs, held, inventories)

    def test_validation_final_overlap_and_wrong_role_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = self.by_role["molecular-validation"]
            inputs, held, inventories = self._fixture(root, {"molecularFinal": [shared]})
            with self.assertRaisesRegex(audit.CorpusAuditError, "overlap"):
                audit.audit_corpora(inputs, held, inventories)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, held, inventories = self._fixture(
                root, {"molecularValidation": [self.by_role["pretrain"]]}
            )
            with self.assertRaisesRegex(audit.CorpusAuditError, "roster role"):
                audit.audit_corpora(inputs, held, inventories)

    def test_input_omissions_extras_and_record_level_trajectory_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, held, inventories = self._fixture(root)
            for changed in (
                {name: value for name, value in inputs.items() if name != "molecularFinal"},
                {**inputs, "unexpected": inputs["pretrain"]},
            ):
                with self.assertRaisesRegex(audit.CorpusAuditError, "must be exactly"):
                    audit.audit_corpora(changed, held, inventories)

            directory = Path(inputs["pretrain"]["path"])
            genes = directory / "trajectory-genes.txt"
            genes.write_bytes(b"")
            manifest_path = directory / "corpus.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["trajectoryGenes"].update(sha256=_sha256(genes), count=0)
            manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
            with self.assertRaisesRegex(audit.CorpusAuditError, "record-level species actions"):
                audit.audit_corpora(inputs, held, inventories)

    def test_benchmark_and_coverage_source_provenance_drift_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, held, inventories = self._fixture(root)
            manifest_path = Path(inputs["pretrain"]["path"]) / "corpus.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["benchmarkLabelsPresent"] = True
            manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
            with self.assertRaisesRegex(audit.CorpusAuditError, "benchmark labels"):
                audit.audit_corpora(inputs, held, inventories)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, held, inventories = self._fixture(root)
            coverage_path = Path(held["path"]) / "coverage.json"
            coverage = json.loads(coverage_path.read_text())
            coverage["sources"].reverse()
            coverage_path.write_bytes(audit.canonical_json_bytes(coverage, newline=True))
            with self.assertRaisesRegex(
                audit.CorpusAuditError, "exactly reproduce|deterministically sorted"
            ):
                audit.audit_corpora(inputs, held, inventories)

    def test_symlink_is_fatal_when_host_permits_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, held, inventories = self._fixture(root)
            directory = Path(inputs["pretrain"]["path"])
            original = directory / "trajectory-genes.txt"
            target = directory / "trajectory-copy.txt"
            target.write_bytes(original.read_bytes())
            original.unlink()
            try:
                os.symlink(target, original)
            except OSError:
                self.skipTest("symlink creation is not permitted on this Windows host")
            with self.assertRaisesRegex(audit.CorpusAuditError, "symlink"):
                audit.audit_corpora(inputs, held, inventories)

    def test_module_workload_and_local_schema_freeze_exact_boundary(self) -> None:
        module = yaml.safe_load((MODULE / "module.yaml").read_text())
        required = set(module["spec"]["contracts"]["input"]["required"])
        self.assertEqual(required, {*audit.EXPECTED_ROLES, "heldRoster"})
        workload_text = (ROOT / "workloads" / "slp-1-1-audit-smoke.yaml").read_text()
        workload = yaml.safe_load(workload_text)
        inputs = workload["spec"]["graph"]["stages"][0]["inputs"]
        protected = {key for key in inputs if key.startswith("protectedInventory")}
        self.assertGreaterEqual(len(protected), 2)
        self.assertEqual(
            set(inputs), {*audit.EXPECTED_ROLES, "heldRoster", *protected}
        )
        self.assertTrue(all(value.startswith("dataset/") for value in inputs.values()))
        self.assertNotIn("omf://", workload_text)
        self.assertFalse(any("benchmark" in key.casefold() for key in inputs))
        self.assertNotIn("artifact:", workload_text)


if __name__ == "__main__":
    unittest.main()
