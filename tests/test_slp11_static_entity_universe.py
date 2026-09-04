from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "modules" / "slp-1-1-static-entity-universe-v1"
SPEC = importlib.util.spec_from_file_location(
    "slp11_static_entity_universe", MODULE_ROOT / "universe.py"
)
assert SPEC is not None and SPEC.loader is not None
universe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = universe
SPEC.loader.exec_module(universe)


class StaticEntityUniverseTest(unittest.TestCase):
    TAXON = 4932
    MAPPING_ID = "sgd:fixture-mapping-v1"
    MAPPING_SHA = "a" * 64
    INTERVENTION_RESOURCE = (
        "omf://fixture/slp/datasetsnapshot/fixture-interventions@sha256:" + "1" * 64
    )
    RELATION_RESOURCE = (
        "omf://fixture/slp/datasetsnapshot/fixture-relations@sha256:" + "2" * 64
    )
    INTERVENTION_OUTER = "sha256:" + "3" * 64
    RELATION_OUTER = "sha256:" + "4" * 64

    @staticmethod
    def _action(identifier: str, *, passing: bool = True, **extra: object) -> dict[str, object]:
        return {
            "schema": universe.INTERVENTION_RECORD_SCHEMA,
            "interventionId": identifier,
            "ncbiTaxon": StaticEntityUniverseTest.TAXON,
            "qcPassing": passing,
            **extra,
        }

    @staticmethod
    def _relation(protein: str, targets: list[str], **extra: object) -> dict[str, object]:
        return {
            "schema": universe.PROTEIN_RECORD_SCHEMA,
            "proteinId": f"UniProtKB:{protein}",
            "sourceAccession": protein,
            "sourceAccessionType": {
                "source": "UniProtKB",
                "type": "UniProtKB ID",
                "namespaceInferred": False,
                "caseNormalization": "none",
            },
            "ncbiTaxon": StaticEntityUniverseTest.TAXON,
            "currentOrfRelations": targets,
            "currentOrfRelationCount": len(targets),
            "chooseFirstAllowed": False,
            **extra,
        }

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, object]]) -> str:
        payload = b"".join(
            (json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()
            for record in records
        )
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _write_manifest(path: Path, value: dict[str, object]) -> str:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def _fixture(
        self,
        root: Path,
        *,
        actions: list[dict[str, object]] | None = None,
        relations: list[dict[str, object]] | None = None,
        intervention_manifest_update: dict[str, object] | None = None,
        relation_manifest_update: dict[str, object] | None = None,
    ) -> dict[str, object]:
        actions = actions if actions is not None else [
            self._action("SGD:S000000001"),
            self._action("SGD:S000000002"),
            self._action("SGD:S000000003"),
            self._action("SGD:S000000001"),
        ]
        relations = relations if relations is not None else [
            self._relation("P00001", ["SGD:S000000001"]),
            self._relation("P00002", ["SGD:S000000001", "SGD:S000000004"]),
        ]
        intervention_name = "fixture-interventions"
        relation_name = "fixture-relations"
        intervention_root = root / "inputs" / "interventionInventory" / intervention_name
        relation_root = root / "inputs" / "proteinRelations" / relation_name
        intervention_root.mkdir(parents=True)
        relation_root.mkdir(parents=True)
        action_sha = self._write_jsonl(intervention_root / "interventions.jsonl", actions)
        relation_sha = self._write_jsonl(relation_root / "relations.jsonl", relations)
        intervention_manifest = {
            "schema": universe.INTERVENTION_MANIFEST_SCHEMA,
            "sourceId": "fixture:proteome",
            "sourceRelease": "fixture-1",
            "ncbiTaxon": self.TAXON,
            "stableIdNamespace": "SGD",
            "identityMappingId": self.MAPPING_ID,
            "identityMappingSha256": self.MAPPING_SHA,
            "inventoryFormat": universe.INTERVENTION_RECORD_SCHEMA,
            "files": [{
                "path": "interventions.jsonl",
                "sha256": action_sha,
                "records": len(actions),
            }],
        }
        relation_manifest = {
            "schema": universe.PROTEIN_MANIFEST_SCHEMA,
            "sourceId": "fixture:proteome",
            "sourceRelease": "fixture-1",
            "ncbiTaxon": self.TAXON,
            "identityMappingId": self.MAPPING_ID,
            "identityMappingSha256": self.MAPPING_SHA,
            "relationFormat": universe.PROTEIN_RECORD_SCHEMA,
            "files": [{
                "path": "relations.jsonl",
                "sha256": relation_sha,
                "records": len(relations),
            }],
        }
        if intervention_manifest_update:
            intervention_manifest.update(intervention_manifest_update)
        if relation_manifest_update:
            relation_manifest.update(relation_manifest_update)
        intervention_manifest_sha = self._write_manifest(
            intervention_root / "inventory.json", intervention_manifest
        )
        relation_manifest_sha = self._write_manifest(
            relation_root / "manifest.json", relation_manifest
        )
        action_states: dict[tuple[int, str], bool] = {}
        for record in actions:
            if set(record) == universe.INTERVENTION_RECORD_FIELDS:
                action_states[(record["ncbiTaxon"], record["interventionId"])] = record["qcPassing"]
        action_keys = {key for key, passing in action_states.items() if passing}
        protein_keys = {
            (record["ncbiTaxon"], record["proteinId"])
            for record in relations
            if set(record) == universe.PROTEIN_RECORD_FIELDS
        }
        target_keys = {
            (record["ncbiTaxon"], target)
            for record in relations
            if set(record) == universe.PROTEIN_RECORD_FIELDS
            for target in record["currentOrfRelations"]
        }
        expected = universe.ExpectedContract(
            intervention=universe.ExpectedSnapshot(
                self.INTERVENTION_RESOURCE,
                self.INTERVENTION_OUTER,
                intervention_manifest_sha,
                action_sha,
            ),
            relations=universe.ExpectedSnapshot(
                self.RELATION_RESOURCE,
                self.RELATION_OUTER,
                relation_manifest_sha,
                relation_sha,
            ),
            mapping_id=self.MAPPING_ID,
            mapping_sha256=self.MAPPING_SHA,
            intervention_records=len(actions),
            action_entities=len(action_keys),
            relation_records=len(relations),
            relation_edges=sum(len(record.get("currentOrfRelations", [])) for record in relations),
            relation_target_genes=len(target_keys),
            relation_targets_in_action_universe=len(target_keys & action_keys),
            relation_support_only=len(target_keys - action_keys),
            one_to_many_relations=sum(
                len(record.get("currentOrfRelations", [])) > 1 for record in relations
            ),
            readout_entities=len(protein_keys),
            total_entities=len(action_keys | target_keys) + len(protein_keys),
            action_id_set_sha256=universe.framed_ascii_set_sha256(key[1] for key in action_keys),
            protein_id_set_sha256=universe.framed_ascii_set_sha256(key[1] for key in protein_keys),
            relation_edge_set_sha256=universe.framed_ascii_set_sha256(
                f"{record['proteinId']}\t{target}"
                for record in relations
                if set(record) == universe.PROTEIN_RECORD_FIELDS
                for target in record["currentOrfRelations"]
            ),
            full_entity_id_set_sha256=universe.framed_ascii_set_sha256(
                key[1] for key in action_keys | target_keys | protein_keys
            ),
            full_entity_key_set_sha256=universe.framed_composite_key_set_sha256(
                action_keys | target_keys | protein_keys
            ),
        )
        intervention_input = {
            "resource": self.INTERVENTION_RESOURCE,
            "mode": "copy",
            "path": str(intervention_root),
            "manifestDigest": self.INTERVENTION_OUTER,
        }
        relation_input = {
            "resource": self.RELATION_RESOURCE,
            "mode": "copy",
            "path": str(relation_root),
            "manifestDigest": self.RELATION_OUTER,
        }
        return {
            "interventionRoot": intervention_root,
            "relationRoot": relation_root,
            "interventionInput": intervention_input,
            "relationInput": relation_input,
            "intervention": universe.resolve_pinned_dataset(
                intervention_input, "interventionInventory"
            ),
            "relations": universe.resolve_pinned_dataset(relation_input, "proteinRelations"),
            "expected": expected,
        }

    def _build(self, fixture: dict[str, object], destination: Path) -> dict[str, object]:
        return universe.build_entity_universe(
            fixture["intervention"],
            fixture["relations"],
            destination,
            universe.Bounds(),
            expected=fixture["expected"],
        )

    @staticmethod
    def _archive_blobs(path: Path) -> dict[str, bytes]:
        with tarfile.open(path, mode="r:") as archive:
            return {
                member.name: archive.extractfile(member).read()
                for member in archive.getmembers()
            }

    def test_relation_closed_deterministic_universe_preserves_one_to_many(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._fixture(root / "first")
            second = self._fixture(root / "second")
            first_out = root / "out-one"
            second_out = root / "out-two"
            result = self._build(first, first_out)
            self._build(second, second_out)
            self.assertEqual(
                (first_out / "entity-universe.tar").read_bytes(),
                (second_out / "entity-universe.tar").read_bytes(),
            )
            self.assertEqual(
                (first_out / "entity-universe-audit.json").read_bytes(),
                (second_out / "entity-universe-audit.json").read_bytes(),
            )
            self.assertEqual(result["actionEntities"], 3)
            self.assertEqual(result["readoutQueryEntities"], 2)
            self.assertEqual(result["relationTargetGenes"], 2)
            self.assertEqual(result["relationTargetsInUniverse"], 2)
            self.assertEqual(result["relationSupportOnly"], 1)
            self.assertEqual(result["totalEntities"], 6)
            blobs = self._archive_blobs(first_out / "entity-universe.tar")
            entities = [json.loads(line) for line in blobs["static-entity-universe/entities.jsonl"].splitlines()]
            by_id = {item["entityId"]: item for item in entities}
            self.assertEqual(by_id["SGD:S000000001"]["usages"], ["action", "relation-support"])
            self.assertEqual(by_id["SGD:S000000004"]["usages"], ["relation-support"])
            relations = [json.loads(line) for line in blobs["static-entity-universe/relations.jsonl"].splitlines()]
            self.assertEqual(relations[1]["currentOrfRelations"], ["SGD:S000000001", "SGD:S000000004"])
            self.assertIs(relations[1]["chooseFirstAllowed"], False)
            audit = json.loads((first_out / "entity-universe-audit.json").read_text())
            self.assertEqual(audit["counts"]["currentModelEligibleKeys"], 5)
            self.assertEqual(audit["counts"]["relationSupportOnly"], 1)
            self.assertEqual(len(audit["oneToManyRelations"]), 1)
            self.assertEqual(
                audit["semanticSetHashes"]["relationEdgeSet"]["sha256"],
                result["relationEdgeSetSha256"],
            )

    def test_production_semantic_hash_constants_and_framing_are_frozen(self) -> None:
        expected = universe.PRODUCTION_CONTRACT
        self.assertEqual(
            expected.action_id_set_sha256,
            "7424b17b63f504b419fb1c52930ede3d1cbb2a0fabf3a0621624df1d098c4d88",
        )
        self.assertEqual(
            expected.protein_id_set_sha256,
            "25ec666023cc97610797b4e915e537bc3f0212b0f3a02972cb3b60eac160d12d",
        )
        self.assertEqual(
            expected.relation_edge_set_sha256,
            "8a75c42d5a0f24a86be16ecea2616d6d13d25d90de18a80d3dd22cd188afc6d1",
        )
        self.assertEqual(
            expected.full_entity_id_set_sha256,
            "e7231d3bb859ca4818364c76d9aa9fee54d6b1d9a64050c2d3ab8af81a9b3eb9",
        )
        self.assertEqual(
            expected.full_entity_key_set_sha256,
            "82b8e2885939577fe6946e3b974a10cb947834118f2070e1bcbe4c2f2e6a5fd9",
        )
        self.assertEqual(
            universe.framed_ascii_set_sha256(["B", "A", "B"]),
            hashlib.sha256(b"A\nB\n").hexdigest(),
        )
        self.assertEqual(
            universe.framed_composite_key_set_sha256(
                [(559292, "SGD:S000000001"), (4932, "SGD:S000000001")]
            ),
            hashlib.sha256(
                b"4932\tSGD:S000000001\n559292\tSGD:S000000001\n"
            ).hexdigest(),
        )

    def test_noncanonical_jsonl_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            path.write_bytes(b'{"b": 2, "a": 1}\r\n')
            with self.assertRaisesRegex(
                universe.StaticEntityUniverseError, "not canonical JSONL"
            ):
                list(universe._jsonl(path, universe.Bounds(), 1, "records.jsonl"))

    def test_input_permutations_canonicalize_payload_but_remain_provenance_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actions = [
                self._action("SGD:S000000001"), self._action("SGD:S000000002"),
                self._action("SGD:S000000003"), self._action("SGD:S000000001"),
            ]
            relations = [
                self._relation("P00001", ["SGD:S000000001"]),
                self._relation("P00002", ["SGD:S000000001", "SGD:S000000004"]),
            ]
            first = self._fixture(root / "first", actions=actions, relations=relations)
            second = self._fixture(
                root / "second", actions=list(reversed(actions)), relations=list(reversed(relations))
            )
            first_out, second_out = root / "one", root / "two"
            self._build(first, first_out)
            self._build(second, second_out)
            first_blobs = self._archive_blobs(first_out / "entity-universe.tar")
            second_blobs = self._archive_blobs(second_out / "entity-universe.tar")
            self.assertEqual(
                first_blobs["static-entity-universe/entities.jsonl"],
                second_blobs["static-entity-universe/entities.jsonl"],
            )
            self.assertEqual(
                first_blobs["static-entity-universe/relations.jsonl"],
                second_blobs["static-entity-universe/relations.jsonl"],
            )
            self.assertNotEqual(
                first_blobs["static-entity-universe/manifest.json"],
                second_blobs["static-entity-universe/manifest.json"],
                "raw source order must remain bound even when semantic output is canonicalized",
            )

    def test_composite_identity_keeps_same_curie_in_distinct_taxa(self) -> None:
        records = [
            {
                "schema": universe.ENTITY_SCHEMA,
                "ncbiTaxon": 4932,
                "entityId": "SGD:S000000001",
                "entityClass": "gene",
                "usages": ["action"],
            },
            {
                "schema": universe.ENTITY_SCHEMA,
                "ncbiTaxon": 559292,
                "entityId": "SGD:S000000001",
                "entityClass": "gene",
                "usages": ["relation-support"],
            },
        ]
        output = universe.canonicalize_entities(reversed(records))
        self.assertEqual([(item["ncbiTaxon"], item["entityId"]) for item in output], [
            (4932, "SGD:S000000001"), (559292, "SGD:S000000001")
        ])

    def test_duplicate_relations_and_conflicting_action_duplicates_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate_relation = self._relation("P00001", ["SGD:S000000001"])
            fixture = self._fixture(
                root / "relations",
                relations=[duplicate_relation, dict(duplicate_relation)],
            )
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "duplicate protein"):
                self._build(fixture, root / "out-relations")

            fixture = self._fixture(
                root / "actions",
                actions=[
                    self._action("SGD:S000000001", passing=True),
                    self._action("SGD:S000000001", passing=False),
                ],
                relations=[self._relation("P00001", ["SGD:S000000001"])],
            )
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "conflicting duplicate"):
                self._build(fixture, root / "out-actions")

    def test_malformed_identifiers_and_ambiguous_relations_fail(self) -> None:
        cases = [
            (
                [self._action("YAL001C")],
                [self._relation("P00001", ["SGD:S000000001"])],
                "canonical SGD",
            ),
            (
                [self._action("SGD:S000000001")],
                [self._relation("p00001", ["SGD:S000000001"])],
                "UniProtKB|sourceAccession",
            ),
            (
                [self._action("SGD:S000000001")],
                [self._relation("P00001", ["SGD:S000000001", "SGD:S000000001"])],
                "sorted unique",
            ),
            (
                [self._action("SGD:S000000001")],
                [self._relation("P00001", ["SGD:S000000001"], chooseFirstAllowed=True)],
                "select a first gene",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (actions, relations, message) in enumerate(cases):
                fixture = self._fixture(root / str(index), actions=actions, relations=relations)
                with self.assertRaisesRegex(universe.StaticEntityUniverseError, message):
                    self._build(fixture, root / f"out-{index}")

    def test_missing_extra_and_forbidden_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = self._action("SGD:S000000001")
            missing.pop("qcPassing")
            fixture = self._fixture(
                root / "missing",
                actions=[missing],
                relations=[self._relation("P00001", ["SGD:S000000001"])],
            )
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "fields do not match"):
                self._build(fixture, root / "out-missing")

            extra = self._action("SGD:S000000001", comment="not permitted")
            fixture = self._fixture(
                root / "extra", actions=[extra],
                relations=[self._relation("P00001", ["SGD:S000000001"])],
            )
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "fields do not match"):
                self._build(fixture, root / "out-extra")

            forbidden = self._action("SGD:S000000001", outcomeLabel=1)
            fixture = self._fixture(
                root / "forbidden", actions=[forbidden],
                relations=[self._relation("P00001", ["SGD:S000000001"])],
            )
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "forbidden non-identity"):
                self._build(fixture, root / "out-forbidden")

    def test_resource_outer_inner_and_record_pins_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root / "resource")
            bad_resource = universe.PinnedDataset(
                input_name=fixture["intervention"].input_name,
                path=fixture["intervention"].path,
                resource=fixture["intervention"].resource.replace("1" * 64, "9" * 64),
                revision="sha256:" + "9" * 64,
                manifest_digest=fixture["intervention"].manifest_digest,
            )
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "resource revision"):
                universe.build_entity_universe(
                    bad_resource, fixture["relations"], root / "out-resource",
                    universe.Bounds(), expected=fixture["expected"],
                )

            fixture = self._fixture(root / "outer")
            bad_outer = universe.PinnedDataset(
                fixture["intervention"].input_name,
                fixture["intervention"].path,
                fixture["intervention"].resource,
                fixture["intervention"].revision,
                "sha256:" + "9" * 64,
            )
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "outer manifest"):
                universe.build_entity_universe(
                    bad_outer, fixture["relations"], root / "out-outer",
                    universe.Bounds(), expected=fixture["expected"],
                )

            fixture = self._fixture(root / "inner")
            (fixture["interventionRoot"] / "inventory.json").write_bytes(b"{}\n")
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "inner manifest digest"):
                self._build(fixture, root / "out-inner")

            fixture = self._fixture(root / "record")
            path = fixture["relationRoot"] / "relations.jsonl"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "record-set digest"):
                self._build(fixture, root / "out-record")

    def test_exact_file_set_materialization_shape_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root / "files")
            (fixture["interventionRoot"] / "extra.txt").write_text("no")
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "file set drift"):
                self._build(fixture, root / "out-extra")

            fixture = self._fixture(root / "shape")
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "spoofed"):
                universe.resolve_pinned_dataset(
                    {**fixture["interventionInput"], "extra": True}, "interventionInventory"
                )
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "copied"):
                universe.resolve_pinned_dataset(
                    {**fixture["interventionInput"], "mode": "reference"}, "interventionInventory"
                )

            fixture = self._fixture(root / "symlink")
            target = fixture["interventionRoot"] / "interventions.jsonl"
            link = fixture["interventionRoot"] / "linked.jsonl"
            try:
                link.symlink_to(target)
            except OSError:
                with mock.patch.object(
                    type(target),
                    "is_symlink",
                    autospec=True,
                    side_effect=lambda value: value.name == "interventions.jsonl",
                ), self.assertRaisesRegex(universe.StaticEntityUniverseError, "symlink"):
                    self._build(fixture, root / "out-link")
            else:
                target.unlink()
                link.rename(target)
                with self.assertRaisesRegex(universe.StaticEntityUniverseError, "symlink"):
                    self._build(fixture, root / "out-link")

    def test_archive_validator_rejects_extra_and_symbolic_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root / "fixture")
            output = root / "output"
            self._build(fixture, output)
            source = output / "entity-universe.tar"
            blobs = self._archive_blobs(source)
            extra = root / "extra.tar"
            with tarfile.open(extra, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name in sorted([*blobs, "static-entity-universe/unexpected"]):
                    payload = blobs.get(name, b"unexpected")
                    archive.addfile(universe._tar_info(name, len(payload)), io.BytesIO(payload))
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "member set or order"):
                universe.validate_archive(extra, universe.Bounds())

            symbolic = root / "symbolic.tar"
            with tarfile.open(symbolic, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name in sorted(blobs):
                    if name.endswith("relations.jsonl"):
                        info = tarfile.TarInfo(name)
                        info.type = tarfile.SYMTYPE
                        info.linkname = "entities.jsonl"
                        archive.addfile(info)
                    else:
                        payload = blobs[name]
                        archive.addfile(universe._tar_info(name, len(payload)), io.BytesIO(payload))
            with self.assertRaisesRegex(universe.StaticEntityUniverseError, "metadata drift"):
                universe.validate_archive(symbolic, universe.Bounds())

    def test_archive_validator_rejects_self_consistent_forbidden_payload_and_semantic_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root / "fixture")
            output = root / "output"
            self._build(fixture, output)
            original = self._archive_blobs(output / "entity-universe.tar")

            forbidden_blobs = dict(original)
            entity_name = "static-entity-universe/entities.jsonl"
            entities = [json.loads(line) for line in forbidden_blobs[entity_name].splitlines()]
            entities[0]["outcomeLabel"] = 1
            forbidden_blobs[entity_name] = b"".join(
                universe._canonical_json_bytes(record) for record in entities
            )
            manifest_name = "static-entity-universe/manifest.json"
            manifest = json.loads(forbidden_blobs[manifest_name])
            manifest["entities"]["file"] = universe._file_ref(
                "entities.jsonl", forbidden_blobs[entity_name], len(entities)
            )
            forbidden_blobs[manifest_name] = universe._pretty_json_bytes(manifest)
            forbidden_archive = root / "forbidden.tar"
            forbidden_archive.write_bytes(universe._tar_bytes(forbidden_blobs))
            with self.assertRaisesRegex(
                universe.StaticEntityUniverseError, "forbidden non-identity"
            ):
                universe.validate_archive(forbidden_archive, universe.Bounds())

            drift_blobs = dict(original)
            manifest = json.loads(drift_blobs[manifest_name])
            manifest["semanticSetHashes"]["fullEntityKeySet"]["sha256"] = "9" * 64
            drift_blobs[manifest_name] = universe._pretty_json_bytes(manifest)
            drift_archive = root / "semantic-drift.tar"
            drift_archive.write_bytes(universe._tar_bytes(drift_blobs))
            with self.assertRaisesRegex(
                universe.StaticEntityUniverseError, "fullEntityKeySet digest drift"
            ):
                universe.validate_archive(drift_archive, universe.Bounds())

    def test_record_byte_bound_is_checked_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root / "fixture")
            with mock.patch.object(
                universe, "_sha256_file", side_effect=AssertionError("hash must not run")
            ), self.assertRaisesRegex(
                universe.StaticEntityUniverseError, "configured byte bound"
            ):
                universe.build_entity_universe(
                    fixture["intervention"],
                    fixture["relations"],
                    root / "output",
                    universe.Bounds(max_line_bytes=128, max_intervention_records=1),
                    expected=fixture["expected"],
                )

    def test_module_and_workload_expose_only_two_pinned_identity_snapshots(self) -> None:
        module = yaml.safe_load((MODULE_ROOT / "module.yaml").read_text())
        input_contract = module["spec"]["contracts"]["input"]
        self.assertEqual(input_contract["required"], ["interventionInventory", "proteinRelations"])
        self.assertFalse(input_contract["additionalProperties"])
        workload_text = (
            ROOT / "workloads" / "slp-1-1-static-entity-universe-v1.yaml.tmpl"
        ).read_text()
        workload = yaml.safe_load(workload_text)
        stage = workload["spec"]["graph"]["stages"][0]
        self.assertEqual(set(stage["inputs"]), {"interventionInventory", "proteinRelations"})
        self.assertEqual(
            stage["inputs"],
            {
                "interventionInventory": "dataset/slp-1-1-proteome-intervention-inventory-v1",
                "proteinRelations": "dataset/slp-1-1-proteome-protein-relations-v1",
            },
        )
        self.assertEqual(
            input_contract["properties"]["interventionInventory"]["properties"]["resource"]["const"],
            universe.PRODUCTION_CONTRACT.intervention.resource,
        )
        self.assertEqual(
            input_contract["properties"]["proteinRelations"]["properties"]["resource"]["const"],
            universe.PRODUCTION_CONTRACT.relations.resource,
        )
        self.assertNotIn("@@", workload_text)
        self.assertNotIn("aliases:", workload_text)
        for forbidden in ("heldRoster", "observation", "benchmark", "reward"):
            self.assertNotIn(forbidden, json.dumps(stage))

    def test_derived_rights_bind_exact_inputs_runs_and_payloads(self) -> None:
        rights = yaml.safe_load(
            (
                ROOT
                / "rights"
                / "slp-1-1-static-entity-universe-v1-cc-by-4.0.yaml"
            ).read_text()
        )
        self.assertTrue(rights["trainingAllowed"])
        self.assertTrue(rights["redistributionAllowed"])
        self.assertEqual(
            rights["derivedFrom"]["datasets"],
            [
                universe.PRODUCTION_CONTRACT.intervention.resource,
                universe.PRODUCTION_CONTRACT.relations.resource,
            ],
        )
        self.assertEqual(len(rights["derivedFrom"]["runResults"]), 2)
        self.assertEqual(
            rights["derivedFrom"]["payloads"],
            {
                "entityUniverseTar": {
                    "bytes": 1_525_760,
                    "sha256": "d947bf618b854dd33a7157ac0f0380c544e9a4377bddb00806c9ca07f689a544",
                },
                "entityUniverseAudit": {
                    "bytes": 4_880,
                    "sha256": "339412ea008cf383db2258d0788d71c2cf357183b331d49f4168aa7f113f1a0f",
                },
            },
        )
        for forbidden in ("quantitative", "molecular target", "benchmark", "embedding"):
            self.assertIn(forbidden, " ".join(rights["exclusions"]).casefold())

    def test_output_schemas_close_every_emitted_object_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root / "fixture")
            output = root / "output"
            self._build(fixture, output)
            blobs = self._archive_blobs(output / "entity-universe.tar")
            manifest = json.loads(blobs["static-entity-universe/manifest.json"])
            audit = json.loads((output / "entity-universe-audit.json").read_text())
            manifest_schema = json.loads(
                (MODULE_ROOT / "entity-universe.schema.json").read_text()
            )
            audit_schema = json.loads(
                (MODULE_ROOT / "entity-universe-audit.schema.json").read_text()
            )

            def assert_closed_shape(schema: dict[str, object], value: dict[str, object]) -> None:
                self.assertEqual(set(schema["required"]), set(value))
                self.assertEqual(set(schema["properties"]), set(value))
                self.assertIs(schema["additionalProperties"], False)

            assert_closed_shape(manifest_schema, manifest)
            assert_closed_shape(manifest_schema["$defs"]["source"], manifest["source"])
            assert_closed_shape(
                manifest_schema["$defs"]["identityMapping"], manifest["identityMapping"]
            )
            assert_closed_shape(
                manifest_schema["$defs"]["semanticSetHashes"],
                manifest["semanticSetHashes"],
            )
            assert_closed_shape(manifest_schema["properties"]["inputs"], manifest["inputs"])
            for input_value in manifest["inputs"].values():
                assert_closed_shape(manifest_schema["$defs"]["pinnedInput"], input_value)
            assert_closed_shape(manifest_schema["properties"]["entities"], manifest["entities"])
            assert_closed_shape(
                manifest_schema["$defs"]["fileReference"], manifest["entities"]["file"]
            )
            assert_closed_shape(
                manifest_schema["properties"]["entities"]["properties"]["counts"],
                manifest["entities"]["counts"],
            )
            assert_closed_shape(manifest_schema["properties"]["relations"], manifest["relations"])
            assert_closed_shape(
                manifest_schema["$defs"]["fileReference"], manifest["relations"]["file"]
            )
            assert_closed_shape(
                manifest_schema["properties"]["contentPolicy"], manifest["contentPolicy"]
            )

            assert_closed_shape(audit_schema, audit)
            assert_closed_shape(audit_schema["properties"]["inputs"], audit["inputs"])
            for input_value in audit["inputs"].values():
                assert_closed_shape(audit_schema["$defs"]["pinnedInput"], input_value)
            assert_closed_shape(audit_schema["$defs"]["source"], audit["source"])
            assert_closed_shape(
                audit_schema["$defs"]["identityMapping"], audit["identityMapping"]
            )
            assert_closed_shape(
                audit_schema["$defs"]["semanticSetHashes"], audit["semanticSetHashes"]
            )
            assert_closed_shape(audit_schema["properties"]["outputs"], audit["outputs"])
            assert_closed_shape(audit_schema["properties"]["counts"], audit["counts"])
            assert_closed_shape(
                audit_schema["properties"]["oneToManyRelations"]["items"],
                audit["oneToManyRelations"][0],
            )
            assert_closed_shape(
                audit_schema["properties"]["accessBoundary"], audit["accessBoundary"]
            )
            self.assertEqual(
                [item["const"] for item in audit_schema["properties"]["limitations"]["prefixItems"]],
                audit["limitations"],
            )
            self.assertEqual(
                manifest_schema["properties"]["contentPolicy"]["required"],
                [
                    "containsDisplaySymbols",
                    "containsNumericFeatures",
                    "containsOutcomesOrLabels",
                    "containsTrainingPartitionAssignments",
                    "crossTaxonIdentityMerge",
                ],
            )
            self.assertNotIn("containsRoleAssignments", json.dumps(manifest_schema))
            self.assertNotIn("roleAssignmentsConsumed", json.dumps(audit_schema))

    def test_output_contains_no_symbols_outcomes_assignments_or_numeric_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root / "fixture")
            output = root / "output"
            self._build(fixture, output)
            blobs = self._archive_blobs(output / "entity-universe.tar")
            entity_text = blobs["static-entity-universe/entities.jsonl"].decode().casefold()
            for forbidden in (
                "symbol", "display", "outcome", "label", "pretrain", "validation",
                "final", "reward", "embedding", "featurevalue", "vector",
            ):
                self.assertNotIn(forbidden, entity_text)
            manifest = json.loads(blobs["static-entity-universe/manifest.json"])
            self.assertFalse(manifest["contentPolicy"]["containsNumericFeatures"])
            self.assertFalse(
                manifest["contentPolicy"]["containsTrainingPartitionAssignments"]
            )
            self.assertTrue(manifest["relations"]["targetsInUniverse"] == manifest["relations"]["targetGenes"])


if __name__ == "__main__":
    unittest.main()
