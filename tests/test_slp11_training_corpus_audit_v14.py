"""Adversarial tests for the composite-key clean-training corpus audit v1.4."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tests import test_slp11_corpus_audit as _v12_fixtures

try:
    import jsonschema
except ModuleNotFoundError:  # pragma: no cover - exercised in the OMF environment
    jsonschema = None

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-training-corpus-audit-v1-4"

AUDIT_SPEC = importlib.util.spec_from_file_location(
    "slp11_training_corpus_audit_v14_audit", MODULE / "audit.py"
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = audit
AUDIT_SPEC.loader.exec_module(audit)

AUTH_SPEC = importlib.util.spec_from_file_location(
    "slp11_training_corpus_audit_v14_attestation", MODULE / "attestation.py"
)
assert AUTH_SPEC is not None and AUTH_SPEC.loader is not None
authorization = importlib.util.module_from_spec(AUTH_SPEC)
sys.modules[AUTH_SPEC.name] = authorization
_prior_audit = sys.modules.get("audit")
sys.modules["audit"] = audit
try:
    AUTH_SPEC.loader.exec_module(authorization)
finally:
    if _prior_audit is None:
        del sys.modules["audit"]
    else:
        sys.modules["audit"] = _prior_audit


class TrainingCorpusAuditTest(unittest.TestCase):
    FACTORY_IDENTITY = "sha256:" + "a" * 64
    CHALLENGE = "b" * 64

    @classmethod
    def setUpClass(cls) -> None:
        _v12_fixtures.CorpusAuditTest.setUpClass()

    def _safe_inputs(
        self,
        root: Path,
        *,
        active_gene_role: str = "pretrain",
        inventory_count: int = 2,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        builder = _v12_fixtures.CorpusAuditTest(
            methodName="test_passing_audit_is_deterministic_and_schema_valid"
        )
        roster_ids = sorted(
            {
                builder.by_role["pretrain"],
                builder.by_role["molecular-validation"],
                builder.by_role["molecular-final"],
            }
        )
        pretrain = builder._corpus(
            root,
            "pretrain",
            "pretrain",
            [builder.by_role[active_gene_role]],
        )
        names = [
            "protectedInventoryAtlas",
            "protectedInventoryProteome",
            "protectedInventoryThird",
        ][:inventory_count]
        inventories = {
            name: builder._protected_inventory(
                root,
                name,
                f"TESTSOURCE:{name.removeprefix('protectedInventory').casefold()}",
                roster_ids,
            )
            for name in names
        }
        held = builder._held_roster(root, roster_ids, inventories)
        self._upgrade_composite_corpus(pretrain, held)
        return pretrain, held, inventories

    @staticmethod
    def _snapshot(name: str) -> dict[str, str]:
        revision = "sha256:" + hashlib.sha256((name + " revision").encode()).hexdigest()
        return {
            "resource": f"omf://abiome/slp/datasetsnapshot/{name}@{revision}",
            "revision": revision,
            "outerManifestDigest": "sha256:" + hashlib.sha256(
                (name + " manifest").encode()
            ).hexdigest(),
            "treeDigest": "sha256:" + hashlib.sha256(
                (name + " tree").encode()
            ).hexdigest(),
        }

    @staticmethod
    def _file_ref(path: Path, root: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    def _upgrade_composite_corpus(
        self, pretrain: dict[str, object], held: dict[str, object]
    ) -> None:
        directory = Path(pretrain["path"])
        manifest_path = directory / "corpus.json"
        manifest = json.loads(manifest_path.read_bytes())

        entity_path = directory / manifest["entityDictionary"]["path"]
        with np.load(entity_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["entity_taxon"] = arrays.pop("entity_species_taxon")
        arrays["entity_taxon"] = np.concatenate(
            [arrays["entity_taxon"], np.asarray([4932], dtype=np.int64)]
        )
        arrays["entity_id"] = np.concatenate(
            [arrays["entity_id"], np.asarray(["SLPCTX:fixture"], dtype="<U32")]
        )
        arrays["entity_type"] = np.concatenate(
            [arrays["entity_type"], np.asarray([2], dtype=np.int64)]
        )
        arrays["entity_feature_value"] = np.concatenate(
            [arrays["entity_feature_value"], np.zeros((1, 1), dtype=np.float32)]
        )
        arrays["entity_feature_present"] = np.concatenate(
            [arrays["entity_feature_present"], np.zeros((1, 1), dtype=np.bool_)]
        )
        np.savez(entity_path, **arrays)

        shard_path = directory / manifest["shards"][0]["path"]
        shard_arrays = self._npz_arrays(shard_path)
        shard_arrays["context_entity_index"][:] = 1
        shard_arrays["context_type"][:] = 0
        shard_arrays["context_mask"][:] = True
        np.savez(shard_path, **shard_arrays)

        query_path = directory / manifest["queryDictionary"]["path"]
        with np.load(query_path, allow_pickle=False) as archive:
            query_arrays = {
                name: archive[name].copy()
                for name in archive.files
                if name != "query_id"
            }
        np.savez(query_path, **query_arrays)

        old_trajectory = manifest.pop("trajectoryGenes")
        old_path = directory / old_trajectory["path"]
        identifiers = old_path.read_text(encoding="ascii").splitlines()
        trajectory_path = directory / "trajectory-interventions.jsonl"
        trajectory_path.write_bytes(
            b"".join(
                audit.canonical_json_bytes(
                    {
                        "schema": audit.TRAJECTORY_INTERVENTION_SCHEMA,
                        "ncbiTaxon": 4932,
                        "entityId": identifier,
                    },
                    newline=True,
                )
                for identifier in identifiers
            )
        )
        old_path.unlink()

        for key in ("entityDictionary", "queryDictionary", "queryPanels"):
            path = directory / manifest[key]["path"]
            manifest[key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest[key]["bytes"] = path.stat().st_size
        trajectory_ref = self._file_ref(trajectory_path, directory)
        manifest["trajectoryInterventions"] = {
            **trajectory_ref,
            "count": len(identifiers),
        }
        for shard in manifest["shards"]:
            shard_path = directory / shard["path"]
            shard["bytes"] = shard_path.stat().st_size

        keys = {
            (int(taxon), str(identifier))
            for taxon, identifier in zip(
                arrays["entity_taxon"], arrays["entity_id"], strict=True
            )
        }
        entity_key_sha = audit._composite_set_sha256(keys)
        static_snapshot = self._snapshot("fixture-static-features")
        block = {
            "id": "TESTFEATURE:sequence-statistics",
            "offset": 0,
            "dimension": 1,
            "datasetSnapshot": static_snapshot,
            "semanticSha256": "1" * 64,
            "entityKeySetSha256": entity_key_sha,
            "files": [self._file_ref(entity_path, directory)],
        }
        feature_pack = {
            "schema": "slp.static-feature-pack/v1",
            "revision": "TESTFEATURE:fixture-v1",
            "sha256": "0" * 64,
            "entityFeatureDim": 1,
            "speciesFeatureDim": 1,
            "blocks": [block],
        }
        digest_basis = dict(feature_pack)
        del digest_basis["sha256"]
        feature_pack["sha256"] = audit.canonical_sha256(digest_basis)
        held_root = Path(held["path"])
        held_files = [
            self._file_ref(held_root / name, held_root)
            for name in sorted({"coverage.json", "held-intervention-roster.tsv"})
        ]
        manifest.update(
            {
                "schema": audit.CORPUS_SCHEMA,
                "rewardEnabled": False,
                "identityKey": ["ncbiTaxon", "entityId"],
                "featurePack": feature_pack,
                "entityTypes": [
                    "slp-entity-type:gene",
                    "slp-entity-type:protein",
                    "slp-entity-type:context",
                ],
                "inputs": {
                    "observations": {
                        "datasetSnapshot": self._snapshot("fixture-observations"),
                        "semanticSha256": "2" * 64,
                        "files": [self._file_ref(directory / "shard-000.npz", directory)],
                    },
                    "staticFeatures": {
                        "datasetSnapshot": static_snapshot,
                        "semanticSha256": block["semanticSha256"],
                        "files": [self._file_ref(entity_path, directory)],
                    },
                    "heldInterventionRoster": {
                        "datasetSnapshot": {
                            "resource": held["resource"],
                            "revision": held["resource"].rsplit("@", 1)[1],
                            "outerManifestDigest": held["manifestDigest"],
                            "treeDigest": "sha256:" + "3" * 64,
                        },
                        "semanticSha256": held_files[1]["sha256"],
                        "files": held_files,
                    },
                },
                "counts": {
                    "entities": len(arrays["entity_id"]),
                    "featureRows": len(arrays["entity_id"]),
                    "contexts": 0,
                    "queries": manifest["queryDictionary"]["count"],
                    "panels": manifest["queryPanels"]["count"],
                    "trajectoryInterventions": len(identifiers),
                    "records": sum(item["records"] for item in manifest["shards"]),
                    "targetValues": sum(
                        item["targetValues"] for item in manifest["shards"]
                    ),
                    "shards": len(manifest["shards"]),
                },
            }
        )
        manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
        self._refresh_corpus(pretrain)

    def _refresh_corpus(self, pretrain: dict[str, object]) -> None:
        directory = Path(pretrain["path"])
        manifest_path = directory / "corpus.json"
        manifest = json.loads(manifest_path.read_bytes())
        entity_path = directory / manifest["entityDictionary"]["path"]
        with np.load(entity_path, allow_pickle=False) as archive:
            entities = {name: archive[name].copy() for name in archive.files}
        keys = [
            (int(taxon), str(identifier))
            for taxon, identifier in zip(
                entities["entity_taxon"], entities["entity_id"], strict=True
            )
        ]
        present = entities["entity_feature_present"]
        complete_rows = np.all(present, axis=1)
        for block in manifest["featurePack"]["blocks"]:
            start = int(block["offset"])
            stop = start + int(block["dimension"])
            block_rows = np.all(present[:, start:stop], axis=1)
            block["entityKeySetSha256"] = audit._composite_set_sha256(
                {key for key, included in zip(keys, block_rows, strict=True) if included}
            )
            block["files"] = [self._file_ref(entity_path, directory)]
        for name in ("entityDictionary", "queryDictionary", "queryPanels"):
            path = directory / manifest[name]["path"]
            with np.load(path, allow_pickle=False) as archive:
                if name == "entityDictionary":
                    count = len(archive["entity_id"])
                elif name == "queryDictionary":
                    count = len(archive["query_entity_index"])
                else:
                    count = len(archive["panel_id"])
            manifest[name].update(
                {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                    "count": count,
                }
            )
        trajectory_path = directory / manifest["trajectoryInterventions"]["path"]
        trajectory_count = len(trajectory_path.read_bytes().splitlines())
        manifest["trajectoryInterventions"].update(
            {
                "sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
                "bytes": trajectory_path.stat().st_size,
                "count": trajectory_count,
            }
        )
        active_contexts: set[int] = set()
        total_records = 0
        total_targets = 0
        observation_files = []
        for shard in manifest["shards"]:
            shard_path = directory / shard["path"]
            shard_arrays = self._npz_arrays(shard_path)
            perturbation_ids = []
            for row in range(len(shard_arrays["record_id"])):
                row_actions = {
                    keys[int(index)]
                    for index, active in zip(
                        shard_arrays["action_entity_index"][row],
                        shard_arrays["action_mask"][row],
                        strict=True,
                    )
                    if bool(active)
                }
                perturbation_ids.append(audit._perturbation_id(row_actions))
            shard_arrays["perturbation_id"] = np.asarray(
                perturbation_ids, dtype="<U96"
            )
            np.savez(shard_path, **shard_arrays)
            with np.load(shard_path, allow_pickle=False) as archive:
                records = len(archive["record_id"])
                targets = len(archive["target_value"])
                total_records += records
                total_targets += targets
                for index, flag in zip(
                    archive["context_entity_index"].flat,
                    archive["context_mask"].flat,
                    strict=True,
                ):
                    if bool(flag):
                        active_contexts.add(int(index))
            shard.update(
                {
                    "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                    "bytes": shard_path.stat().st_size,
                    "records": records,
                    "targetValues": targets,
                }
            )
            observation_files.append(self._file_ref(shard_path, directory))
        digest_basis = dict(manifest["featurePack"])
        del digest_basis["sha256"]
        manifest["featurePack"]["sha256"] = audit.canonical_sha256(digest_basis)
        manifest["inputs"]["staticFeatures"].update(
            {
                "semanticSha256": manifest["featurePack"]["blocks"][0][
                    "semanticSha256"
                ],
                "files": [self._file_ref(entity_path, directory)],
            }
        )
        manifest["inputs"]["observations"]["files"] = sorted(
            observation_files, key=lambda item: item["path"]
        )
        manifest["counts"] = {
            "entities": len(keys),
            "featureRows": int(np.count_nonzero(complete_rows)),
            "contexts": len(active_contexts),
            "queries": manifest["queryDictionary"]["count"],
            "panels": manifest["queryPanels"]["count"],
            "trajectoryInterventions": trajectory_count,
            "records": total_records,
            "targetValues": total_targets,
            "shards": len(manifest["shards"]),
        }
        manifest["bounds"]["maxRecordsPerShard"] = max(
            item["records"] for item in manifest["shards"]
        )
        manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))

    @staticmethod
    def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}

    @staticmethod
    def _write_trajectory(
        directory: Path, manifest: dict[str, object], keys: set[tuple[int, str]]
    ) -> None:
        rows = [
            {
                "schema": audit.TRAJECTORY_INTERVENTION_SCHEMA,
                "ncbiTaxon": taxon,
                "entityId": identifier,
            }
            for taxon, identifier in keys
        ]
        rows.sort(key=lambda row: (row["ncbiTaxon"], row["entityId"]))
        path = directory / manifest["trajectoryInterventions"]["path"]
        path.write_bytes(
            b"".join(audit.canonical_json_bytes(row, newline=True) for row in rows)
        )

    def _retag_active_corpus(
        self, pretrain: dict[str, object], taxon: int
    ) -> None:
        directory = Path(pretrain["path"])
        manifest_path = directory / "corpus.json"
        manifest = json.loads(manifest_path.read_bytes())
        entity_path = directory / manifest["entityDictionary"]["path"]
        entities = self._npz_arrays(entity_path)
        entities["entity_taxon"][:] = taxon
        np.savez(entity_path, **entities)
        shard_path = directory / manifest["shards"][0]["path"]
        shard = self._npz_arrays(shard_path)
        shard["species_taxon"][:] = taxon
        np.savez(shard_path, **shard)
        manifest["species"] = [
            {"taxon": taxon, "featureValue": [1.0], "featurePresent": [True]}
        ]
        manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
        action_keys = {
            (taxon, str(entities["entity_id"][int(index)]))
            for index, active in zip(
                shard["action_entity_index"].flat,
                shard["action_mask"].flat,
                strict=True,
            )
            if bool(active)
        }
        self._write_trajectory(directory, manifest, action_keys)
        self._refresh_corpus(pretrain)

    def _make_dual_taxon_corpus(self, pretrain: dict[str, object]) -> str:
        directory = Path(pretrain["path"])
        manifest_path = directory / "corpus.json"
        manifest = json.loads(manifest_path.read_bytes())
        entity_path = directory / manifest["entityDictionary"]["path"]
        entities = self._npz_arrays(entity_path)
        identifier = str(entities["entity_id"][0])
        for name, array in tuple(entities.items()):
            second_gene = array[:1].copy()
            second_context = array[1:2].copy()
            if name == "entity_taxon":
                second_gene[:] = 9606
                second_context[:] = 9606
            elif name == "entity_feature_value":
                second_gene[:] = 2.0
            entities[name] = np.concatenate(
                [array, second_gene, second_context], axis=0
            )
        np.savez(entity_path, **entities)

        query_path = directory / manifest["queryDictionary"]["path"]
        queries = self._npz_arrays(query_path)
        queries["query_entity_index"] = np.asarray([0, 2], dtype=np.int64)
        queries["query_readout_index"] = np.asarray([0, 0], dtype=np.int64)
        np.savez(query_path, **queries)

        panel_path = directory / manifest["queryPanels"]["path"]
        panels = self._npz_arrays(panel_path)
        panels["panel_indptr"] = np.asarray([0, 2], dtype=np.int64)
        panels["panel_query_index"] = np.asarray([0, 1], dtype=np.int64)
        np.savez(panel_path, **panels)

        shard_path = directory / manifest["shards"][0]["path"]
        shard = self._npz_arrays(shard_path)
        for name, array in tuple(shard.items()):
            if name == "target_indptr":
                shard[name] = np.asarray([0, 1, 2], dtype=np.int64)
            elif name == "target_query_index":
                shard[name] = np.asarray([0, 1], dtype=np.int64)
            elif array.shape[0] == 1:
                shard[name] = np.concatenate([array, array.copy()], axis=0)
        shard["record_id"][1] = "TEST:pretrain-human"
        shard["species_taxon"][1] = 9606
        shard["species_feature_value"][1, 0] = 2.0
        shard["action_entity_index"][1, 0] = 2
        shard["context_entity_index"][1, 0] = 3
        np.savez(shard_path, **shard)

        manifest["species"] = [
            {"taxon": 4932, "featureValue": [1.0], "featurePresent": [True]},
            {"taxon": 9606, "featureValue": [2.0], "featurePresent": [True]},
        ]
        manifest["bounds"]["maxPanelQueries"] = 2
        manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
        self._write_trajectory(
            directory, manifest, {(4932, identifier), (9606, identifier)}
        )
        self._refresh_corpus(pretrain)
        return identifier

    def _content_identities(
        self,
        pretrain: object,
        held: object,
        inventories: dict[str, object],
    ) -> tuple[object, dict[str, object], dict[str, object]]:
        bounds = audit.AuditBounds()
        corpus = audit.load_corpus(
            audit.resolve_dataset_input(pretrain, "pretrain"), "pretrain", bounds
        )
        loaded = {
            name: audit.load_protected_inventory(
                audit.resolve_dataset_input(value, name), bounds
            )
            for name, value in inventories.items()
        }
        ordered = tuple(sorted(loaded.values(), key=lambda item: item.source_id))
        held_identity, _ = audit.load_held_roster(
            audit.resolve_dataset_input(held, "heldRoster"), ordered, bounds
        )
        inventory_identities = {
            name: audit._protected_inventory_identity(value)
            for name, value in loaded.items()
        }
        return corpus, held_identity, inventory_identities

    def _add_static_only_held_gene(
        self, pretrain: dict[str, object], held_gene: str
    ) -> None:
        directory = Path(pretrain["path"])
        manifest_path = directory / "corpus.json"
        manifest = json.loads(manifest_path.read_bytes())
        entity_path = directory / manifest["entityDictionary"]["path"]
        with np.load(entity_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["entity_id"] = np.concatenate(
            [arrays["entity_id"], np.asarray([held_gene], dtype=arrays["entity_id"].dtype)]
        )
        arrays["entity_type"] = np.concatenate(
            [arrays["entity_type"], np.asarray([0], dtype=np.int64)]
        )
        arrays["entity_taxon"] = np.concatenate(
            [arrays["entity_taxon"], np.asarray([4932], dtype=np.int64)]
        )
        arrays["entity_feature_value"] = np.concatenate(
            [arrays["entity_feature_value"], np.ones((1, 1), dtype=np.float32)]
        )
        arrays["entity_feature_present"] = np.concatenate(
            [arrays["entity_feature_present"], np.ones((1, 1), dtype=np.bool_)]
        )
        old_count = len(arrays["entity_id"])
        order = sorted(
            range(old_count),
            key=lambda index: (
                int(arrays["entity_taxon"][index]),
                str(arrays["entity_id"][index]),
            ),
        )
        inverse = np.empty(old_count, dtype=np.int64)
        for new_index, old_index in enumerate(order):
            inverse[old_index] = new_index
        arrays = {name: array[order] for name, array in arrays.items()}
        np.savez(entity_path, **arrays)

        query_path = directory / manifest["queryDictionary"]["path"]
        queries = self._npz_arrays(query_path)
        queries["query_entity_index"] = inverse[queries["query_entity_index"]]
        np.savez(query_path, **queries)
        for shard in manifest["shards"]:
            shard_path = directory / shard["path"]
            shard_arrays = self._npz_arrays(shard_path)
            for axis in ("context", "action"):
                indices = shard_arrays[f"{axis}_entity_index"]
                mask = shard_arrays[f"{axis}_mask"]
                indices[mask] = inverse[indices[mask]]
            np.savez(shard_path, **shard_arrays)
        manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
        self._refresh_corpus(pretrain)

    def _authorization(
        self,
        root: Path,
        pretrain: dict[str, object],
        held: dict[str, object],
        inventories: dict[str, object],
        *,
        statement_mutator=None,
    ) -> dict[str, object]:
        corpus, held_identity, inventory_identities = self._content_identities(
            pretrain, held, inventories
        )
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        key_id = "sha256:" + hashlib.sha256(public_key).hexdigest()
        statement = {
            "schema": authorization.AUTHORIZATION_SCHEMA,
            "authorizationId": "urn:uuid:123e4567-e89b-42d3-a456-426614174000",
            "issuedAt": "2026-09-04T17:30:00Z",
            "issuer": {
                "name": authorization.EXPECTED_ISSUER,
                "keyId": key_id,
            },
            "recipient": {
                "namespace": authorization.EXPECTED_RECIPIENT_NAMESPACE,
                "factoryIdentity": self.FACTORY_IDENTITY,
                "challengeNonce": self.CHALLENGE,
            },
            "purpose": authorization.EXPECTED_PURPOSE,
            "protocol": {
                "auditSchema": audit.AUDIT_SCHEMA,
                "rewardEnabled": False,
                "protectedQuantitativeTruthIncluded": False,
                "benchmarkLabelsIncluded": False,
            },
            "datasets": {
                "pretrain": {
                    "resource": pretrain["resource"],
                    "manifestDigest": pretrain["manifestDigest"],
                    "corpusManifestSha256": corpus.identity[
                        "corpusManifestSha256"
                    ],
                    "contentDigest": corpus.identity["contentDigest"],
                },
                "heldRoster": {
                    "resource": held["resource"],
                    "manifestDigest": held["manifestDigest"],
                    "rosterSha256": held_identity["rosterSha256"],
                    "coverageSha256": held_identity["coverageSha256"],
                },
                "protectedInventories": [
                    {
                        "inputName": name,
                        "resource": inventories[name]["resource"],
                        "manifestDigest": inventories[name]["manifestDigest"],
                        "inventoryManifestSha256": inventory_identities[name][
                            "manifestSha256"
                        ],
                    }
                    for name in sorted(inventories)
                ],
            },
        }
        if statement_mutator is not None:
            statement_mutator(statement)
        statement_bytes = audit.canonical_json_bytes(statement, newline=True)
        message = (
            authorization.SIGNATURE_DOMAIN
            + len(statement_bytes).to_bytes(8, "big")
            + statement_bytes
        )
        signature = private_key.sign(message)

        builder = _v12_fixtures.CorpusAuditTest(
            methodName="test_passing_audit_is_deterministic_and_schema_valid"
        )
        input_name = "custodianBoundaryAttestation"
        resource_name = "fixture-custodian-authorization"
        directory = builder._snapshot_root(root, input_name, resource_name)
        (directory / "authorization.json").write_bytes(statement_bytes)
        (directory / "authorization.ed25519").write_bytes(
            signature.hex().encode("ascii") + b"\n"
        )
        trust_anchor = root / "test-custodian.pub"
        trust_anchor.write_bytes(public_key.hex().encode("ascii") + b"\n")
        return {
            "input": builder._dataset_input(directory, input_name, resource_name),
            "statement": statement,
            "privateKey": private_key,
            "trustAnchor": trust_anchor,
            "keyId": key_id,
            "trustTextSha256": hashlib.sha256(trust_anchor.read_bytes()).hexdigest(),
        }

    def _verify(
        self,
        handoff: dict[str, object],
        pretrain: dict[str, object],
        held: dict[str, object],
        inventories: dict[str, object],
    ):
        return authorization._verify_authorization_with_anchor(
            handoff["input"],
            {"pretrain": pretrain, "heldRoster": held, **inventories},
            recipient_factory_identity=self.FACTORY_IDENTITY,
            challenge_nonce=self.CHALLENGE,
            trust_anchor_path=handoff["trustAnchor"],
            expected_key_id=handoff["keyId"],
            expected_text_sha256=handoff["trustTextSha256"],
        )

    @contextmanager
    def _patched_verifier(self, handoff: dict[str, object]):
        original = authorization.verify_custodian_authorization
        prior_module = sys.modules.get("attestation")

        def ephemeral(dataset_input, actual_inputs, **kwargs):
            return authorization._verify_authorization_with_anchor(
                dataset_input,
                actual_inputs,
                trust_anchor_path=handoff["trustAnchor"],
                expected_key_id=handoff["keyId"],
                expected_text_sha256=handoff["trustTextSha256"],
                **kwargs,
            )

        authorization.verify_custodian_authorization = ephemeral
        sys.modules["attestation"] = authorization
        try:
            yield
        finally:
            authorization.verify_custodian_authorization = original
            if prior_module is None:
                del sys.modules["attestation"]
            else:
                sys.modules["attestation"] = prior_module

    def test_valid_ephemeral_authorization_and_training_audit_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)
            handoff = self._authorization(root, pretrain, held, inventories)
            verified = self._verify(handoff, pretrain, held, inventories)
            self.assertEqual(verified.key_id, handoff["keyId"])
            with self._patched_verifier(handoff):
                outputs = []
                for name in ("first", "second"):
                    report, digest = audit.write_training_audit_artifact(
                        pretrain,
                        held,
                        inventories,
                        handoff["input"],
                        root / name,
                        reward_enabled=False,
                        recipient_factory_identity=self.FACTORY_IDENTITY,
                        challenge_nonce=self.CHALLENGE,
                    )
                    payload = root / name / "corpus-audit.json"
                    self.assertEqual(hashlib.sha256(payload.read_bytes()).hexdigest(), digest)
                    self.assertEqual({item.name for item in payload.parent.iterdir()}, {payload.name})
                    self.assertEqual(set(report["datasets"]), {"pretrain"})
                    self.assertFalse(report["protectedTruthInputsPresent"])
                    self.assertTrue(report["custodianSignatureVerified"])
                    outputs.append(payload.read_bytes())
            self.assertEqual(outputs[0], outputs[1])
            schema = json.loads((MODULE / "corpus-audit.schema.json").read_bytes())
            self.assertEqual(set(schema["required"]), set(json.loads(outputs[0])))

    def test_production_verifier_is_unprovisioned_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)
            handoff = self._authorization(root, pretrain, held, inventories)
            with self.assertRaisesRegex(
                authorization.AuthorizationError, "not provisioned"
            ):
                authorization.verify_custodian_authorization(
                    handoff["input"],
                    {"pretrain": pretrain, "heldRoster": held, **inventories},
                    recipient_factory_identity=self.FACTORY_IDENTITY,
                    challenge_nonce=self.CHALLENGE,
                )
        self.assertIsNone(authorization.PINNED_CUSTODIAN_KEY_ID)
        self.assertFalse((MODULE / "trust" / authorization.TRUST_ANCHOR_NAME).exists())

    def test_signature_framing_key_and_canonical_bytes_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)
            handoff = self._authorization(root, pretrain, held, inventories)
            directory = Path(handoff["input"]["path"])
            signature_path = directory / "authorization.ed25519"
            signature = bytearray.fromhex(signature_path.read_text().strip())
            signature[0] ^= 1
            signature_path.write_bytes(bytes(signature).hex().encode() + b"\n")
            with self.assertRaisesRegex(
                authorization.AuthorizationError, "signature verification"
            ):
                self._verify(handoff, pretrain, held, inventories)

            handoff = self._authorization(
                root / "canonical", pretrain, held, inventories
            )
            statement_path = Path(handoff["input"]["path"]) / "authorization.json"
            value = json.loads(statement_path.read_bytes())
            statement_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(
                authorization.AuthorizationError, "canonical JSON"
            ):
                self._verify(handoff, pretrain, held, inventories)

    def test_recipient_protocol_and_actual_input_binding_reject_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)
            wrong_recipient = self._authorization(
                root / "recipient",
                pretrain,
                held,
                inventories,
                statement_mutator=lambda value: value["recipient"].__setitem__(
                    "factoryIdentity", "sha256:" + "c" * 64
                ),
            )
            with self.assertRaisesRegex(
                authorization.AuthorizationError, "recipient identity"
            ):
                self._verify(wrong_recipient, pretrain, held, inventories)

            handoff = self._authorization(root / "binding", pretrain, held, inventories)
            changed = dict(pretrain)
            changed["manifestDigest"] = "sha256:" + "d" * 64
            with self.assertRaisesRegex(
                authorization.AuthorizationError, "exact pretrain"
            ):
                self._verify(handoff, changed, held, inventories)

    def test_signed_inner_content_claims_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)

            def mutate(value):
                value["datasets"]["pretrain"]["contentDigest"] = "e" * 64

            handoff = self._authorization(
                root, pretrain, held, inventories, statement_mutator=mutate
            )
            verified = self._verify(handoff, pretrain, held, inventories)
            corpus, held_identity, inventory_identities = self._content_identities(
                pretrain, held, inventories
            )
            with self.assertRaisesRegex(
                authorization.AuthorizationError, "audited content"
            ):
                authorization.assert_authorized_content(
                    verified,
                    pretrain_identity=corpus.identity,
                    held_roster_identity=held_identity,
                    inventory_identities=inventory_identities,
                )

    def test_each_inner_content_identity_is_signed(self) -> None:
        mutations = {
            "roster": lambda value: value["datasets"]["heldRoster"].__setitem__(
                "rosterSha256", "1" * 64
            ),
            "coverage": lambda value: value["datasets"]["heldRoster"].__setitem__(
                "coverageSha256", "2" * 64
            ),
            "inventory": lambda value: value["datasets"][
                "protectedInventories"
            ][0].__setitem__("inventoryManifestSha256", "3" * 64),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                pretrain, held, inventories = self._safe_inputs(root)
                handoff = self._authorization(
                    root,
                    pretrain,
                    held,
                    inventories,
                    statement_mutator=mutation,
                )
                verified = self._verify(handoff, pretrain, held, inventories)
                corpus, held_identity, inventory_identities = self._content_identities(
                    pretrain, held, inventories
                )
                with self.assertRaisesRegex(
                    authorization.AuthorizationError, "audited content"
                ):
                    authorization.assert_authorized_content(
                        verified,
                        pretrain_identity=corpus.identity,
                        held_roster_identity=held_identity,
                        inventory_identities=inventory_identities,
                    )

    def test_held_gene_static_features_are_allowed_but_not_active_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)
            held_gene = _v12_fixtures.CorpusAuditTest.by_role[
                "molecular-validation"
            ]
            self._add_static_only_held_gene(pretrain, held_gene)
            handoff = self._authorization(root, pretrain, held, inventories)
            with self._patched_verifier(handoff):
                report = audit.audit_training_corpus(
                    pretrain,
                    held,
                    inventories,
                    handoff["input"],
                    reward_enabled=False,
                    recipient_factory_identity=self.FACTORY_IDENTITY,
                    challenge_nonce=self.CHALLENGE,
                )
            self.assertEqual(report["datasets"]["pretrain"]["entityCount"], 3)
            self.assertEqual(
                report["datasets"]["pretrain"]["trajectoryInterventionCount"], 1
            )

    def test_two_or_more_inventories_are_supported_but_one_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root / "three", inventory_count=3)
            handoff = self._authorization(root / "three", pretrain, held, inventories)
            self.assertEqual(len(self._verify(handoff, pretrain, held, inventories).inventory_claims), 3)

            pretrain, held, inventories = self._safe_inputs(root / "one")

            def keep_one(value):
                value["datasets"]["protectedInventories"] = value["datasets"][
                    "protectedInventories"
                ][:1]

            handoff = self._authorization(
                root / "one",
                pretrain,
                held,
                inventories,
                statement_mutator=keep_one,
            )
            with self.assertRaisesRegex(
                authorization.AuthorizationError, "between 2 and 64"
            ):
                self._verify(handoff, pretrain, held, inventories)

    def test_validation_and_final_actions_are_rejected_without_truth_inputs(self) -> None:
        for role in ("molecular-validation", "molecular-final"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                pretrain, held, inventories = self._safe_inputs(
                    root, active_gene_role=role
                )
                handoff = self._authorization(root, pretrain, held, inventories)
                with self._patched_verifier(handoff), self.assertRaisesRegex(
                    audit.CorpusAuditError, "quantitative fitting trajectories"
                ):
                    audit.audit_training_corpus(
                        pretrain,
                        held,
                        inventories,
                        handoff["input"],
                        reward_enabled=False,
                        recipient_factory_identity=self.FACTORY_IDENTITY,
                        challenge_nonce=self.CHALLENGE,
                    )

    def test_forbidden_inputs_reward_and_workload_contract_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)
            handoff = self._authorization(root, pretrain, held, inventories)
            with self._patched_verifier(handoff), self.assertRaisesRegex(
                audit.CorpusAuditError, "rewardEnabled=false"
            ):
                audit.audit_training_corpus(
                    pretrain,
                    held,
                    inventories,
                    handoff["input"],
                    reward_enabled=True,
                    recipient_factory_identity=self.FACTORY_IDENTITY,
                    challenge_nonce=self.CHALLENGE,
                )
            forbidden = {**inventories, "molecularValidation": pretrain}
            with self.assertRaisesRegex(
                audit.CorpusAuditError, "protected inventory input names"
            ):
                audit.audit_training_corpus(
                    pretrain,
                    held,
                    forbidden,
                    handoff["input"],
                    reward_enabled=False,
                    recipient_factory_identity=self.FACTORY_IDENTITY,
                    challenge_nonce=self.CHALLENGE,
                )

        module = yaml.safe_load((MODULE / "module.yaml").read_text())
        contract = module["spec"]["contracts"]
        self.assertEqual(
            set(contract["input"]["required"]),
            {"pretrain", "heldRoster", "custodianBoundaryAttestation"},
        )
        self.assertEqual(contract["input"]["minProperties"], 5)
        self.assertEqual(contract["config"]["properties"]["rewardEnabled"], {"const": False})
        self.assertNotIn("trust", json.dumps(contract).casefold())
        fixture_inputs = module["spec"]["fixtures"][0]["request"]["inputs"]
        self.assertEqual(set(fixture_inputs), {
            "pretrain", "heldRoster", "custodianBoundaryAttestation",
            "protectedInventoryOne", "protectedInventoryTwo",
        })
        for value in fixture_inputs.values():
            self.assertEqual(set(value), {"resource", "mode", "path", "manifestDigest"})
            self.assertEqual(value["mode"], "copy")

    def test_authorization_directory_and_trust_anchor_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)
            handoff = self._authorization(root, pretrain, held, inventories)
            directory = Path(handoff["input"]["path"])
            (directory / "extra.txt").write_text("not allowed", encoding="ascii")
            with self.assertRaisesRegex(
                authorization.AuthorizationError, "exactly two"
            ):
                self._verify(handoff, pretrain, held, inventories)
            (directory / "extra.txt").unlink()

            trust = Path(handoff["trustAnchor"])
            trust.write_bytes(trust.read_bytes().upper())
            with self.assertRaisesRegex(
                authorization.AuthorizationError, "64 lowercase hex"
            ):
                self._verify(handoff, pretrain, held, inventories)

    def test_authorization_claims_are_immutable_after_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)
            handoff = self._authorization(root, pretrain, held, inventories)
            verified = self._verify(handoff, pretrain, held, inventories)
            with self.assertRaises(TypeError):
                verified.pretrain_claim["contentDigest"] = "0" * 64

    def test_same_content_replay_is_reported_as_external_ledger_responsibility(self) -> None:
        contract = (MODULE / "CONTRACT.md").read_text().casefold()
        self.assertIn("consumed", contract)
        self.assertIn("ledger", contract)
        self.assertIn("cannot prove one-time", contract)

    def test_same_text_id_in_two_taxa_routes_as_two_composite_entities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, _, _ = self._safe_inputs(root)
            identifier = self._make_dual_taxon_corpus(pretrain)
            corpus = audit.load_corpus(
                audit.resolve_dataset_input(pretrain, "pretrain"),
                "pretrain",
                audit.AuditBounds(),
            )
            expected = frozenset({(4932, identifier), (9606, identifier)})
            self.assertEqual(corpus.trajectory_interventions, expected)
            self.assertEqual(corpus.active_actions, expected)
            self.assertEqual(corpus.identity["entityCount"], 4)
            self.assertEqual(corpus.identity["queryCount"], 2)
            self.assertEqual(
                corpus.identity["trajectoryInterventionSetSha256"],
                audit._composite_set_sha256(set(expected)),
            )

    def test_composite_hashes_order_by_taxon_then_entity_id(self) -> None:
        keys = {(4932, "SGD:S999999999"), (9606, "HGNC:1")}
        tuple_ordered = [
            {"ncbiTaxon": 4932, "entityId": "SGD:S999999999"},
            {"ncbiTaxon": 9606, "entityId": "HGNC:1"},
        ]
        entity_id_ordered = list(reversed(tuple_ordered))
        expected = audit.canonical_sha256(tuple_ordered)
        self.assertEqual(audit._composite_set_sha256(keys), expected)
        self.assertNotEqual(expected, audit.canonical_sha256(entity_id_ordered))
        self.assertEqual(
            audit._perturbation_id(keys),
            "slp-perturbation:sha256-" + expected,
        )

    def test_duplicate_identical_composite_entity_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, _, _ = self._safe_inputs(root)
            directory = Path(pretrain["path"])
            manifest = json.loads((directory / "corpus.json").read_bytes())
            entity_path = directory / manifest["entityDictionary"]["path"]
            arrays = self._npz_arrays(entity_path)
            arrays = {
                name: np.concatenate([array, array[:1]], axis=0)
                for name, array in arrays.items()
            }
            np.savez(entity_path, **arrays)
            self._refresh_corpus(pretrain)
            with self.assertRaisesRegex(
                audit.CorpusAuditError, "composite entity keys must be unique"
            ):
                audit.load_corpus(
                    audit.resolve_dataset_input(pretrain, "pretrain"),
                    "pretrain",
                    audit.AuditBounds(),
                )

    def test_entity_and_query_composite_dictionaries_require_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, _, _ = self._safe_inputs(root / "entities")
            directory = Path(pretrain["path"])
            manifest = json.loads((directory / "corpus.json").read_bytes())
            entity_path = directory / manifest["entityDictionary"]["path"]
            arrays = self._npz_arrays(entity_path)
            arrays = {name: array[::-1].copy() for name, array in arrays.items()}
            np.savez(entity_path, **arrays)
            self._refresh_corpus(pretrain)
            with self.assertRaisesRegex(audit.CorpusAuditError, "must be ordered"):
                audit.load_corpus(
                    audit.resolve_dataset_input(pretrain, "pretrain"),
                    "pretrain",
                    audit.AuditBounds(),
                )

            pretrain, _, _ = self._safe_inputs(root / "queries")
            self._make_dual_taxon_corpus(pretrain)
            directory = Path(pretrain["path"])
            manifest = json.loads((directory / "corpus.json").read_bytes())
            query_path = directory / manifest["queryDictionary"]["path"]
            queries = self._npz_arrays(query_path)
            queries["query_entity_index"][1] = queries["query_entity_index"][0]
            queries["query_readout_index"][1] = queries["query_readout_index"][0]
            np.savez(query_path, **queries)
            self._refresh_corpus(pretrain)
            with self.assertRaisesRegex(
                audit.CorpusAuditError, "query dictionary composite identities"
            ):
                audit.load_corpus(
                    audit.resolve_dataset_input(pretrain, "pretrain"),
                    "pretrain",
                    audit.AuditBounds(),
                )

    def test_query_panel_structure_and_indices_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, _, _ = self._safe_inputs(root)
            directory = Path(pretrain["path"])
            manifest = json.loads((directory / "corpus.json").read_bytes())
            panel_path = directory / manifest["queryPanels"]["path"]
            panels = self._npz_arrays(panel_path)
            panels["panel_query_index"][0] = 1
            np.savez(panel_path, **panels)
            self._refresh_corpus(pretrain)
            with self.assertRaisesRegex(
                audit.CorpusAuditError, "out-of-range query index"
            ):
                audit.load_corpus(
                    audit.resolve_dataset_input(pretrain, "pretrain"),
                    "pretrain",
                    audit.AuditBounds(),
                )

    def test_each_target_must_belong_to_its_records_selected_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, _, _ = self._safe_inputs(root)
            self._make_dual_taxon_corpus(pretrain)
            directory = Path(pretrain["path"])
            manifest_path = directory / "corpus.json"
            manifest = json.loads(manifest_path.read_bytes())
            panel_path = directory / manifest["queryPanels"]["path"]
            panels = self._npz_arrays(panel_path)
            panels["panel_id"] = np.asarray(
                ["TESTPANEL:human", "TESTPANEL:yeast"], dtype="<U32"
            )
            panels["panel_indptr"] = np.asarray([0, 1, 2], dtype=np.int64)
            panels["panel_query_index"] = np.asarray([0, 1], dtype=np.int64)
            np.savez(panel_path, **panels)
            shard_path = directory / manifest["shards"][0]["path"]
            shard = self._npz_arrays(shard_path)
            shard["query_panel_index"] = np.asarray([0, 0], dtype=np.int64)
            np.savez(shard_path, **shard)
            manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
            self._refresh_corpus(pretrain)
            with self.assertRaisesRegex(
                audit.CorpusAuditError, "absent from its selected panel"
            ):
                audit.load_corpus(
                    audit.resolve_dataset_input(pretrain, "pretrain"),
                    "pretrain",
                    audit.AuditBounds(),
                )

    def test_all_shard_identity_and_covariate_contracts_are_inspected(self) -> None:
        def wrong_string(arrays, manifest):
            arrays["observation_unit_id"] = np.asarray([1], dtype=np.int64)

        def wrong_species_feature(arrays, manifest):
            arrays["species_feature_value"][0, 0] = 2.0

        def wrong_context_type(arrays, manifest):
            arrays["context_type"][0, 0] = 1

        def wrong_panel(arrays, manifest):
            arrays["query_panel_index"][0] = 1

        def absent_action_covariate_is_nonzero(arrays, manifest):
            manifest["covariates"]["action"] = [
                {"id": "TESTCOV:dose", "unit": "TESTUNIT:arbitrary", "access": "world"}
            ]
            arrays["action_covariate_value"] = np.ones((1, 1, 1), dtype=np.float32)
            arrays["action_covariate_present"] = np.zeros((1, 1, 1), dtype=np.bool_)

        def repeated_target_query(arrays, manifest):
            arrays["target_indptr"] = np.asarray([0, 2], dtype=np.int64)
            arrays["target_query_index"] = np.asarray([0, 0], dtype=np.int64)
            arrays["target_value"] = np.asarray([0.25, 0.5], dtype=np.float32)
            manifest["bounds"]["maxTargetsPerRecord"] = 2

        cases = (
            ("string", wrong_string, "record-aligned Unicode"),
            ("species", wrong_species_feature, "species features"),
            ("context-type", wrong_context_type, "context_type is out of range"),
            ("panel", wrong_panel, "query_panel_index is out of range"),
            ("covariate", absent_action_covariate_is_nonzero, "action covariates"),
            ("target", repeated_target_query, "repeats a target query"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, mutate, message in cases:
                with self.subTest(name=name):
                    pretrain, _, _ = self._safe_inputs(root / name)
                    directory = Path(pretrain["path"])
                    manifest_path = directory / "corpus.json"
                    manifest = json.loads(manifest_path.read_bytes())
                    shard_path = directory / manifest["shards"][0]["path"]
                    arrays = self._npz_arrays(shard_path)
                    mutate(arrays, manifest)
                    np.savez(shard_path, **arrays)
                    manifest_path.write_bytes(
                        audit.canonical_json_bytes(manifest, newline=True)
                    )
                    self._refresh_corpus(pretrain)
                    with self.assertRaisesRegex(audit.CorpusAuditError, message):
                        audit.load_corpus(
                            audit.resolve_dataset_input(pretrain, "pretrain"),
                            "pretrain",
                            audit.AuditBounds(),
                        )

    def test_taxon_substitution_changes_composite_hashes_and_inconsistency_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, _, _ = self._safe_inputs(root / "consistent")
            before = audit.load_corpus(
                audit.resolve_dataset_input(pretrain, "pretrain"),
                "pretrain",
                audit.AuditBounds(),
            ).identity
            self._retag_active_corpus(pretrain, 9606)
            after = audit.load_corpus(
                audit.resolve_dataset_input(pretrain, "pretrain"),
                "pretrain",
                audit.AuditBounds(),
            ).identity
            for field in (
                "entityKeySetSha256",
                "trajectoryInterventionSetSha256",
                "contentDigest",
            ):
                self.assertNotEqual(before[field], after[field])

            pretrain, _, _ = self._safe_inputs(root / "inconsistent")
            directory = Path(pretrain["path"])
            manifest = json.loads((directory / "corpus.json").read_bytes())
            entity_path = directory / manifest["entityDictionary"]["path"]
            arrays = self._npz_arrays(entity_path)
            arrays["entity_taxon"][:] = 9606
            np.savez(entity_path, **arrays)
            manifest["species"].append(
                {"taxon": 9606, "featureValue": [2.0], "featurePresent": [True]}
            )
            (directory / "corpus.json").write_bytes(
                audit.canonical_json_bytes(manifest, newline=True)
            )
            self._refresh_corpus(pretrain)
            with self.assertRaisesRegex(
                audit.CorpusAuditError,
                "composite key outside entityDictionary|taxon does not match",
            ):
                audit.load_corpus(
                    audit.resolve_dataset_input(pretrain, "pretrain"),
                    "pretrain",
                    audit.AuditBounds(),
                )

    def test_held_yeast_key_does_not_block_same_text_id_in_human_taxon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(
                root, active_gene_role="molecular-validation"
            )
            held_id = _v12_fixtures.CorpusAuditTest.by_role[
                "molecular-validation"
            ]
            self._retag_active_corpus(pretrain, 9606)
            handoff = self._authorization(root, pretrain, held, inventories)
            with self._patched_verifier(handoff):
                report = audit.audit_training_corpus(
                    pretrain,
                    held,
                    inventories,
                    handoff["input"],
                    reward_enabled=False,
                    recipient_factory_identity=self.FACTORY_IDENTITY,
                    challenge_nonce=self.CHALLENGE,
                )
            self.assertTrue(report["auditPassed"])
            corpus = audit.load_corpus(
                audit.resolve_dataset_input(pretrain, "pretrain"),
                "pretrain",
                audit.AuditBounds(),
            )
            self.assertEqual(corpus.active_actions, frozenset({(9606, held_id)}))

    def test_legacy_trajectory_genes_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, _, _ = self._safe_inputs(root)
            directory = Path(pretrain["path"])
            manifest_path = directory / "corpus.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["trajectoryGenes"] = manifest.pop("trajectoryInterventions")
            manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
            with self.assertRaisesRegex(audit.CorpusAuditError, "fields mismatch"):
                audit.load_corpus(
                    audit.resolve_dataset_input(pretrain, "pretrain"),
                    "pretrain",
                    audit.AuditBounds(),
                )

    def test_perturbation_id_must_bind_sorted_unique_composite_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, _, _ = self._safe_inputs(root)
            directory = Path(pretrain["path"])
            manifest_path = directory / "corpus.json"
            manifest = json.loads(manifest_path.read_bytes())
            shard_path = directory / manifest["shards"][0]["path"]
            arrays = self._npz_arrays(shard_path)
            arrays["perturbation_id"] = np.asarray(
                ["slp-perturbation:sha256-" + "0" * 64], dtype="<U96"
            )
            np.savez(shard_path, **arrays)
            shard_ref = self._file_ref(shard_path, directory)
            manifest["shards"][0].update(shard_ref)
            manifest["inputs"]["observations"]["files"] = [shard_ref]
            manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))
            with self.assertRaisesRegex(audit.CorpusAuditError, "perturbation_id"):
                audit.load_corpus(
                    audit.resolve_dataset_input(pretrain, "pretrain"),
                    "pretrain",
                    audit.AuditBounds(),
                )

    def test_single_uncompressed_composite_corpus_tar_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)
            directory = Path(pretrain["path"])
            source_files = sorted(path for path in directory.rglob("*") if path.is_file())
            temporary_tar = directory.parent / "corpus-v1-2.tar"
            with tarfile.open(temporary_tar, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for path in source_files:
                    relative = path.relative_to(directory).as_posix()
                    archive.add(
                        path,
                        arcname=f"composite-corpus/{relative}",
                        recursive=False,
                    )
            for path in source_files:
                path.unlink()
            temporary_tar.replace(directory / "corpus-v1-2.tar")
            handoff = self._authorization(root, pretrain, held, inventories)
            with self._patched_verifier(handoff):
                report = audit.audit_training_corpus(
                    pretrain,
                    held,
                    inventories,
                    handoff["input"],
                    reward_enabled=False,
                    recipient_factory_identity=self.FACTORY_IDENTITY,
                    challenge_nonce=self.CHALLENGE,
                )
            self.assertTrue(report["auditPassed"])

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_reports_and_signed_statement_match_strict_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, held, inventories = self._safe_inputs(root)
            handoff = self._authorization(root, pretrain, held, inventories)
            with self._patched_verifier(handoff):
                report = audit.audit_training_corpus(
                    pretrain,
                    held,
                    inventories,
                    handoff["input"],
                    reward_enabled=False,
                    recipient_factory_identity=self.FACTORY_IDENTITY,
                    challenge_nonce=self.CHALLENGE,
                )
            report_schema = json.loads(
                (MODULE / "corpus-audit.schema.json").read_bytes()
            )
            statement_schema = json.loads(
                (MODULE / "custodian-boundary-attestation.schema.json").read_bytes()
            )
            jsonschema.Draft202012Validator(report_schema).validate(report)
            statement = json.loads(
                (
                    Path(handoff["input"]["path"]) / "authorization.json"
                ).read_bytes()
            )
            jsonschema.Draft202012Validator(statement_schema).validate(statement)


if __name__ == "__main__":
    unittest.main()
