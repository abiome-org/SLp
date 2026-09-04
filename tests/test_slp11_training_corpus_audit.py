"""Adversarial tests for the clean-training corpus audit v1.3."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import numpy as np
import yaml

from tests import test_slp11_corpus_audit as _v12_fixtures


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-training-corpus-audit-v1-3"

AUDIT_SPEC = importlib.util.spec_from_file_location("audit", MODULE / "audit.py")
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = audit
AUDIT_SPEC.loader.exec_module(audit)

AUTH_SPEC = importlib.util.spec_from_file_location(
    "attestation", MODULE / "attestation.py"
)
assert AUTH_SPEC is not None and AUTH_SPEC.loader is not None
authorization = importlib.util.module_from_spec(AUTH_SPEC)
sys.modules[AUTH_SPEC.name] = authorization
AUTH_SPEC.loader.exec_module(authorization)


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
        return pretrain, held, inventories

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
        arrays["entity_species_taxon"] = np.concatenate(
            [arrays["entity_species_taxon"], np.asarray([4932], dtype=np.int64)]
        )
        arrays["entity_feature_value"] = np.concatenate(
            [arrays["entity_feature_value"], np.ones((1, 1), dtype=np.float32)]
        )
        arrays["entity_feature_present"] = np.concatenate(
            [arrays["entity_feature_present"], np.ones((1, 1), dtype=np.bool_)]
        )
        np.savez(entity_path, **arrays)
        manifest["entityDictionary"]["count"] += 1
        manifest["entityDictionary"]["sha256"] = hashlib.sha256(
            entity_path.read_bytes()
        ).hexdigest()
        manifest_path.write_bytes(audit.canonical_json_bytes(manifest, newline=True))

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
        try:
            yield
        finally:
            authorization.verify_custodian_authorization = original

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
            self.assertEqual(report["datasets"]["pretrain"]["entityCount"], 2)
            self.assertEqual(
                report["datasets"]["pretrain"]["trajectoryGeneCount"], 1
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


if __name__ == "__main__":
    unittest.main()
