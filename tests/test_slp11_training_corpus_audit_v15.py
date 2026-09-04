"""Focused adversarial tests for packaged clean-training corpus audit v1.5."""

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
from types import SimpleNamespace

try:
    import jsonschema
except ModuleNotFoundError:  # pragma: no cover - exercised in the OMF environment
    jsonschema = None

from tests import test_slp11_training_corpus_audit_v14 as _v14_fixtures

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-training-corpus-audit-v1-5"

AUDIT_SPEC = importlib.util.spec_from_file_location(
    "slp11_training_corpus_audit_v15_audit", MODULE / "audit.py"
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = audit
AUDIT_SPEC.loader.exec_module(audit)
sys.modules["audit"] = audit

AUTH_SPEC = importlib.util.spec_from_file_location(
    "slp11_training_corpus_audit_v15_attestation", MODULE / "attestation.py"
)
assert AUTH_SPEC is not None and AUTH_SPEC.loader is not None
attestation = importlib.util.module_from_spec(AUTH_SPEC)
sys.modules[AUTH_SPEC.name] = attestation
AUTH_SPEC.loader.exec_module(attestation)


class PackagedTrainingCorpusAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _v14_fixtures.TrainingCorpusAuditTest.setUpClass()

    def _raw_inputs(self, root: Path) -> tuple[dict[str, object], object, object]:
        builder = _v14_fixtures.TrainingCorpusAuditTest(
            methodName="test_valid_ephemeral_authorization_and_training_audit_are_deterministic"
        )
        return builder._safe_inputs(root)

    @staticmethod
    def _pretty(value: object) -> bytes:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _tar_info(name: str, size: int) -> tarfile.TarInfo:
        info = tarfile.TarInfo(name)
        info.size = size
        info.mode = 0o644
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        return info

    def _package(self, pretrain: dict[str, object]) -> dict[str, object]:
        directory = Path(pretrain["path"])
        dataset = audit.resolve_dataset_input(pretrain, "pretrain")
        corpus = audit._load_corpus_root(
            dataset, "pretrain", audit.AuditBounds(), directory
        )
        facts = corpus.composition_facts
        members = {
            path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        buffer = io.BytesIO()
        with tarfile.open(
            fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive:
            for name in sorted(members):
                payload = members[name]
                archive.addfile(
                    self._tar_info(f"composite-corpus/{name}", len(payload)),
                    io.BytesIO(payload),
                )
        archive_payload = buffer.getvalue()
        companion = {
            "schema": audit.COMPOSITION_AUDIT_SCHEMA,
            "archive": {
                "path": "corpus-v1-2.tar",
                "bytes": len(archive_payload),
                "sha256": hashlib.sha256(archive_payload).hexdigest(),
            },
            "corpusManifestSha256": facts["corpusManifestSha256"],
            "inputs": facts["inputs"],
            "counts": facts["counts"],
            "identity": {
                "key": ["ncbiTaxon", "entityId"],
                "corpusEntityKeySetSha256": facts["entityKeySetSha256"],
                "featureEntityKeySetSha256": facts[
                    "featureEntityKeySetSha256"
                ],
                "contextEntity": facts["contextEntities"][0],
            },
            "featurePackSha256": facts["featurePackSha256"],
            "featurePreservation": {
                "rows": facts["counts"]["featureRows"],
                "dimension": corpus.identity["featurePack"]["entityFeatureDim"],
                "sourceValueBytesSha256": facts["featureValueBytesSha256"],
                "composedValueBytesSha256": facts["featureValueBytesSha256"],
                "sourcePresentBytesSha256": facts["featurePresentBytesSha256"],
                "composedPresentBytesSha256": facts["featurePresentBytesSha256"],
                "byteExact": True,
            },
            "targetPreservation": {
                "dtype": "little-endian-float32",
                "values": facts["counts"]["targetValues"],
                "sourceBytesSha256": facts["targetValueBytesSha256"],
                "composedBytesSha256": facts["targetValueBytesSha256"],
                "byteExact": True,
            },
            "leakage": {
                "heldRosterChecked": True,
                "protectedInterventionOverlap": 0,
                "benchmarkLabelsPresent": False,
                "rewardDataPresent": False,
            },
            "formats": {
                "archive": "canonical-USTAR",
                "arrays": "deterministic-uncompressed-NPZ",
            },
            "limitations": list(audit.COMPOSITION_LIMITATIONS),
        }
        for path in sorted(directory.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        (directory / "corpus-v1-2.tar").write_bytes(archive_payload)
        (directory / "corpus-compose-audit.json").write_bytes(
            self._pretty(companion)
        )
        return companion

    @staticmethod
    def _load(pretrain: dict[str, object]):
        return audit.load_corpus(
            audit.resolve_dataset_input(pretrain, "pretrain"),
            "pretrain",
            audit.AuditBounds(),
        )

    def _rewrite_companion(
        self,
        pretrain: dict[str, object],
        mutate,
        *,
        canonical: bool = True,
    ) -> None:
        path = Path(pretrain["path"]) / "corpus-compose-audit.json"
        value = json.loads(path.read_bytes())
        mutate(value)
        payload = self._pretty(value) if canonical else audit.canonical_json_bytes(value)
        path.write_bytes(payload)

    def _repack_tar(
        self, pretrain: dict[str, object], *, reverse: bool = False
    ) -> None:
        directory = Path(pretrain["path"])
        path = directory / "corpus-v1-2.tar"
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
            payloads = {
                member.name: archive.extractfile(member).read()
                for member in members
            }
        order = sorted(payloads, reverse=reverse)
        buffer = io.BytesIO()
        with tarfile.open(
            fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive:
            for name in order:
                payload = payloads[name]
                info = self._tar_info(name, len(payload))
                if not reverse:
                    info.mtime = 1
                archive.addfile(info, io.BytesIO(payload))
        repacked = buffer.getvalue()
        path.write_bytes(repacked)

        def bind_archive(value):
            value["archive"] = {
                "path": "corpus-v1-2.tar",
                "bytes": len(repacked),
                "sha256": hashlib.sha256(repacked).hexdigest(),
            }

        self._rewrite_companion(pretrain, bind_archive)

    def test_exact_two_file_bundle_is_valid_and_signed_identity_binds_both(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pretrain, _, _ = self._raw_inputs(Path(temporary))
            self._package(pretrain)
            corpus = self._load(pretrain)
            identity = corpus.identity
            self.assertEqual(
                {path.name for path in Path(pretrain["path"]).iterdir()},
                {"corpus-v1-2.tar", "corpus-compose-audit.json"},
            )
            self.assertEqual(identity["compositionAuditSchema"], audit.COMPOSITION_AUDIT_SCHEMA)
            self.assertTrue(identity["compositionCompanionValidated"])
            self.assertFalse(identity["sourcePreservationIndependentlyRecomputed"])
            self.assertEqual(
                identity["bundleArchiveSha256"],
                hashlib.sha256(
                    (Path(pretrain["path"]) / "corpus-v1-2.tar").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                identity["compositionAuditSha256"],
                hashlib.sha256(
                    (
                        Path(pretrain["path"])
                        / "corpus-compose-audit.json"
                    ).read_bytes()
                ).hexdigest(),
            )
            signed_claim = {
                "resource": pretrain["resource"],
                "manifestDigest": pretrain["manifestDigest"],
                **{
                    key: identity[key]
                    for key in (
                        "corpusManifestSha256",
                        "contentDigest",
                        "bundleArchiveSha256",
                        "compositionAuditSha256",
                    )
                },
            }
            self.assertEqual(
                attestation._dataset_claim(
                    signed_claim, "datasets.pretrain", kind="pretrain"
                ),
                signed_claim,
            )
            incomplete_claim = dict(signed_claim)
            del incomplete_claim["compositionAuditSha256"]
            with self.assertRaisesRegex(
                attestation.AuthorizationError, "fields mismatch"
            ):
                attestation._dataset_claim(
                    incomplete_claim, "datasets.pretrain", kind="pretrain"
                )

            signed = dict(identity)
            signed["bundleArchiveSha256"] = "0" * 64
            authorization = SimpleNamespace(
                pretrain_claim={
                    key: signed[key]
                    for key in (
                        "corpusManifestSha256",
                        "contentDigest",
                        "bundleArchiveSha256",
                        "compositionAuditSha256",
                    )
                },
                held_roster_claim={},
                inventory_claims=(),
            )
            with self.assertRaisesRegex(
                attestation.AuthorizationError, "bundleArchiveSha256"
            ):
                attestation.assert_authorized_content(
                    authorization,
                    pretrain_identity=identity,
                    held_roster_identity={},
                    inventory_identities={},
                )

    def test_missing_tar_missing_companion_and_extra_file_fail_closed(self) -> None:
        cases = ("corpus-v1-2.tar", "corpus-compose-audit.json", "extra")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in cases:
                with self.subTest(name=name):
                    pretrain, _, _ = self._raw_inputs(root / name)
                    self._package(pretrain)
                    directory = Path(pretrain["path"])
                    if name == "extra":
                        (directory / "extra.txt").write_text("forbidden", encoding="ascii")
                    else:
                        (directory / name).unlink()
                    with self.assertRaisesRegex(
                        audit.CorpusAuditError, "must contain exactly"
                    ):
                        self._load(pretrain)

    def test_companion_must_be_canonical_bounded_closed_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, _, _ = self._raw_inputs(root / "noncanonical")
            self._package(pretrain)
            self._rewrite_companion(pretrain, lambda value: None, canonical=False)
            with self.assertRaisesRegex(audit.CorpusAuditError, "canonical sorted pretty"):
                self._load(pretrain)

            pretrain, _, _ = self._raw_inputs(root / "extra-field")
            self._package(pretrain)
            self._rewrite_companion(
                pretrain, lambda value: value.__setitem__("untrusted", True)
            )
            with self.assertRaisesRegex(audit.CorpusAuditError, "fields mismatch"):
                self._load(pretrain)

    def test_companion_archive_counts_identity_and_preservation_must_match(self) -> None:
        def archive(value):
            value["archive"]["sha256"] = "0" * 64

        def counts(value):
            value["counts"]["records"] += 1

        def corpus_manifest(value):
            value["corpusManifestSha256"] = "0" * 64

        def inputs(value):
            value["inputs"]["observations"]["semanticSha256"] = "0" * 64

        def identity(value):
            value["identity"]["featureEntityKeySetSha256"] = "0" * 64

        def feature(value):
            value["featurePreservation"]["composedValueBytesSha256"] = "0" * 64

        def target(value):
            value["targetPreservation"]["composedBytesSha256"] = "0" * 64

        def feature_pack(value):
            value["featurePackSha256"] = "0" * 64

        cases = (
            ("archive", archive, "archive or corpus digest"),
            ("corpus-manifest", corpus_manifest, "archive or corpus digest"),
            ("inputs", inputs, "lineage or counts"),
            ("counts", counts, "lineage or counts"),
            ("identity", identity, "identity mismatch"),
            ("feature-pack", feature_pack, "feature-pack digest"),
            ("feature", feature, "feature preservation"),
            ("target", target, "target preservation"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, mutate, message in cases:
                with self.subTest(name=name):
                    pretrain, _, _ = self._raw_inputs(root / name)
                    self._package(pretrain)
                    self._rewrite_companion(pretrain, mutate)
                    with self.assertRaisesRegex(audit.CorpusAuditError, message):
                        self._load(pretrain)

    def test_companion_cannot_assert_protected_reward_or_benchmark_content(self) -> None:
        def protected(value):
            value["leakage"]["protectedInterventionOverlap"] = 1

        def reward(value):
            value["leakage"]["rewardDataPresent"] = True

        def benchmark(value):
            value["leakage"]["benchmarkLabelsPresent"] = True

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, mutate in (
                ("protected", protected),
                ("reward", reward),
                ("benchmark", benchmark),
            ):
                with self.subTest(name=name):
                    pretrain, _, _ = self._raw_inputs(root / name)
                    self._package(pretrain)
                    self._rewrite_companion(pretrain, mutate)
                    with self.assertRaisesRegex(
                        audit.CorpusAuditError, "leakage declaration"
                    ):
                        self._load(pretrain)

    def test_tampered_tar_is_rejected_even_if_companion_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pretrain, _, _ = self._raw_inputs(Path(temporary))
            self._package(pretrain)
            path = Path(pretrain["path"]) / "corpus-v1-2.tar"
            payload = bytearray(path.read_bytes())
            payload[1024] ^= 1
            path.write_bytes(payload)
            with self.assertRaises(audit.CorpusAuditError):
                self._load(pretrain)

    def test_identical_content_with_noncanonical_tar_representation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, reverse, message in (
                ("metadata", False, "metadata is not canonical USTAR"),
                ("order", True, "exact sorted order"),
            ):
                with self.subTest(name=name):
                    pretrain, _, _ = self._raw_inputs(root / name)
                    self._package(pretrain)
                    self._repack_tar(pretrain, reverse=reverse)
                    with self.assertRaisesRegex(audit.CorpusAuditError, message):
                        self._load(pretrain)

    def test_companion_bool_integer_type_confusion_is_rejected(self) -> None:
        def count_bool(value):
            value["counts"]["panels"] = True

        def overlap_bool(value):
            value["leakage"]["protectedInterventionOverlap"] = False

        def checked_integer(value):
            value["leakage"]["heldRosterChecked"] = 1

        def benchmark_integer(value):
            value["leakage"]["benchmarkLabelsPresent"] = 0

        def reward_integer(value):
            value["leakage"]["rewardDataPresent"] = 0

        cases = (
            ("count", count_bool, "counts.panels must be a"),
            ("overlap", overlap_bool, "protectedInterventionOverlap must be a"),
            ("checked", checked_integer, "leakage declaration mismatch"),
            ("benchmark", benchmark_integer, "leakage declaration mismatch"),
            ("reward", reward_integer, "leakage declaration mismatch"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, mutate, message in cases:
                with self.subTest(name=name):
                    pretrain, _, _ = self._raw_inputs(root / name)
                    self._package(pretrain)
                    self._rewrite_companion(pretrain, mutate)
                    with self.assertRaisesRegex(audit.CorpusAuditError, message):
                        self._load(pretrain)

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_v15_report_and_authorization_schemas_are_valid(self) -> None:
        for name in (
            "corpus-audit.schema.json",
            "custodian-boundary-attestation.schema.json",
        ):
            schema = json.loads((MODULE / name).read_bytes())
            jsonschema.Draft202012Validator.check_schema(schema)
        with tempfile.TemporaryDirectory() as temporary:
            pretrain, _, _ = self._raw_inputs(Path(temporary))
            self._package(pretrain)
            identity = self._load(pretrain).identity
            report_schema = json.loads(
                (MODULE / "corpus-audit.schema.json").read_bytes()
            )
            identity_schema = {
                "$schema": report_schema["$schema"],
                "$ref": "#/$defs/corpusIdentity",
                "$defs": report_schema["$defs"],
            }
            jsonschema.Draft202012Validator(identity_schema).validate(identity)


if __name__ == "__main__":
    unittest.main()
