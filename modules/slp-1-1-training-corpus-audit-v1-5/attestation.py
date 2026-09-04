"""Strict recipient-bound Ed25519 authorization for a clean training corpus.

The production trust anchor is deliberately unprovisioned. An independently
controlled custodian key ceremony must add the public-key file and freeze both
digest constants below in a new immutable module revision. Runtime config and
attestation inputs can never supply or override the trust anchor.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from audit import DatasetInput, canonical_json_bytes, resolve_dataset_input
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

AUTHORIZATION_SCHEMA = "slp.training-corpus-handoff/v1"
AUDIT_SCHEMA = "slp.corpus-audit/v1.5"
SIGNATURE_DOMAIN = b"abiome-org/SLp/training-corpus-handoff/v1\x00"
EXPECTED_ISSUER = "abiome-protected-data-custodian"
EXPECTED_RECIPIENT_NAMESPACE = "abiome/slp"
EXPECTED_PURPOSE = "slp-1.1-clean-training-boundary"
TRUST_ANCHOR_NAME = "custodian-ed25519-v1.pub"

# These must be provisioned together by an independently controlled key
# ceremony. None is an intentional hard stop, never a wildcard.
PINNED_CUSTODIAN_KEY_ID: str | None = None
PINNED_CUSTODIAN_PUBLIC_KEY_TEXT_SHA256: str | None = None

MAX_STATEMENT_BYTES = 64 * 1024
MAX_INVENTORIES = 64
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PREFIXED_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
RESOURCE = re.compile(
    r"^omf://abiome/slp/datasetsnapshot/[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    r"@sha256:[0-9a-f]{64}$"
)
INPUT_NAME = re.compile(r"^protectedInventory[A-Za-z0-9_-]+$")
ISSUED_AT = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)


class AuthorizationError(ValueError):
    """Raised whenever the training handoff cannot be proven exactly."""


@dataclass(frozen=True)
class AuthorizationIdentity:
    dataset: DatasetInput
    authorization_id: str
    issued_at: str
    key_id: str
    recipient_factory_identity: str
    challenge_nonce: str
    statement_sha256: str
    signature_sha256: str
    public_key_text_sha256: str
    pretrain_claim: Mapping[str, Any]
    held_roster_claim: Mapping[str, Any]
    inventory_claims: tuple[Mapping[str, Any], ...]

    @property
    def report_identity(self) -> dict[str, Any]:
        return {
            "resource": self.dataset.resource,
            "revision": self.dataset.revision,
            "manifestDigest": self.dataset.manifest_digest,
            "authorizationId": self.authorization_id,
            "issuedAt": self.issued_at,
            "keyId": self.key_id,
            "recipientNamespace": EXPECTED_RECIPIENT_NAMESPACE,
            "recipientFactoryIdentity": self.recipient_factory_identity,
            "challengeNonce": self.challenge_nonce,
            "statementSha256": self.statement_sha256,
            "signatureSha256": self.signature_sha256,
            "publicKeyTextSha256": self.public_key_text_sha256,
        }


def _strict_dict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = set(value) if isinstance(value, dict) else set()
        raise AuthorizationError(
            f"{label} fields mismatch; missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AuthorizationError(f"duplicate JSON member: {key}")
        value[key] = item
    return value


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise AuthorizationError(f"{label} must be a regular file")
        size = path.stat().st_size
        if not 0 < size <= maximum:
            raise AuthorizationError(f"{label} exceeds its positive byte bound")
        return path.read_bytes()
    except AuthorizationError:
        raise
    except OSError as error:
        raise AuthorizationError(f"could not read {label}") from error


def _canonical_document(payload: bytes, label: str) -> dict[str, Any]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise AuthorizationError(f"{label} must not contain a UTF-8 BOM")
    try:
        decoded = payload.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_duplicate_rejecting_object)
    except AuthorizationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise AuthorizationError(f"{label} must be bounded UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise AuthorizationError(f"{label} must be a JSON object")
    try:
        canonical = canonical_json_bytes(value, newline=True)
    except (TypeError, ValueError, RecursionError) as error:
        raise AuthorizationError(f"{label} contains an unsupported JSON value") from error
    if payload != canonical:
        raise AuthorizationError(f"{label} must use canonical JSON with one LF")
    return value


def _sha256(value: object, label: str, *, prefixed: bool = False) -> str:
    pattern = PREFIXED_SHA256 if prefixed else SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        prefix = "sha256:" if prefixed else ""
        raise AuthorizationError(f"{label} must be {prefix}lowercase SHA-256")
    return value


def _dataset_claim(value: object, label: str, *, kind: str) -> dict[str, Any]:
    extra_fields = {
        "pretrain": {
            "corpusManifestSha256",
            "contentDigest",
            "bundleArchiveSha256",
            "compositionAuditSha256",
        },
        "heldRoster": {"rosterSha256", "coverageSha256"},
        "inventory": {"inputName", "inventoryManifestSha256"},
    }[kind]
    item = _strict_dict(
        value,
        {"resource", "manifestDigest", *extra_fields},
        label,
    )
    if (
        not isinstance(item["resource"], str)
        or RESOURCE.fullmatch(item["resource"]) is None
    ):
        raise AuthorizationError(
            f"{label}.resource must be an immutable project DatasetSnapshot"
        )
    _sha256(item["manifestDigest"], f"{label}.manifestDigest", prefixed=True)
    for key in extra_fields - {"inputName"}:
        _sha256(item[key], f"{label}.{key}")
    if kind == "inventory":
        input_name = item["inputName"]
        if not isinstance(input_name, str) or INPUT_NAME.fullmatch(input_name) is None:
            raise AuthorizationError(
                f"{label}.inputName must be protectedInventory* with a suffix"
            )
    return item


def _validate_statement(statement: dict[str, Any]) -> tuple[
    str,
    str,
    str,
    str,
    str,
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], ...],
]:
    value = _strict_dict(
        statement,
        {
            "schema", "authorizationId", "issuedAt", "issuer", "recipient",
            "purpose", "protocol", "datasets",
        },
        "training corpus authorization",
    )
    if value["schema"] != AUTHORIZATION_SCHEMA:
        raise AuthorizationError("authorization schema mismatch")
    authorization_id = value["authorizationId"]
    if (
        not isinstance(authorization_id, str)
        or not authorization_id.startswith("urn:uuid:")
    ):
        raise AuthorizationError(
            "authorizationId must be a canonical lowercase UUIDv4 URN"
        )
    try:
        parsed_uuid = uuid.UUID(authorization_id.removeprefix("urn:uuid:"))
    except (ValueError, AttributeError) as error:
        raise AuthorizationError(
            "authorizationId must be a canonical lowercase UUIDv4 URN"
        ) from error
    if parsed_uuid.version != 4 or authorization_id != f"urn:uuid:{parsed_uuid}":
        raise AuthorizationError(
            "authorizationId must be a canonical lowercase UUIDv4 URN"
        )

    issued_at = value["issuedAt"]
    if not isinstance(issued_at, str) or ISSUED_AT.fullmatch(issued_at) is None:
        raise AuthorizationError("issuedAt must use UTC whole-second RFC 3339 form")
    try:
        datetime.strptime(issued_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise AuthorizationError("issuedAt is not a real UTC timestamp") from error

    issuer = _strict_dict(value["issuer"], {"name", "keyId"}, "issuer")
    if issuer["name"] != EXPECTED_ISSUER:
        raise AuthorizationError("issuer name mismatch")
    key_id = _sha256(issuer["keyId"], "issuer.keyId", prefixed=True)

    recipient = _strict_dict(
        value["recipient"],
        {"namespace", "factoryIdentity", "challengeNonce"},
        "recipient",
    )
    if recipient["namespace"] != EXPECTED_RECIPIENT_NAMESPACE:
        raise AuthorizationError("recipient namespace mismatch")
    factory_identity = _sha256(
        recipient["factoryIdentity"], "recipient.factoryIdentity", prefixed=True
    )
    challenge_nonce = _sha256(
        recipient["challengeNonce"], "recipient.challengeNonce"
    )

    if value["purpose"] != EXPECTED_PURPOSE:
        raise AuthorizationError("authorization purpose mismatch")
    protocol = _strict_dict(
        value["protocol"],
        {
            "auditSchema", "rewardEnabled", "protectedQuantitativeTruthIncluded",
            "benchmarkLabelsIncluded",
        },
        "protocol",
    )
    if protocol != {
        "auditSchema": AUDIT_SCHEMA,
        "rewardEnabled": False,
        "protectedQuantitativeTruthIncluded": False,
        "benchmarkLabelsIncluded": False,
    }:
        raise AuthorizationError("authorization protocol is not training-safe")

    datasets = _strict_dict(
        value["datasets"],
        {"pretrain", "heldRoster", "protectedInventories"},
        "datasets",
    )
    pretrain = _dataset_claim(
        datasets["pretrain"], "datasets.pretrain", kind="pretrain"
    )
    held_roster = _dataset_claim(
        datasets["heldRoster"], "datasets.heldRoster", kind="heldRoster"
    )
    raw_inventories = datasets["protectedInventories"]
    if (
        not isinstance(raw_inventories, list)
        or not 2 <= len(raw_inventories) <= MAX_INVENTORIES
    ):
        raise AuthorizationError(
            "protectedInventories must contain between 2 and 64 entries"
        )
    inventories = tuple(
        _dataset_claim(
            item, f"datasets.protectedInventories[{index}]", kind="inventory"
        )
        for index, item in enumerate(raw_inventories)
    )
    names = [item["inputName"] for item in inventories]
    resources = [item["resource"] for item in inventories]
    if names != sorted(names) or len(set(names)) != len(names):
        raise AuthorizationError(
            "protectedInventories must have sorted unique inputName values"
        )
    if len(set(resources)) != len(resources):
        raise AuthorizationError("protectedInventories must have unique resources")
    all_resources = [pretrain["resource"], held_roster["resource"], *resources]
    if len(set(all_resources)) != len(all_resources):
        raise AuthorizationError("authorization DatasetSnapshot resources must be unique")
    return (
        authorization_id,
        issued_at,
        key_id,
        factory_identity,
        challenge_nonce,
        pretrain,
        held_roster,
        inventories,
    )


def _load_trust_anchor(
    path: Path,
    *,
    expected_key_id: str,
    expected_text_sha256: str,
) -> tuple[bytes, str]:
    payload = _read_bounded(path, 65, "custodian public key")
    if re.fullmatch(rb"[0-9a-f]{64}\n", payload) is None:
        raise AuthorizationError(
            "custodian public key must be 64 lowercase hex characters plus LF"
        )
    text_sha256 = hashlib.sha256(payload).hexdigest()
    if text_sha256 != expected_text_sha256:
        raise AuthorizationError("custodian public-key text digest mismatch")
    public_key = bytes.fromhex(payload[:-1].decode("ascii"))
    derived_key_id = "sha256:" + hashlib.sha256(public_key).hexdigest()
    if derived_key_id != expected_key_id:
        raise AuthorizationError("custodian public-key identity mismatch")
    return public_key, text_sha256


def _verify_authorization_with_anchor(
    dataset_input: object,
    actual_inputs: Mapping[str, object],
    *,
    recipient_factory_identity: object,
    challenge_nonce: object,
    trust_anchor_path: str | Path,
    expected_key_id: str,
    expected_text_sha256: str,
) -> AuthorizationIdentity:
    expected_factory = _sha256(
        recipient_factory_identity,
        "configured recipientFactoryIdentity",
        prefixed=True,
    )
    expected_nonce = _sha256(challenge_nonce, "configured challengeNonce")
    expected_key_id = _sha256(
        expected_key_id, "pinned custodian keyId", prefixed=True
    )
    expected_text_sha256 = _sha256(
        expected_text_sha256, "pinned custodian public-key text digest"
    )

    try:
        dataset = resolve_dataset_input(
            dataset_input, "custodianBoundaryAttestation"
        )
    except ValueError as error:
        raise AuthorizationError("custodian authorization input is invalid") from error
    expected_files = {"authorization.json", "authorization.ed25519"}
    try:
        actual_files = {path.name for path in dataset.root.iterdir()}
    except OSError as error:
        raise AuthorizationError("could not enumerate custodian authorization") from error
    if actual_files != expected_files or any(
        (dataset.root / name).is_symlink() for name in expected_files
    ):
        raise AuthorizationError(
            "custodian authorization must contain exactly two regular files"
        )

    statement_bytes = _read_bounded(
        dataset.root / "authorization.json",
        MAX_STATEMENT_BYTES,
        "authorization statement",
    )
    signature_text = _read_bounded(
        dataset.root / "authorization.ed25519", 129, "authorization signature"
    )
    if re.fullmatch(rb"[0-9a-f]{128}\n", signature_text) is None:
        raise AuthorizationError(
            "authorization signature must be 128 lowercase hex characters plus LF"
        )
    signature = bytes.fromhex(signature_text[:-1].decode("ascii"))
    statement = _canonical_document(statement_bytes, "authorization statement")
    (
        authorization_id,
        issued_at,
        key_id,
        signed_factory,
        signed_nonce,
        pretrain_claim,
        held_claim,
        inventory_claims,
    ) = _validate_statement(statement)
    if signed_factory != expected_factory or signed_nonce != expected_nonce:
        raise AuthorizationError(
            "authorization recipient identity or challenge mismatch"
        )

    public_key, text_sha256 = _load_trust_anchor(
        Path(trust_anchor_path),
        expected_key_id=expected_key_id,
        expected_text_sha256=expected_text_sha256,
    )
    if key_id != expected_key_id:
        raise AuthorizationError(
            "authorization issuer keyId does not match the pinned key"
        )
    message = (
        SIGNATURE_DOMAIN
        + len(statement_bytes).to_bytes(8, "big")
        + statement_bytes
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except (InvalidSignature, ValueError) as error:
        raise AuthorizationError(
            "custodian Ed25519 signature verification failed"
        ) from error

    expected_names = {
        "pretrain",
        "heldRoster",
        *(item["inputName"] for item in inventory_claims),
    }
    if set(actual_inputs) != expected_names:
        raise AuthorizationError(
            f"authorization input names mismatch; expected={sorted(expected_names)}, "
            f"actual={sorted(actual_inputs)}"
        )
    try:
        resolved = {
            name: resolve_dataset_input(value, name)
            for name, value in actual_inputs.items()
        }
    except ValueError as error:
        raise AuthorizationError("an authorized DatasetSnapshot input is invalid") from error
    claims = {
        "pretrain": pretrain_claim,
        "heldRoster": held_claim,
        **{item["inputName"]: item for item in inventory_claims},
    }
    for name, claimed in claims.items():
        actual = resolved[name]
        if (
            claimed["resource"] != actual.resource
            or claimed["manifestDigest"] != actual.manifest_digest
        ):
            raise AuthorizationError(
                f"authorization does not bind the exact {name} input"
            )

    return AuthorizationIdentity(
        dataset=dataset,
        authorization_id=authorization_id,
        issued_at=issued_at,
        key_id=key_id,
        recipient_factory_identity=signed_factory,
        challenge_nonce=signed_nonce,
        statement_sha256=hashlib.sha256(statement_bytes).hexdigest(),
        signature_sha256=hashlib.sha256(signature).hexdigest(),
        public_key_text_sha256=text_sha256,
        pretrain_claim=MappingProxyType(dict(pretrain_claim)),
        held_roster_claim=MappingProxyType(dict(held_claim)),
        inventory_claims=tuple(
            MappingProxyType(dict(item)) for item in inventory_claims
        ),
    )


def verify_custodian_authorization(
    dataset_input: object,
    actual_inputs: Mapping[str, object],
    *,
    recipient_factory_identity: object,
    challenge_nonce: object,
) -> AuthorizationIdentity:
    """Verify using only the immutable source-pinned production trust anchor."""
    if (
        PINNED_CUSTODIAN_KEY_ID is None
        or PINNED_CUSTODIAN_PUBLIC_KEY_TEXT_SHA256 is None
    ):
        raise AuthorizationError(
            "production custodian trust anchor is not provisioned; "
            "an independent key ceremony is required"
        )
    return _verify_authorization_with_anchor(
        dataset_input,
        actual_inputs,
        recipient_factory_identity=recipient_factory_identity,
        challenge_nonce=challenge_nonce,
        trust_anchor_path=Path(__file__).with_name("trust") / TRUST_ANCHOR_NAME,
        expected_key_id=PINNED_CUSTODIAN_KEY_ID,
        expected_text_sha256=PINNED_CUSTODIAN_PUBLIC_KEY_TEXT_SHA256,
    )


def assert_authorized_content(
    authorization: AuthorizationIdentity,
    *,
    pretrain_identity: Mapping[str, Any],
    held_roster_identity: Mapping[str, Any],
    inventory_identities: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind signed safe-input claims to independently recomputed identities."""
    pretrain_expected = {
        "corpusManifestSha256": pretrain_identity.get("corpusManifestSha256"),
        "contentDigest": pretrain_identity.get("contentDigest"),
        "bundleArchiveSha256": pretrain_identity.get("bundleArchiveSha256"),
        "compositionAuditSha256": pretrain_identity.get("compositionAuditSha256"),
    }
    for key, expected in pretrain_expected.items():
        if authorization.pretrain_claim[key] != expected:
            raise AuthorizationError(
                f"signed pretrain {key} does not match audited content"
            )
    held_expected = {
        "rosterSha256": held_roster_identity.get("rosterSha256"),
        "coverageSha256": held_roster_identity.get("coverageSha256"),
    }
    for key, expected in held_expected.items():
        if authorization.held_roster_claim[key] != expected:
            raise AuthorizationError(
                f"signed held-roster {key} does not match audited content"
            )
    signed_inventories = {
        item["inputName"]: item for item in authorization.inventory_claims
    }
    if set(signed_inventories) != set(inventory_identities):
        raise AuthorizationError(
            "signed and audited protected-inventory names differ"
        )
    for name, identity in inventory_identities.items():
        if (
            signed_inventories[name]["inventoryManifestSha256"]
            != identity.get("manifestSha256")
        ):
            raise AuthorizationError(
                f"signed {name} inventoryManifestSha256 does not match audited content"
            )
