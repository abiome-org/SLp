"""Adversarial production-boundary checks for sparse OMF training."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-sparse"
sys.path.insert(0, str(MODULE))

from slp_sparse_artifacts import (  # noqa: E402
    AUDIT_DATASET_ROLES,
    AUDIT_SCHEMA,
    CHECKPOINT_MAGIC,
    _audit_dataset_identity,
    attested_dataset_identity,
    build_artifact_report,
    canonical_json_bytes,
    canonical_sha256,
    load_sparse_checkpoint,
    validate_admitted_training_evidence,
    write_canonical_report,
    write_sparse_checkpoint,
    write_target_free_predictions,
)
from slp_sparse_corpus import CorpusIndex  # noqa: E402
from slp_sparse_training import model_parameter_sha256, train_sparse_world  # noqa: E402
from tests.test_slp11_sparse_candidate import _write_corpus  # noqa: E402
from tests.test_slp11_sparse_training import (  # noqa: E402
    FINAL_GENE,
    VALIDATION_GENE,
    _assigned_gene,
    _config,
    _make_pretrain,
    _make_query,
)

ROSTER_DOMAIN = b"slp-1.1-yeast-global-held-v1\x00"
AUDIT_MANIFEST_DIGEST = "sha256:" + "3" * 64
AUDIT_MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-corpus-audit"


def _dataset_input(path: Path, name: str, digit: str) -> dict[str, str]:
    digest = digit * 64
    return {
        "manifestDigest": f"sha256:{digest}",
        "mode": "copy",
        "path": str(path),
        "resource": f"omf://abiome/slp/datasetsnapshot/{name}@sha256:{digest}",
    }


def _artifact_input(path: Path, digit: str) -> dict[str, object]:
    digest = f"sha256:{digit * 64}"
    return {
        "resource": f"artifact:{digest}",
        "kind": "artifact",
        "artifacts": {"payload": digest},
        "paths": {"payload": str(path)},
        "path": str(path),
    }


def _write_roster(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for gene in (VALIDATION_GENE, FINAL_GENE):
        digest = hashlib.sha256(ROSTER_DOMAIN + gene.encode("ascii")).hexdigest()
        bucket = int(digest[:16], 16) % 100
        role = "molecular-final" if bucket < 10 else "molecular-validation" if bucket < 30 else "pretrain"
        rows.append(f"{gene}\t{role}\t{digest}\n")
    path = root / "held-intervention-roster.tsv"
    path.write_text("".join(sorted(rows)), encoding="ascii", newline="")
    (root / "coverage.json").write_bytes(b"{}\n")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_audit(
    root: Path, pretrain: CorpusIndex, query: object,
    pretrain_input: dict[str, str], query_input: dict[str, str],
    roster_input: dict[str, str], roster_path: Path,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    validation = [VALIDATION_GENE]
    final = [FINAL_GENE]
    union = sorted(validation + final)
    pretrain_identity = _audit_dataset_identity(pretrain, pretrain_input)

    def held_identity(name: str, role: str, digit: str, genes: list[str]) -> dict[str, object]:
        digest = digit * 64
        return {
            **pretrain_identity,
            "resource": f"omf://abiome/slp/datasetsnapshot/{name}@sha256:{digest}",
            "revision": f"sha256:{digest}",
            "manifestDigest": f"sha256:{digest}",
            "datasetId": f"TEST:{name}",
            "role": role,
            "corpusManifestSha256": digest,
            "contentDigest": digest,
            "trajectoryGenesSha256": digest,
            "trajectoryGeneSetSha256": canonical_sha256(genes),
            "trajectoryGeneCount": len(genes),
        }
    audit = {
        "schema": AUDIT_SCHEMA,
        "auditPassed": True,
        "strictInterventionIsolation": True,
        "leakageViolations": 0,
        "leakedTrajectoryGenes": [],
        "benchmarkLabelRecords": 0,
        "omfPriorAdmissionRequired": True,
        "datasets": {
            "pretrain": pretrain_identity,
            "molecularReward": held_identity("reward", "molecular-reward", "5", []),
            "molecularValidation": held_identity(
                "validation", "molecular-validation", "6", validation
            ),
            "molecularFinal": held_identity("final", "molecular-final", "7", final),
        },
        "heldRoster": {
            "resource": roster_input["resource"],
            "revision": roster_input["resource"].rpartition("@")[2],
            "manifestDigest": roster_input["manifestDigest"],
            "rosterSha256": _sha256(roster_path),
            "coverageSha256": _sha256(roster_path.parent / "coverage.json"),
            "assignmentDomainHex": ROSTER_DOMAIN.hex(),
            "bucketRule": "int(first-16-lowercase-hex,16) mod 100",
            "identityMappingId": "TEST:mapping-v1",
            "identityMappingSha256": "b" * 64,
            "sourceInventories": [
                {
                    "resource": (
                        "omf://abiome/slp/datasetsnapshot/"
                        f"protected-inventory-{index}@sha256:{digit * 64}"
                    ),
                    "revision": f"sha256:{digit * 64}",
                    "artifactManifestDigest": f"sha256:{digit * 64}",
                    "sourceId": f"TEST:inventory-{index}",
                    "sourceRelease": "fixture-v1",
                    "identityMappingId": "TEST:mapping-v1",
                    "identityMappingSha256": "b" * 64,
                    "manifestSha256": digit * 64,
                    "records": 2,
                    "duplicateRecords": 0,
                    "uniqueInterventions": 2,
                    "qcPassing": 2,
                    "qcFailed": 0,
                    "intersectionCoverage": 2,
                }
                for index, digit in enumerate(("c", "d"))
            ],
            "intersectionSize": 2,
            "pretrainGeneCount": 0,
            "validationGeneSetSha256": canonical_sha256(validation),
            "validationGeneCount": 1,
            "finalGeneSetSha256": canonical_sha256(final),
            "finalGeneCount": 1,
            "unionGeneSetSha256": canonical_sha256(union),
            "unionGeneCount": 2,
        },
    }
    path = root / "payload"
    path.write_bytes(canonical_json_bytes(audit, newline=True))
    return path


def _case(
    root: Path, *, query_gene: str = VALIDATION_GENE,
    pretrain_active_gene: str = "SGD:S0002",
):
    pretrain_root = root / "inputs" / "pretrain" / "pretrain-snapshot"
    query_root = root / "inputs" / "molecularPredictionQuery" / "query-snapshot"
    pretrain_root.parent.mkdir(parents=True, exist_ok=True)
    query_root.parent.mkdir(parents=True, exist_ok=True)
    pretrain = _make_pretrain(pretrain_root, active_gene=pretrain_active_gene)
    query = _make_query(query_root, pretrain, query_gene)
    pretrain_input = _dataset_input(pretrain_root, "pretrain-snapshot", "1")
    query_input = _dataset_input(query_root, "query-snapshot", "2")
    roster_path = _write_roster(root / "inputs" / "heldRosterEvidence" / "held-roster")
    roster_input = _dataset_input(roster_path.parent, "held-roster", "4")
    audit_path = _write_audit(
        root / "evidence" / "audit" / "payload", pretrain, query,
        pretrain_input, query_input, roster_input, roster_path,
    )
    audit_input = _artifact_input(audit_path, "3")
    evidence = validate_admitted_training_evidence(
        pretrain, query, pretrain_input, query_input, audit_input, roster_input,
        expected_corpus_audit_manifest_digest=AUDIT_MANIFEST_DIGEST,
    )
    return pretrain, query, pretrain_input, query_input, audit_input, roster_input, evidence


class SparseOmfTrainingArtifactsTest(unittest.TestCase):
    def test_rechecks_query_membership_and_full_held_union_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "outside the admitted validation roster"):
                _case(root / "wrong-query", query_gene=FINAL_GENE)
            with self.assertRaisesRegex(ValueError, "held validation/final genes occur"):
                _case(root / "held-leak", pretrain_active_gene=FINAL_GENE)

    def test_audit_must_bind_the_exact_roster_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, query, pretrain_input, query_input, audit_input, roster_input, _ = _case(root)
            audit_path = Path(audit_input["path"])
            audit = json.loads(audit_path.read_text())
            replacement = "sha256:" + "e" * 64
            audit["heldRoster"]["resource"] = (
                "omf://abiome/slp/datasetsnapshot/other-roster@" + replacement
            )
            audit["heldRoster"]["revision"] = replacement
            audit_path.write_bytes(canonical_json_bytes(audit, newline=True))
            with self.assertRaisesRegex(ValueError, "held-roster provenance"):
                validate_admitted_training_evidence(
                    pretrain, query, pretrain_input, query_input, audit_input, roster_input,
                    expected_corpus_audit_manifest_digest=AUDIT_MANIFEST_DIGEST,
                )

    def test_query_domain_must_cover_every_validation_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, query, pretrain_input, query_input, audit_input, roster_input, _ = _case(root)
            second_validation = _assigned_gene("molecular-validation", start=20_000)
            self.assertNotEqual(second_validation, VALIDATION_GENE)
            roster_path = Path(roster_input["path"]) / "held-intervention-roster.tsv"
            digest = hashlib.sha256(
                ROSTER_DOMAIN + second_validation.encode("ascii")
            ).hexdigest()
            lines = roster_path.read_text(encoding="ascii").splitlines()
            lines.append(f"{second_validation}\tmolecular-validation\t{digest}")
            roster_path.write_text(
                "\n".join(sorted(lines)) + "\n", encoding="ascii", newline=""
            )
            audit_path = Path(audit_input["path"])
            audit = json.loads(audit_path.read_text())
            validation = sorted([VALIDATION_GENE, second_validation])
            union = sorted(validation + [FINAL_GENE])
            held = audit["heldRoster"]
            held["rosterSha256"] = _sha256(roster_path)
            held["intersectionSize"] = 3
            held["validationGeneSetSha256"] = canonical_sha256(validation)
            held["validationGeneCount"] = 2
            held["unionGeneSetSha256"] = canonical_sha256(union)
            held["unionGeneCount"] = 3
            for inventory in held["sourceInventories"]:
                inventory["records"] = 3
                inventory["uniqueInterventions"] = 3
                inventory["qcPassing"] = 3
                inventory["intersectionCoverage"] = 3
            validation_identity = audit["datasets"]["molecularValidation"]
            validation_identity["trajectoryGeneSetSha256"] = canonical_sha256(validation)
            validation_identity["trajectoryGeneCount"] = 2
            audit_path.write_bytes(canonical_json_bytes(audit, newline=True))
            with self.assertRaisesRegex(ValueError, "does not exactly equal"):
                validate_admitted_training_evidence(
                    pretrain, query, pretrain_input, query_input, audit_input, roster_input,
                    expected_corpus_audit_manifest_digest=AUDIT_MANIFEST_DIGEST,
                )

    def test_evidence_artifacts_have_exact_bounded_layout_and_frozen_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = _case(root / "audit-sibling")
            pretrain, query, pretrain_input, query_input, audit_input, roster_input, _ = case
            audit_input["path"] = str(Path(audit_input["path"]).parent)
            audit_input["paths"] = {"payload": audit_input["path"]}
            with self.assertRaisesRegex(ValueError, "file artifact semantics"):
                validate_admitted_training_evidence(
                    pretrain, query, pretrain_input, query_input, audit_input, roster_input,
                    expected_corpus_audit_manifest_digest=AUDIT_MANIFEST_DIGEST,
                )

            case = _case(root / "roster-sibling")
            pretrain, query, pretrain_input, query_input, audit_input, roster_input, _ = case
            (Path(roster_input["path"]) / "nested").mkdir()
            with self.assertRaisesRegex(ValueError, "undeclared entries|non-regular or nested"):
                validate_admitted_training_evidence(
                    pretrain, query, pretrain_input, query_input, audit_input, roster_input,
                    expected_corpus_audit_manifest_digest=AUDIT_MANIFEST_DIGEST,
                )

            case = _case(root / "digest")
            pretrain, query, pretrain_input, query_input, audit_input, roster_input, _ = case
            with self.assertRaisesRegex(ValueError, "frozen artifact manifest digest"):
                validate_admitted_training_evidence(
                    pretrain, query, pretrain_input, query_input, audit_input, roster_input,
                    expected_corpus_audit_manifest_digest="sha256:" + "f" * 64,
                )

    def test_matches_actual_corpus_audit_v11_roles_and_schema(self) -> None:
        module_name = "slp11_corpus_audit_contract_integration"
        spec = importlib.util.spec_from_file_location(module_name, AUDIT_MODULE / "audit.py")
        assert spec is not None and spec.loader is not None
        producer = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {module_name: producer}):
            spec.loader.exec_module(producer)
        schema = json.loads((AUDIT_MODULE / "corpus-audit.schema.json").read_text())
        self.assertEqual(producer.AUDIT_SCHEMA, AUDIT_SCHEMA)
        self.assertEqual(producer.EXPECTED_ROLES, AUDIT_DATASET_ROLES)
        self.assertEqual(
            set(schema["properties"]["datasets"]["required"]),
            set(AUDIT_DATASET_ROLES),
        )
        self.assertEqual(
            set(schema["$defs"]["corpusIdentity"]["properties"]["role"]["enum"]),
            set(AUDIT_DATASET_ROLES.values()),
        )
        self.assertEqual(
            set(schema["$defs"]["sourceInventory"]["required"]),
            {
                "resource", "revision", "artifactManifestDigest", "sourceId",
                "sourceRelease", "identityMappingId", "identityMappingSha256",
                "manifestSha256", "records", "duplicateRecords",
                "uniqueInterventions", "qcPassing", "qcFailed",
                "intersectionCoverage",
            },
        )
        held_properties = schema["$defs"]["heldRosterIdentity"]["properties"]
        self.assertEqual(held_properties["validationGeneCount"]["minimum"], 1)
        self.assertEqual(held_properties["finalGeneCount"]["minimum"], 1)

    def test_checkpoint_is_invariant_to_protected_truth_content_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, query, pretrain_input, query_input, audit_input, roster_input, first_evidence = _case(root / "case")
            audit_path = Path(audit_input["path"])
            audit = json.loads(audit_path.read_text())
            for name, digit in (("molecularValidation", "c"), ("molecularFinal", "d")):
                audit["datasets"][name]["corpusManifestSha256"] = digit * 64
                audit["datasets"][name]["contentDigest"] = digit * 64
            audit_path.write_bytes(canonical_json_bytes(audit, newline=True))
            second_evidence = validate_admitted_training_evidence(
                pretrain, query, pretrain_input, query_input, audit_input, roster_input,
                expected_corpus_audit_manifest_digest=AUDIT_MANIFEST_DIGEST,
            )
            self.assertEqual(first_evidence, second_evidence)
            outcome = train_sparse_world(pretrain, _config())
            (root / "one").mkdir(); (root / "two").mkdir()
            one, one_digest = write_sparse_checkpoint(
                root / "one", outcome.model, outcome.report, first_evidence
            )
            two, two_digest = write_sparse_checkpoint(
                root / "two", outcome.model, outcome.report, second_evidence
            )
            self.assertEqual(one_digest, two_digest)
            self.assertEqual(one.read_bytes(), two.read_bytes())

    def test_requires_literal_admitted_evidence_and_frozen_roster_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = _case(Path(temporary))
            pretrain, query, pretrain_input, query_input, audit_input, roster_input, _ = case
            with self.assertRaisesRegex(ValueError, "materialized admitted OMF artifact"):
                validate_admitted_training_evidence(
                    pretrain, query, pretrain_input, query_input, {}, roster_input,
                    expected_corpus_audit_manifest_digest=AUDIT_MANIFEST_DIGEST,
                )
            roster_path = Path(roster_input["path"]) / "held-intervention-roster.tsv"
            rows = roster_path.read_text(encoding="ascii").splitlines()
            columns = rows[0].split("\t")
            columns[1] = "pretrain"
            rows[0] = "\t".join(columns)
            roster_path.write_text("\n".join(rows) + "\n", encoding="ascii", newline="")
            with self.assertRaisesRegex(ValueError, "frozen assignment"):
                validate_admitted_training_evidence(
                    pretrain, query, pretrain_input, query_input, audit_input, roster_input,
                    expected_corpus_audit_manifest_digest=AUDIT_MANIFEST_DIGEST,
                )

    def test_checkpoint_is_deterministic_load_equivalent_and_preallocation_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, query, _, _, _, _, evidence = _case(root / "case")
            first = train_sparse_world(pretrain, _config())
            second = train_sparse_world(pretrain, _config())
            (root / "one").mkdir(); (root / "two").mkdir()
            one, one_digest = write_sparse_checkpoint(root / "one", first.model, first.report, evidence)
            two, two_digest = write_sparse_checkpoint(root / "two", second.model, second.report, evidence)
            self.assertEqual(one_digest, two_digest)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            rng = torch.random.get_rng_state().clone()
            loaded, metadata = load_sparse_checkpoint(one, expected_sha256=one_digest)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(model_parameter_sha256(loaded), model_parameter_sha256(first.model))
            self.assertEqual(metadata["checkpointContentSha256"], one_digest)
            with self.assertRaises(TypeError):
                load_sparse_checkpoint(one)
            with self.assertRaisesRegex(ValueError, "expected checkpoint content digest"):
                load_sparse_checkpoint(one, expected_sha256="not-a-digest")

            raw = one.read_bytes()
            header_size = struct.unpack(">Q", raw[len(CHECKPOINT_MAGIC):len(CHECKPOINT_MAGIC) + 8])[0]
            start = len(CHECKPOINT_MAGIC) + 8
            header = json.loads(raw[start:start + header_size])
            header["modelConfig"]["d_model"] = 4096
            malicious_header = canonical_json_bytes(header)
            malicious = root / "oversized-config.slpc"
            malicious.write_bytes(
                CHECKPOINT_MAGIC + struct.pack(">Q", len(malicious_header))
                + malicious_header + raw[start + header_size:]
            )
            with self.assertRaisesRegex(ValueError, "d_model violates its bound"):
                load_sparse_checkpoint(malicious, expected_sha256=_sha256(malicious))

            trailing = root / "trailing.slpc"
            trailing.write_bytes(raw + b"x")
            with self.assertRaisesRegex(ValueError, "exactly match"):
                load_sparse_checkpoint(trailing, expected_sha256=_sha256(trailing))

    def test_typed_prediction_artifact_has_exact_query_coverage_and_no_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain, query, _, query_input, _, _, evidence = _case(root / "case")
            outcome = train_sparse_world(pretrain, _config())
            root.mkdir(exist_ok=True)
            checkpoint, checkpoint_digest = write_sparse_checkpoint(root, outcome.model, outcome.report, evidence)
            prediction = write_target_free_predictions(
                root, outcome.model, query, query_input, checkpoint_digest, batch_size=1
            )
            self.assertTrue(prediction[0].is_file())
            self.assertTrue(prediction[0].name.endswith(".tar"))
            with tarfile.open(prediction[0], mode="r:") as archive:
                self.assertEqual(
                    [member.name for member in archive.getmembers()],
                    ["evaluation.json", "profiles-000.jsonl"],
                )
                manifest_stream = archive.extractfile("evaluation.json")
                profiles_stream = archive.extractfile("profiles-000.jsonl")
                assert manifest_stream is not None and profiles_stream is not None
                manifest = json.loads(manifest_stream.read())
                produced = [
                    json.loads(line) for line in profiles_stream.read().splitlines()
                ]
            self.assertEqual(manifest["modelCheckpointContentSha256"], checkpoint_digest)
            self.assertFalse(manifest["targetValuesPresent"])
            self.assertFalse(manifest["observedMaskPresent"])
            self.assertNotIn("sourceCorpusContentDigest", json.dumps(manifest))
            expected = [json.loads(line) for line in (query.root / "profiles-query.jsonl").read_text().splitlines()]
            self.assertEqual(len(produced), len(expected))
            for actual, query_row in zip(produced, expected, strict=True):
                self.assertEqual(
                    {name: actual[name] for name in query_row}, query_row
                )
                self.assertEqual(set(actual), set(query_row) | {"predictionParameters"})
                self.assertEqual(len(actual["predictionParameters"]), len(actual["readoutIds"]))
                self.assertNotIn("target", json.dumps(actual).lower())
                for distribution, parameters in zip(actual["distributionTypes"], actual["predictionParameters"], strict=True):
                    expected_fields = {"mean", "logScale"} if distribution == "gaussian" else {"logMean", "logInverseDispersion"}
                    self.assertEqual(set(parameters), expected_fields)
            report = build_artifact_report(
                outcome.report, evidence=evidence, molecular_query=query,
                molecular_query_input=query_input, checkpoint_content_sha256=checkpoint_digest,
                prediction_content_sha256=prediction[1], prediction_records=prediction[2],
                prediction_queries=prediction[3],
            )
            report_path, _ = write_canonical_report(root, report)
            serialized = report_path.read_text()
            self.assertNotIn("molecularValidation", serialized)
            self.assertNotIn("sourceCorpusContentDigest", serialized)
            self.assertFalse(report["artifactBoundary"]["heldTruthAccessible"])
            self.assertEqual(len(checkpoint.name), len("slp-world-sparse-.slpc") + 64)

    def test_omf_run_repeats_and_rejects_any_truth_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"
            pretrain, query, pretrain_input, query_input, audit_input, roster_input, _ = _case(stage)
            request = types.SimpleNamespace(
                inputs={
                    "pretrain": pretrain_input,
                    "molecularPredictionQuery": query_input,
                    "corpusAuditEvidence": audit_input,
                    "heldRosterEvidence": roster_input,
                },
                config={
                    "expectedCorpusAuditArtifactManifestDigest": AUDIT_MANIFEST_DIGEST,
                    "seed": 83, "epochs": 2, "drawsPerEpoch": 8, "batchSize": 4,
                    "learningRate": 0.01, "predictionBatchSize": 2, "dModel": 8,
                    "nhead": 2, "encoderLayers": 1, "decoderLayers": 1,
                    "ffnMultiplier": 2, "dropout": 0.0,
                },
            )
            sparse_main = _load_main_with_sdk_stub()
            with patch.dict(os.environ, {"OMF_RESULT_FILE": str(stage / "result.json")}):
                result = sparse_main.run(request)
            repeat = stage / "repeat"; repeat.mkdir()
            with patch.dict(os.environ, {"OMF_RESULT_FILE": str(repeat / "result.json")}):
                repeated = sparse_main.run(request)
            self.assertEqual(result.outputs, repeated.outputs)
            self.assertEqual(result.state, repeated.state)
            self.assertEqual(len([x for x in result.artifacts if x["kind"] == "checkpoint"]), 1)
            self.assertEqual(len(result.artifacts), 3)
            self.assertTrue(result.outputs["predictionsTargetFree"])
            self.assertFalse(result.outputs["heldTruthAccessible"])
            self.assertEqual(result.state["releaseBlockers"], [
                "hash-pinned-offline-wheelhouse-required",
                "omf-1.0-artifact-to-inference-adapter-gap",
                "omf-corpus-audit-producer-lineage-policy-required",
            ])
            self.assertNotIn(str(stage), json.dumps(result.state))
            missing_audit_pin = deepcopy(request)
            missing_audit_pin.config.pop("expectedCorpusAuditArtifactManifestDigest")
            with self.assertRaisesRegex(ValueError, "must be sha256-pinned"):
                sparse_main.run(missing_audit_pin)
            adversarial = deepcopy(request)
            adversarial.inputs["molecularValidation"] = {"target": [1.0]}
            with self.assertRaisesRegex(ValueError, "requires only"):
                sparse_main.run(adversarial)


def _load_main_with_sdk_stub():
    sdk = types.ModuleType("omf.sdk")

    class Result:
        def __init__(self, **values):
            self.__dict__.update(values)

    sdk.ProtocolRequest = object
    sdk.ProtocolResult = Result
    sdk.main = lambda _handlers: 0
    package = types.ModuleType("omf")
    package.sdk = sdk
    spec = importlib.util.spec_from_file_location("slp_sparse_omf_main_test", MODULE / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"omf": package, "omf.sdk": sdk}):
        spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
