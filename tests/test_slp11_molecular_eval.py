from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "modules" / "slp-1-1-molecular-eval"
AUDIT_ROOT = ROOT / "modules" / "slp-1-1-corpus-audit"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load("slp11_evaluator_v2", MODULE_ROOT / "evaluator.py")
renderer = _load("slp11_eval_renderer_v2", MODULE_ROOT / "render_workload.py")
audit_emitter = _load("slp11_corpus_audit_contract", AUDIT_ROOT / "audit.py")
audit_schema = json.loads((AUDIT_ROOT / "corpus-audit.schema.json").read_text())


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.inputs_root = root / "run" / "stages" / "evaluate" / "inputs"
        self.source = "SOURCE:one"
        self.readouts = ["READOUT:a", "READOUT:b", "READOUT:c"]
        self.distributions = ["gaussian", "negative-binomial", "gaussian"]
        roles: dict[str, list[str]] = {"pretrain": [], "molecular-validation": [], "molecular-final": []}
        index = 1
        while any(len(roles[name]) < count for name, count in {
            "pretrain": 2, "molecular-validation": 2, "molecular-final": 1
        }.items()):
            identifier = f"SGD:S{index:09d}"
            digest = hashlib.sha256(evaluator.ROSTER_ASSIGNMENT_DOMAIN + identifier.encode("ascii")).hexdigest()
            role, _ = evaluator._role_from_digest(digest)
            if len(roles[role]) < 2:
                roles[role].append(identifier)
            index += 1
        self.roles = roles
        self.center_records = [
            self.target_record(roles["pretrain"][0], [0.0, 4, 0.0]),
            self.target_record(roles["pretrain"][1], [2.0, 6, 2.0]),
        ]
        self.truth_records = [
            self.target_record(roles["molecular-validation"][0], [2.0, 7, None]),
            self.target_record(roles["molecular-validation"][1], [4.0, 9, 3.0]),
        ]
        self.query_records = [self.query_record(record) for record in self.truth_records]
        self.checkpoint_bytes = b"deterministic checkpoint bytes\n"
        self.center = self.dataset_input("molecularCenteringReference", "centering", "1")
        self.truth = self.dataset_input("molecularTruth", "truth", "2")
        self.query = self.dataset_input("molecularQuery", "query", "3")
        self.audit = self.dataset_input("corpusAudit", "audit", "4")
        self.roster = self.dataset_input("heldRoster", "roster", "5")
        self._write_snapshot(Path(self.center["path"]), evaluator.CENTERING_ROLE, self.center_records)
        self.query_sha = self._write_query()
        self._write_snapshot(Path(self.truth["path"]), evaluator.TRUTH_ROLE, self.truth_records)
        self.prediction_root = root / "prediction"
        self._write_snapshot(
            self.prediction_root,
            evaluator.PREDICTION_ROLE,
            [self.prediction_record(record) for record in self.truth_records],
        )
        self.checkpoint = self._write_checkpoint()
        self._write_roster()
        self._write_audit()

    def dataset_input(self, input_name: str, resource_name: str, digit: str) -> dict[str, object]:
        path = self.inputs_root / input_name / resource_name
        path.mkdir(parents=True)
        revision = f"sha256:{digit * 64}"
        return {
            "resource": f"omf://fixture/datasetsnapshot/{resource_name}@{revision}",
            "mode": "copy",
            "path": str(path),
            "manifestDigest": f"sha256:{digit * 64}",
        }

    def target_record(self, intervention: str, target: list[float | int | None]) -> dict[str, object]:
        perturbation = evaluator.canonical_perturbation_id((intervention,))
        profile = evaluator.canonical_profile_id(4932, self.source, "GROUP:one", perturbation)
        return {
            "profileId": profile, "speciesTaxon": 4932, "sourceId": self.source,
            "centeringGroup": "GROUP:one", "perturbationId": perturbation,
            "interventionIds": [intervention], "readoutIds": list(self.readouts),
            "distributionTypes": list(self.distributions), "target": target,
        }

    @staticmethod
    def query_record(record: dict[str, object]) -> dict[str, object]:
        return {key: copy.deepcopy(value) for key, value in record.items() if key != "target"}

    def prediction_record(self, record: dict[str, object]) -> dict[str, object]:
        parameters: list[dict[str, float]] = []
        for distribution, value in zip(self.distributions, record["target"]):
            predicted = 1.0 if value is None else float(value)
            parameters.append(
                {"mean": predicted, "logScale": 0.0}
                if distribution == "gaussian"
                else {"logMean": math.log(predicted), "logInverseDispersion": 0.0}
            )
        return {**self.query_record(record), "predictionParameters": parameters}

    def _write_jsonl(self, root: Path, records: list[dict[str, object]], *, gzip_file: bool = False) -> dict[str, object]:
        root.mkdir(parents=True, exist_ok=True)
        name = "profiles-000.jsonl.gz" if gzip_file else "profiles-000.jsonl"
        path = root / name
        payload = b"".join(_canonical(record) + b"\n" for record in records)
        if gzip_file:
            with gzip.open(path, "wb", mtime=0) as stream:
                stream.write(payload)
        else:
            path.write_bytes(payload)
        return {"path": name, "sha256": _sha(path), "bytes": path.stat().st_size, "records": len(records)}

    def _write_snapshot(self, root: Path, role: str, records: list[dict[str, object]]) -> None:
        shard = self._write_jsonl(root, records)
        manifest: dict[str, object] = {
            "schema": evaluator.SCHEMA,
            "datasetId": "fixture-query" if role == evaluator.PREDICTION_ROLE else f"fixture-{role}",
            "version": "1",
            "role": role, "labelClass": "none" if role == evaluator.PREDICTION_ROLE else "molecular",
            "benchmarkLabelsPresent": False, "valueSpace": "fixture-space-v1",
            "speciesTaxa": [4932], "sourceIds": [self.source], "shards": [shard],
        }
        if role == evaluator.CENTERING_ROLE:
            manifest["fittingOnly"] = True
        elif role == evaluator.TRUTH_ROLE:
            manifest["evaluatorOnly"] = True
            manifest.update({
                "queryResource": self.query["resource"],
                "queryDatasetManifestDigest": self.query["manifestDigest"],
                "queryManifestSha256": self.query_sha,
            })
        else:
            manifest.update({
                "modelCheckpointContentSha256": hashlib.sha256(self.checkpoint_bytes).hexdigest(),
                "queryResource": self.query["resource"],
                "queryDatasetManifestDigest": self.query["manifestDigest"],
                "queryManifestSha256": self.query_sha,
                "targetValuesPresent": False, "observedMaskPresent": False,
            })
        (root / "evaluation.json").write_bytes(_canonical(manifest))

    def _write_query(self) -> str:
        root = Path(self.query["path"])
        shard = self._write_jsonl(root, self.query_records)
        manifest = {
            "schema": evaluator.QUERY_SCHEMA, "datasetId": "fixture-query", "version": "1",
            "role": "molecular-validation-query", "labelClass": "none",
            "targetValuesPresent": False, "observedMaskPresent": False,
            "valueSpace": "fixture-space-v1", "speciesTaxa": [4932],
            "sourceIds": [self.source], "shards": [shard],
        }
        path = root / "query.json"
        path.write_bytes(_canonical(manifest))
        return _sha(path)

    def _write_checkpoint(self) -> dict[str, object]:
        path = self.inputs_root / "modelCheckpoint" / "payload" / "payload"
        path.parent.mkdir(parents=True)
        path.write_bytes(self.checkpoint_bytes)
        digest = f"sha256:{'a' * 64}"
        return {
            "resource": f"artifact:{digest}", "kind": "artifact",
            "artifacts": {"payload": digest}, "paths": {"payload": str(path)}, "path": str(path),
        }

    def _write_roster(self) -> None:
        root = Path(self.roster["path"])
        all_ids = sorted(identifier for values in self.roles.values() for identifier in values)
        roster_bytes = b"".join(
            f"{identifier}\t{evaluator._role_from_digest(hashlib.sha256(evaluator.ROSTER_ASSIGNMENT_DOMAIN + identifier.encode('ascii')).hexdigest())[0]}\t{hashlib.sha256(evaluator.ROSTER_ASSIGNMENT_DOMAIN + identifier.encode('ascii')).hexdigest()}\n".encode()
            for identifier in all_ids
        )
        (root / "held-intervention-roster.tsv").write_bytes(roster_bytes)
        coverage = {
            "schema": evaluator.ROSTER_SCHEMA,
            "assignment": {"domainHex": evaluator.ROSTER_ASSIGNMENT_DOMAIN.hex(), "digest": "sha256",
                "bucketRule": "int(first-16-lowercase-hex,16) mod 100",
                "roles": {"0-9": "molecular-final", "10-29": "molecular-validation", "30-99": "pretrain"}},
            "sourceCount": 2, "identityMapping": {"id": "SGD-MAP:fixture", "sha256": "d" * 64},
            "minimumIntersectionSize": 1, "intersectionSize": len(all_ids),
            "roleCounts": {role: len(self.roles[role]) for role in ("pretrain", "molecular-validation", "molecular-final")},
            "rejectionCounts": {"qcFailed": 0, "notPassingAllProtectedSources": 0, "identicalDuplicatesCollapsed": 0},
            "rosterPath": "held-intervention-roster.tsv", "rosterSha256": hashlib.sha256(roster_bytes).hexdigest(),
            "sources": [],
        }
        for index in range(2):
            revision = f"sha256:{str(index + 4) * 64}"
            coverage["sources"].append({
                "resource": (
                    f"omf://fixture/datasetsnapshot/protected-inventory-{index}@{revision}"
                ),
                "revision": revision,
                "artifactManifestDigest": f"sha256:{str(index + 8) * 64}",
                "sourceId": f"INVENTORY:{index}", "sourceRelease": f"release-{index}",
                "identityMappingId": "SGD-MAP:fixture", "identityMappingSha256": "d" * 64,
                "manifestSha256": str(index + 6) * 64, "records": len(all_ids),
                "duplicateRecords": 0, "uniqueInterventions": len(all_ids), "qcPassing": len(all_ids),
                "qcFailed": 0, "intersectionCoverage": len(all_ids), "exclusions": [],
            })
        (root / "coverage.json").write_bytes(_canonical(coverage))

    def _write_audit(self) -> None:
        roster_root = Path(self.roster["path"])
        held_sets = {role: set(values) for role, values in self.roles.items()}
        all_ids = set().union(*held_sets.values())
        datasets = {}
        for index, role in enumerate(("pretrain", "molecularReward", "molecularValidation", "molecularFinal"), 6):
            trajectory_genes = (
                held_sets["molecular-validation"] if role == "molecularValidation" else {f"GENE:{role}"}
            )
            datasets[role] = {
                "resource": f"omf://fixture/datasetsnapshot/{role}@sha256:{str(index) * 64}",
                "revision": f"sha256:{str(index) * 64}", "manifestDigest": f"sha256:{str(index) * 64}",
                "datasetId": f"DATASET:fixture-{role}", "version": "1",
                "role": audit_emitter.EXPECTED_ROLES[role],
                "corpusManifestSha256": str(index) * 64, "contentDigest": str(index) * 64,
                "trajectoryGenesSha256": str(index) * 64, "trajectoryGeneSetSha256": str(index) * 64,
                "trajectoryGeneCount": len(trajectory_genes), "records": 1, "targetValues": 1,
                "modalities": ["MODALITY:transcript"], "sourceIds": [self.source], "speciesTaxa": [4932],
            }
            datasets[role]["trajectoryGeneSetSha256"] = evaluator._gene_set_sha256(trajectory_genes)
        coverage = json.loads((roster_root / "coverage.json").read_text())
        source_inventories = [
            {key: value for key, value in source.items() if key != "exclusions"}
            for source in coverage["sources"]
        ]
        held_union = held_sets["molecular-validation"] | held_sets["molecular-final"]
        audit = {
            "schema": audit_emitter.AUDIT_SCHEMA, "auditPassed": True,
            "strictInterventionIsolation": True, "leakageViolations": 0,
            "leakedTrajectoryGenes": [], "benchmarkLabelRecords": 0,
            "omfPriorAdmissionRequired": True, "datasets": datasets,
            "heldRoster": {
                "resource": self.roster["resource"],
                "revision": self.roster["resource"].rpartition("@")[2],
                "manifestDigest": self.roster["manifestDigest"],
                "rosterSha256": _sha(roster_root / "held-intervention-roster.tsv"),
                "coverageSha256": _sha(roster_root / "coverage.json"),
                "assignmentDomainHex": evaluator.ROSTER_ASSIGNMENT_DOMAIN.hex(),
                "bucketRule": "int(first-16-lowercase-hex,16) mod 100",
                "identityMappingId": "SGD-MAP:fixture", "identityMappingSha256": "d" * 64,
                "sourceInventories": source_inventories,
                "intersectionSize": len(all_ids), "pretrainGeneCount": len(held_sets["pretrain"]),
                "validationGeneSetSha256": evaluator._gene_set_sha256(held_sets["molecular-validation"]),
                "validationGeneCount": len(held_sets["molecular-validation"]),
                "finalGeneSetSha256": evaluator._gene_set_sha256(held_sets["molecular-final"]),
                "finalGeneCount": len(held_sets["molecular-final"]),
                "unionGeneSetSha256": evaluator._gene_set_sha256(held_union), "unionGeneCount": len(held_union),
            },
        }
        (Path(self.audit["path"]) / "corpus-audit.json").write_bytes(_canonical(audit))

    def evaluate(self, **kwargs):
        return evaluator.evaluate_molecular_predictions(
            self.center, self.prediction_root, self.truth, self.query, self.audit,
            self.roster, self.checkpoint, **kwargs,
        )

    @staticmethod
    def rewrite_snapshot(root: Path, mutate) -> None:
        manifest_path = root / "evaluation.json"
        manifest = json.loads(manifest_path.read_text())
        shard_path = root / manifest["shards"][0]["path"]
        records = [json.loads(line) for line in shard_path.read_text().splitlines()]
        mutate(records, manifest)
        payload = b"".join(_canonical(record) + b"\n" for record in records)
        shard_path.write_bytes(payload)
        manifest["shards"][0].update(sha256=_sha(shard_path), bytes=len(payload), records=len(records))
        manifest_path.write_bytes(_canonical(manifest))


class MolecularEvaluatorV2Tests(unittest.TestCase):
    def test_file_valued_prediction_tar_is_canonical_and_streamed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            bundle = Path(temporary) / "predictions.tar"
            with tarfile.open(bundle, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name in ("evaluation.json", "profiles-000.jsonl"):
                    source = fixture.prediction_root / name
                    member = tarfile.TarInfo(name)
                    member.size = source.stat().st_size
                    member.mode = 0o644
                    member.mtime = member.uid = member.gid = 0
                    member.uname = member.gname = ""
                    with source.open("rb") as stream:
                        archive.addfile(member, stream)
            manifest = evaluator.SnapshotManifest.load(
                bundle, evaluator.PREDICTION_ROLE
            )
            profiles, records = evaluator._load_profiles(manifest, 1_048_576, 20.0)
            self.assertEqual(records, len(fixture.truth_records))
            self.assertEqual(set(profiles), {row["profileId"] for row in fixture.truth_records})

            noncanonical = Path(temporary) / "noncanonical.tar"
            with tarfile.open(noncanonical, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name in ("evaluation.json", "profiles-000.jsonl"):
                    source = fixture.prediction_root / name
                    member = tarfile.TarInfo(name)
                    member.size = source.stat().st_size
                    member.mode = 0o644
                    member.mtime = 1
                    with source.open("rb") as stream:
                        archive.addfile(member, stream)
            with self.assertRaisesRegex(
                evaluator.MolecularEvaluationError, "metadata is not canonical"
            ):
                evaluator.SnapshotManifest.load(noncanonical, evaluator.PREDICTION_ROLE)

    def fixture(self, temporary: str) -> Fixture:
        return Fixture(Path(temporary))

    def test_accepts_finalized_corpus_audit_emitter_contract(self) -> None:
        # Use the corpus-audit module's own end-to-end synthetic DatasetSnapshots;
        # this catches drift that a parallel hand-built audit document would hide.
        from tests.test_slp11_corpus_audit import CorpusAuditTest

        CorpusAuditTest.setUpClass()
        builder = CorpusAuditTest(methodName="test_passing_audit_is_deterministic_and_schema_valid")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpora, held_roster, protected_inventories = builder._fixture(root)
            destination = (
                root / "run" / "stages" / "evaluate" / "inputs"
                / "corpusAudit" / "emitted-corpus-audit"
            )
            destination.parent.mkdir(parents=True)
            report, digest = audit_emitter.write_audit_artifact(
                corpora, held_roster, protected_inventories, destination
            )
            revision = "sha256:" + "e" * 64
            pinned = evaluator.resolve_pinned_dataset_input(
                {
                    "resource": (
                        "omf://abiome/slp/datasetsnapshot/emitted-corpus-audit@"
                        + revision
                    ),
                    "mode": "copy",
                    "path": str(destination),
                    "manifestDigest": "sha256:" + "f" * 64,
                },
                "corpusAudit",
            )
            loaded, observed_digest = evaluator._load_corpus_audit(pinned)
            self.assertEqual(loaded, report)
            self.assertEqual(observed_digest, digest)
            self.assertEqual(
                set(loaded["heldRoster"]["sourceInventories"][0]),
                evaluator.AUDIT_SOURCE_INVENTORY_FIELDS,
            )

    def test_protected_inventory_snapshot_provenance_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            audit_path = Path(fixture.audit["path"]) / "corpus-audit.json"
            audit = json.loads(audit_path.read_text())
            audit["heldRoster"]["sourceInventories"][0]["revision"] = "sha256:" + "0" * 64
            audit_path.write_bytes(_canonical(audit))
            with self.assertRaisesRegex(
                evaluator.MolecularEvaluationError,
                "protected-inventory revision",
            ):
                fixture.evaluate()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            coverage_path = Path(fixture.roster["path"]) / "coverage.json"
            coverage = json.loads(coverage_path.read_text())
            replacement_revision = "sha256:" + "1" * 64
            coverage["sources"][0]["resource"] = (
                "omf://fixture/datasetsnapshot/replaced-protected-inventory@"
                + replacement_revision
            )
            coverage["sources"][0]["revision"] = replacement_revision
            coverage_path.write_bytes(_canonical(coverage))
            audit_path = Path(fixture.audit["path"]) / "corpus-audit.json"
            audit = json.loads(audit_path.read_text())
            audit["heldRoster"]["coverageSha256"] = _sha(coverage_path)
            audit_path.write_bytes(_canonical(audit))
            with self.assertRaisesRegex(
                evaluator.MolecularEvaluationError,
                "source inventory bindings mismatch",
            ):
                fixture.evaluate()

    def test_success_scores_gaussian_and_nb_and_skips_null_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            report = fixture.evaluate()
            self.assertTrue(report["diagnostic"]["diagnosticPassed"])
            self.assertEqual(report["audit"]["nullTruthValuesNotScored"], 1)
            self.assertEqual(report["audit"]["predictionsCoverPreregisteredValues"], 6)
            self.assertEqual(report["overall"]["ordinary"]["distributionTargets"], {
                "gaussian": 3, "negative-binomial": 2
            })
            self.assertTrue(math.isfinite(report["overall"]["ordinary"]["meanNll"]))

    def test_prediction_target_or_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            for forged in ("target", "observedMask"):
                with self.subTest(forged=forged):
                    copied = Path(temporary) / f"prediction-{forged}"
                    copied.mkdir()
                    for item in fixture.prediction_root.iterdir():
                        (copied / item.name).write_bytes(item.read_bytes())
                    Fixture.rewrite_snapshot(copied, lambda records, _manifest: records[0].__setitem__(forged, []))
                    with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "fields do not match"):
                        evaluator.evaluate_molecular_predictions(
                            fixture.center, copied, fixture.truth, fixture.query, fixture.audit,
                            fixture.roster, fixture.checkpoint,
                        )

    def test_missing_extra_and_readout_panel_drift_fail_exact_join(self) -> None:
        mutations = {
            "missing": lambda records, _manifest: records.pop(),
            "extra": lambda records, _manifest: records.append(copy.deepcopy(records[0])),
            "readout": lambda records, _manifest: records[0]["readoutIds"].__setitem__(0, "READOUT:forged"),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(temporary)
                Fixture.rewrite_snapshot(fixture.prediction_root, mutation)
                with self.assertRaises(evaluator.MolecularEvaluationError):
                    fixture.evaluate()

    def test_stale_query_and_checkpoint_digests_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            Fixture.rewrite_snapshot(
                fixture.prediction_root,
                lambda _records, manifest: manifest.__setitem__("queryManifestSha256", "f" * 64),
            )
            with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "query binding"):
                fixture.evaluate()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            Path(fixture.checkpoint["path"]).write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "checkpoint bytes"):
                fixture.evaluate()

    def test_held_overlap_and_forged_roster_role_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            held = fixture.roles["molecular-validation"][0]
            Fixture.rewrite_snapshot(
                Path(fixture.center["path"]),
                lambda records, _manifest: records[0].update(fixture.target_record(held, [0.0, 4, 0.0])),
            )
            with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "fitting centering contains"):
                fixture.evaluate()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            roster_path = Path(fixture.roster["path"]) / "held-intervention-roster.tsv"
            roster_path.write_text(roster_path.read_text().replace("molecular-validation", "pretrain", 1))
            with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "bucket/role"):
                fixture.evaluate()

    def test_incomplete_roster_coverage_and_audit_query_domain_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            coverage_path = Path(fixture.roster["path"]) / "coverage.json"
            coverage = json.loads(coverage_path.read_text())
            coverage["sources"][0]["intersectionCoverage"] -= 1
            coverage_path.write_bytes(_canonical(coverage))
            audit_path = Path(fixture.audit["path"]) / "corpus-audit.json"
            audit = json.loads(audit_path.read_text())
            audit["heldRoster"]["coverageSha256"] = _sha(coverage_path)
            audit_path.write_bytes(_canonical(audit))
            with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "complete intersection"):
                fixture.evaluate()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            audit_path = Path(fixture.audit["path"]) / "corpus-audit.json"
            audit = json.loads(audit_path.read_text())
            audit["datasets"]["molecularValidation"]["trajectoryGeneSetSha256"] = "f" * 64
            audit_path.write_bytes(_canonical(audit))
            with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "query intervention domain"):
                fixture.evaluate()

    def test_full_centering_support_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            Fixture.rewrite_snapshot(
                Path(fixture.center["path"]),
                lambda records, _manifest: [record["target"].__setitem__(2, None) for record in records],
            )
            with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "full fitting centering support"):
                fixture.evaluate()

    def test_mixed_species_is_explicitly_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            Fixture.rewrite_snapshot(
                fixture.prediction_root,
                lambda _records, manifest: manifest.__setitem__("speciesTaxa", [4932, 9606]),
            )
            with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "yeast-only"):
                fixture.evaluate()

    def test_bounded_readline_rejects_large_decompressed_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            manifest_path = fixture.prediction_root / "evaluation.json"
            manifest = json.loads(manifest_path.read_text())
            plain = fixture.prediction_root / manifest["shards"][0]["path"]
            records = [json.loads(line) for line in plain.read_text().splitlines()]
            records[0]["padding"] = "x" * 3000
            decompressed = b"".join(_canonical(record) + b"\n" for record in records)
            compressed = fixture.prediction_root / "profiles-000.jsonl.gz"
            with gzip.open(compressed, "wb") as stream:
                stream.write(decompressed)
            plain.unlink()
            manifest["shards"] = [{"path": compressed.name, "sha256": _sha(compressed),
                "bytes": compressed.stat().st_size, "records": len(records)}]
            manifest_path.write_bytes(_canonical(manifest))
            with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "exceeds maxLineBytes"):
                fixture.evaluate(max_line_bytes=1024)

    def test_undefined_source_correlation_fails_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            def constant(records, _manifest):
                for record in records:
                    record["predictionParameters"] = [
                        {"mean": 1.0, "logScale": 0.0},
                        {"logMean": 0.0, "logInverseDispersion": 0.0},
                        {"mean": 1.0, "logScale": 0.0},
                    ]
            Fixture.rewrite_snapshot(fixture.prediction_root, constant)
            self.assertFalse(fixture.evaluate()["diagnostic"]["diagnosticPassed"])

    def test_finite_gaussian_mean_is_not_mistaken_for_a_log_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            Fixture.rewrite_snapshot(
                fixture.prediction_root,
                lambda records, _manifest: records[0]["predictionParameters"][0].__setitem__("mean", 25.0),
            )
            report = fixture.evaluate(maximum_absolute_log_scale=20.0)
            self.assertEqual(report["overall"]["ordinary"]["targets"], 5)

    def test_file_valued_omf_payload_and_checkpoint_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            payload = Path(temporary) / "artifact-input" / "payload" / "payload"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"file artifact\n")
            digest = f"sha256:{'e' * 64}"
            wrapped = {"resource": f"artifact:{digest}", "kind": "artifact",
                "artifacts": {"payload": digest}, "paths": {"payload": str(payload)}, "path": str(payload)}
            resolved, observed = evaluator.resolve_literal_omf_artifact(wrapped, "fileArtifact")
            self.assertEqual(Path(resolved), payload.resolve())
            self.assertEqual(observed, digest)
            checkpoint = evaluator.resolve_pinned_checkpoint_input(fixture.checkpoint)
            self.assertEqual(checkpoint.content_sha256, hashlib.sha256(fixture.checkpoint_bytes).hexdigest())

    def test_dataset_snapshot_and_renderer_require_exact_frozen_query_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            with self.assertRaisesRegex(evaluator.MolecularEvaluationError, "immutable copied"):
                forged = {**fixture.truth, "mode": "mount"}
                evaluator.resolve_pinned_dataset_input(forged, "molecularTruth")
            text = renderer.render_workload_text(
                "dataset/centering", f"sha256:{'a' * 64}", "dataset/truth", "dataset/audit",
                "dataset/roster", "dataset/query", fixture.query["resource"],
                fixture.query["manifestDigest"], f"sha256:{'b' * 64}",
            )
            self.assertIn(fixture.query["resource"], text)
            self.assertIn(fixture.query["manifestDigest"], text)
            with self.assertRaises(renderer.WorkloadRenderError):
                renderer.render_workload_text(
                    "dataset/centering", f"sha256:{'a' * 64}", "dataset/truth", "dataset/audit",
                    "dataset/roster", "dataset/query", "omf://mutable/query", fixture.query["manifestDigest"],
                    f"sha256:{'b' * 64}",
                )

    def test_schemas_and_module_are_v2_target_separated(self) -> None:
        evaluation_schema = json.loads((ROOT / "schemas" / "slp-molecular-evaluation-v2.schema.json").read_text())
        query_schema = json.loads((ROOT / "schemas" / "slp-molecular-query-manifest-v1.schema.json").read_text())
        serialized = json.dumps(evaluation_schema)
        self.assertIn("negative-binomial", serialized)
        self.assertIn("modelCheckpointContentSha256", serialized)
        self.assertNotIn("sourceCorpusContentDigest", serialized + json.dumps(query_schema))
        module_text = (MODULE_ROOT / "module.yaml").read_text()
        self.assertIn("diagnosticPassed", module_text)
        self.assertNotIn("\n        passed:", module_text)
        emitted_roles = audit_emitter.EXPECTED_ROLES
        schema_roles = set(audit_schema["$defs"]["corpusIdentity"]["properties"]["role"]["enum"])
        self.assertEqual(evaluator.AUDIT_DATASET_ROLES, emitted_roles)
        self.assertEqual(set(emitted_roles.values()), schema_roles)


if __name__ == "__main__":
    unittest.main()
