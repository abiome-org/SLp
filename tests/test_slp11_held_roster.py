"""Determinism and fail-closed tests for the held-intervention roster."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-held-roster"
WORKLOAD = Path(__file__).resolve().parents[1] / "workloads" / "slp-1-1-held-roster.yaml"
sys.path.insert(0, str(MODULE))

from roster import (  # noqa: E402
    ASSIGNMENT_DOMAIN,
    HeldRosterError,
    RosterBounds,
    assign_intervention,
    build_held_roster,
    resolve_pinned_dataset_input,
    role_from_digest,
)


class HeldRosterTest(unittest.TestCase):
    @staticmethod
    def _record(identifier: str, passing: bool = True, **extra: object) -> dict[str, object]:
        return {
            "schema": "slp.intervention-identity-record/v1",
            "interventionId": identifier,
            "ncbiTaxon": 4932,
            "qcPassing": passing,
            **extra,
        }

    def _inventory(
        self,
        root: Path,
        name: str,
        records: list[dict[str, object]],
        *,
        taxon: int = 4932,
        namespace: str = "SGD",
        mapping_id: str = "sgd:fixture-mapping-v1",
        mapping_sha256: str | None = None,
    ) -> Path:
        source = root / name
        source.mkdir()
        shard = source / "inventory-000.jsonl"
        shard.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
            newline="\n",
        )
        manifest = {
            "schema": "slp.intervention-identity-inventory/v1",
            "sourceId": f"fixture:{name}",
            "sourceRelease": f"{name}-2026-09-03",
            "ncbiTaxon": taxon,
            "stableIdNamespace": namespace,
            "identityMappingId": mapping_id,
            "identityMappingSha256": mapping_sha256
            or hashlib.sha256(b"fixture SGD mapping").hexdigest(),
            "inventoryFormat": "slp.intervention-identity-record/v1",
            "files": [
                {
                    "path": shard.name,
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                    "records": len(records),
                }
            ],
        }
        (source / "inventory.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8", newline="\n"
        )
        return source

    @staticmethod
    def _bounds(minimum: int = 1) -> RosterBounds:
        return RosterBounds(minimum_intersection_size=minimum)

    def test_assignment_is_exact_and_boundary_roles_are_frozen(self) -> None:
        identifier = "SGD:S000000001"
        assigned = assign_intervention(identifier)
        expected = hashlib.sha256(ASSIGNMENT_DOMAIN + identifier.encode("ascii")).hexdigest()
        self.assertEqual(assigned.digest, expected)
        self.assertEqual(assigned.bucket, int(expected[:16], 16) % 100)
        self.assertEqual((assigned.role, assigned.bucket), role_from_digest(expected))

        for bucket, role in (
            (0, "molecular-final"),
            (9, "molecular-final"),
            (10, "molecular-validation"),
            (29, "molecular-validation"),
            (30, "pretrain"),
            (99, "pretrain"),
        ):
            with self.subTest(bucket=bucket):
                prefix = f"{bucket:016x}"
                self.assertEqual(role_from_digest(prefix + "0" * 48), (role, bucket))

    def test_output_is_deterministic_under_source_and_record_reordering(self) -> None:
        common = [f"SGD:S{index:09d}" for index in range(1, 9)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha = self._inventory(
                root, "alpha", [self._record(item) for item in reversed(common)]
            )
            beta = self._inventory(root, "beta", [self._record(item) for item in common])
            first = root / "first"
            second = root / "second"
            report_one = build_held_roster([alpha, beta], first, self._bounds(8))
            report_two = build_held_roster([beta, alpha], second, self._bounds(8))
            self.assertEqual(
                (first / "held-intervention-roster.tsv").read_bytes(),
                (second / "held-intervention-roster.tsv").read_bytes(),
            )
            self.assertEqual(
                (first / "coverage.json").read_bytes(),
                (second / "coverage.json").read_bytes(),
            )
            self.assertEqual(report_one["rosterSha256"], report_two["rosterSha256"])
            rows = (first / "held-intervention-roster.tsv").read_text().splitlines()
            self.assertEqual([row.split("\t")[0] for row in rows], common)

    def test_intersection_and_per_source_exclusions_use_only_qc_passing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha = self._inventory(
                root,
                "alpha",
                [
                    self._record("SGD:S000000001"),
                    self._record("SGD:S000000002"),
                    self._record("SGD:S000000003", False),
                ],
            )
            beta = self._inventory(
                root,
                "beta",
                [
                    self._record("SGD:S000000002"),
                    self._record("SGD:S000000003"),
                    self._record("SGD:S000000004"),
                ],
            )
            output = root / "output"
            report = build_held_roster([alpha, beta], output, self._bounds())
            self.assertEqual(report["intersectionSize"], 1)
            self.assertEqual(
                (output / "held-intervention-roster.tsv").read_text().split("\t", 1)[0],
                "SGD:S000000002",
            )
            by_source = {item["sourceId"]: item for item in report["sources"]}
            alpha_exclusions = {
                (item["interventionId"], item["reason"])
                for item in by_source["fixture:alpha"]["exclusions"]
            }
            self.assertIn(
                ("SGD:S000000001", "not-qc-passing-in-all-protected-sources"),
                alpha_exclusions,
            )
            self.assertIn(("SGD:S000000003", "qc-failed"), alpha_exclusions)

    def test_frozen_expectations_accept_exact_roster_and_reject_drift(self) -> None:
        common = [f"SGD:S{index:09d}" for index in range(1, 9)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha = self._inventory(root, "alpha", [self._record(item) for item in common])
            beta = self._inventory(root, "beta", [self._record(item) for item in common])
            observed = build_held_roster([alpha, beta], root / "observed", self._bounds(8))
            counts = observed["roleCounts"]
            frozen = RosterBounds(
                minimum_intersection_size=8,
                expected_intersection_size=8,
                expected_pretrain_count=counts["pretrain"],
                expected_validation_count=counts["molecular-validation"],
                expected_final_count=counts["molecular-final"],
                expected_roster_sha256=observed["rosterSha256"],
            )
            build_held_roster([alpha, beta], root / "accepted", frozen)
            with self.assertRaisesRegex(HeldRosterError, "frozen roster expectation mismatch"):
                build_held_roster(
                    [alpha, beta],
                    root / "rejected",
                    RosterBounds(
                        minimum_intersection_size=8,
                        expected_intersection_size=8,
                        expected_pretrain_count=counts["pretrain"],
                        expected_validation_count=counts["molecular-validation"],
                        expected_final_count=counts["molecular-final"],
                        expected_roster_sha256="0" * 64,
                    ),
                )

    def test_production_workload_freezes_exact_outcome_blind_inputs_and_roster(self) -> None:
        workload = yaml.safe_load(WORKLOAD.read_text(encoding="utf-8"))
        stage = workload["spec"]["graph"]["stages"][0]
        self.assertEqual(stage["module"], "modules/slp-1-1-held-roster/module.yaml")
        self.assertEqual(
            stage["inputs"],
            {
                "proteomeInventory": "dataset/slp-1-1-proteome-intervention-inventory-v1",
                "atlasInventory": "dataset/slp-1-1-atlas-intervention-inventory-v1",
            },
        )
        self.assertEqual(
            stage["config"],
            {
                "minimumIntersectionSize": 2700,
                "expectedIntersectionSize": 2700,
                "expectedPretrainCount": 1903,
                "expectedValidationCount": 529,
                "expectedFinalCount": 268,
                "expectedRosterSha256": "c27eb11a20f593235131f28fc29d8fbd69735f8a0aea88736104850bb875117a",
                "maxSources": 2,
                "maxFilesPerSource": 1,
                "maxRecordsPerSource": 5000,
                "maxLineBytes": 4096,
            },
        )
        self.assertNotIn("benchmark", WORKLOAD.read_text(encoding="utf-8").casefold())

    def test_protected_sources_must_share_the_exact_identity_mapping(self) -> None:
        for field in ("id", "digest"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                alpha = self._inventory(
                    root, "alpha", [self._record("SGD:S000000001")]
                )
                options = (
                    {"mapping_id": "sgd:different-mapping-v2"}
                    if field == "id"
                    else {"mapping_sha256": "f" * 64}
                )
                beta = self._inventory(
                    root,
                    "beta",
                    [self._record("SGD:S000000001")],
                    **options,
                )
                with self.assertRaisesRegex(HeldRosterError, "exact same identityMapping"):
                    build_held_roster([alpha, beta], root / "output", self._bounds())

    def test_literal_pinned_dataset_input_resolver_accepts_actual_omf_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "stages" / "roster" / "inputs" / "proteome" / "proteome-v2"
            payload.mkdir(parents=True)
            revision = "sha256:" + "a" * 64
            manifest = "sha256:" + "b" * 64
            value = {
                "resource": (
                    "omf://abiome/slp/datasetsnapshot/proteome-v2@" + revision
                ),
                "mode": "copy",
                "path": str(payload),
                "manifestDigest": manifest,
            }
            resolved = resolve_pinned_dataset_input(value, "proteome")
            self.assertEqual(resolved.path, str(payload.resolve()))
            self.assertEqual(resolved.revision, revision)
            self.assertEqual(resolved.manifest_digest, manifest)

    def test_literal_pinned_dataset_input_resolver_rejects_spoofed_or_mutable_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "inputs" / "proteome" / "proteome-v2"
            payload.mkdir(parents=True)
            base = {
                "resource": (
                    "omf://abiome/slp/datasetsnapshot/proteome-v2@sha256:" + "a" * 64
                ),
                "mode": "copy",
                "path": str(payload),
                "manifestDigest": "sha256:" + "b" * 64,
            }
            cases = (
                ("bare", {"path": str(payload)}, "exact materialized"),
                ("extra-kind", {**base, "kind": "DatasetSnapshot"}, "exact materialized"),
                (
                    "wrong-kind",
                    {**base, "resource": base["resource"].replace("datasetsnapshot", "artifact")},
                    "kind must be DatasetSnapshot",
                ),
                (
                    "mutable-revision",
                    {**base, "resource": base["resource"].split("@")[0] + "@latest"},
                    "admission-pinned SHA-256",
                ),
                ("mounted", {**base, "mode": "mount"}, "immutable copied"),
                (
                    "mutable-manifest",
                    {**base, "manifestDigest": "latest"},
                    "admission-pinned SHA-256",
                ),
                (
                    "wrong-name",
                    {
                        **base,
                        "resource": base["resource"].replace("proteome-v2@", "other-v2@"),
                    },
                    "path is inconsistent",
                ),
            )
            for name, value, message in cases:
                with self.subTest(name=name), self.assertRaisesRegex(HeldRosterError, message):
                    resolve_pinned_dataset_input(value, "proteome")

    def test_outcome_bearing_inventory_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = self._inventory(
                root,
                "bad",
                [self._record("SGD:S000000001", target=-0.5)],
            )
            good = self._inventory(root, "good", [self._record("SGD:S000000001")])
            with self.assertRaisesRegex(HeldRosterError, "outcome-like.*target"):
                build_held_roster([bad, good], root / "output", self._bounds())

    def test_symbol_taxon_namespace_and_conflicting_duplicate_are_fatal(self) -> None:
        cases = (
            (
                "symbol",
                [self._record("RAD52")],
                {},
                "canonical SGD CURIE",
            ),
            (
                "taxon",
                [self._record("SGD:S000000001")],
                {"taxon": 9606},
                "taxon 4932",
            ),
            (
                "namespace",
                [self._record("SGD:S000000001")],
                {"namespace": "symbol"},
                "exactly SGD",
            ),
            (
                "duplicate",
                [
                    self._record("SGD:S000000001", True),
                    self._record("SGD:S000000001", False),
                ],
                {},
                "conflicting duplicate",
            ),
        )
        for name, records, options, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bad = self._inventory(root, "bad", records, **options)
                good = self._inventory(root, "good", [self._record("SGD:S000000001")])
                with self.assertRaisesRegex(HeldRosterError, message):
                    build_held_roster([bad, good], root / "output", self._bounds())

    def test_digest_count_and_path_escape_drift_are_fatal(self) -> None:
        for mutation, message in (
            ("digest", "digest drift"),
            ("count", "record count drift"),
            ("escape", "canonical relative POSIX path"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bad = self._inventory(root, "bad", [self._record("SGD:S000000001")])
                good = self._inventory(root, "good", [self._record("SGD:S000000001")])
                manifest_path = bad / "inventory.json"
                manifest = json.loads(manifest_path.read_text())
                if mutation == "digest":
                    manifest["files"][0]["sha256"] = "0" * 64
                elif mutation == "count":
                    manifest["files"][0]["records"] = 2
                else:
                    outside = root / "outside.jsonl"
                    outside.write_text("{}\n")
                    manifest["files"][0]["path"] = "../outside.jsonl"
                    manifest["files"][0]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
                manifest_path.write_text(json.dumps(manifest, sort_keys=True))
                with self.assertRaisesRegex(HeldRosterError, message):
                    build_held_roster([bad, good], root / "output", self._bounds())

    def test_symlink_and_undersized_intersection_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha = self._inventory(root, "alpha", [self._record("SGD:S000000001")])
            beta = self._inventory(root, "beta", [self._record("SGD:S000000002")])
            with self.assertRaisesRegex(HeldRosterError, "undersized"):
                build_held_roster([alpha, beta], root / "output", self._bounds())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha = self._inventory(root, "alpha", [self._record("SGD:S000000001")])
            beta = self._inventory(root, "beta", [self._record("SGD:S000000001")])
            target = alpha / "inventory-000.jsonl"
            link = alpha / "linked.jsonl"
            try:
                os.symlink(target, link)
            except OSError:
                self.skipTest("symlink creation is not permitted on this Windows host")
            manifest_path = alpha / "inventory.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"][0]["path"] = link.name
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            with self.assertRaisesRegex(HeldRosterError, "must not .*symlink"):
                build_held_roster([alpha, beta], root / "output", self._bounds())


if __name__ == "__main__":
    unittest.main()
