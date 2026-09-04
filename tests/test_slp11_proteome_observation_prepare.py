"""Synthetic contracts for leakage-safe proteome observation preparation."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from io import BytesIO
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-proteome-observation-prepare-v1"
SPEC = importlib.util.spec_from_file_location(
    "slp11_proteome_observation_prepare", MODULE / "observation_prepare.py"
)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> str:
    payload = _canonical(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> tuple[str, int]:
    payload = b"".join(_canonical(value) for value in values)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _role(identifier: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        b"slp-1.1-yeast-global-held-v1\x00" + identifier.encode("ascii")
    ).hexdigest()
    bucket = int(digest[:16], 16) % 100
    if bucket <= 9:
        role = "molecular-final"
    elif bucket <= 29:
        role = "molecular-validation"
    else:
        role = "pretrain"
    return role, digest


class Fixture:
    genes = {
        "YAL001C": "SGD:S000000001",
        "YAL002W": "SGD:S000000002",
        "YAL011W": "SGD:S000000011",
        "YAL014C": "SGD:S000000014",
    }

    def __init__(self, root: Path, excluded_token: str = "opaque-excluded") -> None:
        self.root = root
        self.raw = root / "raw"
        self.interventions = root / "interventions"
        self.proteins = root / "proteins"
        self.roster = root / "roster"
        for directory in (self.raw, self.interventions, self.proteins, self.roster):
            directory.mkdir(parents=True)
        self.current = root / "current-orfs.jsonl"
        self.mapping_manifest = root / "mapping-manifest.json"
        self._write_mapping()
        self._write_interventions()
        self._write_proteins()
        self._write_roster()
        self._write_raw(excluded_token)
        self.contract = prepare.SourceContract(
            raw_files=tuple(
                prepare.FileSpec(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in sorted(self.raw.iterdir())
            ),
            intervention_manifest_sha256=hashlib.sha256(
                (self.interventions / "inventory.json").read_bytes()
            ).hexdigest(),
            protein_manifest_sha256=hashlib.sha256(
                (self.proteins / "manifest.json").read_bytes()
            ).hexdigest(),
            protein_records_sha256=hashlib.sha256(
                (self.proteins / "relations.jsonl").read_bytes()
            ).hexdigest(),
            roster_sha256=hashlib.sha256(
                (self.roster / "held-intervention-roster.tsv").read_bytes()
            ).hexdigest(),
            coverage_sha256=hashlib.sha256(
                (self.roster / "coverage.json").read_bytes()
            ).hexdigest(),
            current_orfs_sha256=hashlib.sha256(self.current.read_bytes()).hexdigest(),
            mapping_manifest_sha256=hashlib.sha256(self.mapping_manifest.read_bytes()).hexdigest(),
            mapping_id=self.mapping_id,
            mapping_sha256=self.mapping_sha,
            minimum_control_fraction=0.8,
        )
        selected = sorted((self.genes["YAL001C"], self.genes["YAL002W"]))
        trajectory = "".join(item + "\n" for item in selected).encode("ascii")
        self.expected = prepare.ExpectedCounts(
            metadata_rows=12,
            eligible_rows=5,
            eligible_genes=4,
            pretrain_records=3,
            pretrain_genes=2,
            target_values=11,
            missing_values=1,
            trajectory_genes_sha256=hashlib.sha256(trajectory).hexdigest(),
            trajectory_gene_set_sha256=hashlib.sha256(
                json.dumps(selected, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest(),
            protein_readouts=4,
            basal_controls=5,
            basal_observed_values=17,
            basal_supported_readouts=3,
            validation_genes=1,
            validation_rows=1,
            final_genes=1,
            final_rows=1,
            quarantine_rows=1,
            qc_rows=1,
        )
        self.bounds = prepare.Bounds(
            max_metadata_rows=32,
            max_readouts=16,
            max_mapping_records=32,
        )

    def _write_mapping(self) -> None:
        records = []
        for systematic, curie in sorted(self.genes.items(), key=lambda item: item[1]):
            records.append(
                {
                    "schema": "slp.sgd-current-orf/v1",
                    "canonicalSgdCurie": curie,
                    "systematicName": systematic,
                    "featureQualifier": "Verified",
                    "ncbiTaxon": 4932,
                    "secondaryIdentifiers": [],
                    "secondaryIdentifiersResolve": False,
                    "displayMetadata": {
                        "aliases": [],
                        "resolvesIdentity": False,
                        "standardGeneName": None,
                    },
                }
            )
        current_sha, current_bytes = _write_jsonl(self.current, records)
        basis = {
            "schema": "slp.sgd-stable-id-mapping-digest/v1",
            "identityMappingId": "slp-sgd-map:fixture-v1",
            "ncbiTaxon": 4932,
            "normalizationPolicy": {
                "systematicNameMatch": "exact-case-sensitive",
            },
            "inputFiles": [],
            "outputFiles": [
                {
                    "name": "current-orfs.jsonl",
                    "records": len(records),
                    "bytes": current_bytes,
                    "sha256": current_sha,
                }
            ],
        }
        self.mapping_id = "slp-sgd-map:fixture-v1"
        self.mapping_sha = hashlib.sha256(_canonical(basis)).hexdigest()
        _write_json(
            self.mapping_manifest,
            {
                "schema": "slp.sgd-stable-id-mapping/v1",
                "identityMappingId": self.mapping_id,
                "identityMappingSha256": self.mapping_sha,
                "ncbiTaxon": 4932,
                "digestBasis": basis,
            },
        )

    def _write_interventions(self) -> None:
        identifiers = sorted(
            [
                self.genes["YAL001C"],
                self.genes["YAL001C"],
                self.genes["YAL002W"],
                self.genes["YAL011W"],
                self.genes["YAL014C"],
            ]
        )
        records = [
            {
                "schema": "slp.intervention-identity-record/v1",
                "interventionId": identifier,
                "ncbiTaxon": 4932,
                "qcPassing": True,
            }
            for identifier in identifiers
        ]
        records_sha, _ = _write_jsonl(self.interventions / "interventions.jsonl", records)
        _write_json(
            self.interventions / "inventory.json",
            {
                "schema": "slp.intervention-identity-inventory/v1",
                "sourceId": "mendeley:w8jtmnszd9.2",
                "sourceRelease": "10.17632/w8jtmnszd9.2",
                "ncbiTaxon": 4932,
                "stableIdNamespace": "SGD",
                "identityMappingId": self.mapping_id,
                "identityMappingSha256": self.mapping_sha,
                "inventoryFormat": "slp.intervention-identity-record/v1",
                "files": [
                    {
                        "path": "interventions.jsonl",
                        "sha256": records_sha,
                        "records": len(records),
                    }
                ],
            },
        )

    def _write_proteins(self) -> None:
        accessions = ["P00001", "P00002", "P00003", "PAMB01"]
        rows = []
        for accession in accessions:
            relations = (
                [self.genes["YAL001C"], self.genes["YAL002W"]]
                if accession == "PAMB01"
                else [self.genes["YAL001C"]]
            )
            rows.append(
                {
                    "schema": "slp.proteome-protein-relation/v1",
                    "proteinId": f"UniProtKB:{accession}",
                    "sourceAccession": accession,
                    "sourceAccessionType": {
                        "caseNormalization": "none",
                        "namespaceInferred": False,
                        "source": "UniProtKB",
                        "type": "UniProtKB ID",
                    },
                    "ncbiTaxon": 4932,
                    "currentOrfRelations": relations,
                    "currentOrfRelationCount": len(relations),
                    "chooseFirstAllowed": False,
                }
            )
        records_sha, _ = _write_jsonl(self.proteins / "relations.jsonl", rows)
        _write_json(
            self.proteins / "manifest.json",
            {
                "schema": "slp.proteome-protein-relation-inventory/v1",
                "sourceId": "mendeley:w8jtmnszd9.2",
                "sourceRelease": "10.17632/w8jtmnszd9.2",
                "ncbiTaxon": 4932,
                "identityMappingId": self.mapping_id,
                "identityMappingSha256": self.mapping_sha,
                "relationFormat": "slp.proteome-protein-relation/v1",
                "files": [
                    {
                        "path": "relations.jsonl",
                        "sha256": records_sha,
                        "records": len(rows),
                    }
                ],
            },
        )

    def _write_roster(self) -> None:
        identifiers = [
            self.genes["YAL001C"],
            self.genes["YAL011W"],
            self.genes["YAL014C"],
        ]
        lines = []
        roles = Counter()
        for identifier in sorted(identifiers):
            role, digest = _role(identifier)
            roles[role] += 1
            lines.append(f"{identifier}\t{role}\t{digest}\n")
        roster_bytes = "".join(lines).encode("ascii")
        (self.roster / "held-intervention-roster.tsv").write_bytes(roster_bytes)
        _write_json(
            self.roster / "coverage.json",
            {
                "schema": "slp.held-intervention-roster-report/v1",
                "rosterPath": "held-intervention-roster.tsv",
                "rosterSha256": hashlib.sha256(roster_bytes).hexdigest(),
                "intersectionSize": len(identifiers),
                "identityMapping": {"id": self.mapping_id, "sha256": self.mapping_sha},
                "assignment": {
                    "domainHex": b"slp-1.1-yeast-global-held-v1\x00".hex(),
                    "digest": "sha256",
                    "bucketRule": "int(first-16-lowercase-hex,16) mod 100",
                    "roles": {
                        "0-9": "molecular-final",
                        "10-29": "molecular-validation",
                        "30-99": "pretrain",
                    },
                },
                "roleCounts": dict(roles),
            },
        )

    def _write_raw(self, excluded_token: str) -> None:
        metadata = [
            ["WT,DUR1,2", "1", "1", "hpr1", "HIS3", "YOR202W"],
            ['WT"quoted', "2", "2", "hpr1", "HIS3", "YOR202W"],
            ["WT3", "3", "3", "hpr2", "HIS3", "YOR202W"],
            ["WT4", "4", "4", "hpr2", "HIS3", "YOR202W"],
            ["WT5", "5", "5", "hpr3", "HIS3", "YOR202W"],
            ["QC1", "6", "0", "hpr3", "qc", "qc"],
            ["KO-A-1", "7", "7", "hpr4", "ko", "YAL001C"],
            ["KO-A-2", "8", "8", "hpr4", "ko", "YAL001C"],
            ["KO-V", "9", "9", "hpr5", "ko", "YAL011W"],
            ["KO-F", "10", "10", "hpr5", "ko", "YAL014C"],
            ["KO-SOURCE", "11", "11", "hpr6", "ko", "YAL002W"],
            ["KO-UNKNOWN", "12", "12", "hpr6", "ko", "YZZ999C"],
        ]
        with (self.raw / "yeast5k_metadata.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(prepare.METADATA_COLUMNS)
            writer.writerows(metadata)
        rows = [
            ["P00001", "1", "2", "4", "8", "16", excluded_token, "8", "16", excluded_token, excluded_token, "32", excluded_token],
            ["P00002", "1", "4", "16", "64", "NA", excluded_token, "4", "NA", excluded_token, excluded_token, "8", excluded_token],
            ["P00003", "1", "2", "4", "NA", "NA", excluded_token, "2", "2", excluded_token, excluded_token, "2", excluded_token],
            ["PAMB01", "2", "2", "2", "2", "2", excluded_token, "1", "2", excluded_token, excluded_token, "4", excluded_token],
        ]
        with (self.raw / "yeast5k_noimpute_wide.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["Protein.Group", *(row[0] for row in metadata)])
            writer.writerows(rows)
        (self.raw / "Detection_of_KO_proteins.csv").write_text("identity-only\n", encoding="utf-8", newline="\n")
        (self.raw / "summary_fileupload.pdf").write_bytes(b"fixture documentation only")

    def build(self, destination: Path) -> dict[str, object]:
        return prepare.build_pretrain_observations(
            self.raw,
            self.interventions,
            self.proteins,
            self.roster,
            self.current,
            self.mapping_manifest,
            destination,
            source_contract=self.contract,
            expected=self.expected,
            bounds=self.bounds,
        )

    def validate_observation(self, path: Path) -> dict[str, object]:
        return prepare.validate_observation_archive(
            path,
            self.bounds,
            source_contract=self.contract,
            expected=self.expected,
        )

    def validate_basal(self, path: Path) -> dict[str, object]:
        return prepare.validate_basal_archive(
            path,
            self.bounds,
            source_contract=self.contract,
            expected=self.expected,
        )


def _tar_blobs(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, mode="r:") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
        }


def _write_tar_blobs(path: Path, blobs: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(blobs):
            payload = blobs[name]
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            archive.addfile(info, BytesIO(payload))


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    payload = BytesIO()
    with zipfile.ZipFile(
        payload, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.save(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue())
    return payload.getvalue()


def _npz_arrays(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(payload), allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


class ProteomeObservationPreparationTest(unittest.TestCase):
    def test_systematic_name_contract_includes_exact_mitochondrial_orfs(self) -> None:
        self.assertIsNotNone(prepare.SYSTEMATIC_NAME.fullmatch("Q0010"))
        self.assertIsNotNone(prepare.SYSTEMATIC_NAME.fullmatch("Q0297"))
        for invalid in ("q0010", "Q010", "Q00010", "R0010", "Q00A0"):
            self.assertIsNone(prepare.SYSTEMATIC_NAME.fullmatch(invalid))

    def test_happy_path_preserves_sparse_targets_basal_and_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary) / "fixture")
            output = Path(temporary) / "output"
            report = fixture.build(output)
            self.assertEqual(report["records"], 3)
            self.assertEqual(report["interventionGenes"], 2)
            self.assertEqual(report["targetValues"], 11)
            self.assertEqual(report["missingValues"], 1)
            self.assertEqual(report["basalSupportedReadouts"], 3)
            self.assertEqual(report["excludedValidationRows"], 1)
            self.assertEqual(report["excludedFinalRows"], 1)
            audit = json.loads((output / "preparation-audit.json").read_text())
            self.assertFalse(audit["accessBoundary"]["excludedNumericValuesInspectedOrValidated"])
            self.assertFalse(audit["accessBoundary"]["knockoutOutcomesUsedForBasal"])
            manifest = fixture.validate_observation(output / "observation-corpus.tar")
            self.assertEqual(manifest["counts"]["observedValues"], 11)
            self.assertEqual(manifest["assayedPanel"]["readouts"], 4)
            self.assertNotIn("featurePack", manifest)
            self.assertNotIn("likelihood", manifest["measurement"])
            blobs = _tar_blobs(output / "observation-corpus.tar")
            shard = next(value for name, value in blobs.items() if name.endswith(".npz"))
            with np.load(BytesIO(shard), allow_pickle=False) as arrays:
                self.assertEqual(arrays["action_id"].tolist(), [
                    "SGD:S000000001", "SGD:S000000001", "SGD:S000000002"
                ])
                self.assertEqual(arrays["target_indptr"].tolist(), [0, 4, 7, 11])
                self.assertEqual(arrays["target_readout_index"].tolist(), [0, 1, 2, 3, 0, 2, 3, 0, 1, 2, 3])
                self.assertTrue(np.allclose(arrays["target_value"], [3, 2, 1, 0, 4, 1, 1, 5, 3, 1, 2]))
            basal_blobs = _tar_blobs(output / "basal-control.tar")
            with np.load(BytesIO(basal_blobs["basal-control/basal.npz"]), allow_pickle=False) as basal:
                self.assertEqual(basal["control_observed"].tolist(), [5, 4, 3, 5])
                self.assertEqual(basal["value_present"].tolist(), [True, True, False, True])
                self.assertTrue(np.allclose(basal["value"], [2, 3, 0, 1]))

    def test_outputs_are_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root / "fixture")
            first, second = root / "first", root / "second"
            fixture.build(first)
            fixture.build(second)
            for name in ("observation-corpus.tar", "basal-control.tar", "preparation-audit.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            with tarfile.open(first / "observation-corpus.tar", mode="r:") as archive:
                members = archive.getmembers()
                self.assertEqual([item.name for item in members], sorted(item.name for item in members))
                self.assertTrue(all(item.isfile() and item.mtime == 0 and item.mode == 0o644 for item in members))
            blobs = _tar_blobs(first / "observation-corpus.tar")
            shard = next(value for name, value in blobs.items() if name.endswith(".npz"))
            with zipfile.ZipFile(BytesIO(shard)) as archive:
                self.assertEqual(archive.namelist(), sorted(archive.namelist()))
                self.assertTrue(all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist()))

    def test_outputs_are_byte_deterministic_across_fresh_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = (
                "from pathlib import Path\n"
                "import sys\n"
                "from tests.test_slp11_proteome_observation_prepare import Fixture\n"
                "Fixture(Path(sys.argv[1])).build(Path(sys.argv[2]))\n"
            )
            outputs = []
            for index, hash_seed in enumerate(("1", "8675309")):
                fixture = root / f"fixture-{index}"
                output = root / f"output-{index}"
                environment = os.environ.copy()
                environment["PYTHONHASHSEED"] = hash_seed
                subprocess.run(
                    [sys.executable, "-c", script, str(fixture), str(output)],
                    cwd=ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                outputs.append(output)
            for name in ("observation-corpus.tar", "basal-control.tar", "preparation-audit.json"):
                self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())

    def test_observation_validator_rejects_action_outside_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary) / "fixture")
            output = Path(temporary) / "output"
            fixture.build(output)
            archive_path = output / "observation-corpus.tar"
            blobs = _tar_blobs(archive_path)
            manifest_name = "proteome-observations/manifest.json"
            manifest = json.loads(blobs[manifest_name])
            shard_ref = manifest["shards"][0]
            shard_name = "proteome-observations/" + shard_ref["path"]
            arrays = _npz_arrays(blobs[shard_name])
            arrays["action_id"][0] = "SGD:S000000999"
            shard_payload = _npz_bytes(arrays)
            blobs[shard_name] = shard_payload
            shard_ref["bytes"] = len(shard_payload)
            shard_ref["sha256"] = hashlib.sha256(shard_payload).hexdigest()
            blobs[manifest_name] = _canonical(manifest)
            _write_tar_blobs(archive_path, blobs)
            with self.assertRaisesRegex(
                prepare.ProteomeObservationError, "action|trajectory|identity"
            ):
                fixture.validate_observation(archive_path)

    def test_observation_validator_rejects_duplicate_record_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary) / "fixture")
            output = Path(temporary) / "output"
            fixture.build(output)
            archive_path = output / "observation-corpus.tar"
            blobs = _tar_blobs(archive_path)
            manifest_name = "proteome-observations/manifest.json"
            manifest = json.loads(blobs[manifest_name])
            shard_ref = manifest["shards"][0]
            shard_name = "proteome-observations/" + shard_ref["path"]
            arrays = _npz_arrays(blobs[shard_name])
            arrays["metadata_row"][1] = arrays["metadata_row"][0]
            arrays["record_id"][1] = arrays["record_id"][0]
            arrays["observation_unit_id"][1] = arrays["observation_unit_id"][0]
            arrays["replicate_id"][1] = arrays["replicate_id"][0]
            shard_payload = _npz_bytes(arrays)
            blobs[shard_name] = shard_payload
            shard_ref["bytes"] = len(shard_payload)
            shard_ref["sha256"] = hashlib.sha256(shard_payload).hexdigest()
            blobs[manifest_name] = _canonical(manifest)
            _write_tar_blobs(archive_path, blobs)
            with self.assertRaisesRegex(
                prepare.ProteomeObservationError, "record|duplicate|unique"
            ):
                fixture.validate_observation(archive_path)

    def test_basal_validator_rejects_provenance_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary) / "fixture")
            output = Path(temporary) / "output"
            fixture.build(output)
            archive_path = output / "basal-control.tar"
            blobs = _tar_blobs(archive_path)
            manifest_name = "basal-control/basal.json"
            manifest = json.loads(blobs[manifest_name])
            manifest["source"]["immutableRelease"] = "10.17632/not-the-frozen-release"
            blobs[manifest_name] = _canonical(manifest)
            _write_tar_blobs(archive_path, blobs)
            with self.assertRaisesRegex(
                prepare.ProteomeObservationError, "source|provenance|identity"
            ):
                fixture.validate_basal(archive_path)

    def test_basal_validator_rejects_support_mask_inconsistent_with_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary) / "fixture")
            output = Path(temporary) / "output"
            fixture.build(output)
            archive_path = output / "basal-control.tar"
            blobs = _tar_blobs(archive_path)
            manifest_name = "basal-control/basal.json"
            profile_name = "basal-control/basal.npz"
            manifest = json.loads(blobs[manifest_name])
            arrays = _npz_arrays(blobs[profile_name])
            self.assertEqual(arrays["control_observed"].tolist(), [5, 4, 3, 5])
            arrays["value_present"][1] = False
            arrays["value_present"][2] = True
            profile_payload = _npz_bytes(arrays)
            blobs[profile_name] = profile_payload
            manifest["profileFile"]["bytes"] = len(profile_payload)
            manifest["profileFile"]["sha256"] = hashlib.sha256(profile_payload).hexdigest()
            blobs[manifest_name] = _canonical(manifest)
            _write_tar_blobs(archive_path, blobs)
            with self.assertRaisesRegex(
                prepare.ProteomeObservationError, "support|observed|mask"
            ):
                fixture.validate_basal(archive_path)

    def test_abrupt_termination_never_exposes_final_destination(self) -> None:
        class AbruptTermination(BaseException):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary) / "fixture")
            output = Path(temporary) / "output"
            original_write_tar = prepare._write_tar
            calls = 0

            def interrupt_second_archive(source: Path, destination: Path, prefix: str) -> None:
                nonlocal calls
                calls += 1
                self.assertFalse(
                    output.exists(),
                    "the final destination became visible before all outputs were complete",
                )
                if calls == 2:
                    raise AbruptTermination
                original_write_tar(source, destination, prefix)

            prepare._write_tar = interrupt_second_archive
            try:
                with self.assertRaises(AbruptTermination):
                    fixture.build(output)
            finally:
                prepare._write_tar = original_write_tar
            self.assertFalse(output.exists())

    def test_excluded_numeric_mutation_cannot_change_targets_or_basal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_fixture = Fixture(root / "first-fixture", excluded_token="opaque-one")
            second_fixture = Fixture(root / "second-fixture", excluded_token="different opaque, value")
            first, second = root / "first", root / "second"
            first_fixture.build(first)
            second_fixture.build(second)
            first_blobs = _tar_blobs(first / "observation-corpus.tar")
            second_blobs = _tar_blobs(second / "observation-corpus.tar")
            compared = [name for name in first_blobs if name.endswith(".npz") or name.endswith("readouts.jsonl") or name.endswith("trajectory-genes.txt")]
            self.assertTrue(compared)
            for name in compared:
                self.assertEqual(first_blobs[name], second_blobs[name])
            first_basal = _tar_blobs(first / "basal-control.tar")
            second_basal = _tar_blobs(second / "basal-control.tar")
            self.assertEqual(
                first_basal["basal-control/basal.npz"],
                second_basal["basal-control/basal.npz"],
            )
            self.assertNotEqual(
                (first / "basal-control.tar").read_bytes(),
                (second / "basal-control.tar").read_bytes(),
            )
            self.assertNotEqual((first / "preparation-audit.json").read_bytes(), (second / "preparation-audit.json").read_bytes())

    def test_selected_nonpositive_or_nonnumeric_value_fails_atomically(self) -> None:
        for token in ("0", "-1", "NaN", "Inf", "", " NA"):
            with self.subTest(token=token), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary) / "fixture")
                matrix = fixture.raw / "yeast5k_noimpute_wide.csv"
                with matrix.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream))
                rows[1][7] = token
                with matrix.open("w", newline="", encoding="utf-8") as stream:
                    csv.writer(stream, lineterminator="\n").writerows(rows)
                fixture.contract = replace(
                    fixture.contract,
                    raw_files=tuple(
                        prepare.FileSpec(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
                        for path in sorted(fixture.raw.iterdir())
                    ),
                )
                output = Path(temporary) / "output"
                with self.assertRaises(prepare.ProteomeObservationError):
                    fixture.build(output)
                self.assertFalse(output.exists())

    def test_inventory_cannot_be_order_zipped_or_omit_a_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary) / "fixture")
            records_path = fixture.interventions / "interventions.jsonl"
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
            records.pop()
            records_sha, _ = _write_jsonl(records_path, records)
            manifest = json.loads((fixture.interventions / "inventory.json").read_text())
            manifest["files"][0]["sha256"] = records_sha
            manifest["files"][0]["records"] = len(records)
            fixture.contract = replace(
                fixture.contract,
                intervention_manifest_sha256=_write_json(fixture.interventions / "inventory.json", manifest),
            )
            with self.assertRaisesRegex(prepare.ProteomeObservationError, "exactly reproduce"):
                fixture.build(Path(temporary) / "output")

    def test_omf_input_shapes_are_literal_and_revision_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = prepare.DatasetContract(
                "frozen-v1", "sha256:" + "1" * 64, "sha256:" + "2" * 64
            )
            directory = root / "inputs" / "rawProteome" / contract.name
            directory.mkdir(parents=True)
            value = {
                "resource": f"omf://abiome/slp/datasetsnapshot/{contract.name}@{contract.revision}",
                "mode": "copy",
                "path": str(directory),
                "manifestDigest": contract.manifest_digest,
            }
            resolved = prepare.resolve_pinned_dataset_input(value, "rawProteome", contract)
            self.assertEqual(resolved.path, directory.resolve())
            for mutation in (
                {**value, "mode": "reference"},
                {**value, "manifestDigest": "sha256:" + "3" * 64},
                {**value, "extra": True},
                {
                    **value,
                    "resource": (
                        f"omf://foreign-authority/other-project/datasetsnapshot/"
                        f"{contract.name}@{contract.revision}"
                    ),
                },
            ):
                with self.assertRaises(prepare.ProteomeObservationError):
                    prepare.resolve_pinned_dataset_input(mutation, "rawProteome", contract)
            artifact_path = root / "inputs" / "sgdCurrentOrfs" / "payload" / "payload"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b"x")
            digest = "sha256:" + "4" * 64
            artifact = {
                "resource": "artifact:" + digest,
                "kind": "artifact",
                "artifacts": {"payload": digest},
                "paths": {"payload": str(artifact_path)},
                "path": str(artifact_path),
            }
            self.assertEqual(
                prepare.resolve_literal_artifact(artifact, "sgdCurrentOrfs", digest).path,
                artifact_path.resolve(),
            )
            artifact["path"] = str(artifact_path.parent)
            with self.assertRaises(prepare.ProteomeObservationError):
                prepare.resolve_literal_artifact(artifact, "sgdCurrentOrfs", digest)

    def test_static_module_and_workload_are_pretrain_only_and_file_valued(self) -> None:
        module = yaml.safe_load((MODULE / "module.yaml").read_text())
        inputs = module["spec"]["contracts"]["input"]
        self.assertEqual(
            set(inputs["required"]),
            {
                "rawProteome", "interventionInventory", "proteinRelations",
                "heldRoster", "sgdCurrentOrfs", "sgdMappingManifest",
            },
        )
        self.assertNotIn("molecularReward", json.dumps(module))
        workload_path = ROOT / "workloads" / "slp-1-1-proteome-observation-pretrain-v1.yaml"
        workload = yaml.safe_load(workload_path.read_text())
        stage = workload["spec"]["graph"]["stages"][0]
        self.assertEqual(stage["config"]["role"], "pretrain")
        semantic_workload = json.dumps(workload).casefold()
        self.assertNotIn("benchmark", semantic_workload)
        self.assertNotIn("reward", semantic_workload)
        main_text = (MODULE / "main.py").read_text()
        for path in (
            "proteome-observation-pretrain-v1/observation-corpus.tar",
            "proteome-observation-pretrain-v1/basal-control.tar",
            "proteome-observation-pretrain-v1/preparation-audit.json",
        ):
            self.assertIn(path, main_text)


if __name__ == "__main__":
    unittest.main()
