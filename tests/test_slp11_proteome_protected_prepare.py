"""Adversarial contracts for separately prepared proteome held truth."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import numpy as np
import yaml

from tests.test_slp11_proteome_observation_prepare import Fixture


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-proteome-protected-observation-prepare-v1"
PRETRAIN_MODULE = ROOT / "modules" / "slp-1-1-proteome-observation-prepare-v1"
SPEC = importlib.util.spec_from_file_location(
    "slp11_proteome_protected_prepare", MODULE / "protected_prepare.py"
)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def _rehash_raw(fixture: Fixture) -> None:
    fixture.contract = replace(
        fixture.contract,
        raw_files=tuple(
            prepare.FileSpec(
                path.name,
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(fixture.raw.iterdir())
        ),
    )


def _configure_role_matrix(fixture: Fixture, role: str, *, selected_token: str | None = None) -> None:
    matrix_path = fixture.raw / "yeast5k_noimpute_wide.csv"
    with matrix_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    header = rows[0]
    selected_name = "KO-V" if role == prepare.ROLE_VALIDATION else "KO-F"
    selected_column = header.index(selected_name)
    selected_values = ["2", "4", "NA", "8"]
    if selected_token is not None:
        selected_values[0] = selected_token
    for row, value in zip(rows[1:], selected_values):
        for column in range(1, len(row)):
            row[column] = "opaque-unselected"
        row[selected_column] = value
    with matrix_path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    _rehash_raw(fixture)


def _fixture_role_contract(fixture: Fixture, role: str) -> prepare.ObservationRoleContract:
    systematic = "YAL011W" if role == prepare.ROLE_VALIDATION else "YAL014C"
    genes = [fixture.genes[systematic]]
    filename = "KO-V" if role == prepare.ROLE_VALIDATION else "KO-F"
    with (fixture.raw / "yeast5k_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    selected_index = next(index for index, row in enumerate(rows) if row["Filename"] == filename)
    action_sequence = genes
    locator_sequence = [
        {
            "actionId": genes[0],
            "matrixColumn": selected_index + 1,
            "metadataRow": selected_index + 2,
        }
    ]
    trajectory = "".join(identifier + "\n" for identifier in genes).encode("ascii")
    return prepare.ObservationRoleContract(
        role=role,
        records=1,
        intervention_genes=1,
        target_values=3,
        missing_values=1,
        trajectory_genes_sha256=hashlib.sha256(trajectory).hexdigest(),
        trajectory_gene_set_sha256=prepare.canonical_sha256(genes),
        action_sequence_sha256=prepare.canonical_sha256(action_sequence),
        locator_sequence_sha256=prepare.canonical_sha256(locator_sequence),
    )


def _build(fixture: Fixture, destination: Path, role: str) -> dict[str, object]:
    return prepare.build_protected_observations(
        fixture.raw,
        fixture.interventions,
        fixture.proteins,
        fixture.roster,
        fixture.current,
        fixture.mapping_manifest,
        destination,
        role=role,
        role_contract=_fixture_role_contract(fixture, role),
        source_contract=fixture.contract,
        expected=fixture.expected,
        bounds=fixture.bounds,
    )


def _archive_blobs(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, mode="r:") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
        }


class ProteomeProtectedPreparationTest(unittest.TestCase):
    def test_each_role_decodes_only_its_selected_columns(self) -> None:
        for role, expected_action in (
            (prepare.ROLE_VALIDATION, "SGD:S000000011"),
            (prepare.ROLE_FINAL, "SGD:S000000014"),
        ):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary) / "fixture")
                _configure_role_matrix(fixture, role)
                output = Path(temporary) / "output"
                report = _build(fixture, output, role)
                self.assertEqual(report["role"], role)
                self.assertEqual(report["records"], 1)
                self.assertEqual(report["targetValues"], 3)
                self.assertEqual(report["missingValues"], 1)
                audit = json.loads((output / "preparation-audit.json").read_text())
                self.assertEqual(audit["schema"], prepare.PROTECTED_AUDIT_SCHEMA)
                boundary = audit["accessBoundary"]
                self.assertEqual(boundary["numericColumnsConverted"], {role: 1})
                self.assertFalse(
                    boundary["unselectedNumericTokensConvertedOrSemanticallyValidated"]
                )
                self.assertFalse(boundary["fittingOrRewardInputsPresent"])
                self.assertFalse(boundary["fittingOrRewardOperationsPresent"])
                self.assertFalse(boundary["benchmarkLabelsPresent"])
                self.assertFalse(boundary["controlsDecoded"])
                contract = _fixture_role_contract(fixture, role)
                self.assertEqual(
                    audit["outputs"]["actionSequenceSha256"],
                    contract.action_sequence_sha256,
                )
                self.assertEqual(
                    audit["outputs"]["locatorSequenceSha256"],
                    contract.locator_sequence_sha256,
                )
                blobs = _archive_blobs(output / "observation-corpus.tar")
                manifest = json.loads(blobs["proteome-observations/manifest.json"])
                self.assertEqual(manifest["role"], role)
                self.assertEqual(manifest["archiveId"], prepare._archive_id(role))
                shard_name = next(name for name in blobs if name.endswith(".npz"))
                with np.load(BytesIO(blobs[shard_name]), allow_pickle=False) as shard:
                    self.assertEqual(shard["action_id"].tolist(), [expected_action])
                    self.assertEqual(shard["target_indptr"].tolist(), [0, 3])

    def test_protected_outputs_are_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root / "fixture")
            _configure_role_matrix(fixture, prepare.ROLE_VALIDATION)
            first, second = root / "first", root / "second"
            _build(fixture, first, prepare.ROLE_VALIDATION)
            _build(fixture, second, prepare.ROLE_VALIDATION)
            for name in ("observation-corpus.tar", "preparation-audit.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_invalid_selected_value_fails_without_publishing(self) -> None:
        for token in ("opaque-selected", "0", "NaN"):
            with self.subTest(token=token), tempfile.TemporaryDirectory() as temporary:
                fixture = Fixture(Path(temporary) / "fixture")
                _configure_role_matrix(
                    fixture,
                    prepare.ROLE_FINAL,
                    selected_token=token,
                )
                output = Path(temporary) / "output"
                with self.assertRaisesRegex(
                    prepare.ProteomeObservationError, prepare.ROLE_FINAL
                ):
                    _build(fixture, output, prepare.ROLE_FINAL)
                self.assertFalse(output.exists())

    def test_role_contract_cannot_be_crossed_or_spoofed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary) / "fixture")
            _configure_role_matrix(fixture, prepare.ROLE_VALIDATION)
            validation = _fixture_role_contract(fixture, prepare.ROLE_VALIDATION)
            with self.assertRaisesRegex(prepare.ProteomeObservationError, "one exact held role"):
                prepare.build_protected_observations(
                    fixture.raw,
                    fixture.interventions,
                    fixture.proteins,
                    fixture.roster,
                    fixture.current,
                    fixture.mapping_manifest,
                    Path(temporary) / "output",
                    role=prepare.ROLE_FINAL,
                    role_contract=validation,
                    source_contract=fixture.contract,
                    expected=fixture.expected,
                    bounds=fixture.bounds,
                )

    def test_module_and_workloads_expose_one_protected_role_per_run(self) -> None:
        module = yaml.safe_load((MODULE / "module.yaml").read_text())
        role_schema = module["spec"]["contracts"]["config"]["properties"]["role"]
        self.assertEqual(
            role_schema["enum"],
            [prepare.ROLE_VALIDATION, prepare.ROLE_FINAL],
        )
        self.assertNotIn("pretrain", role_schema["enum"])
        for role in (prepare.ROLE_VALIDATION, prepare.ROLE_FINAL):
            workload_path = ROOT / "workloads" / f"slp-1-1-proteome-observation-{role}-v1.yaml"
            workload = yaml.safe_load(workload_path.read_text())
            stage = workload["spec"]["graph"]["stages"][0]
            self.assertEqual(stage["config"], {"role": role})
            self.assertEqual(
                stage["module"],
                "modules/slp-1-1-proteome-protected-observation-prepare-v1/module.yaml",
            )
            serialized = json.dumps(workload).casefold()
            self.assertNotIn("benchmark", serialized)
            self.assertNotIn("reward", serialized)
            self.assertNotIn("world", serialized)

    def test_proven_pretrain_module_contains_no_protected_entry_point(self) -> None:
        self.assertEqual(
            {path.name for path in PRETRAIN_MODULE.iterdir() if path.is_file()},
            {
                "CONTRACT.md",
                "main.py",
                "module.yaml",
                "observation_prepare.py",
                "README.md",
                "requirements.lock",
            },
        )
        self.assertNotIn(
            "build_protected_observations",
            (PRETRAIN_MODULE / "observation_prepare.py").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
