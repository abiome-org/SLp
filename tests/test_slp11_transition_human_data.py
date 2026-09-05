from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from human_data import (
    CONTEXT_IDS,
    HumanDataError,
    SourceSpec,
    _split_name,
    build_human_development,
)


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _gene_for(group: str) -> str:
    for number in range(1, 100_000):
        gene = f"ENSG{number:011d}"
        if _split_name(gene) == group:
            return gene
    raise AssertionError(f"no synthetic gene found for {group}")


def _write_source(
    path: Path,
    action_ids: list[str],
    genes: list[str],
    values: np.ndarray,
    *,
    unresolved: bool = False,
) -> SourceSpec:
    record_ids = [f"{index}_GENE_P1_{gene}" for index, gene in enumerate(action_ids)]
    record_ids.append("100_non-targeting_non-targeting_non-targeting")
    if unresolved:
        record_ids.append("101_UNRESOLVED_P1_nan")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.asarray(values, dtype=np.float32))
        obs = handle.create_group("obs")
        obs.create_dataset(
            "gene_transcript",
            data=np.asarray(record_ids, dtype=h5py.string_dtype("utf-8")),
        )
        var = handle.create_group("var")
        var.create_dataset(
            "gene_id", data=np.asarray(genes, dtype=h5py.string_dtype("utf-8"))
        )
    return SourceSpec(
        CONTEXT_IDS[0],
        path.name,
        path.stat().st_size,
        _hash(path),
        _hash(path, "md5"),
    )


class SyntheticHumanAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train_gene = _gene_for("train")
        self.validation_gene = _gene_for("validation")
        self.test_gene = _gene_for("test")
        self.actions = [self.train_gene, self.validation_gene, self.test_gene]
        self.shared = ["ENSG00001000001", "ENSG00001000002", "ENSG00001000003"]

        k562 = self.root / "k562.h5ad"
        rpe1 = self.root / "rpe1.h5ad"
        k_values = np.asarray(
            [
                [1.5, 2.5, 6.0, 1000.0],
                [2.0, 3.0, 5.0, 1000.0],
                [4.0, 1.0, 5.0, 1000.0],
                [2.0, 2.0, 6.0, 1000.0],
            ]
        )
        r_values = np.asarray(
            [
                [1.5, 2.5, 6.0, 2000.0],
                [2.0, 3.0, 5.0, 2000.0],
                [4.0, 1.0, 5.0, 2000.0],
                [3.0, 2.0, 5.0, 2000.0],
                [1.0, 1.0, 1.0, 2000.0],
            ]
        )
        spec0 = _write_source(
            k562, self.actions, [*self.shared, "ENSG00001000004"], k_values
        )
        spec1_raw = _write_source(
            rpe1,
            self.actions,
            [*self.shared, "ENSG00001000005"],
            r_values,
            unresolved=True,
        )
        spec1 = SourceSpec(
            CONTEXT_IDS[1],
            spec1_raw.filename,
            spec1_raw.bytes,
            spec1_raw.sha256,
            spec1_raw.upstream_md5,
        )
        self.result = build_human_development(
            k562,
            rpe1,
            self.root / "output",
            source_specs=(spec0, spec1),
            expected_query_count=3,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_shared_panel_transform_uses_intersection_denominator(self) -> None:
        output = self.root / "output"
        with np.load(output / "replogle-k562-rpe1-development-v1.npz", allow_pickle=False) as data:
            row = np.flatnonzero(data["action_ids"] == self.train_gene)[0]
            expected = np.log2(1.0 + 10_000.0 * np.asarray([1.5, 2.5, 6.0]) / 10.0)
            np.testing.assert_allclose(data["targets"][row], expected, rtol=1e-6)
            self.assertTrue(data["observed"].all())
            self.assertEqual(data["query_ids"].tolist(), sorted(self.shared))

    def test_development_excludes_test_and_groups_gene_across_contexts(self) -> None:
        output = self.root / "output"
        with np.load(output / "replogle-k562-rpe1-development-v1.npz", allow_pickle=False) as data:
            self.assertEqual(len(data["split_test"]), 0)
            self.assertEqual(set(data["action_ids"]), {self.train_gene, self.validation_gene})
            for gene in (self.train_gene, self.validation_gene):
                rows = np.flatnonzero(data["action_ids"] == gene)
                self.assertEqual(set(data["context_index"][rows]), {0, 1})
            self.assertEqual(data["basal_control"].shape, (2, 3))

    def test_test_artifact_is_separate_and_manifest_checksum_pinned(self) -> None:
        manifest = self.result["manifest"]
        output = self.root / "output"
        test_ref = manifest["outputs"]["testOnly"]
        self.assertEqual(test_ref["contains"], ["test"])
        self.assertEqual(
            _hash(output / test_ref["path"]),
            test_ref["sha256"],
        )
        # Deliberately do not load the test-only NPZ.

    def test_unresolved_action_is_quarantined_and_controls_are_not_targets(self) -> None:
        manifest = self.result["manifest"]
        self.assertEqual(
            manifest["sources"][1]["unresolvedActionsQuarantined"],
            ["101_UNRESOLVED_P1_nan"],
        )
        self.assertEqual(manifest["counts"]["controlsUsedForBasal"], [1, 1])
        self.assertEqual(manifest["counts"]["records"], 6)

    def test_rebuild_is_byte_deterministic(self) -> None:
        first = self.result["manifest"]
        second = build_human_development(
            self.root / "k562.h5ad",
            self.root / "rpe1.h5ad",
            self.root / "second",
            source_specs=(
                SourceSpec(
                    CONTEXT_IDS[0],
                    "k562.h5ad",
                    (self.root / "k562.h5ad").stat().st_size,
                    _hash(self.root / "k562.h5ad"),
                    _hash(self.root / "k562.h5ad", "md5"),
                ),
                SourceSpec(
                    CONTEXT_IDS[1],
                    "rpe1.h5ad",
                    (self.root / "rpe1.h5ad").stat().st_size,
                    _hash(self.root / "rpe1.h5ad"),
                    _hash(self.root / "rpe1.h5ad", "md5"),
                ),
            ),
            expected_query_count=3,
        )["manifest"]
        self.assertEqual(
            first["outputs"]["development"]["sha256"],
            second["outputs"]["development"]["sha256"],
        )
        self.assertEqual(
            first["outputs"]["testOnly"]["sha256"],
            second["outputs"]["testOnly"]["sha256"],
        )

    def test_negative_source_value_is_rejected(self) -> None:
        path = self.root / "k562.h5ad"
        with h5py.File(path, "r+") as handle:
            handle["X"][0, 0] = -1.0
        spec = SourceSpec(
            CONTEXT_IDS[0],
            path.name,
            path.stat().st_size,
            _hash(path),
            _hash(path, "md5"),
        )
        with self.assertRaisesRegex(HumanDataError, "negative or non-finite"):
            build_human_development(
                path,
                self.root / "rpe1.h5ad",
                self.root / "bad-output",
                source_specs=(
                    spec,
                    SourceSpec(
                        CONTEXT_IDS[1],
                        "rpe1.h5ad",
                        (self.root / "rpe1.h5ad").stat().st_size,
                        _hash(self.root / "rpe1.h5ad"),
                        _hash(self.root / "rpe1.h5ad", "md5"),
                    ),
                ),
                expected_query_count=3,
            )


class ProductionHumanArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = ROOT / "data" / "derived" / "slp11-human"
        self.manifest_path = self.output / "manifest.json"
        if not self.manifest_path.is_file():
            raise unittest.SkipTest("derived human development artifacts are unavailable")
        self.manifest = json.loads(self.manifest_path.read_text())

    def test_production_counts_and_identity_hashes(self) -> None:
        counts = self.manifest["counts"]
        self.assertEqual(counts["queries"], 7_226)
        self.assertEqual(counts["actions"], 2_392)
        self.assertEqual(counts["records"], 4_720)
        self.assertEqual(counts["splitRecords"], {"train": 3281, "validation": 726, "test": 713})
        self.assertEqual(
            self.manifest["identityLists"]["entity-ids.txt"]["sha256"],
            "c6836645dcfc24788f2c06110ddc08ee4949d97f710dd117db12db1949d9b33e",
        )

    def test_production_development_contract_without_opening_test(self) -> None:
        reference = self.manifest["outputs"]["development"]
        path = self.output / reference["path"]
        self.assertEqual(_hash(path), reference["sha256"])
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(data["targets"].shape, (4007, 7226))
            self.assertEqual(data["targets"].dtype, np.float32)
            self.assertEqual(data["observed"].dtype, np.bool_)
            self.assertTrue(data["observed"].all())
            self.assertEqual(len(data["split_train"]), 3281)
            self.assertEqual(len(data["split_validation"]), 726)
            self.assertEqual(len(data["split_test"]), 0)
        test_ref = self.manifest["outputs"]["testOnly"]
        self.assertTrue((self.output / test_ref["path"]).is_file())
        # Test-only contents remain unopened.


if __name__ == "__main__":
    unittest.main()
