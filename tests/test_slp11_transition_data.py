from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from transition_data import (
    PRODUCTION_ARCHIVE_SHA256,
    CorpusLoadError,
    _load_tar,
    _read_npz,
    _ref_payload,
    load_corpus,
    split_by_gene,
)


class CorpusSafetyTests(unittest.TestCase):
    def test_rejects_pickle_backed_npz(self) -> None:
        output = io.BytesIO()
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
            array = io.BytesIO()
            np.save(array, np.asarray([{"hidden": "object"}], dtype=object), allow_pickle=True)
            archive.writestr("unsafe.npy", array.getvalue())
        with self.assertRaisesRegex(CorpusLoadError, "pickle-free"):
            _read_npz(output.getvalue(), {"unsafe"}, "fixture")

    def test_rejects_manifest_digest_drift(self) -> None:
        payload = b"bounded"
        reference = {
            "path": "value.bin",
            "bytes": len(payload),
            "sha256": "0" * 64,
        }
        with self.assertRaisesRegex(CorpusLoadError, "SHA-256"):
            _ref_payload({"composite-corpus/value.bin": payload}, reference, "fixture")

    def test_rejects_traversal_tar_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tar"
            with tarfile.open(path, mode="w") as archive:
                payload = b"bad"
                info = tarfile.TarInfo("composite-corpus/../escape")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(CorpusLoadError, "canonical relative path"):
                _load_tar(path)


class GeneSplitTests(unittest.TestCase):
    def test_exact_hash_rule_and_duplicate_gene_grouping(self) -> None:
        keys = [
            (4932, "SGD:S000000001"),
            (4932, "SGD:S000000002"),
            (4932, "SGD:S000000001"),
            (9606, "SGD:S000000001"),
        ]
        actual = split_by_gene(keys)
        membership = {
            int(index): group for group, indices in actual.items() for index in indices
        }
        self.assertEqual(membership[0], membership[2])
        self.assertEqual(set(membership), set(range(len(keys))))
        for index, (taxon, identifier) in enumerate(keys):
            payload = f"slp11-development-v1|731|{taxon}|{identifier}".encode()
            bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
            expected = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
            self.assertEqual(membership[index], expected)
        self.assertTrue(all(indices.dtype == np.int64 for indices in actual.values()))

    def test_taxon_is_part_of_split_identity(self) -> None:
        keys = [(4932, "same-id"), (9606, "same-id")]
        # Check the exact digest inputs, without assuming two buckets cannot collide.
        yeast = hashlib.sha256(b"slp11-development-v1|731|4932|same-id").digest()
        human = hashlib.sha256(b"slp11-development-v1|731|9606|same-id").digest()
        self.assertNotEqual(yeast, human)
        self.assertEqual(sum(len(value) for value in split_by_gene(keys).values()), 2)


class ProductionCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.archive = (
            ROOT
            / ".omf"
            / "runs"
            / "01a06e28-2f3f-7ccb-b71d-8b7654fc26ca"
            / "stages"
            / "compose"
            / "proteome-corpus-compose-v1"
            / "corpus-v1-2.tar"
        )
        if not cls.archive.is_file():
            raise unittest.SkipTest("exact local fitting-only corpus is unavailable")
        cls.corpus = load_corpus(cls.archive)

    def test_exact_archive_loads_with_expected_counts_and_dtypes(self) -> None:
        corpus = self.corpus
        self.assertEqual(corpus["entity_features"].shape, (7038, 21))
        self.assertEqual(corpus["entity_present"].shape, (7038, 21))
        self.assertEqual(corpus["targets"].shape, (3811, 1850))
        self.assertEqual(corpus["observed"].shape, (3811, 1850))
        self.assertEqual(corpus["entity_features"].dtype, np.float32)
        self.assertEqual(corpus["entity_present"].dtype, np.bool_)
        self.assertEqual(corpus["action_index"].dtype, np.int64)
        self.assertEqual(corpus["query_entity_index"].dtype, np.int64)
        self.assertEqual(corpus["targets"].dtype, np.float32)
        self.assertEqual(corpus["observed"].dtype, np.bool_)
        self.assertEqual(int(corpus["observed"].sum()), 6_865_493)
        self.assertEqual(int((~corpus["observed"]).sum()), 184_857)
        self.assertTrue(np.all(corpus["targets"][~corpus["observed"]] == 0))

    def test_composite_identity_and_unique_record_contract(self) -> None:
        corpus = self.corpus
        self.assertEqual(len(corpus["entity_keys"]), len(set(corpus["entity_keys"])))
        self.assertEqual(len(corpus["record_ids"]), len(set(corpus["record_ids"])))
        self.assertEqual(len(corpus["action_keys"]), 3811)
        for index, key in zip(corpus["action_index"], corpus["action_keys"]):
            self.assertEqual(corpus["entity_keys"][int(index)], key)

    def test_audit_covariates_are_declared_but_not_returned(self) -> None:
        corpus = self.corpus
        self.assertNotIn("observation_covariates", corpus)
        self.assertNotIn("record_covariates", corpus)
        masked = corpus["metadata"]["masked_audit_covariates"]
        self.assertEqual(len(masked), 5)
        self.assertTrue(all(identifier.startswith("slp-covariate:") for identifier in masked))

    def test_source_hashes_pin_the_fitting_archive(self) -> None:
        metadata = self.corpus["metadata"]
        self.assertTrue(metadata["production_archive_match"])
        self.assertEqual(metadata["source_hashes"]["archive"], PRODUCTION_ARCHIVE_SHA256)
        self.assertEqual(self.corpus["source_id"], "mendeley:w8jtmnszd9.2")


if __name__ == "__main__":
    unittest.main()
