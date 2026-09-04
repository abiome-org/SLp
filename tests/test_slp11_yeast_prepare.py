"""Rights, identity, and reproducibility checks for yeast corpus preparation."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-yeast-prepare"
WORLD_MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world"
sys.path.insert(0, str(MODULE))
sys.path.insert(1, str(WORLD_MODULE))

from prepare import (  # noqa: E402
    YeastPreparationError,
    prepare_yeast_snapshot,
    write_deterministic_tar,
)
from trainer import CorpusIndex, REQUIRED_ARRAYS, _validate_corpus  # noqa: E402


class YeastPreparationTest(unittest.TestCase):
    def _record(self, record_id: str, action_id: str = "SGD:S000000002") -> dict[str, object]:
        return {
            "schema": "slp.yeast-molecular-record/v1",
            "recordId": record_id,
            "perturbationId": f"synthetic-perturbation:{action_id}",
            "ncbiTaxon": 4932,
            "modality": "quantitative-genetic-interaction",
            "assay": "synthetic-fixture",
            "protocol": "temperature-30C",
            "endpoint": "epsilon-score",
            "normalization": "fixture-centered",
            "experimentalMetadata": {
                "allele": "deletion",
                "arrayType": "DMA",
                "direction": "query-to-array",
                "replicateUncertainty": 0.05,
                "temperatureC": 30,
            },
            "speciesFeatures": [1.0, 0.0],
            "context": [
                {"entityId": "SGD:S000000001", "features": [0.1, 0.2, 0.3]}
            ],
            "actions": [
                {
                    "entityId": action_id,
                    "features": [0.4, 0.5, 0.6],
                    "covariates": [1.0, 0.0, 30.0, 1.0],
                }
            ],
            "queries": [
                {
                    "entityId": "SGD:S000000003",
                    "features": [0.7, 0.8, 0.9],
                    "readoutType": "epsilon",
                    "target": -0.25,
                    "observed": True,
                }
            ],
        }

    def _source(
        self,
        root: Path,
        records: list[dict[str, object]],
        *,
        training_allowed: bool = True,
    ) -> Path:
        source = root / "source"
        source.mkdir()
        rights = source / "rights.yaml"
        rights.write_text(
            "license: CC0-1.0\n"
            f"trainingAllowed: {'true' if training_allowed else 'false'}\n"
            "redistributionAllowed: true\n"
            "source: generated synthetic fixture\n",
            encoding="utf-8",
            newline="\n",
        )
        raw = source / "records.jsonl"
        raw.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
            encoding="utf-8",
            newline="\n",
        )
        manifest = {
            "schema": "slp.yeast-source/v1",
            "sourceId": "synthetic-yeast",
            "sourceRelease": "fixture-2026-09-03",
            "ncbiTaxon": 4932,
            "stableIdNamespace": "SGD",
            "labelClass": "molecular",
            "benchmarkLabelsPresent": False,
            "modalities": ["quantitative-genetic-interaction"],
            "rightsFile": rights.name,
            "rightsSha256": hashlib.sha256(rights.read_bytes()).hexdigest(),
            "rawFormat": "slp.yeast-molecular-jsonl/v1",
            "rawFiles": [
                {
                    "path": raw.name,
                    "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    "records": len(records),
                }
            ],
        }
        (source / "source.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8", newline="\n"
        )
        return source

    @staticmethod
    def _config() -> dict[str, object]:
        return {
            "datasetId": "fixture-yeast-pretrain",
            "version": "fixture-v1",
            "role": "pretrain",
            "entityFeatureDim": 3,
            "speciesFeatureDim": 2,
            "speciesFeatureVector": [1.0, 0.0],
            "actionCovariateDim": 4,
            "readoutTypes": ["epsilon"],
            "maxContextTokens": 2,
            "maxActionTokens": 2,
            "maxQueryTokens": 2,
            "shardRecords": 1,
        }

    def test_output_is_species_native_streamable_and_byte_deterministic(self) -> None:
        records = [
            self._record("record-0001"),
            self._record("record-0002", "SGD:S000000004"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, records)
            first = root / "first"
            second = root / "second"
            first_report = prepare_yeast_snapshot(source, first, self._config())
            second_report = prepare_yeast_snapshot(source, second, self._config())
            first_files = {
                path.relative_to(first).as_posix(): path.read_bytes()
                for path in first.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second).as_posix(): path.read_bytes()
                for path in second.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            self.assertEqual(first_report["contentSha256"], second_report["contentSha256"])
            with np.load(first / "shard-00000.npz", allow_pickle=False) as shard:
                self.assertTrue(REQUIRED_ARRAYS.issubset(shard.files))
                for name in ("record_id", "source_id", "perturbation_id", "action_curies"):
                    self.assertEqual(shard[name].dtype.kind, "U")
                self.assertEqual(shard["species_taxon"].tolist(), [4932])
                self.assertEqual(shard["source_id"].tolist(), ["synthetic-yeast"])
                self.assertEqual(
                    shard["perturbation_id"].tolist(),
                    ["synthetic-perturbation:SGD:S000000002"],
                )
                self.assertEqual(shard["action_curies"][0, 0], "SGD:S000000002")
                self.assertEqual(shard["action_curies"][0, 1], "")
                self.assertEqual(shard["action_mask"].tolist(), [[True, False]])
                self.assertEqual(shard["experimental_metadata_json"].shape, (1,))
                self.assertEqual(shard["target"].shape, (1, 2))
            manifest = json.loads((first / "corpus.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["actionCovariateDim"], 4)
            self.assertEqual(manifest["speciesFeatureVectors"], {"4932": [1.0, 0.0]})
            corpus = CorpusIndex.load(first, "pretrain")
            _validate_corpus(corpus)
            archive_one = root / "one.tar"
            archive_two = root / "two.tar"
            self.assertEqual(
                write_deterministic_tar(first, archive_one),
                write_deterministic_tar(second, archive_two),
            )
            self.assertEqual(archive_one.read_bytes(), archive_two.read_bytes())
            self.assertEqual(
                (first / "trajectory-genes.txt").read_text(encoding="utf-8"),
                "SGD:S000000002\nSGD:S000000004\n",
            )

    def test_record_species_vector_must_match_manifest_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self._record("record-0001")
            record["speciesFeatures"] = [0.0, 1.0]
            source = self._source(root, [record])
            with self.assertRaisesRegex(YeastPreparationError, "speciesFeatureVector"):
                prepare_yeast_snapshot(source, root / "output", self._config())

    def test_training_rights_false_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, [self._record("record-0001")], training_allowed=False)
            with self.assertRaisesRegex(YeastPreparationError, "do not explicitly allow"):
                prepare_yeast_snapshot(source, root / "output", self._config())

    def test_rights_digest_drift_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, [self._record("record-0001")])
            (source / "rights.yaml").write_text(
                "license: CC0-1.0\ntrainingAllowed: false\nsource: changed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(YeastPreparationError, "rights declaration digest"):
                prepare_yeast_snapshot(source, root / "output", self._config())

    def test_missing_rights_declaration_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, [self._record("record-0001")])
            (source / "rights.yaml").unlink()
            with self.assertRaisesRegex(YeastPreparationError, "missing or symlinked rightsFile"):
                prepare_yeast_snapshot(source, root / "output", self._config())

    def test_species_relabeling_and_symbol_identity_are_fatal(self) -> None:
        for mutation, message in (
            (("ncbiTaxon", 9606), "not NCBI taxon 4932"),
            (("actionSymbol", "RAD52"), "stable SGD CURIE"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                record = self._record("record-0001")
                if mutation[0] == "ncbiTaxon":
                    record["ncbiTaxon"] = mutation[1]
                else:
                    record["actions"][0]["entityId"] = mutation[1]
                source = self._source(root, [record])
                with self.assertRaisesRegex(YeastPreparationError, message):
                    prepare_yeast_snapshot(source, root / "output", self._config())

    def test_unsorted_records_and_raw_digest_drift_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(
                root, [self._record("record-0002"), self._record("record-0001")]
            )
            with self.assertRaisesRegex(YeastPreparationError, "strictly sorted"):
                prepare_yeast_snapshot(source, root / "output", self._config())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, [self._record("record-0001")])
            with (source / "records.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(YeastPreparationError, "raw file digest mismatch"):
                prepare_yeast_snapshot(source, root / "output", self._config())

    def test_duplicate_stable_record_identity_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(
                root, [self._record("record-0001"), self._record("record-0001")]
            )
            with self.assertRaisesRegex(YeastPreparationError, "unique and strictly sorted"):
                prepare_yeast_snapshot(source, root / "output", self._config())


if __name__ == "__main__":
    unittest.main()
