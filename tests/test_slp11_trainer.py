"""End-to-end CPU smoke check for the SLp-1.1 training module."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world"
sys.path.insert(0, str(MODULE))

from trainer import train_world  # noqa: E402


class WorldTrainerTest(unittest.TestCase):
    def _corpus(self, root: Path, name: str, role: str, offset: float) -> Path:
        directory = root / name
        directory.mkdir()
        rng = np.random.default_rng(31 + int(offset * 10))
        records = 8
        action = rng.normal(size=(records, 2, 5)).astype("float32")
        target = (action[:, :1, 0] + offset).astype("float32")
        shard = directory / "shard-000.npz"
        np.savez(
            shard,
            context_features=rng.normal(size=(records, 2, 5)).astype("float32"),
            context_mask=np.ones((records, 2), dtype=bool),
            action_features=action,
            action_covariates=np.zeros((records, 2, 4), dtype="float32"),
            action_mask=np.ones((records, 2), dtype=bool),
            query_features=rng.normal(size=(records, 1, 5)).astype("float32"),
            query_mask=np.ones((records, 1), dtype=bool),
            readout_type=np.zeros((records, 1), dtype="int64"),
            species_features=np.tile(np.array([[0.0, 1.0]], dtype="float32"), (records, 1)),
            species_taxon=np.where(np.arange(records) % 2, 4932, 9606).astype("int64"),
            target=target,
            target_mask=np.ones((records, 1), dtype=bool),
        )
        (directory / "trajectory-genes.txt").write_text(
            f"ENSEMBL:ENSG{name}\n", encoding="utf-8"
        )
        manifest = {
            "schema": "slp.corpus/v1",
            "datasetId": name,
            "version": "test-v1",
            "role": role,
            "labelClass": "molecular",
            "benchmarkLabelsPresent": False,
            "speciesTaxa": [4932, 9606],
            "modalities": ["synthetic-test"],
            "trajectoryGenes": "trajectory-genes.txt",
            "entityFeatureDim": 5,
            "speciesFeatureDim": 2,
            "readoutTypes": ["effect"],
            "shards": [
                {
                    "path": shard.name,
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                    "records": records,
                }
            ],
        }
        (directory / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    def test_one_epoch_produces_finite_molecular_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, report, baselines = train_world(
                {
                    "pretrain": self._corpus(root, "pretrain", "pretrain", 0.0),
                    "molecularValidation": self._corpus(
                        root, "validation", "molecular-validation", 0.1
                    ),
                    "molecularReward": self._corpus(
                        root, "reward", "molecular-reward", 0.2
                    ),
                },
                {
                    "seed": 41,
                    "epochs": 1,
                    "reinforcementEpochs": 0,
                    "batchSize": 4,
                    "dModel": 8,
                    "nhead": 2,
                    "encoderLayers": 1,
                    "decoderLayers": 1,
                    "dropout": 0.0,
                },
            )
        self.assertGreater(model.count_parameters(), 0)
        self.assertEqual(report["schema"], "slp.training-report/v1.1")
        self.assertTrue(np.isfinite(report["selected"]["nll"]))
        self.assertEqual(baselines.mean.shape, (1,))
        self.assertGreater(baselines.linear_weight.numel(), 1)


if __name__ == "__main__":
    unittest.main()
