"""End-to-end CPU smoke check for the SLp-1.1 training module."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world"
AUDIT_MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-corpus-audit"
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(AUDIT_MODULE))

from trainer import train_world  # noqa: E402
from audit import audit_corpora  # noqa: E402


class WorldTrainerTest(unittest.TestCase):
    def _rewrite_shard(self, corpus: Path, mutate) -> None:
        shard = corpus / "shard-000.npz"
        with np.load(shard, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        mutate(arrays)
        np.savez(shard, **arrays)
        manifest_path = corpus / "corpus.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def _corpus(self, root: Path, name: str, role: str, offset: float) -> Path:
        directory = root / name
        directory.mkdir()
        rng = np.random.default_rng(31 + int(offset * 10))
        records = 8
        action = rng.normal(size=(records, 2, 5)).astype("float32")
        target = (action[:, :1, 0] + offset).astype("float32")
        taxon = np.where(np.arange(records) % 2, 4932, 9606).astype("int64")
        species_features = np.asarray(
            [[1.0, 0.0] if item == 4932 else [0.0, 1.0] for item in taxon],
            dtype="float32",
        )
        human_gene = f"ENSEMBL:ENSG{name}"
        yeast_gene = f"SGD:S{name}"
        action_mask = np.zeros((records, 2), dtype=bool)
        action_mask[:, 0] = True
        action_curies = np.full((records, 2), "", dtype="<U96")
        action_curies[:, 0] = np.where(taxon == 4932, yeast_gene, human_gene)
        shard = directory / "shard-000.npz"
        np.savez(
            shard,
            record_id=np.asarray([f"TEST:{name}:{i}" for i in range(records)], dtype="<U96"),
            source_id=np.full(records, "TEST:synthetic-source", dtype="<U96"),
            perturbation_id=np.asarray(
                [f"TEST:{name}:perturbation:{i}" for i in range(records)], dtype="<U96"
            ),
            context_features=rng.normal(size=(records, 2, 5)).astype("float32"),
            context_mask=np.ones((records, 2), dtype=bool),
            action_features=action,
            action_covariates=np.zeros((records, 2, 4), dtype="float32"),
            action_curies=action_curies,
            action_mask=action_mask,
            query_features=rng.normal(size=(records, 1, 5)).astype("float32"),
            query_mask=np.ones((records, 1), dtype=bool),
            readout_type=np.zeros((records, 1), dtype="int64"),
            species_features=species_features,
            species_taxon=taxon,
            target=target,
            target_mask=np.ones((records, 1), dtype=bool),
        )
        (directory / "trajectory-genes.txt").write_text(
            human_gene + "\n" + yeast_gene + "\n", encoding="utf-8"
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
            "speciesFeatureVectors": {"4932": [1.0, 0.0], "9606": [0.0, 1.0]},
            "actionCovariateDim": 4,
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
            roots = {
                "pretrain": self._corpus(root, "pretrain", "pretrain", 0.0),
                "molecularValidation": self._corpus(
                    root, "validation", "molecular-validation", 0.1
                ),
                "molecularReward": self._corpus(
                    root, "reward", "molecular-reward", 0.2
                ),
            }
            model, report, baselines = train_world(
                roots,
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
                audit_corpora(roots),
            )
        self.assertGreater(model.count_parameters(), 0)
        self.assertEqual(report["schema"], "slp.training-report/v1.1")
        self.assertTrue(np.isfinite(report["selected"]["nll"]))
        self.assertEqual(baselines.mean.shape, (1,))
        self.assertGreater(baselines.linear_weight.numel(), 1)

    def test_audit_is_bound_to_exact_corpus_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {
                "pretrain": self._corpus(root, "pretrain-a", "pretrain", 0.0),
                "molecularValidation": self._corpus(
                    root, "validation-a", "molecular-validation", 0.1
                ),
                "molecularReward": self._corpus(
                    root, "reward-a", "molecular-reward", 0.2
                ),
            }
            audit = audit_corpora(roots)
            roots["molecularValidation"] = self._corpus(
                root, "validation-b", "molecular-validation", 0.3
            )
            with self.assertRaisesRegex(ValueError, "audit identity mismatch"):
                train_world(roots, {"epochs": 1, "reinforcementEpochs": 0}, audit)

    def test_nonfinite_observed_features_are_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {
                "pretrain": self._corpus(root, "pretrain", "pretrain", 0.0),
                "molecularValidation": self._corpus(
                    root, "validation", "molecular-validation", 0.1
                ),
                "molecularReward": self._corpus(
                    root, "reward", "molecular-reward", 0.2
                ),
            }
            corpus = roots["pretrain"]
            self._rewrite_shard(
                corpus,
                lambda arrays: arrays["context_features"].__setitem__((0, 0, 0), np.nan),
            )
            audit = audit_corpora(roots)
            with self.assertRaisesRegex(ValueError, "finite floating-point"):
                train_world(roots, {"epochs": 1, "reinforcementEpochs": 0}, audit)

    def test_reinforcement_is_fail_closed_until_control_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {
                "pretrain": self._corpus(root, "pretrain", "pretrain", 0.0),
                "molecularValidation": self._corpus(
                    root, "validation", "molecular-validation", 0.1
                ),
                "molecularReward": self._corpus(
                    root, "reward", "molecular-reward", 0.2
                ),
            }
            with self.assertRaisesRegex(ValueError, "reinforcement is disabled"):
                train_world(
                    roots,
                    {"epochs": 1, "reinforcementEpochs": 1},
                    audit_corpora(roots),
                )

    def test_shard_action_inventory_must_match_audited_genes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {
                "pretrain": self._corpus(root, "pretrain", "pretrain", 0.0),
                "molecularValidation": self._corpus(
                    root, "validation", "molecular-validation", 0.1
                ),
                "molecularReward": self._corpus(
                    root, "reward", "molecular-reward", 0.2
                ),
            }
            self._rewrite_shard(
                roots["pretrain"],
                lambda arrays: arrays["action_curies"].__setitem__(
                    (0, 0), "ENSEMBL:UNDECLARED"
                ),
            )
            with self.assertRaisesRegex(ValueError, "trajectoryGenes does not exactly match"):
                train_world(
                    roots,
                    {"epochs": 1, "reinforcementEpochs": 0},
                    audit_corpora(roots),
                )

    def test_species_vector_must_match_declared_taxon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {
                "pretrain": self._corpus(root, "pretrain", "pretrain", 0.0),
                "molecularValidation": self._corpus(
                    root, "validation", "molecular-validation", 0.1
                ),
                "molecularReward": self._corpus(
                    root, "reward", "molecular-reward", 0.2
                ),
            }
            self._rewrite_shard(
                roots["pretrain"],
                lambda arrays: arrays["species_features"].__setitem__((0, 0), 1.0),
            )
            with self.assertRaisesRegex(ValueError, "species_features do not match"):
                train_world(
                    roots,
                    {"epochs": 1, "reinforcementEpochs": 0},
                    audit_corpora(roots),
                )

    def test_each_species_requires_observed_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {
                "pretrain": self._corpus(root, "pretrain", "pretrain", 0.0),
                "molecularValidation": self._corpus(
                    root, "validation", "molecular-validation", 0.1
                ),
                "molecularReward": self._corpus(
                    root, "reward", "molecular-reward", 0.2
                ),
            }
            def remove_yeast_targets(arrays) -> None:
                arrays["target_mask"][arrays["species_taxon"] == 4932] = False

            self._rewrite_shard(roots["pretrain"], remove_yeast_targets)
            with self.assertRaisesRegex(ValueError, "every represented species"):
                train_world(
                    roots,
                    {"epochs": 1, "reinforcementEpochs": 0},
                    audit_corpora(roots),
                )


if __name__ == "__main__":
    unittest.main()
