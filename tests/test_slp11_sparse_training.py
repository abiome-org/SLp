"""Deterministic maximum-likelihood checks for the sparse world candidate."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world-sparse"
sys.path.insert(0, str(MODULE))

from slp_sparse_architecture import WorldPrediction  # noqa: E402
from slp_sparse_corpus import CorpusIndex  # noqa: E402
from slp_sparse_training import (  # noqa: E402
    TrainingConfig,
    equal_record_negative_log_likelihood,
    iter_sparse_predictions,
    train_sparse_world,
)
from tests.test_slp11_sparse_candidate import _sha256, _write_corpus  # noqa: E402


def _make_validation(root: Path, *, extra_entities: int = 0) -> CorpusIndex:
    _write_corpus(root, extra_entities=extra_entities)
    entity_path = root / "entity-table.npz"
    with np.load(entity_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    action = np.flatnonzero(arrays["entity_id"] == "SGD:S0002")
    if action.shape != (1,):
        raise AssertionError("fixture action entity is ambiguous")
    arrays["entity_id"][action[0]] = "SGD:S1002"
    np.savez(entity_path, **arrays)
    gene_path = root / "trajectory-genes.txt"
    gene_path.write_text("SGD:S1002\n", encoding="utf-8")
    manifest_path = root / "corpus.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["datasetId"] = "TEST:sparse-validation"
    manifest["role"] = "molecular-validation"
    manifest["entityDictionary"]["sha256"] = _sha256(entity_path)
    manifest["trajectoryGenes"]["sha256"] = _sha256(gene_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return CorpusIndex.load(root)


class SparseTrainingTest(unittest.TestCase):
    def _config(self) -> TrainingConfig:
        return TrainingConfig(
            seed=83,
            epochs=18,
            draws_per_epoch=8,
            batch_size=4,
            learning_rate=0.01,
            evaluation_batch_size=2,
            d_model=8,
            nhead=2,
            encoder_layers=1,
            decoder_layers=1,
            ffn_multiplier=2,
            dropout=0.0,
        )

    def test_fixed_training_reduces_held_nll_and_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain = CorpusIndex.load(_write_corpus(root / "pretrain"))
            validation = _make_validation(root / "validation")
            initial_rng_state = torch.random.get_rng_state().clone()
            first = train_sparse_world(pretrain, validation, self._config())
            self.assertTrue(torch.equal(initial_rng_state, torch.random.get_rng_state()))
            second = train_sparse_world(pretrain, validation, self._config())
            self.assertTrue(torch.equal(initial_rng_state, torch.random.get_rng_state()))
            first_predictions = list(
                iter_sparse_predictions(first.model, validation, batch_size=1)
            )
            second_predictions = list(
                iter_sparse_predictions(second.model, validation, batch_size=1)
            )
            validation_batch = validation.materialize_batch(
                validation.load_shard(0), [0, 1]
            ).world

        first_report = first.report
        second_report = second.report
        overall = first_report["validation"]["overall"]
        self.assertLess(
            overall["finalPerObservedTargetNll"],
            overall["initializationPerObservedTargetNll"],
        )
        self.assertGreater(overall["descriptiveImprovement"], 0.0)
        self.assertFalse(first_report["validation"]["scientificBaselineComparison"])
        self.assertEqual(
            first_report["validation"]["decisionUse"],
            "frozen-molecular-gate-only",
        )
        self.assertEqual(
            first_report["modelParameterSha256"],
            second_report["modelParameterSha256"],
        )
        self.assertEqual(
            first_report["validationPredictionSha256"],
            second_report["validationPredictionSha256"],
        )
        self.assertEqual(first_report["reportSha256"], second_report["reportSha256"])
        for name, tensor in first.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.model.state_dict()[name]), name)
        self.assertEqual(
            [batch.record_id for batch in first_predictions],
            [batch.record_id for batch in second_predictions],
        )
        self.assertEqual(
            [batch.query_id for batch in first_predictions],
            [batch.query_id for batch in second_predictions],
        )
        for first_batch, second_batch in zip(first_predictions, second_predictions):
            self.assertTrue(
                torch.equal(first_batch.parameters, second_batch.parameters)
            )
        self.assertEqual(
            set(first_report["validation"]["bySource"]),
            {"TESTSOURCE:a", "TESTSOURCE:b"},
        )
        self.assertEqual(set(first_report["validation"]["bySpecies"]), {"4932"})
        self.assertEqual(
            set(first_report["validation"]["bySpeciesSource"]),
            {"4932|TESTSOURCE:a", "4932|TESTSOURCE:b"},
        )
        self.assertEqual(first_report["isolation"]["trajectoryGeneOverlap"], [])
        self.assertFalse(first_report["isolation"]["validationUsedForOptimization"])
        self.assertFalse(first_report["checkpointProduced"])
        source_improvements = [
            value["descriptiveImprovement"]
            for value in first_report["validation"]["bySource"].values()
        ]
        self.assertEqual(
            first_report["validation"]["minimumSourceDescriptiveImprovement"],
            min(source_improvements),
        )
        self.assertGreater(
            first_report["validation"]["minimumSourceDescriptiveImprovement"],
            0.0,
        )
        self.assertGreater(
            first_report["validation"]["minimumSpeciesDescriptiveImprovement"],
            0.0,
        )
        self.assertFalse(
            first_report["streaming"]["denseRecordByDictionaryTargetAllocated"]
        )
        self.assertLessEqual(
            first_report["streaming"]["maxShardRecordsLoaded"],
            pretrain.bounds["maxRecordsPerShard"],
        )
        self.assertLessEqual(
            first_report["streaming"]["maxShardTargetValuesLoaded"],
            pretrain.bounds["maxRecordsPerShard"]
            * pretrain.bounds["maxTargetsPerRecord"],
        )
        self.assertEqual(
            first_report["training"]["sourceDraws"],
            {"TESTSOURCE:a": 72, "TESTSOURCE:b": 72},
        )
        self.assertEqual(
            first_report["training"]["objective"],
            {
                "name": "mean-per-record-observed-typed-nll",
                "scheduledRecordWeighting": "equal",
                "withinRecordTargetWeighting": "equal-observed-target",
                "scheduleHierarchy": "source-perturbation-replicate-record",
            },
        )
        self.assertEqual(
            first_report["training"]["sourceDrawsByEpoch"],
            [{"TESTSOURCE:a": 4, "TESTSOURCE:b": 4}] * self._config().epochs,
        )
        schedule_hashes = first_report["training"]["epochScheduleSha256"]
        self.assertEqual(len(schedule_hashes), self._config().epochs)
        self.assertGreater(len(set(schedule_hashes)), 1)
        self.assertEqual(
            schedule_hashes,
            second_report["training"]["epochScheduleSha256"],
        )

        first.model.eval()
        expected = first.model(validation_batch)
        chunks = []
        for index in (0, 1):
            selection = torch.tensor([index])
            chunk = replace(
                validation_batch,
                query_features=validation_batch.query_features[:, selection],
                query_feature_present=validation_batch.query_feature_present[:, selection],
                query_entity_type=validation_batch.query_entity_type[:, selection],
                readout_type=validation_batch.readout_type[:, selection],
                likelihood_type=validation_batch.likelihood_type[:, selection],
                query_mask=validation_batch.query_mask[:, selection],
            )
            chunks.append(first.model(chunk).parameters)
        self.assertTrue(torch.equal(torch.cat(chunks, dim=1), expected.parameters))

    def test_optimization_weights_records_equally_despite_target_density(self) -> None:
        parameters = torch.zeros((2, 3, 2), dtype=torch.float32, requires_grad=True)
        prediction = WorldPrediction(
            parameters=parameters,
            likelihood_type=torch.zeros((2, 3), dtype=torch.long),
            query_mask=torch.ones((2, 3), dtype=torch.bool),
        )
        target = torch.tensor(
            [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        )
        observed = torch.tensor(
            [[True, False, False], [True, True, True]], dtype=torch.bool
        )

        objective = equal_record_negative_log_likelihood(
            prediction, target, observed
        )
        gaussian_constant = 0.5 * np.log(2.0 * np.pi)
        expected_equal_record = 0.5 * (
            gaussian_constant + 8.0 + gaussian_constant
        )
        target_weighted = gaussian_constant + 2.0
        self.assertAlmostEqual(objective.item(), expected_equal_record, places=6)
        self.assertNotAlmostEqual(objective.item(), target_weighted, places=6)

        objective.backward()
        self.assertAlmostEqual(parameters.grad[0, 0, 1].item(), -7.5, places=6)
        for query_index in range(3):
            self.assertAlmostEqual(
                parameters.grad[1, query_index, 1].item(), 1.0 / 6.0, places=6
            )

    def test_dictionary_growth_does_not_change_training_or_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact_pretrain = CorpusIndex.load(_write_corpus(root / "compact-pretrain"))
            compact_validation = _make_validation(root / "compact-validation")
            large_pretrain = CorpusIndex.load(
                _write_corpus(root / "large-pretrain", extra_entities=19)
            )
            large_validation = _make_validation(
                root / "large-validation", extra_entities=23
            )
            compact = train_sparse_world(
                compact_pretrain, compact_validation, self._config()
            )
            large = train_sparse_world(large_pretrain, large_validation, self._config())
        self.assertNotEqual(len(compact_pretrain.entity_id), len(large_pretrain.entity_id))
        self.assertEqual(
            compact.report["parameterCount"], large.report["parameterCount"]
        )
        self.assertEqual(
            compact.report["modelParameterSha256"],
            large.report["modelParameterSha256"],
        )
        self.assertEqual(
            compact.report["validationPredictionSha256"],
            large.report["validationPredictionSha256"],
        )

    def test_role_and_intervention_isolation_fail_before_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain = CorpusIndex.load(_write_corpus(root / "pretrain"))
            wrong_role = CorpusIndex.load(_write_corpus(root / "wrong-role"))
            with self.assertRaisesRegex(ValueError, "molecular-validation role"):
                train_sparse_world(pretrain, wrong_role, self._config())
            model = train_sparse_world(
                pretrain, _make_validation(root / "prediction-validation"), self._config()
            ).model
            with self.assertRaisesRegex(ValueError, "molecular-validation role"):
                list(iter_sparse_predictions(model, wrong_role))

            overlap_root = root / "overlap"
            _write_corpus(overlap_root)
            manifest_path = overlap_root / "corpus.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["role"] = "molecular-validation"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            overlapping_validation = CorpusIndex.load(overlap_root)
            with self.assertRaisesRegex(
                ValueError, "validation intervention genes occur in pretrain"
            ):
                train_sparse_world(pretrain, overlapping_validation, self._config())

    def test_benchmark_like_manifest_field_is_rejected_not_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _write_corpus(Path(temporary) / "corpus")
            manifest_path = root / "corpus.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["syntheticLethalityLabel"] = [1, 0]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corpus manifest fields mismatch"):
                CorpusIndex.load(root)


if __name__ == "__main__":
    unittest.main()
