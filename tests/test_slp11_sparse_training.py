"""Pretrain-only optimizer and target-free prediction checks."""

from dataclasses import replace
import hashlib
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
from slp_sparse_corpus import CorpusIndex, PredictionQueryIndex  # noqa: E402
from slp_sparse_training import (  # noqa: E402
    TrainingConfig,
    equal_record_negative_log_likelihood,
    iter_sparse_predictions,
    train_sparse_world,
)
from tests.test_slp11_sparse_candidate import _sha256, _write_corpus  # noqa: E402


def _assigned_gene(role: str, start: int = 1000) -> str:
    domain = b"slp-1.1-yeast-global-held-v1\x00"
    for number in range(start, start + 10000):
        candidate = f"SGD:S{number:09d}"
        digest = hashlib.sha256(domain + candidate.encode("ascii")).hexdigest()
        bucket = int(digest[:16], 16) % 100
        observed = "molecular-final" if bucket < 10 else "molecular-validation" if bucket < 30 else "pretrain"
        if observed == role:
            return candidate
    raise AssertionError("could not construct assigned fixture identifier")


VALIDATION_GENE = _assigned_gene("molecular-validation")
FINAL_GENE = _assigned_gene("molecular-final")


def _make_pretrain(
    root: Path, *, extra_entities: int = 0, active_gene: str = "SGD:S0002",
) -> CorpusIndex:
    _write_corpus(root, extra_entities=extra_entities + 1)
    entity_path = root / "entity-table.npz"
    with np.load(entity_path, allow_pickle=False) as source:
        entities = {name: source[name] for name in source.files}
    validation = np.flatnonzero(entities["entity_id"] == "SGD:S0003")
    entities["entity_id"][validation[0]] = VALIDATION_GENE
    if active_gene != "SGD:S0002":
        active = np.flatnonzero(entities["entity_id"] == "SGD:S0002")
        entities["entity_id"][active[0]] = active_gene
    else:
        held_static = np.flatnonzero(entities["entity_id"] == "SGD:SX0000")
        entities["entity_id"][held_static[0]] = FINAL_GENE
    np.savez(entity_path, **entities)
    gene_path = root / "trajectory-genes.txt"
    gene_path.write_text(active_gene + "\n", encoding="utf-8")
    manifest_path = root / "corpus.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entityDictionary"]["sha256"] = _sha256(entity_path)
    manifest["trajectoryGenes"]["sha256"] = _sha256(gene_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return CorpusIndex.load(root)


def _canonical_id(prefix: str, value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return prefix + hashlib.sha256(payload).hexdigest()


def _make_query(
    root: Path, feature_corpus: CorpusIndex, intervention: str = VALIDATION_GENE,
) -> PredictionQueryIndex:
    root.mkdir()
    interventions = [intervention]
    perturbation = _canonical_id("PERTURBATION:", interventions)
    rows = []
    for index, source in enumerate(("TESTSOURCE:a", "TESTSOURCE:b")):
        group = f"TEST:centering-{index}"
        identity = {
            "speciesTaxon": 4932,
            "sourceId": source,
            "centeringGroup": group,
            "perturbationId": perturbation,
        }
        rows.append({
            "profileId": _canonical_id("PROFILE:", identity),
            **identity,
            "interventionIds": interventions,
            "readoutIds": [VALIDATION_GENE, "SGD:S0004"],
            "distributionTypes": ["negative-binomial", "gaussian"],
        })
    shard_path = root / "profiles-query.jsonl"
    shard_path.write_bytes(b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    ))
    manifest = {
        "schema": "slp.molecular-query-manifest/v1",
        "datasetId": "TEST:sparse-query",
        "version": "fixture-v1",
        "role": "molecular-validation-query",
        "labelClass": "none",
        "targetValuesPresent": False,
        "observedMaskPresent": False,
        "valueSpace": feature_corpus.value_space,
        "speciesTaxa": [4932],
        "sourceIds": ["TESTSOURCE:a", "TESTSOURCE:b"],
        "shards": [{
            "path": shard_path.name,
            "sha256": _sha256(shard_path),
            "bytes": shard_path.stat().st_size,
            "records": len(rows),
        }],
    }
    (root / "query.json").write_text(json.dumps(manifest), encoding="utf-8")
    return PredictionQueryIndex.load(root, feature_corpus)


def _config() -> TrainingConfig:
    return TrainingConfig(
        seed=83, epochs=3, draws_per_epoch=8, batch_size=4,
        learning_rate=0.01, prediction_batch_size=2, d_model=8, nhead=2,
        encoder_layers=1, decoder_layers=1, ffn_multiplier=2, dropout=0.0,
    )


class SparseTrainingTest(unittest.TestCase):
    def test_pretrain_only_training_and_target_free_prediction_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain = _make_pretrain(root / "pretrain")
            query = _make_query(root / "query", pretrain)
            rng = torch.random.get_rng_state().clone()
            first = train_sparse_world(pretrain, _config())
            second = train_sparse_world(pretrain, _config())
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            first_predictions = list(iter_sparse_predictions(first.model, query, 1))
            second_predictions = list(iter_sparse_predictions(second.model, query, 1))
        self.assertEqual(first.report, second.report)
        self.assertNotIn("validation", json.dumps(first.report).lower())
        self.assertFalse(first.report["isolation"]["heldTruthAccessible"])
        self.assertEqual(first.report["training"]["sourceDraws"], {"TESTSOURCE:a": 12, "TESTSOURCE:b": 12})
        self.assertGreater(len(set(first.report["training"]["epochScheduleSha256"])), 1)
        for left, right in zip(first_predictions, second_predictions, strict=True):
            self.assertEqual(left.profile_id, right.profile_id)
            self.assertEqual(left.readout_ids, right.readout_ids)
            self.assertTrue(torch.equal(left.parameters, right.parameters))

    def test_optimization_weights_records_equally_despite_target_density(self) -> None:
        parameters = torch.zeros((2, 3, 2), dtype=torch.float32, requires_grad=True)
        prediction = WorldPrediction(
            parameters=parameters,
            likelihood_type=torch.zeros((2, 3), dtype=torch.long),
            query_mask=torch.ones((2, 3), dtype=torch.bool),
        )
        target = torch.tensor([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        observed = torch.tensor([[True, False, False], [True, True, True]])
        objective = equal_record_negative_log_likelihood(prediction, target, observed)
        constant = 0.5 * np.log(2.0 * np.pi)
        self.assertAlmostEqual(objective.item(), 0.5 * (constant + 8.0 + constant), places=6)

    def test_query_is_structurally_target_free_and_exactly_chunkable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain = _make_pretrain(root / "pretrain")
            query = _make_query(root / "query", pretrain)
            self.assertEqual(
                {item.name for item in query.root.iterdir()},
                {"query.json", "profiles-query.jsonl"},
            )
            batch = query.materialize(list(query.iter_records(0))).world
            model = train_sparse_world(pretrain, _config()).model.eval()
            expected = model(batch).parameters
            chunks = []
            for index in range(expected.shape[1]):
                selection = torch.tensor([index])
                chunks.append(model(replace(
                    batch,
                    query_features=batch.query_features[:, selection],
                    query_feature_present=batch.query_feature_present[:, selection],
                    query_entity_type=batch.query_entity_type[:, selection],
                    readout_type=batch.readout_type[:, selection],
                    likelihood_type=batch.likelihood_type[:, selection],
                    query_mask=batch.query_mask[:, selection],
                )).parameters)
            self.assertTrue(torch.equal(torch.cat(chunks, dim=1), expected))

    def test_query_identity_or_hidden_target_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrain = _make_pretrain(root / "pretrain")
            query = _make_query(root / "query", pretrain)
            query_path = query.root / "query.json"
            manifest = json.loads(query_path.read_text())
            manifest["targetValuesPresent"] = True
            query_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "target-free"):
                PredictionQueryIndex.load(query.root, pretrain)


if __name__ == "__main__":
    unittest.main()
