"""Focused contract tests for the typed sparse SLp-1.1 candidate."""

from dataclasses import fields, replace
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

from slp_sparse_architecture import (  # noqa: E402
    SparseTypedWorldModel,
    WorldBatch,
    negative_log_likelihood,
)
from slp_sparse_corpus import (  # noqa: E402
    CorpusIndex,
    DeterministicHierarchicalSampler,
    pinned_dataset_path,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_corpus(
    root: Path,
    *,
    permute_entities: bool = False,
    permute_queries: bool = False,
    extra_entities: int = 0,
) -> Path:
    root.mkdir()
    logical_entity_id = [
        "SGD:S0001",
        "SGD:S0002",
        "SGD:S0003",
        "SGD:S0004",
    ] + [f"SGD:SX{index:04d}" for index in range(extra_entities)]
    logical_entity_type = np.zeros(len(logical_entity_id), dtype=np.int64)
    logical_taxon = np.full(len(logical_entity_id), 4932, dtype=np.int64)
    logical_value = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [0.5, 1.0, 0.0],
            [-0.5, 1.0, 0.0],
        ]
        + [[float(index + 2), 0.0, 0.0] for index in range(extra_entities)],
        dtype=np.float32,
    )
    logical_present = np.asarray(
        [
            [True, False, True],
            [True, True, True],
            [True, True, False],
            [True, True, False],
        ]
        + [[True, False, False] for _ in range(extra_entities)],
        dtype=np.bool_,
    )
    entity_order = list(range(len(logical_entity_id)))
    if permute_entities:
        entity_order[:4] = [2, 0, 3, 1]
    logical_to_entity = np.empty(len(entity_order), dtype=np.int64)
    for physical, logical in enumerate(entity_order):
        logical_to_entity[logical] = physical
    entity_path = root / "entity-table.npz"
    np.savez(
        entity_path,
        entity_id=np.asarray([logical_entity_id[item] for item in entity_order], dtype="<U32"),
        entity_type=logical_entity_type[entity_order],
        entity_species_taxon=logical_taxon[entity_order],
        entity_feature_value=logical_value[entity_order],
        entity_feature_present=logical_present[entity_order],
    )

    logical_query_id = ["SLPQ:umi-S0003", "SLPQ:effect-S0004"]
    logical_query_entity = [2, 3]
    logical_query_readout = [0, 1]
    query_order = [1, 0] if permute_queries else [0, 1]
    logical_to_query = np.empty(2, dtype=np.int64)
    for physical, logical in enumerate(query_order):
        logical_to_query[logical] = physical
    query_path = root / "query-table.npz"
    np.savez(
        query_path,
        query_id=np.asarray([logical_query_id[item] for item in query_order], dtype="<U32"),
        query_entity_index=np.asarray(
            [logical_to_entity[logical_query_entity[item]] for item in query_order],
            dtype=np.int64,
        ),
        query_readout_index=np.asarray(
            [logical_query_readout[item] for item in query_order], dtype=np.int64
        ),
    )
    panel_path = root / "query-panels.npz"
    panel_queries = np.asarray(
        [logical_to_query[0], logical_to_query[1]], dtype=np.int64
    )
    np.savez(
        panel_path,
        panel_id=np.asarray(["SLPPANEL:joint-small"], dtype="<U32"),
        panel_indptr=np.asarray([0, 2], dtype=np.int64),
        panel_query_index=panel_queries,
    )

    shard_path = root / "shard-000.npz"
    context_ref = np.asarray(
        [[logical_to_entity[0]], [logical_to_entity[0]]], dtype=np.int64
    )
    action_ref = np.asarray(
        [
            [logical_to_entity[1], -1],
            [-1, -1],
        ],
        dtype=np.int64,
    )
    np.savez(
        shard_path,
        record_id=np.asarray(["TEST:record-0", "TEST:record-1"], dtype="<U32"),
        observation_unit_id=np.asarray(["TEST:cell-0", "TEST:cell-1"], dtype="<U32"),
        source_index=np.asarray([0, 1], dtype=np.int64),
        replicate_id=np.asarray(["TEST:replicate-a", "TEST:replicate-b"], dtype="<U32"),
        perturbation_id=np.asarray(
            ["TEST:perturbation-a", "TEST:perturbation-b"], dtype="<U32"
        ),
        species_taxon=np.asarray([4932, 4932], dtype=np.int64),
        species_feature_value=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        species_feature_present=np.asarray(
            [[True, False], [True, False]], dtype=np.bool_
        ),
        context_entity_index=context_ref,
        context_type=np.zeros((2, 1), dtype=np.int64),
        context_mask=np.ones((2, 1), dtype=np.bool_),
        context_covariate_value=np.asarray([[[30.0]], [[30.0]]], dtype=np.float32),
        context_covariate_present=np.ones((2, 1, 1), dtype=np.bool_),
        record_covariate_value=np.asarray([[0.0], [1.0]], dtype=np.float32),
        record_covariate_present=np.ones((2, 1), dtype=np.bool_),
        action_entity_index=action_ref,
        action_type=np.asarray([[0, -1], [-1, -1]], dtype=np.int64),
        action_mask=np.asarray([[True, False], [False, False]], dtype=np.bool_),
        action_covariate_value=np.asarray(
            [[[15.0, 0.4], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=np.float32,
        ),
        action_covariate_present=np.asarray(
            [[[True, True], [False, False]], [[False, False], [False, False]]],
            dtype=np.bool_,
        ),
        observation_covariate_value=np.asarray(
            [[1200.0, 0.0], [900.0, 1.0]], dtype=np.float32
        ),
        observation_covariate_present=np.ones((2, 2), dtype=np.bool_),
        query_panel_index=np.zeros(2, dtype=np.int64),
        target_indptr=np.asarray([0, 2, 2], dtype=np.int64),
        target_query_index=np.asarray(
            [logical_to_query[0], logical_to_query[1]], dtype=np.int64
        ),
        target_value=np.asarray([3.0, 0.0], dtype=np.float32),
    )
    gene_path = root / "trajectory-genes.txt"
    gene_path.write_text("SGD:S0002\n", encoding="utf-8")
    manifest = {
        "schema": "slp.corpus/v1.1",
        "datasetId": "TEST:sparse-corpus",
        "version": "fixture-v1",
        "role": "pretrain",
        "labelClass": "molecular",
        "benchmarkLabelsPresent": False,
        "rights": {
            "revision": "TESTRIGHTS:cc0-v1",
            "trainingAllowed": True,
            "redistributionAllowed": True,
        },
        "modalities": ["EFO:0002691"],
        "sources": [{"id": "TESTSOURCE:a"}, {"id": "TESTSOURCE:b"}],
        "sampling": {
            "scheme": "slp.source-intervention-replicate-record/v1",
            "sourceWeights": [1.0, 1.0],
        },
        "species": [
            {
                "taxon": 4932,
                "featureValue": [1.0, 0.0],
                "featurePresent": [True, False],
            }
        ],
        "featurePack": {
            "revision": "SLPFEATURE:fixture-v1",
            "sha256": "0" * 64,
            "entityFeatureDim": 3,
            "speciesFeatureDim": 2,
        },
        "entityTypes": ["SLPET:gene"],
        "contextTypes": ["SLPCTX:basal-state"],
        "actionTypes": ["SLPACT:gene-deletion"],
        "covariates": {
            "record": [
                {"id": "SLPCOV:medium", "unit": "UCUM:1", "access": "world"}
            ],
            "context": [
                {
                    "id": "SLPCOV:temperature",
                    "unit": "UCUM:Cel",
                    "access": "world",
                }
            ],
            "action": [
                {"id": "SLPCOV:duration", "unit": "UCUM:min", "access": "world"},
                {
                    "id": "SLPCOV:concentration",
                    "unit": "UCUM:mol/L",
                    "access": "world",
                },
            ],
            "observation": [
                {
                    "id": "SLPCOV:library-size",
                    "unit": "UCUM:1",
                    "access": "likelihood",
                },
                {"id": "SLPCOV:state", "unit": "UCUM:1", "access": "world"},
            ],
        },
        "readoutTypes": [
            {
                "id": "SLPRO:raw-umi",
                "likelihood": "negative-binomial",
                "unit": "UCUM:1",
                "implicitZero": True,
            },
            {
                "id": "SLPRO:continuous-effect",
                "likelihood": "gaussian",
                "unit": "UCUM:1",
                "implicitZero": False,
            },
        ],
        "entityDictionary": {
            "path": entity_path.name,
            "sha256": _sha256(entity_path),
            "count": len(logical_entity_id),
        },
        "queryDictionary": {
            "path": query_path.name,
            "sha256": _sha256(query_path),
            "count": 2,
        },
        "queryPanels": {
            "path": panel_path.name,
            "sha256": _sha256(panel_path),
            "count": 1,
        },
        "trajectoryGenes": {
            "path": gene_path.name,
            "sha256": _sha256(gene_path),
            "count": 1,
        },
        "normalization": {"id": "SLPNORM:none", "valueSpace": "SLPVS:mixed"},
        "bounds": {
            "maxRecordsPerShard": 8,
            "maxContextTokens": 2,
            "maxActionTokens": 2,
            "maxPanelQueries": 4,
            "maxTargetsPerRecord": 2,
        },
        "shards": [
            {
                "path": shard_path.name,
                "sha256": _sha256(shard_path),
                "records": 2,
                "targetValues": 2,
            }
        ],
    }
    (root / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _rewrite_shard(root: Path, mutate) -> None:
    path = root / "shard-000.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    mutate(arrays)
    np.savez(path, **arrays)
    manifest_path = root / "corpus.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_trajectory_genes(root: Path, genes: list[str]) -> None:
    path = root / "trajectory-genes.txt"
    path.write_text("".join(f"{gene}\n" for gene in sorted(genes)), encoding="utf-8")
    manifest_path = root / "corpus.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trajectoryGenes"]["sha256"] = _sha256(path)
    manifest["trajectoryGenes"]["count"] = len(genes)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class SparseCorpusContractTest(unittest.TestCase):
    def test_absent_count_is_zero_but_absent_continuous_target_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = CorpusIndex.load(_write_corpus(Path(temporary) / "corpus"))
            batch = corpus.materialize_batch(corpus.load_shard(0), [0, 1])
        self.assertEqual(batch.target_value.tolist(), [[3.0, 0.0], [0.0, 0.0]])
        self.assertEqual(
            batch.target_observed.tolist(), [[True, True], [True, False]]
        )
        self.assertEqual(batch.world.observation_covariates.shape, (2, 1))
        self.assertEqual(batch.likelihood_covariates.observation_value.shape, (2, 1))

    def test_missing_zero_and_observed_zero_are_distinct_model_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = CorpusIndex.load(_write_corpus(Path(temporary) / "corpus"))
            batch = corpus.materialize_batch(corpus.load_shard(0), [0])
        torch.manual_seed(19)
        model = SparseTypedWorldModel(
            corpus.world_config(
                d_model=16,
                nhead=4,
                encoder_layers=1,
                decoder_layers=1,
                dropout=0.0,
            )
        ).eval()
        missing = model(batch.world)
        observed_presence = batch.world.context_feature_present.clone()
        self.assertFalse(observed_presence[0, 0, 1])
        self.assertEqual(batch.world.context_features[0, 0, 1], 0)
        observed_presence[0, 0, 1] = True
        observed = model(replace(batch.world, context_feature_present=observed_presence))
        self.assertFalse(torch.equal(missing.parameters, observed.parameters))

    def test_dictionary_permutation_and_growth_do_not_reach_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = CorpusIndex.load(_write_corpus(root / "first"))
            second = CorpusIndex.load(
                _write_corpus(
                    root / "second",
                    permute_entities=True,
                    permute_queries=True,
                    extra_entities=17,
                )
            )
            first_batch = first.materialize_batch(first.load_shard(0), [0, 1]).world
            second_batch = second.materialize_batch(second.load_shard(0), [0, 1]).world
        for field in fields(WorldBatch):
            self.assertNotIn("id", field.name)
            torch.testing.assert_close(
                getattr(first_batch, field.name), getattr(second_batch, field.name)
            )
        torch.manual_seed(23)
        first_model = SparseTypedWorldModel(
            first.world_config(
                d_model=8,
                nhead=2,
                encoder_layers=1,
                decoder_layers=1,
                dropout=0.0,
            )
        ).eval()
        torch.manual_seed(23)
        second_model = SparseTypedWorldModel(
            second.world_config(
                d_model=8,
                nhead=2,
                encoder_layers=1,
                decoder_layers=1,
                dropout=0.0,
            )
        ).eval()
        self.assertEqual(first_model.count_parameters(), second_model.count_parameters())
        self.assertNotEqual(len(first.entity_id), len(second.entity_id))
        self.assertTrue(
            torch.equal(
                first_model(first_batch).parameters,
                second_model(second_batch).parameters,
            )
        )

    def test_queries_are_exactly_panel_order_and_chunk_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = CorpusIndex.load(_write_corpus(Path(temporary) / "corpus"))
            world = corpus.materialize_batch(corpus.load_shard(0), [0, 1]).world
        torch.manual_seed(29)
        model = SparseTypedWorldModel(
            corpus.world_config(
                d_model=16,
                nhead=4,
                encoder_layers=1,
                decoder_layers=2,
                dropout=0.0,
            )
        ).eval()
        expected = model(world)

        def select(indices: list[int]) -> WorldBatch:
            index = torch.tensor(indices)
            return replace(
                world,
                query_features=world.query_features[:, index],
                query_feature_present=world.query_feature_present[:, index],
                query_entity_type=world.query_entity_type[:, index],
                readout_type=world.readout_type[:, index],
                likelihood_type=world.likelihood_type[:, index],
                query_mask=world.query_mask[:, index],
            )

        reversed_prediction = model(select([1, 0]))
        self.assertTrue(
            torch.equal(reversed_prediction.parameters, expected.parameters[:, [1, 0]])
        )
        chunks = torch.cat(
            [model(select([0])).parameters, model(select([1])).parameters], dim=1
        )
        self.assertTrue(torch.equal(chunks, expected.parameters))

    def test_gaussian_and_negative_binomial_loss_are_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = CorpusIndex.load(_write_corpus(Path(temporary) / "corpus"))
            batch = corpus.materialize_batch(corpus.load_shard(0), [0, 1])
        torch.manual_seed(31)
        model = SparseTypedWorldModel(
            corpus.world_config(
                d_model=8,
                nhead=2,
                encoder_layers=1,
                decoder_layers=1,
                dropout=0.0,
            )
        )
        loss = negative_log_likelihood(
            model(batch.world), batch.target_value, batch.target_observed
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_sparse_shard_has_no_dense_record_by_query_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = CorpusIndex.load(_write_corpus(Path(temporary) / "corpus"))
            shard = corpus.load_shard(0)
            batch = corpus.materialize_batch(shard, [1])
        self.assertNotIn("target", shard.arrays)
        self.assertEqual(shard.arrays["target_value"].ndim, 1)
        self.assertEqual(shard.arrays["target_indptr"].shape, (shard.records + 1,))
        self.assertEqual(batch.target_value.shape, (1, 2))

    def test_active_species_action_missing_from_inventory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _write_corpus(Path(temporary) / "corpus")

            def add_undeclared_action(arrays: dict[str, np.ndarray]) -> None:
                arrays["action_entity_index"][0, 1] = 2
                arrays["action_type"][0, 1] = 0
                arrays["action_mask"][0, 1] = True

            _rewrite_shard(root, add_undeclared_action)
            with self.assertRaisesRegex(
                ValueError, r"missing=\['SGD:S0003'\], extra=\[\]"
            ):
                CorpusIndex.load(root)

    def test_inventory_superset_without_active_action_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _write_corpus(Path(temporary) / "corpus")
            _rewrite_trajectory_genes(root, ["SGD:S0001", "SGD:S0002"])
            with self.assertRaisesRegex(
                ValueError, r"missing=\[\], extra=\['SGD:S0001'\]"
            ):
                CorpusIndex.load(root)

    def test_equal_source_weights_schedule_exactly_fifty_fifty(self) -> None:
        source = [0] * 1000 + [1] * 10
        perturbation = [f"TEST:p-{index % 5}" for index in range(1010)]
        replicate = [f"TEST:r-{index % 3}" for index in range(1010)]
        sampler = DeterministicHierarchicalSampler(
            source,
            perturbation,
            replicate,
            source_weights=[1.0, 1.0],
            seed=731,
        )
        schedule = sampler.schedule(200)
        self.assertEqual(schedule, sampler.schedule(200))
        counts = [sum(source[position] == item for position in schedule) for item in (0, 1)]
        self.assertEqual(counts, [100, 100])

    def test_schema_is_valid_json_and_names_the_new_contract(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "slp-corpus-v1-1.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema"]["const"], "slp.corpus/v1.1")
        trajectory_ref = schema["properties"]["trajectoryGenes"]["$ref"]
        self.assertEqual(trajectory_ref, "#/$defs/possiblyEmptyFileReference")
        self.assertEqual(
            schema["$defs"]["possiblyEmptyFileReference"]["properties"]["count"]["minimum"],
            0,
        )
        self.assertFalse(schema["additionalProperties"])

    def test_pinned_dataset_resolver_accepts_actual_omf_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload = (
                Path(temporary)
                / "stages"
                / "train"
                / "inputs"
                / "corpus"
                / "sparse-v1"
            )
            payload.mkdir(parents=True)
            value = {
                "resource": (
                    "omf://abiome/slp/datasetsnapshot/sparse-v1@sha256:" + "a" * 64
                ),
                "mode": "copy",
                "path": str(payload),
                "manifestDigest": "sha256:" + "b" * 64,
            }
            self.assertEqual(pinned_dataset_path(value), str(payload.resolve()))

    def test_pinned_dataset_resolver_rejects_spoofed_path_and_mutable_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "inputs" / "corpus" / "sparse-v1"
            valid.mkdir(parents=True)
            arbitrary = root / "arbitrary" / "sparse-v1"
            arbitrary.mkdir(parents=True)
            base = {
                "resource": (
                    "omf://abiome/slp/datasetsnapshot/sparse-v1@sha256:" + "a" * 64
                ),
                "mode": "copy",
                "path": str(valid),
                "manifestDigest": "sha256:" + "b" * 64,
            }
            cases = (
                ({**base, "path": str(arbitrary)}, "path is inconsistent"),
                (
                    {**base, "resource": base["resource"].replace("sparse-v1@", "other@")},
                    "path is inconsistent",
                ),
                (
                    {**base, "resource": base["resource"].split("@")[0] + "@latest"},
                    "admission-pinned revision",
                ),
                ({**base, "mode": "mount"}, "immutable copied"),
            )
            for value, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    pinned_dataset_path(value)


if __name__ == "__main__":
    unittest.main()
