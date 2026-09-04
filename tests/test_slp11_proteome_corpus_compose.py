from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-proteome-corpus-compose-v1"
sys.path.insert(0, str(MODULE))

from composer import (
    PRODUCTION_CONTRACT,
    Bounds,
    CorpusComposeError,
    PinnedDataset,
    build_composite_corpus,
    composite_key_sha256,
    composite_perturbation_id,
    deterministic_npz_bytes,
    deterministic_tar_bytes,
    read_canonical_tar,
    read_deterministic_npz,
    validate_composite_corpus_archive,
    validate_held_intervention_boundary,
)


class CompositeIdentityTests(unittest.TestCase):
    def test_same_identifier_in_two_taxa_remains_two_keys(self) -> None:
        keys = [(4932, "X:one"), (9606, "X:one")]
        digest = composite_key_sha256(keys)
        self.assertNotEqual(digest, composite_key_sha256([(4932, "X:one")]))
        self.assertEqual(digest, composite_key_sha256(reversed(keys)))

    def test_taxon_change_changes_perturbation_identity(self) -> None:
        yeast = composite_perturbation_id([(4932, "SGD:S000000001")])
        human = composite_perturbation_id([(9606, "SGD:S000000001")])
        self.assertNotEqual(yeast, human)
        self.assertRegex(yeast, r"^slp-perturbation:sha256-[0-9a-f]{64}$")

    def test_protected_roster_action_fails(self) -> None:
        key = (4932, "SGD:S000000001")
        with self.assertRaisesRegex(CorpusComposeError, "protected roster"):
            validate_held_intervention_boundary([key], {key: "molecular-validation"})


class ProductionCompositeCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        observations = (
            ROOT / ".omf" / "staging" / "slp-1-1-proteome-observation-pretrain-v1"
        )
        roster = ROOT / ".omf" / "staging" / "slp-1-1-held-roster-v1"
        features = (
            ROOT
            / ".omf"
            / "runs"
            / "01a06df0-2427-7737-9321-1615583dedd8"
            / "stages"
            / "build"
            / "sequence-statistics-feature-block-v1"
        )
        if not all(path.is_dir() for path in (observations, roster, features)):
            raise unittest.SkipTest("exact local OMF production inputs are unavailable")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls.temporary.name)
        expected = PRODUCTION_CONTRACT
        obs = PinnedDataset(
            "observations",
            observations.resolve(),
            expected.observations.resource,
            expected.observations.resource.rsplit("@", 1)[1],
            expected.observations.manifest_digest,
        )
        feat = PinnedDataset(
            "staticFeatures",
            features.resolve(),
            expected.features.resource,
            expected.features.resource.rsplit("@", 1)[1],
            expected.features.manifest_digest,
        )
        held = PinnedDataset(
            "heldInterventionRoster",
            roster.resolve(),
            expected.roster.resource,
            expected.roster.resource.rsplit("@", 1)[1],
            expected.roster.manifest_digest,
        )
        cls.first = build_composite_corpus(obs, feat, held, cls.temp_root / "first")
        cls.second = build_composite_corpus(obs, feat, held, cls.temp_root / "second")
        cls.archive = cls.temp_root / "first" / "corpus-v1-2.tar"
        cls.bounds = Bounds()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def _mutated_archive(self, name: str, mutate) -> Path:
        blobs = read_canonical_tar(self.archive, self.bounds, "test corpus")
        manifest = json.loads(blobs["composite-corpus/corpus.json"])
        mutate(blobs, manifest)
        blobs["composite-corpus/corpus.json"] = (
            json.dumps(
                manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            )
            + "\n"
        ).encode("ascii")
        path = self.temp_root / name
        path.write_bytes(deterministic_tar_bytes(blobs))
        return path

    def test_production_compose_is_byte_deterministic_and_composite(self) -> None:
        self.assertEqual(self.first, self.second)
        self.assertEqual(
            (self.temp_root / "first" / "corpus-v1-2.tar").read_bytes(),
            (self.temp_root / "second" / "corpus-v1-2.tar").read_bytes(),
        )
        summary = validate_composite_corpus_archive(self.archive)
        self.assertEqual(summary["counts"]["entities"], 7038)
        blobs = read_canonical_tar(self.archive, self.bounds, "test corpus")
        entities = read_deterministic_npz(
            blobs["composite-corpus/entities.npz"],
            {
                "entity_taxon",
                "entity_id",
                "entity_type",
                "entity_feature_value",
                "entity_feature_present",
            },
            "entities",
        )
        self.assertTrue(entities["entity_feature_present"][:7037].all())
        self.assertFalse(entities["entity_feature_present"][7037].any())
        queries = read_deterministic_npz(
            blobs["composite-corpus/queries.npz"],
            {"query_entity_index", "query_readout_index"},
            "queries",
        )
        self.assertEqual(set(queries), {"query_entity_index", "query_readout_index"})
        first_trajectory = json.loads(
            blobs["composite-corpus/trajectory-interventions.jsonl"].splitlines()[0]
        )
        self.assertEqual(set(first_trajectory), {"schema", "ncbiTaxon", "entityId"})
        audit = json.loads(
            (self.temp_root / "first" / "corpus-compose-audit.json").read_text()
        )
        self.assertTrue(audit["featurePreservation"]["byteExact"])
        self.assertEqual(
            audit["featurePreservation"]["sourceValueBytesSha256"],
            audit["featurePreservation"]["composedValueBytesSha256"],
        )
        self.assertEqual(
            audit["targetPreservation"]["sourceBytesSha256"],
            audit["targetPreservation"]["composedBytesSha256"],
        )

    def test_duplicate_composite_entity_key_is_rejected(self) -> None:
        def mutate(blobs, manifest):
            arrays = read_deterministic_npz(
                blobs["composite-corpus/entities.npz"],
                {
                    "entity_taxon",
                    "entity_id",
                    "entity_type",
                    "entity_feature_value",
                    "entity_feature_present",
                },
                "entities",
            )
            arrays["entity_id"] = arrays["entity_id"].copy()
            arrays["entity_id"][1] = arrays["entity_id"][0]
            payload = deterministic_npz_bytes(arrays)
            blobs["composite-corpus/entities.npz"] = payload
            manifest["entityDictionary"] = {
                "path": "entities.npz",
                "bytes": len(payload),
                "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                "count": 7038,
            }

        path = self._mutated_archive("duplicate.tar", mutate)
        with self.assertRaisesRegex(CorpusComposeError, "duplicated or unordered"):
            validate_composite_corpus_archive(path)

    def test_swapped_record_taxon_is_rejected(self) -> None:
        def mutate(blobs, manifest):
            member = "composite-corpus/shards/shard-00000.npz"
            arrays = read_deterministic_npz(blobs[member], SOURCE_OUTPUT_NAMES, "shard")
            arrays["species_taxon"] = arrays["species_taxon"].copy()
            arrays["species_taxon"][0] = 9606
            payload = deterministic_npz_bytes(arrays)
            blobs[member] = payload
            manifest["shards"][0].update(
                bytes=len(payload),
                sha256=__import__("hashlib").sha256(payload).hexdigest(),
            )

        path = self._mutated_archive("swapped-taxon.tar", mutate)
        with self.assertRaisesRegex(CorpusComposeError, "species"):
            validate_composite_corpus_archive(path)

    def test_bare_derived_perturbation_identity_is_rejected(self) -> None:
        def mutate(blobs, manifest):
            member = "composite-corpus/shards/shard-00000.npz"
            arrays = read_deterministic_npz(blobs[member], SOURCE_OUTPUT_NAMES, "shard")
            arrays["perturbation_id"] = arrays["perturbation_id"].copy()
            arrays["perturbation_id"][0] = "slp-perturbation:SGD:S000000001"
            payload = deterministic_npz_bytes(arrays)
            blobs[member] = payload
            manifest["shards"][0].update(
                bytes=len(payload),
                sha256=__import__("hashlib").sha256(payload).hexdigest(),
            )

        path = self._mutated_archive("perturbation-tamper.tar", mutate)
        with self.assertRaisesRegex(CorpusComposeError, "composite action key"):
            validate_composite_corpus_archive(path)

    def test_feature_pack_digest_tamper_is_rejected(self) -> None:
        path = self._mutated_archive(
            "feature-pack-tamper.tar",
            lambda _blobs, manifest: manifest["featurePack"].update(sha256="0" * 64),
        )
        with self.assertRaisesRegex(CorpusComposeError, "featurePack canonical digest"):
            validate_composite_corpus_archive(path)


SOURCE_OUTPUT_NAMES = {
    "record_id",
    "observation_unit_id",
    "source_index",
    "replicate_id",
    "perturbation_id",
    "species_taxon",
    "species_feature_value",
    "species_feature_present",
    "context_entity_index",
    "context_type",
    "context_mask",
    "context_covariate_value",
    "context_covariate_present",
    "record_covariate_value",
    "record_covariate_present",
    "action_entity_index",
    "action_type",
    "action_mask",
    "action_covariate_value",
    "action_covariate_present",
    "observation_covariate_value",
    "observation_covariate_present",
    "query_panel_index",
    "target_indptr",
    "target_query_index",
    "target_value",
}


if __name__ == "__main__":
    unittest.main()
