from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError
except ModuleNotFoundError:  # The pinned OMF runtime carries jsonschema.
    Draft202012Validator = None
    ValidationError = ValueError


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "slp-corpus-v1-2.schema.json"


def _sha(character: str) -> str:
    return character * 64


def _dataset(name: str, revision: str, outer: str, tree: str) -> dict[str, Any]:
    return {
        "resource": (
            f"omf://abiome/slp/datasetsnapshot/{name}@sha256:{_sha(revision)}"
        ),
        "revision": f"sha256:{_sha(revision)}",
        "outerManifestDigest": f"sha256:{_sha(outer)}",
        "treeDigest": f"sha256:{_sha(tree)}",
    }


def _lineage_file(path: str, digest: str, size: int = 128) -> dict[str, Any]:
    return {"path": path, "sha256": _sha(digest), "bytes": size}


def _positive_fixture() -> dict[str, Any]:
    feature_dataset = _dataset(
        "slp-1-1-sequence-statistics-feature-block-v1", "4", "5", "6"
    )
    feature_block = {
        "id": "slp-feature-block:sequence-statistics-v1",
        "offset": 0,
        "dimension": 21,
        "datasetSnapshot": feature_dataset,
        "semanticSha256": _sha("7"),
        "entityKeySetSha256": _sha("8"),
        "files": [
            _lineage_file("sequence-statistics-feature-block.tar", "9", 1024),
            _lineage_file("sequence-statistics-feature-block-audit.json", "a", 256),
        ],
    }
    feature_pack = {
        "schema": "slp.static-feature-pack/v1",
        "revision": "slp-feature-pack:sequence-statistics-v1",
        "entityFeatureDim": 21,
        "speciesFeatureDim": 1,
        "blocks": [feature_block],
    }
    feature_pack["sha256"] = hashlib.sha256(
        json.dumps(
            feature_pack,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema": "slp.corpus/v1.2",
        "datasetId": "slp-corpus:yeast-proteome-pretrain-v1-2",
        "version": "v1.2",
        "role": "pretrain",
        "labelClass": "molecular",
        "benchmarkLabelsPresent": False,
        "rewardEnabled": False,
        "identityKey": ["ncbiTaxon", "entityId"],
        "rights": {
            "revision": "slp-rights:fixture-v1",
            "trainingAllowed": True,
            "redistributionAllowed": False,
        },
        "modalities": ["slp-modality:quantitative-proteome"],
        "sources": [{"id": "slp-source:mendeley-w8jtmnszd9-v2"}],
        "sampling": {
            "scheme": "slp.source-intervention-replicate-record/v1",
            "sourceWeights": [1.0],
        },
        "species": [
            {
                "taxon": 4932,
                "featureValue": [1.0],
                "featurePresent": [True],
            }
        ],
        "featurePack": feature_pack,
        "entityTypes": ["slp-entity:gene", "slp-entity:protein"],
        "contextTypes": ["slp-context-type:cell-state"],
        "actionTypes": ["slp-action:gene-deletion"],
        "covariates": {
            "record": [
                {
                    "id": "slp-covariate:replicate-index",
                    "unit": "unit:dimensionless",
                    "access": "audit",
                }
            ],
            "context": [],
            "action": [],
            "observation": [],
        },
        "readoutTypes": [
            {
                "id": "slp-readout:protein-abundance",
                "likelihood": "gaussian",
                "unit": "unit:log2-ratio",
                "implicitZero": False,
            }
        ],
        "entityDictionary": {
            "path": "entity-dictionary.npz",
            "sha256": _sha("f"),
            "bytes": 4096,
            "count": 7038,
        },
        "queryDictionary": {
            "path": "query-dictionary.npz",
            "sha256": _sha("0"),
            "bytes": 2048,
            "count": 1850,
        },
        "queryPanels": {
            "path": "query-panels.npz",
            "sha256": _sha("1"),
            "bytes": 1024,
            "count": 1,
        },
        "trajectoryInterventions": {
            "path": "trajectory-interventions.jsonl",
            "sha256": _sha("2"),
            "bytes": 512,
            "count": 4,
        },
        "normalization": {
            "id": "slp-normalization:source-log2-ratio-v1",
            "valueSpace": "slp-value-space:source-log2-ratio",
        },
        "bounds": {
            "maxRecordsPerShard": 4096,
            "maxContextTokens": 4,
            "maxActionTokens": 4,
            "maxPanelQueries": 4096,
            "maxTargetsPerRecord": 4096,
        },
        "counts": {
            "entities": 7038,
            "featureRows": 7037,
            "contexts": 1,
            "queries": 1850,
            "panels": 1,
            "trajectoryInterventions": 4,
            "records": 16,
            "targetValues": 128,
            "shards": 1,
        },
        "shards": [
            {
                "path": "shards/shard-00000.npz",
                "sha256": _sha("3"),
                "bytes": 8192,
                "records": 16,
                "targetValues": 128,
            }
        ],
        "inputs": {
            "observations": {
                "datasetSnapshot": _dataset(
                    "slp-1-1-proteome-observation-pretrain-v1", "a", "b", "c"
                ),
                "semanticSha256": _sha("6"),
                "files": [_lineage_file("payload.tar", "7")],
            },
            "staticFeatures": {
                "datasetSnapshot": _dataset(
                    "slp-1-1-sequence-statistics-feature-block-v1", "e", "f", "0"
                ),
                "semanticSha256": _sha("8"),
                "files": [_lineage_file("payload.tar", "9")],
            },
            "heldInterventionRoster": {
                "datasetSnapshot": _dataset(
                    "slp-1-1-held-roster-v1", "2", "3", "4"
                ),
                "semanticSha256": _sha("a"),
                "files": [_lineage_file("held-roster.tsv", "b")],
            },
        },
    }


@unittest.skipUnless(Draft202012Validator is not None, "jsonschema is unavailable")
class CorpusV12SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_positive_composite_corpus_fixture_passes(self) -> None:
        document = _positive_fixture()
        self.validator.validate(document)
        pack = dict(document["featurePack"])
        declared = pack.pop("sha256")
        self.assertEqual(
            hashlib.sha256(
                json.dumps(
                    pack,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            ).hexdigest(),
            declared,
        )

    def test_every_object_schema_is_closed(self) -> None:
        stack: list[Any] = [self.schema]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if value.get("type") == "object":
                    self.assertIs(
                        value.get("additionalProperties"),
                        False,
                        msg=f"open object schema: {value}",
                    )
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)

    def test_missing_taxon_is_rejected(self) -> None:
        for mutation in ("top-level", "species"):
            document = _positive_fixture()
            if mutation == "top-level":
                document["identityKey"] = ["entityId"]
            else:
                del document["species"][0]["taxon"]
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                self.validator.validate(document)

    def test_legacy_bare_trajectory_genes_is_rejected(self) -> None:
        document = _positive_fixture()
        document["trajectoryGenes"] = document.pop("trajectoryInterventions")
        with self.assertRaises(ValidationError):
            self.validator.validate(document)

    def test_malformed_lineage_and_file_digests_are_rejected(self) -> None:
        mutations = []

        missing_prefix = _positive_fixture()
        missing_prefix["inputs"]["observations"]["datasetSnapshot"][
            "treeDigest"
        ] = _sha("a")
        mutations.append(missing_prefix)

        prefixed_semantic_digest = _positive_fixture()
        prefixed_semantic_digest["featurePack"]["blocks"][0][
            "semanticSha256"
        ] = "sha256:" + _sha("a")
        mutations.append(prefixed_semantic_digest)

        uppercase_file_digest = _positive_fixture()
        uppercase_file_digest["featurePack"]["blocks"][0]["files"][0][
            "sha256"
        ] = "A" * 64
        mutations.append(uppercase_file_digest)

        short_pack_digest = _positive_fixture()
        short_pack_digest["featurePack"]["sha256"] = "0" * 63
        mutations.append(short_pack_digest)

        for index, document in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValidationError):
                self.validator.validate(document)

    def test_unknown_nested_fields_are_rejected(self) -> None:
        mutations = []

        input_extra = _positive_fixture()
        input_extra["inputs"]["observations"]["displayName"] = "not provenance"
        mutations.append(input_extra)

        species_extra = _positive_fixture()
        species_extra["species"][0]["symbol"] = "yeast"
        mutations.append(species_extra)

        block_file_extra = _positive_fixture()
        block_file_extra["featurePack"]["blocks"][0]["files"][0]["dtype"] = (
            "float32"
        )
        mutations.append(block_file_extra)

        for index, document in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValidationError):
                self.validator.validate(document)

    def test_non_strict_feature_block_is_rejected(self) -> None:
        for missing in (
            "offset",
            "dimension",
            "datasetSnapshot",
            "semanticSha256",
            "files",
        ):
            document = _positive_fixture()
            del document["featurePack"]["blocks"][0][missing]
            with self.subTest(missing=missing), self.assertRaises(ValidationError):
                self.validator.validate(document)

    def test_file_references_reject_self_asserted_formats_or_arrays(self) -> None:
        for reference, extra_field, value in (
            ("entityDictionary", "arrays", ["entity_id"]),
            ("queryDictionary", "format", "numpy-npz"),
            ("queryPanels", "arrays", ["panel_id"]),
            ("trajectoryInterventions", "recordSchema", "unverified"),
        ):
            document = _positive_fixture()
            document[reference][extra_field] = value
            with self.subTest(reference=reference), self.assertRaises(
                ValidationError
            ):
                self.validator.validate(document)

    def test_benchmark_reward_and_non_pretrain_roles_are_rejected(self) -> None:
        for field, value in (
            ("benchmarkLabelsPresent", True),
            ("rewardEnabled", True),
            ("role", "molecular-validation"),
            ("labelClass", "synthetic-lethality"),
        ):
            document = _positive_fixture()
            document[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.validator.validate(document)


if __name__ == "__main__":
    unittest.main()
