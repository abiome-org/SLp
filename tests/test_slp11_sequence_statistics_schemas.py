from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

import yaml

try:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError
    from referencing import Registry, Resource
except ModuleNotFoundError:  # The OMF runtime carries these; lean host test envs may not.
    Draft202012Validator = None
    ValidationError = ValueError
    Registry = None
    Resource = None

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-sequence-statistics-feature-block-v1"


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return schema
    value: Any = root
    for part in reference.removeprefix("#/").split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    merged = _resolve(copy.deepcopy(value), root)
    for key, item in schema.items():
        if key == "$ref":
            continue
        if key == "properties":
            merged.setdefault("properties", {}).update(copy.deepcopy(item))
        else:
            merged[key] = copy.deepcopy(item)
    return merged


def _example(schema: dict[str, Any], root: dict[str, Any]) -> Any:
    schema = _resolve(schema, root)
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if "oneOf" in schema:
        return _example(schema["oneOf"][0], root)
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties", {})
        return {name: _example(properties[name], root) for name in schema.get("required", [])}
    if kind == "array":
        if "prefixItems" in schema:
            return [_example(item, root) for item in schema["prefixItems"]]
        return []
    if kind == "integer":
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    if schema.get("pattern") == "^[0-9a-f]{64}$":
        return "0" * 64
    return "fixture"


@unittest.skipUnless(Draft202012Validator is not None, "jsonschema is unavailable")
class SequenceStatisticsArtifactSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_schema = json.loads(
            (MODULE / "feature-block.schema.json").read_text(encoding="utf-8")
        )
        cls.audit_schema = json.loads(
            (MODULE / "feature-block-audit.schema.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(cls.manifest_schema)
        Draft202012Validator.check_schema(cls.audit_schema)
        registry = Registry().with_resource(
            cls.manifest_schema["$id"], Resource.from_contents(cls.manifest_schema)
        )
        cls.manifest_validator = Draft202012Validator(cls.manifest_schema)
        cls.audit_validator = Draft202012Validator(cls.audit_schema, registry=registry)
        cls.manifest = _example(cls.manifest_schema, cls.manifest_schema)
        cls.audit = _example(cls.audit_schema, cls.audit_schema)
        for name in ("inputs", "source", "identityMapping", "featureDefinition", "counts"):
            cls.audit[name] = copy.deepcopy(cls.manifest[name])

    def test_production_shaped_manifest_and_audit_pass(self) -> None:
        self.manifest_validator.validate(self.manifest)
        self.audit_validator.validate(self.audit)

    def test_manifest_rejects_semantic_type_and_nested_property_drift(self) -> None:
        mutations = []
        formula = copy.deepcopy(self.manifest)
        formula["featureDefinition"]["formula"] = "learned embedding"
        mutations.append(formula)
        bool_as_int = copy.deepcopy(self.manifest)
        bool_as_int["featureDefinition"]["clipping"] = 0
        mutations.append(bool_as_int)
        extra = copy.deepcopy(self.manifest)
        extra["inputs"]["sgdProteinSequences"]["source"]["displayName"] = "yeast"
        mutations.append(extra)
        swapped = copy.deepcopy(self.manifest)
        swapped["inputs"]["staticEntityUniverse"] = copy.deepcopy(
            swapped["inputs"]["sgdProteinSequences"]
        )
        mutations.append(swapped)
        for document in mutations:
            with self.subTest(mutation=len(mutations)), self.assertRaises(ValidationError):
                self.manifest_validator.validate(document)

    def test_audit_rejects_boundary_consensus_and_output_drift(self) -> None:
        mutations = []
        bool_as_int = copy.deepcopy(self.audit)
        bool_as_int["accessBoundary"]["heldRosterConsumed"] = 0
        mutations.append(bool_as_int)
        reordered = copy.deepcopy(self.audit)
        reordered["multiTargetPeptideConsensus"].reverse()
        mutations.append(reordered)
        changed_output = copy.deepcopy(self.audit)
        changed_output["outputs"]["archive"]["bytes"] += 1
        mutations.append(changed_output)
        extra = copy.deepcopy(self.audit)
        extra["outputs"]["checkpointSha256"] = "0" * 64
        mutations.append(extra)
        for document in mutations:
            with self.assertRaises(ValidationError):
                self.audit_validator.validate(document)


@unittest.skipUnless(Draft202012Validator is not None, "jsonschema is unavailable")
class SequenceStatisticsModuleSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = yaml.safe_load((MODULE / "module.yaml").read_text(encoding="utf-8"))
        contracts = cls.module["spec"]["contracts"]
        Draft202012Validator.check_schema(contracts["input"])
        Draft202012Validator.check_schema(contracts["output"])
        cls.input_validator = Draft202012Validator(contracts["input"])
        cls.output_validator = Draft202012Validator(contracts["output"])
        fixture = cls.module["spec"]["fixtures"][0]
        cls.inputs = fixture["request"]["inputs"]
        cls.outputs = fixture["result"]["outputs"]

    def test_exact_pinned_module_inputs_and_validation_output_pass(self) -> None:
        self.input_validator.validate(self.inputs)
        self.output_validator.validate(self.outputs)

    def test_module_rejects_swapped_or_unpinned_literal_artifacts(self) -> None:
        swapped = copy.deepcopy(self.inputs)
        swapped["sgdCurrentOrfs"], swapped["sgdMappingManifest"] = (
            swapped["sgdMappingManifest"],
            swapped["sgdCurrentOrfs"],
        )
        with self.assertRaises(ValidationError):
            self.input_validator.validate(swapped)
        unpinned = copy.deepcopy(self.inputs)
        unpinned["sgdCurrentOrfs"]["artifacts"]["payload"] = "sha256:" + "0" * 64
        with self.assertRaises(ValidationError):
            self.input_validator.validate(unpinned)

    def test_validation_summary_is_closed_and_boolean_typed(self) -> None:
        extra = copy.deepcopy(self.outputs)
        extra["auditSummary"]["note"] = "not allowed"
        with self.assertRaises(ValidationError):
            self.output_validator.validate(extra)
        bool_as_int = copy.deepcopy(self.outputs)
        bool_as_int["auditSummary"]["validationOnly"] = 1
        with self.assertRaises(ValidationError):
            self.output_validator.validate(bool_as_int)

        production = copy.deepcopy(self.outputs)
        production["auditSummary"] = {
            "schema": "slp.sequence-statistics-feature-block-summary/v1",
            "validationOnly": False,
            "inputNames": [
                "sgdCurrentOrfs",
                "sgdMappingManifest",
                "sgdProteinSequences",
                "staticEntityUniverse",
            ],
            "heldRosterConsumed": False,
            "quantitativeOutcomesConsumed": False,
            "benchmarkDataConsumed": False,
            "rows": 7037,
            "featureDimension": 21,
            "presentValues": 147777,
            "excludedNonCurrentSequences": 109,
            "currentOrfsOutsideUniverse": 1426,
            "multiTargetProteinConsensus": 5,
        }
        self.output_validator.validate(production)
        production["auditSummary"]["source"] = {"unreviewed": True}
        with self.assertRaises(ValidationError):
            self.output_validator.validate(production)


if __name__ == "__main__":
    unittest.main()
