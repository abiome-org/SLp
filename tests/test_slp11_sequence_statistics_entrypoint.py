from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import ANY, patch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-sequence-statistics-feature-block-v1"
INPUT_NAMES = {
    "staticEntityUniverse",
    "sgdProteinSequences",
    "sgdCurrentOrfs",
    "sgdMappingManifest",
}
OUTPUT_KEYS = {
    "auditSummary",
    "archiveSha256",
    "auditSha256",
    "manifestSha256",
    "entityRowsSha256",
    "valuesNpySha256",
    "presentNpySha256",
    "sequenceProvenanceSha256",
    "excludedNonCurrentSha256",
    "featureDefinitionSha256",
    "entityKeySetSha256",
    "rows",
    "featureDimension",
    "presentValues",
    "excludedNonCurrentSequences",
    "currentOrfsOutsideUniverse",
    "multiTargetProteinConsensus",
}


class _Result:
    def __init__(self, **values: object) -> None:
        self.protocol = "omf.module/v1"
        self.outputs: dict[str, object] = {}
        self.state: dict[str, object] = {}
        self.metrics: dict[str, object] = {}
        self.artifacts: list[dict[str, object]] = []
        self.__dict__.update(values)


def _load_entrypoint() -> types.ModuleType:
    sdk = types.ModuleType("omf.sdk")
    sdk.ProtocolRequest = object
    sdk.ProtocolResult = _Result
    sdk.main = lambda _handlers: 0
    package = types.ModuleType("omf")
    package.sdk = sdk
    spec = importlib.util.spec_from_file_location(
        "slp_sequence_statistics_entrypoint_test", MODULE / "main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"omf": package, "omf.sdk": sdk}):
        spec.loader.exec_module(module)
    return module


def _request(
    operation: str,
    *,
    inputs: dict[str, object] | None = None,
    config: dict[str, object] | None = None,
    state: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        operation=operation,
        inputs={name: {"input": name} for name in INPUT_NAMES}
        if inputs is None
        else inputs,
        config={} if config is None else config,
        state={} if state is None else state,
        context={"runId": "wiring-test"},
    )


class _Bounds:
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **values: object) -> None:
        self.values = values
        type(self).calls.append(values)
        if any(type(value) is not int for value in values.values()):
            raise ValueError("bounds require integers")


def _feature_block_stub(**members: object) -> types.ModuleType:
    stub = types.ModuleType("feature_block")
    stub.Bounds = members.pop("Bounds", _Bounds)
    for name, value in members.items():
        setattr(stub, name, value)
    return stub


class SequenceStatisticsEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        _Bounds.calls.clear()
        self.entrypoint = _load_entrypoint()

    def test_validate_checks_stateless_request_surface_without_materializing(self) -> None:
        def forbidden_resolver(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("validate must not resolve or read input paths")

        feature_block = _feature_block_stub(
            resolve_pinned_dataset=forbidden_resolver,
            resolve_literal_artifact=forbidden_resolver,
            build_sequence_feature_block=forbidden_resolver,
        )
        request = _request("validate", config={"maxRecords": 9})
        with patch.dict(sys.modules, {"feature_block": feature_block}):
            result = self.entrypoint.validate(request)

        self.assertEqual(result.status, "ok")
        self.assertEqual(set(result.outputs), OUTPUT_KEYS)
        self.assertEqual(
            result.outputs["auditSummary"],
            {
                "schema": "slp.sequence-statistics-feature-block-validation/v1",
                "validationOnly": True,
            },
        )
        self.assertEqual(result.outputs["featureDimension"], 21)
        self.assertEqual(_Bounds.calls[-1]["max_records"], 9)
        self.assertEqual(_Bounds.calls[-1]["max_manifest_bytes"], 1_048_576)

        invalid_requests = (
            _request("run"),
            _request("validate", inputs={}),
            _request(
                "validate",
                inputs={
                    **{name: {} for name in INPUT_NAMES},
                    "molecularValidation": {},
                },
            ),
            _request("validate", config={"newBehavior": 1}),
            _request("validate", config={"maxRecords": True}),
            _request("validate", state={"resume": "forbidden"}),
        )
        with patch.dict(sys.modules, {"feature_block": feature_block}):
            for invalid in invalid_requests:
                with self.subTest(request=invalid), self.assertRaises(ValueError):
                    self.entrypoint.validate(invalid)

    def test_run_wires_pinned_inputs_bounds_outputs_metrics_and_artifacts(self) -> None:
        resolver_calls: list[tuple[str, object, str]] = []
        builder_calls: list[tuple[object, ...]] = []

        def resolve_dataset(value: object, name: str) -> str:
            resolver_calls.append(("dataset", value, name))
            return f"resolved-dataset:{name}"

        def resolve_artifact(value: object, name: str) -> str:
            resolver_calls.append(("artifact", value, name))
            return f"resolved-artifact:{name}"

        audit = {
            "schema": "slp.sequence-statistics-feature-block-audit/v1",
            "accessBoundary": {
                "heldRosterConsumed": False,
                "quantitativeOutcomesConsumed": False,
                "benchmarkDataConsumed": False,
            },
        }
        hashes = {
            "archiveSha256": "1" * 64,
            "auditSha256": "2" * 64,
            "manifestSha256": "3" * 64,
            "entityRowsSha256": "4" * 64,
            "valuesNpySha256": "5" * 64,
            "presentNpySha256": "6" * 64,
            "sequenceProvenanceSha256": "7" * 64,
            "excludedNonCurrentSha256": "8" * 64,
            "featureDefinitionSha256": "9" * 64,
            "entityKeySetSha256": "a" * 64,
        }
        counts = {
            "rows": 7_037,
            "featureDimension": 21,
            "presentValues": 147_777,
            "excludedNonCurrentSequences": 109,
            "currentOrfsOutsideUniverse": 1_426,
            "multiTargetProteinConsensus": 5,
        }

        def build(*args: object) -> dict[str, object]:
            builder_calls.append(args)
            destination = args[4]
            assert isinstance(destination, Path)
            destination.mkdir(parents=True)
            (destination / "sequence-feature-block.tar").touch()
            (destination / "sequence-feature-block-audit.json").touch()
            return {**hashes, **counts, "audit": audit}

        feature_block = _feature_block_stub(
            resolve_pinned_dataset=resolve_dataset,
            resolve_literal_artifact=resolve_artifact,
            build_sequence_feature_block=build,
        )
        request = _request("run", config={"maxArchiveBytes": 2_048})
        with TemporaryDirectory() as temporary:
            result_file = Path(temporary) / "stage" / "result.json"
            with (
                patch.dict(sys.modules, {"feature_block": feature_block}),
                patch.dict(os.environ, {"OMF_RESULT_FILE": str(result_file)}),
            ):
                result = self.entrypoint.run(request)

            self.assertEqual(
                builder_calls,
                [
                    (
                        "resolved-dataset:staticEntityUniverse",
                        "resolved-dataset:sgdProteinSequences",
                        "resolved-artifact:sgdCurrentOrfs",
                        "resolved-artifact:sgdMappingManifest",
                        result_file.parent
                        / "sequence-statistics-feature-block-v1",
                        ANY,
                    )
                ],
            )

        self.assertEqual(
            [(kind, name) for kind, _value, name in resolver_calls],
            [
                ("dataset", "staticEntityUniverse"),
                ("dataset", "sgdProteinSequences"),
                ("artifact", "sgdCurrentOrfs"),
                ("artifact", "sgdMappingManifest"),
            ],
        )
        self.assertEqual(_Bounds.calls[-1]["max_archive_bytes"], 2_048)
        self.assertEqual(result.status, "ok")
        self.assertEqual(set(result.outputs), OUTPUT_KEYS)
        self.assertEqual(
            result.outputs["auditSummary"],
            {
                "schema": "slp.sequence-statistics-feature-block-summary/v1",
                "validationOnly": False,
                "inputNames": sorted(INPUT_NAMES),
                "heldRosterConsumed": False,
                "quantitativeOutcomesConsumed": False,
                "benchmarkDataConsumed": False,
                **counts,
            },
        )
        self.assertEqual(
            {key: result.outputs[key] for key in hashes | counts},
            hashes | counts,
        )
        self.assertEqual(
            result.metrics,
            {
                "rows": 7_037,
                "feature_dimension": 21,
                "present_values": 147_777,
                "excluded_non_current_sequences": 109,
                "multi_target_protein_consensus": 5,
            },
        )
        self.assertEqual(
            result.artifacts,
            [
                {
                    "name": "sequenceStatisticsFeatureBlock",
                    "kind": "dataset",
                    "path": (
                        "sequence-statistics-feature-block-v1/"
                        "sequence-feature-block.tar"
                    ),
                },
                {
                    "name": "sequenceStatisticsFeatureBlockAudit",
                    "kind": "audit",
                    "path": (
                        "sequence-statistics-feature-block-v1/"
                        "sequence-feature-block-audit.json"
                    ),
                },
            ],
        )

    def test_run_rejects_missing_result_placement_environment(self) -> None:
        feature_block = _feature_block_stub(
            resolve_pinned_dataset=lambda value, name: (value, name),
            resolve_literal_artifact=lambda value, name: (value, name),
            build_sequence_feature_block=lambda *_args: {},
        )
        with (
            patch.dict(sys.modules, {"feature_block": feature_block}),
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "OMF_RESULT_FILE"),
        ):
            self.entrypoint.run(_request("run"))

    def test_admitted_source_is_self_contained_and_dependency_free(self) -> None:
        source = (MODULE / "main.py").read_text(encoding="utf-8")
        import_nodes = [
            node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)
        ]
        imports = {node.module for node in import_nodes}
        self.assertEqual(imports, {"__future__", "pathlib", "omf.sdk", "feature_block"})
        self.assertTrue(all(node.level == 0 for node in import_nodes))
        self.assertTrue((MODULE / "feature_block.py").is_file())
        self.assertEqual((MODULE / "requirements.lock").read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
