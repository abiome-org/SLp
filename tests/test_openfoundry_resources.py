"""Exercise SLp's working definitions against the pinned OpenFoundry runtime.

Run with data/tooling/openfoundry-runtime/bin/python -m unittest discover
-s tests -p test_openfoundry_resources.py. No datasets or accelerator are needed.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def document(path):
    return yaml.safe_load(path.read_text())


@unittest.skipUnless(importlib.util.find_spec("openfoundry"), "use the pinned factory runtime")
class OpenFoundryContracts(unittest.TestCase):
    def test_all_authored_resource_manifests_validate(self):
        from openfoundry.schema_registry import default_registry

        paths = [ROOT / "openfoundry.yaml"]
        for directory in ("policies", "bindings", "evaluations", "modules", "workloads"):
            paths.extend((ROOT / directory).rglob("*.yaml"))
        for path in paths:
            value = document(path)
            if not isinstance(value, dict) or "apiVersion" not in value:
                continue
            with self.subTest(path=str(path.relative_to(ROOT))):
                default_registry.validate(value)

    def test_all_experiments_and_generated_resources_validate(self):
        from openfoundry.experiment_definition import evaluation_spec, read_definition
        from openfoundry.schema_registry import default_registry

        for path in sorted(ROOT.glob("experiment*.yaml")):
            with self.subTest(path=path.name):
                definition = read_definition(path)
                default_registry.validate(evaluation_spec(definition))
                self.assertEqual(definition.train.network, "deny")
                self.assertEqual(definition.evaluate.network, "deny")

    def test_promotion_profiles_enforce_bounds_and_preserve_authorization(self):
        from openfoundry.policy import ProjectPolicy, promotion_gate

        project = document(ROOT / "openfoundry.yaml")
        default = ProjectPolicy.load(ROOT, project)
        paths = [
            ROOT / "policies/default.yaml",
            *sorted((ROOT / "policies/profiles").rglob("default.yaml")),
        ]
        for path in paths:
            with self.subTest(profile=str(path.parent.relative_to(ROOT))):
                selected = copy.deepcopy(project)
                selected["spec"]["extensions"] = {
                    "policyDirectory": str(path.parent.relative_to(ROOT))
                }
                policy = ProjectPolicy.load(ROOT, selected)
                self.assertEqual(policy.engine.rules, default.engine.rules)
                self.assertEqual(policy.dirty_worktree, "archive")
                self.assertEqual(
                    policy.authorize({"actor": "slp-researcher", "resource": "abiome/slp"}).outcome,
                    "allow",
                )
                self.assertEqual(
                    policy.authorize({"actor": "unknown", "resource": "abiome/slp"}).outcome, "deny"
                )
                requirements = policy.config["promotion"]
                bounds = requirements["thresholds"]
                scores = {
                    name: limits.get("minimum", limits.get("maximum"))
                    for name, limits in bounds.items()
                }
                evidence = {
                    "lineage_complete": True,
                    "rights_valid": True,
                    "evaluation_passed": True,
                    "metric_scores": scores,
                }
                self.assertEqual(promotion_gate(evidence, requirements).outcome, "allow")
                for name, limits in bounds.items():
                    invalid = dict(scores)
                    invalid.pop(name)
                    self.assertEqual(
                        promotion_gate(
                            {**evidence, "metric_scores": invalid}, requirements
                        ).outcome,
                        "deny",
                    )
                    invalid[name] = (
                        limits["minimum"] - 1 if "minimum" in limits else limits["maximum"] + 1
                    )
                    self.assertEqual(
                        promotion_gate(
                            {**evidence, "metric_scores": invalid}, requirements
                        ).outcome,
                        "deny",
                    )
                self.assertEqual(
                    promotion_gate({**evidence, "evaluation_passed": False}, requirements).outcome,
                    "deny",
                )

    def test_local_binding_keeps_bounded_resources(self):
        binding = document(ROOT / "bindings/local-linux.yaml")
        self.assertEqual(
            binding["spec"],
            {
                "executor": "local",
                "resources": {
                    "timeoutSeconds": 3600,
                    "processes": 512,
                    "fileSizeBytes": 107374182400,
                },
            },
        )

    def test_standalone_protocol_interoperates_with_upstream_sdk(self):
        from openfoundry.sdk import ProtocolRequest, ProtocolResult

        path = ROOT / "modules/slp-1-1-atlas-genotype-inventory/omf_protocol.py"
        spec = importlib.util.spec_from_file_location("atlas_openfoundry_protocol", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        request = ProtocolRequest(operation="validate", inputs={"example": 3})
        with tempfile.TemporaryDirectory() as directory:
            source, result = Path(directory) / "request.json", Path(directory) / "result.json"
            source.write_text(request.model_dump_json())
            self.assertEqual(
                module.dispatch({"validate": lambda req: {"outputs": req.inputs}}, source, result),
                0,
            )
            response = ProtocolResult.model_validate_json(result.read_bytes())
            self.assertEqual(response.outputs, {"example": 3})
            source.write_text(
                json.dumps({"protocol": "invalid.module/v1", "operation": "validate"})
            )
            self.assertEqual(module.dispatch({}, source, result), 1)
            self.assertEqual(
                ProtocolResult.model_validate_json(result.read_bytes()).status, "error"
            )


if __name__ == "__main__":
    unittest.main()
