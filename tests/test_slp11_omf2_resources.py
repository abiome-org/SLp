"""Pin active project governance resources to the vendored OMF 2 contracts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_FACTORY = ROOT / "data" / "tooling" / "omf-2.0-75f002b" / "factory"


@pytest.fixture(scope="module")
def omf_modules():
    if not (UPSTREAM_FACTORY / "omf" / "schema_registry.py").is_file():
        pytest.skip("ignored pinned OMF 2 source checkout is not present")
    sys.path.insert(0, str(UPSTREAM_FACTORY))
    try:
        try:
            registry = importlib.import_module("omf.schema_registry").default_registry
            policy = importlib.import_module("omf.policy")
        except ModuleNotFoundError as error:
            pytest.skip(f"OMF 2 schema dependency is unavailable: {error.name}")
        yield registry, policy
    finally:
        sys.path.remove(str(UPSTREAM_FACTORY))


def _document(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_active_project_policy_and_bindings_validate_with_omf2(omf_modules) -> None:
    registry, _ = omf_modules
    resources = [ROOT / "omf.yaml"]
    resources.extend(sorted((ROOT / "policies").glob("*.yaml")))
    resources.extend(sorted((ROOT / "bindings").glob("*.yaml")))

    assert resources
    for path in resources:
        document = _document(path)
        validated = registry.validate(document)
        assert validated["apiVersion"] == "omf.dev/v1alpha1"
        assert validated["kind"] == document["kind"]


def test_policy_loads_under_omf2_and_retains_only_enforced_requirements(
    omf_modules,
) -> None:
    registry, policy_module = omf_modules
    project = registry.validate(_document(ROOT / "omf.yaml"))
    policy = policy_module.ProjectPolicy.load(ROOT, project)

    assert policy.enforced
    assert policy.dirty_worktree == "archive"
    assert policy.config == {
        "dirtyWorktree": "archive",
        "promotion": {"requireEvaluationPass": True},
    }
    serialized = (ROOT / "policies" / "default.yaml").read_text(encoding="utf-8")
    for retired in ("unsignedModules", "sync:", "requireCompleteLineage"):
        assert retired not in serialized


def test_local_linux_binding_preserves_bounded_executor_resources() -> None:
    binding = _document(ROOT / "bindings" / "local-linux.yaml")
    assert binding["spec"] == {
        "executor": "local",
        "resources": {
            "timeoutSeconds": 3600,
            "processes": 512,
            "fileSizeBytes": 107374182400,
        },
    }
