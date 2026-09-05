from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_slp11_count_world_response_query33.py"


def load():
    spec = importlib.util.spec_from_file_location("test_response_query_runner", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOD = load()


def write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return MOD.sha256(path)


def test_actual_panels_have_matched_width_and_only_query_arm_differs():
    arms, original = MOD.load_panels_by_arm()
    for source in ("k562", "rpe1"):
        zero = arms["static-zero33"][source]
        response = arms["response33"][source]
        assert zero.query_features.shape == (len(zero.query_ids), 610)
        assert zero.gene_action_features.shape == (len(zero.gene_ids), 610)
        np.testing.assert_array_equal(zero.query_features[:, :577], response.query_features[:, :577])
        np.testing.assert_array_equal(zero.query_features[:, 577:], 0)
        np.testing.assert_array_equal(zero.gene_action_features, response.gene_action_features)
        assert zero.counts is original[source].counts
        assert response.population_targets is original[source].population_targets


def test_internal_freeze_rejects_model_mutation_before_truth_access(tmp_path):
    protocol_sha = write(tmp_path / "protocol.json", b"protocol")
    model_hashes, references, forecasts = {}, {}, {}
    manifest = {"arms": {}, "sha256": {}}
    for arm in MOD.ARMS:
        model_name = f"arms/{arm}.safetensors"
        model_hashes[arm] = write(tmp_path / model_name, arm.encode())
        manifest["sha256"][model_name] = model_hashes[arm]
        manifest["arms"][arm] = {"modelPath": model_name, "panels": {}}
        references[arm] = {}
        for source in ("k562", "rpe1"):
            name = f"reference-{arm}-{source}.npz"
            digest = write(tmp_path / name, f"{arm}/{source}".encode())
            manifest["sha256"][name] = digest
            manifest["arms"][arm]["panels"][source] = {"referencePath": name}
            references[arm][source] = {"sha256": digest}
    (tmp_path / "artifact-manifest.json").write_text(json.dumps(manifest))
    for source in ("k562", "rpe1"):
        name = f"development-forecast-{source}.npz"
        forecasts[source] = {"path": name, "sha256": write(tmp_path / name, source.encode())}
    forecast_receipt = {
        "forecastsFrozenBeforeDevelopmentCountAccess": True,
        "developmentCountMembersOpened": False,
        "forecasts": forecasts,
        "armSpecificReferences": references,
    }
    (tmp_path / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json").write_text(json.dumps(forecast_receipt))
    frozen = {
        "protocolSha256": protocol_sha,
        "artifactManifestSha256": MOD.sha256(tmp_path / "artifact-manifest.json"),
        "forecastFreezeSha256": MOD.sha256(tmp_path / "FORECASTS-FROZEN-BEFORE-DEVELOPMENT.json"),
        "models": model_hashes,
    }
    (tmp_path / "FROZEN-FITTING-ONLY.json").write_text(json.dumps(frozen))
    MOD.validate_internal_freeze(tmp_path)
    with (tmp_path / "arms/static-zero33.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="model changed"):
        MOD.validate_internal_freeze(tmp_path)
