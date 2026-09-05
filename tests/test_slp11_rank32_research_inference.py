from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PATH = (
    ROOT / "modules/slp-1-1-reduced-rank-response-inference-v1/inference.py"
)
RESPONSE_PATH = ROOT / "modules/slp-1-1-reduced-rank-response-v1/response_model.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INFERENCE = load(INFERENCE_PATH, "test_rank32_research_inference")
RESPONSE = load(RESPONSE_PATH, "test_rank32_research_response")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, object]:
    (path / "source").mkdir(parents=True)
    shutil.copyfile(INFERENCE_PATH, path / "source/inference.py")
    shutil.copyfile(RESPONSE_PATH, path / "source/response_model.py")
    training_features = np.asarray(
        [[-1.0, 0.5], [0.0, -0.5], [1.0, 1.5], [2.0, -1.5]], np.float32
    )
    targets = np.column_stack(
        (
            -4.0 + 0.75 * training_features[:, 0],
            0.25 - 0.5 * training_features[:, 1],
            1.0 + training_features[:, 0] - training_features[:, 1],
        )
    )
    model = RESPONSE.fit(training_features, targets, rank=2, alpha=0.5)
    RESPONSE.save(
        path / "model-s.npz",
        model,
        query_ids=np.asarray(["q0", "q1", "q2"]),
        source_id="s",
    )
    basal = np.asarray([[1.0, 3.0, 8.0], [9.0, 7.0, 2.0]], np.float32)
    np.savez_compressed(
        path / "reference-s.npz",
        schema=np.asarray("slp.rank32-local-native-control-reference/v1"),
        source_id=np.asarray("s"),
        query_ids=np.asarray(["q0", "q1", "q2"]),
        context_ids=np.asarray(["c0", "c1"]),
        gem_group_ids=np.asarray([11, 12]),
        basal_rate=basal,
    )
    np.savez_compressed(
        path / "static-actions.npz",
        schema=np.asarray("slp.rank32-local-static-action-cache/v1"),
        entity_taxon=np.asarray([9606, 9606]),
        entity_id=np.asarray(["g0", "g1"]),
        feature_values=training_features[[0, 3]],
    )
    names = [
        "model-s.npz",
        "reference-s.npz",
        "static-actions.npz",
        "source/response_model.py",
    ]
    manifest = {
        "schema": "slp.rank32-local-research-inference-bundle/v1",
        "sources": {"s": {"model": "model-s.npz", "reference": "reference-s.npz"}},
        "sha256": {name: sha256(path / name) for name in names},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return training_features, basal, model


def test_control_is_rate_mixture_before_log_and_does_not_mutate_weights(tmp_path):
    _, basal, _ = make_bundle(tmp_path)
    predictor = INFERENCE.ResearchPredictor(tmp_path, "s")
    weights = np.asarray([[1.0, 3.0], [2.0, 2.0]], np.float64)
    original = weights.copy()
    result = predictor.control(weights, query_indices=np.asarray([2, 0]))
    normalized = weights / weights.sum(1, keepdims=True)
    expected_rate = normalized @ basal[:, [2, 0]]
    np.testing.assert_array_equal(weights, original)
    np.testing.assert_allclose(result["control_rate_cp10k"], expected_rate)
    np.testing.assert_allclose(result["control_log1p_cp10k"], np.log1p(expected_rate))
    np.testing.assert_array_equal(result["query_ids"], np.asarray(["q2", "q0"]))


def test_feature_and_gene_paths_are_identical_and_keep_signed_output(tmp_path):
    features, _, model = make_bundle(tmp_path)
    predictor = INFERENCE.ResearchPredictor(tmp_path, "s")
    weights = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    feature_result = predictor.predict_features(features[[0, 3]], weights)
    gene_result = predictor.predict_genes(np.asarray(["g0", "g1"]), weights)
    np.testing.assert_array_equal(
        feature_result["mean_log1p_cp10k"], gene_result["mean_log1p_cp10k"]
    )
    expected_residual = model.predict(features[[0, 3]])
    np.testing.assert_allclose(
        feature_result["residual_log1p_profile"], expected_residual, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        feature_result["mean_log1p_cp10k"],
        feature_result["control_log1p_cp10k"] + expected_residual,
        rtol=0,
        atol=0,
    )
    assert np.any(feature_result["mean_log1p_cp10k"] < 0)


def test_reload_and_checksum_failure_are_fail_closed(tmp_path):
    make_bundle(tmp_path)
    first = INFERENCE.ResearchPredictor(tmp_path, "s")
    expected = first.predict_genes(["g1"], [[0.5, 0.5]])["mean_log1p_cp10k"]
    second = INFERENCE.ResearchPredictor(tmp_path, "s")
    np.testing.assert_array_equal(
        second.predict_genes(["g1"], [[0.5, 0.5]])["mean_log1p_cp10k"], expected
    )
    with (tmp_path / "static-actions.npz").open("ab") as stream:
        stream.write(b"mutation")
    with pytest.raises(ValueError, match="checksum"):
        INFERENCE.ResearchPredictor(tmp_path, "s")


def test_invalid_context_or_feature_inputs_fail_closed(tmp_path):
    make_bundle(tmp_path)
    predictor = INFERENCE.ResearchPredictor(tmp_path, "s")
    with pytest.raises(ValueError, match="positive rows"):
        predictor.control([[0.0, 0.0]])
    with pytest.raises(ValueError, match="feature and GEM-weight rows"):
        predictor.predict_features(np.zeros((2, 2), np.float32), [[1.0, 0.0]])
    with pytest.raises(KeyError, match="absent"):
        predictor.predict_genes(["not-a-gene"], [[1.0, 0.0]])
