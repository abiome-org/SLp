import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modules/slp-1-1-count-world-evaluation-v1/evaluator.py"
SPEC = importlib.util.spec_from_file_location("slp11_count_world_evaluator_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RUNNER_SOURCE = ROOT / "scripts/evaluate_slp11_count_world_shared_context.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("slp11_count_world_evaluation_runner_test", RUNNER_SOURCE)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


def test_cp10k_aggregation_matches_independent_cell_average():
    raw = np.array([[1, 3, 0], [4, 0, 1], [0, 2, 2], [5, 1, 0]], dtype=np.int32)
    library = raw.sum(1)
    gene = np.array([1, 0, 1, 0])
    sums = np.zeros((2, 3), dtype=np.float64)
    cells = np.zeros(2, dtype=np.int64)
    MODULE.accumulate_cp10k(sums, cells, raw[:2], library[:2], gene[:2])
    MODULE.accumulate_cp10k(sums, cells, raw[2:], library[2:], gene[2:])
    expected = np.stack([
        np.log1p(np.mean(raw[gene == g] * (10000 / library[gene == g, None]), axis=0))
        for g in range(2)
    ])
    np.testing.assert_allclose(MODULE.aggregate_truth(sums, cells), expected, atol=1e-12)
    np.testing.assert_array_equal(cells, [2, 2])


def test_control_prediction_uses_gene_gem_composition_before_log():
    basal = np.array([[1.0, 9.0], [5.0, 1.0]])
    count = np.array([[3, 1], [0, 4]])
    expected = np.log1p(np.array([[2.0, 7.0], [5.0, 1.0]]))
    np.testing.assert_allclose(MODULE.control_prediction(basal, count), expected)


def test_stable_centering_removes_common_profile_and_flags_constant_forecast():
    rng = np.random.default_rng(3)
    anchor = rng.normal(size=(7, 23))
    effect = rng.normal(size=(7, 23))
    common = rng.normal(size=(1, 23))
    truth = anchor + effect
    prediction = anchor + effect + common
    score, _, correlation = MODULE.score_prediction(truth, prediction, anchor)
    assert score["independentlyQueryCenteredResidualPearson"] == pytest.approx(1.0)
    constant = np.broadcast_to(anchor[0] - anchor[0], anchor.shape) + anchor
    score2, _, correlation2 = MODULE.score_prediction(truth, constant, anchor)
    assert score2["independentlyQueryCenteredResidualPearson"] is None
    assert np.isnan(correlation2).all()


def test_forecast_contract_rejects_cell_and_identity_drift():
    arrays = {
        "gene_ids": np.array(["ENSG1", "ENSG2"]),
        "query_ids": np.array(["ENSGQ2", "ENSGQ1"]),
        "cell_count": np.array([3, 2]),
        "gem_group_ids": np.array([1, 2]),
        "gem_cell_count": np.array([[2, 1], [1, 1]]),
    }
    for key in MODULE.PREDICTION_KEYS:
        arrays[key] = np.zeros((2, 2))
    assert MODULE.validate_forecast_arrays(arrays) == (2, 2, 2)
    arrays["gem_cell_count"][1, 1] = 2
    with pytest.raises(ValueError, match="close exactly"):
        MODULE.validate_forecast_arrays(arrays)


def test_advancement_requires_every_fixed_check():
    def item(mse, r):
        return {"geneProfileMse": mse, "independentlyQueryCenteredResidualPearson": r}
    metrics = {
        source: {
            "joint_prediction": item(0.80, 0.30),
            "anchored_mean_prediction": item(1.0, None),
            "static_ridge_prediction": item(0.90, 0.20),
            "k562_only_prediction": item(0.90, 0.25),
        }
        for source in ("k562", "rpe1")
    }
    assert MODULE.advancement(metrics)["passes"]
    metrics["rpe1"]["joint_prediction"] = item(0.895, 0.30)
    assert not MODULE.advancement(metrics)["passes"]


def test_authoritative_freeze_hashes_every_dependency_before_access(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    artifact = tmp_path / "run"
    artifact.mkdir()
    sections = {
        "forecasts": ("k562", "rpe1"),
        "models": ("k562-only", "joint-alternating"),
        "references": ("k562", "rpe1"),
        "baselines": ("k562", "rpe1"),
        "routingMetadata": ("k562", "rpe1"),
    }
    freeze = {
        "forecastsFrozenBeforeDevelopmentCountAccess": True,
        "developmentCountMembersOpened": False,
        "testOpened": False,
    }
    for section, names in sections.items():
        freeze[section] = {}
        for name in names:
            path = artifact / f"{section}-{name}.bin"
            path.write_bytes(f"{section}-{name}".encode())
            item = {"path": path.name, "sha256": RUNNER.sha256(path)}
            if section == "forecasts":
                expected = RUNNER.EXPECTED[name]
                item.update({
                    "sourceId": expected["sourceId"],
                    "contextId": expected["contextId"],
                    "genes": expected["genes"],
                    "queries": expected["queries"],
                    "cellsRepresentedByMetadata": expected["cells"],
                })
            freeze[section][name] = item
    (artifact / RUNNER.FREEZE_NAME).write_text(__import__("json").dumps(freeze))
    _, pins = RUNNER.validate_freeze(artifact, {"pins": {}})
    assert len(pins) == 11
    (artifact / "models-k562-only.bin").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash mismatch before development access"):
        RUNNER.validate_freeze(artifact, {"pins": {}})
