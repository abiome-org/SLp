from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HELPER = load("slp11_nystrom_dimension_test", ROOT / "scripts/run_slp11_nystrom_rbf_baseline.py")
WRAPPER = load("slp11_bp_nystrom_test", ROOT / "scripts/run_slp11_bp_nystrom_screen.py")


def test_generic_nystrom_accepts_augmented_dimension() -> None:
    generator = np.random.default_rng(19)
    ids = tuple(f"ENSG{index:011d}" for index in range(520))
    values = generator.normal(size=(520, 1285)).astype(np.float32)
    model, report = HELPER.fit_nystrom(ids, values, bandwidth_sample=32)
    assert model.feature_mean.shape == (1285,)
    assert report["landmarks"] == 512


def test_query_centering_removes_shared_profile() -> None:
    truth = np.asarray([[1.0, 4.0, 2.0], [3.0, 2.0, 8.0], [7.0, 6.0, 5.0]])
    prediction = truth + np.asarray([10.0, -5.0, 3.0])
    assert WRAPPER.score_profiles(prediction, truth)[
        "independentlyQueryCenteredPearson"
    ] == pytest.approx(1.0)


def test_generic_helper_preserves_frozen_1156_forecast() -> None:
    artifact = ROOT / "results/slp11-transition/human-gwps-nystrom-rbf512-physical-seed731-v1"
    if not artifact.exists():
        pytest.skip("local frozen Nyström artifact unavailable")
    with np.load(
        ROOT / "data/derived/slp11-human-gwps-fixed-panel-context-v1/replogle-k562-rpe1-gwps-complete-panel-development-v2-fixed-control-context.npz",
        allow_pickle=False,
    ) as data:
        validation = data["split_validation"].astype(np.int64)
        action_ids = data["action_ids"].astype(str)
        contexts = data["context_index"].astype(np.int64)
    local = np.flatnonzero(contexts[validation] == 0)
    genes = action_ids[validation[local]]
    with np.load(
        ROOT / "data/derived/slp11-human-physical/direct-experiments700-v1/human-esm-go-physical-features.npz",
        allow_pickle=False,
    ) as pack:
        entity_ids = pack["entity_id"].astype(str)
        feature_values = pack["feature_values"].astype(np.float32)
    feature_index = {gene: index for index, gene in enumerate(entity_ids)}
    values = np.stack([feature_values[feature_index[gene]] for gene in genes])
    with np.load(artifact / "model-context-0.npz", allow_pickle=False) as saved:
        kernel = HELPER.NystromMap(
            feature_mean=saved["feature_mean"],
            feature_scale=saved["feature_scale"],
            bandwidth=float(saved["bandwidth"]),
            landmark_ids=tuple(saved["landmark_ids"].astype(str)),
            landmarks=saved["standardized_landmarks"],
            kernel_basis=saved["kernel_basis"],
            eigenvalues=saved["kernel_eigenvalues"],
        )
        mapped = kernel.transform(values)
        rotated = (mapped - saved["ridge_feature_mean"]) @ saved["ridge_eigenvectors"]
        prediction = saved["target_mean"] + (
            rotated / (saved["ridge_eigenvalues"] + float(str(saved["selected_alpha"])))
        ) @ saved["ridge_rhs"]
    with np.load(artifact / "development-predictions.npz", allow_pickle=False) as frozen:
        maximum_error = float(np.max(np.abs(prediction - frozen["mean"][local])))
    assert maximum_error <= 1e-4
