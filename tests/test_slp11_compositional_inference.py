import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
MODULE = Path(__file__).parents[1] / "modules" / "slp-1-1-compositional-state-v1"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def artifact(tmp_path):
    core = load_module("composition_inference_test_core", MODULE / "operator.py")
    config = core.Config(zero_init_delta=False)
    torch.manual_seed(19)
    model = core.CompositionalStateOperator(config).eval()
    (tmp_path / "protocol.json").write_text(json.dumps({"seeds": [731], "config": vars(config)}))
    rng = np.random.default_rng(19)
    basis, _ = np.linalg.qr(rng.normal(size=(40, 32)))
    basis = basis.T
    feature_mean = rng.normal(size=577)
    feature_scale = rng.uniform(0.05, 2.0, size=577)
    np.savez(tmp_path / "fold0-basis.npz", basis=basis, zscale=np.linspace(0.5, 1.5, 32),
             feature_mean=feature_mean, feature_scale=feature_scale,
             query_ids=np.asarray([f"q{i}" for i in range(40)]))
    safetensors.save_file(model.state_dict(), tmp_path / "fold0-observed_operator-seed731.safetensors")
    inference = load_module("composition_inference_test", MODULE / "inference.py")
    return inference.load(tmp_path, 0, 731)


def test_load_predict_is_symmetric_and_uses_stored_coordinates(tmp_path):
    predictor = artifact(tmp_path)
    rng = np.random.default_rng(23)
    ya, yb = rng.normal(size=40), rng.normal(size=40)
    fa, fb = rng.normal(size=577), rng.normal(size=577)
    first = predictor.predict(ya, yb, fa, fb)
    second = predictor.predict(yb, ya, fb, fa)
    assert first.shape == (40,)
    np.testing.assert_allclose(first, second, atol=1e-6, rtol=0)
    assert predictor.query_ids.tolist() == [f"q{i}" for i in range(40)]


def test_zero_delta_checkpoint_returns_exact_observed_additive(tmp_path):
    predictor = artifact(tmp_path)
    with torch.no_grad():
        predictor.model.delta_head[-1].weight.zero_()
        predictor.model.delta_head[-1].bias.zero_()
    ya, yb = np.arange(40, dtype=np.float32), np.arange(40, dtype=np.float32) / 3
    feature = np.zeros(577, dtype=np.float32)
    np.testing.assert_array_equal(predictor.predict(ya, yb, feature, feature),
                                  ya.astype(np.float64) + yb.astype(np.float64))


def test_rejects_wrong_query_or_feature_axis(tmp_path):
    predictor = artifact(tmp_path)
    with pytest.raises(ValueError, match="stored query axis"):
        predictor.predict(np.zeros(39), np.zeros(40), np.zeros(577), np.zeros(577))
    with pytest.raises(ValueError, match="577-dimensional"):
        predictor.predict(np.zeros(40), np.zeros(40), np.zeros(576), np.zeros(577))


def test_float64_feature_normalization_matches_direct_model_replay(tmp_path):
    predictor = artifact(tmp_path)
    rng = np.random.default_rng(31)
    ya, yb = rng.normal(size=40), rng.normal(size=40)
    raw_a, raw_b = rng.normal(size=577), rng.normal(size=577)
    actual = predictor.predict(ya, yb, raw_a, raw_b)

    normalized_a = ((raw_a.astype(np.float64) - predictor.feature_mean) /
                    predictor.feature_scale).astype(np.float32)
    normalized_b = ((raw_b.astype(np.float64) - predictor.feature_mean) /
                    predictor.feature_scale).astype(np.float32)
    ya32, yb32 = ya.astype(np.float32), yb.astype(np.float32)
    za = torch.tensor(((ya32 @ predictor.basis.T) / predictor.zscale)[None], dtype=torch.float32)
    zb = torch.tensor(((yb32 @ predictor.basis.T) / predictor.zscale)[None], dtype=torch.float32)
    zero = torch.zeros_like(za)
    mask = torch.tensor([[True, False]])
    aa, ab = torch.zeros(1, 2, 577), torch.zeros(1, 2, 577)
    aa[0, 0], ab[0, 0] = torch.from_numpy(normalized_a), torch.from_numpy(normalized_b)
    with torch.no_grad():
        residual = 0.5 * (
            (predictor.model(za, ab, mask) - za) - predictor.model(zero, ab, mask)
            + (predictor.model(zb, aa, mask) - zb) - predictor.model(zero, aa, mask)
        )
    expected = ya32.astype(np.float64) + yb32.astype(np.float64) + residual.numpy()[0] @ predictor.decoder
    np.testing.assert_array_equal(actual, expected)


def test_import_disables_global_mha_fastpath():
    assert not torch.backends.mha.get_fastpath_enabled()
