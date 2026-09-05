from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modules/slp-1-1-count-world-response-query-inference-v1/inference.py"


def load():
    spec = importlib.util.spec_from_file_location("test_response_query_inference", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOD = load()


def test_static_actions_are_padded_normalized_and_input_is_unchanged():
    rng = np.random.default_rng(731)
    raw = rng.normal(size=(3, 2, 577))
    before = raw.copy()
    mean = np.concatenate((np.linspace(-1, 1, 577), np.zeros(33)))
    scale = np.concatenate((np.linspace(0.5, 2, 577), np.ones(33)))
    actual = MOD.normalize_static_actions(raw, mean, scale, 100.0)
    expected = np.concatenate(((raw - mean[:577]) / scale[:577], np.zeros((3, 2, 33))), 2)
    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=0, atol=0)
    np.testing.assert_array_equal(raw, before)


def test_nonzero_response_coordinate_normalizer_is_rejected():
    with pytest.raises(ValueError, match="normalizer"):
        MOD.normalize_static_actions(
            np.zeros((1, 577)), np.ones(610), np.ones(610), 10.0
        )


def test_wrong_raw_width_is_rejected():
    with pytest.raises(ValueError, match="static577"):
        MOD.normalize_static_actions(
            np.zeros((1, 610)), np.zeros(610), np.ones(610), 10.0
        )
