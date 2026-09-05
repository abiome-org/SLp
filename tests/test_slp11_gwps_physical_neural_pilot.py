"""Focused contracts for the frozen physical-neighbor neural pilot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_slp11_gwps_physical_neural_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_slp11_gwps_physical_neural_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def test_feature_extension_requires_exact_base_prefix() -> None:
    taxon = np.full(10_231, 9606, dtype=np.int64)
    ids = np.asarray([f"ENSG{index:011d}" for index in range(10_231)])
    base = np.zeros((10_231, 577), dtype=np.float32)
    physical = np.zeros((10_231, 1_156), dtype=np.float32)
    PILOT.validate_feature_extension(taxon, ids, base, taxon.copy(), ids.copy(), physical)
    physical[4, 7] = 1.0
    with pytest.raises(PILOT.PhysicalNeuralPilotError, match="base feature"):
        PILOT.validate_feature_extension(taxon, ids, base, taxon, ids, physical)


def test_rule_requires_primary_and_no_regression_in_every_context() -> None:
    results = {
        "a": {
            "development_rule_passed": True,
            "world": {
                "gene_macro_nll": -1.1,
                "gene_macro_profile_centroid_adjusted_pearson_mean": 0.2,
            },
        },
        "b": {
            "development_rule_passed": True,
            "world": {
                "gene_macro_nll": -0.9,
                "gene_macro_profile_centroid_adjusted_pearson_mean": 0.2,
            },
        },
    }
    prior = {
        "a": {
            "world": {
                "gene_macro_nll": -1.0,
                "gene_macro_profile_centroid_adjusted_pearson_mean": 0.1,
            }
        },
        "b": {
            "world": {
                "gene_macro_nll": -1.0,
                "gene_macro_profile_centroid_adjusted_pearson_mean": 0.1,
            }
        },
    }
    decision = PILOT.evaluate_rules(results, prior)
    assert decision["primaryRulePassedAllContexts"] is True
    assert decision["noRegressionPassedAllContexts"] is False
    assert decision["hypothesisPassed"] is False
