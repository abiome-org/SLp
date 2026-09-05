"""Focused contracts for the frozen ESM2-t33 feasibility profile."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/profile_slp11_esm2_t33.py"
SPEC = importlib.util.spec_from_file_location("esm2_t33_profile_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_representative_selection_is_length_then_stable_identity() -> None:
    lengths = np.arange(1, 101, dtype=np.int64)
    identifiers = [f"ENSG{i:011d}" for i in range(100)]
    first = MODULE.representative_indices(lengths, identifiers)
    second = MODULE.representative_indices(lengths, identifiers)
    assert first == second
    assert [int(lengths[index]) for index in first] == [1, 11, 26, 51, 75, 90, 95, 99, 100]


def test_runtime_curve_extrapolates_exact_window_lengths() -> None:
    observations = [
        {"residues": float(length), "seconds": 0.01 + 1e-6 * length**2}
        for length in (100, 300, 500, 800, 1022)
    ]
    windows = np.asarray([100, 500, 1022], dtype=np.int64)
    report = MODULE.fit_runtime_curve(observations, windows)
    expected = sum(0.01 + 1e-6 * length**2 for length in windows)
    np.testing.assert_allclose(report["estimatedInferenceSeconds"], expected, rtol=1e-10)
    assert report["conservativeSecondsWith25PercentMargin"] == (
        report["estimatedInferenceSeconds"] * 1.25
    )


def test_source_and_rights_pin_mit_revision_and_exact_safetensors() -> None:
    source = yaml.safe_load(
        (ROOT / "sources/esm2-t33-650m-ur50d-static-protein-model.yaml").read_text(
            encoding="utf-8"
        )
    )
    rights = yaml.safe_load(
        (ROOT / "rights/esm2-t33-650m-ur50d-mit.yaml").read_text(encoding="utf-8")
    )
    assert source["source"]["revision"] == MODULE.MODEL_REVISION
    assert rights["revision"] == MODULE.MODEL_REVISION
    assert rights["license"] == "MIT"
    assert rights["trainingAllowed"] is True
    model_file = next(item for item in source["allowlist"] if item["name"] == "model.safetensors")
    assert model_file["sha256"] == MODULE.MODEL_FILES["model.safetensors"][1]
    assert model_file["bytes"] == MODULE.MODEL_FILES["model.safetensors"][0]
