"""Contracts for target-free HepG2 world forecast generation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/generate_slp11_hepg2_world_forecasts.py"
SPEC = importlib.util.spec_from_file_location("hepg2_world_forecast_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def roster() -> dict[str, np.ndarray]:
    return {
        "population_ids": np.asarray([f"population-{i}" for i in range(2544)]),
        "source_construct_ids": np.asarray([f"construct-{i}" for i in range(2544)]),
        "action_ids": np.asarray([f"ENSG{i:011d}" for i in range(2544)]),
        "query_ids": np.asarray([f"ENSG{i:011d}" for i in range(7036)]),
        "fitting_gene_seen": np.zeros(2544, dtype=np.bool_),
        "source_query_measured": np.ones(7036, dtype=np.bool_),
        "context_control_query_observed": np.ones(7036, dtype=np.bool_),
    }


def test_roster_contract_preserves_record_and_query_axes() -> None:
    value = roster()
    MODULE.validate_roster(value)
    value["population_ids"][-1] = value["population_ids"][0]
    try:
        MODULE.validate_roster(value)
    except ValueError as error:
        assert "identity contract" in str(error)
    else:
        raise AssertionError("duplicate population identity was accepted")


def test_selected_npz_does_not_open_omitted_outcome_array(tmp_path: Path) -> None:
    path = tmp_path / "bounded.npz"
    np.savez(
        path,
        query_ids=np.asarray(["ENSG00000000001"]),
        forbidden_outcomes=np.asarray([{"must": "require pickle"}], dtype=object),
    )
    result = MODULE.selected_npz(path, ("query_ids",))
    assert result["query_ids"].tolist() == ["ENSG00000000001"]
    try:
        MODULE.selected_npz(path, ("forbidden_outcomes",))
    except ValueError as error:
        assert "Object arrays cannot be loaded" in str(error)
    else:
        raise AssertionError("pickle-backed omitted outcome was opened")


def test_generator_pins_failed_candidate_and_target_free_control() -> None:
    source = PATH.read_text(encoding="utf-8")
    assert 'candidate_report.get("advancement", {}).get("passed") is not False' in source
    assert 'int(control["perturbed_expression_rows_read"]) != 0' in source
    assert '"hepg2OutcomesRead": False' in source
    assert '"uncertaintyCalibrated": False' in source
    assert MODULE.EXPECTED["inference_source"] == (
        "da120d2dd8655d6cf90c684e5dbaa6a6aedd42bfefc1090f8bab121de6cd0d1b"
    )
