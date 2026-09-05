"""Focused contracts for the genome-scale development pilot launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_slp11_gwps_genome_scale_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_slp11_gwps_genome_scale_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def test_partition_contract_rejects_intervention_overlap() -> None:
    actions = np.asarray([f"G{index}" for index in range(13_058)])
    train = np.arange(10_719)
    validation = np.arange(10_719, 13_058)
    PILOT.validate_partitions(actions, train, validation, np.empty(0, dtype=np.int64), 13_058)
    actions[-1] = actions[0]
    with pytest.raises(PILOT.GenomeScalePilotError, match="crosses"):
        PILOT.validate_partitions(
            actions, train, validation, np.empty(0, dtype=np.int64), 13_058
        )


def test_partition_contract_rejects_test_or_nonexhaustive_rows() -> None:
    actions = np.asarray([f"G{index}" for index in range(13_058)])
    with pytest.raises(PILOT.GenomeScalePilotError, match="counts"):
        PILOT.validate_partitions(
            actions,
            np.arange(10_719),
            np.arange(10_719, 13_057),
            np.asarray([13_057]),
            13_058,
        )
