"""Pre-access and fixed-rule checks for reserved human molecular confirmation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_slp11_human_molecular_confirmation.py"
SPEC = importlib.util.spec_from_file_location("run_slp11_human_molecular_confirmation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONFIRMATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIRMATION)


def test_test_loader_requires_frozen_protocol_and_synthetic_checks(tmp_path: Path) -> None:
    test_path = tmp_path / "test.npz"
    np.savez(test_path, targets=np.ones((1, 1)))
    protocol = tmp_path / "protocol.json"
    checks = tmp_path / "checks.json"
    with pytest.raises(CONFIRMATION.MolecularConfirmationError, match="must precede"):
        CONFIRMATION.load_test_after_protocol(test_path, protocol, checks)
    protocol.write_text("{}")
    checks.write_text(json.dumps({"passed": False, "testArtifactOpened": False}))
    with pytest.raises(CONFIRMATION.MolecularConfirmationError, match="incomplete"):
        CONFIRMATION.load_test_after_protocol(test_path, protocol, checks)


def test_fixed_rule_uses_unchanged_point_thresholds() -> None:
    passed = CONFIRMATION.fixed_rule(-1.03, -1.0, -1.01, 0.10)
    assert passed["deltaNllVsMean"] == pytest.approx(0.03)
    assert passed["deltaNllVsRidge"] == pytest.approx(0.02)
    assert passed["passed"] is True
    failed = CONFIRMATION.fixed_rule(-1.029, -1.0, -1.01, 0.20)
    assert failed["passed"] is False
