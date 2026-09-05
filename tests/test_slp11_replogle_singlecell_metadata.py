from __future__ import annotations

import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1] / "scripts/audit_replogle_k562_singlecell_metadata.py"
SPEC = importlib.util.spec_from_file_location("replogle_sc_metadata", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_intervention_split_is_stable_and_fail_closed_for_test():
    assert MODULE.action_role("") == "control"
    observed = {MODULE.action_role(f"ENSG{i:011d}") for i in range(100)}
    assert observed == {"train", "validation", "test-excluded"}
    assert MODULE.action_role("ENSG00000158545") == MODULE.action_role("ENSG00000158545")


def test_reconstruction_split_never_routes_held_interventions():
    for role in ("validation", "test-excluded"):
        assert MODULE.reconstruction_role("AAAC-1", role) == "none"
    assert MODULE.reconstruction_role("AAAC-1", "train") in {"train", "validation"}
    assert MODULE.reconstruction_role("AAAC-1", "control") in {"train", "validation"}
