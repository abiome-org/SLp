from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script():
    path = (
        Path(__file__).parents[1]
        / "modules"
        / "slp-1-1-world-transition-v1"
        / "norman_data.py"
    )
    spec = importlib.util.spec_from_file_location("slp11_norman_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NORMAN = _load_script()


def test_combination_routing_uses_most_restricted_constituent() -> None:
    candidates = [f"ENSG{index:011d}" for index in range(1, 2000)]
    by_role = {}
    for candidate in candidates:
        by_role.setdefault(NORMAN._split_name(candidate), candidate)
    assert set(by_role) == {"train", "validation", "test"}
    assert NORMAN.route_actions((by_role["train"], by_role["train"])) == "train"
    assert NORMAN.route_actions((by_role["train"], by_role["validation"])) == "validation"
    assert NORMAN.route_actions((by_role["validation"], by_role["test"])) == "test"


def test_author_guide_identity_parsing_keeps_one_or_two_actions() -> None:
    assert NORMAN._parse_target_symbols("GENE1_NegCtrl0__GENE1_NegCtrl0") == ("GENE1",)
    assert NORMAN._parse_target_symbols("GENE1_GENE2__GENE1_GENE2") == (
        "GENE1",
        "GENE2",
    )
    assert NORMAN._parse_target_symbols("NegCtrl0_NegCtrl1__x") == ()
    assert NORMAN._parse_target_symbols("RHOXF2_NegCtrl0__x")[0] == "RHOXF2B"
