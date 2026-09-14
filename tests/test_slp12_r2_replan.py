"""The selected checkpoint schedule must fit the original allocation."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("r2_replan", Path(__file__).resolve().parents[1] / "scripts/slp12_r2_replan.py")
replan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replan)


def setup_budget():
    state = {"created_at": 0, "terminate_at": 191000, "gpu_and_disk_hourly_estimate_usd": .761918,
             "r2_first_month_reserve_usd": 9.25, "shutdown_reserve_usd": .25, "total_campaign_ceiling_usd": 50}
    recipes = {name: {"adapt": {"updates": 4000, "stop_at_update": 1000}} for name in replan.FAMILIES}
    recipes["mixed_pretraining"]["pretrain"] = {"updates": 32000, "stop_at_update": 8000}
    return state, recipes


def test_retained_point_preserves_schedule_and_counts_all_inner_outer_fits():
    state, recipes = setup_budget()
    original = copy.deepcopy(recipes)
    cost = replan.estimate(state, recipes, now=20000, pretrain_seconds=.1, adapt_seconds=.08)
    assert recipes == original
    assert cost["pretraining_updates_total_with_outer_refits_upper"] == 480000
    assert cost["adaptation_updates_total_with_probes_and_outer_refits"] == 180000
    assert cost["total_campaign_estimate_including_reserve_usd"] < 50
    assert cost["elapsed_gpu_disk_estimate_usd"] > 4


def test_full_32k_suite_cannot_escape_cap_or_original_deadline():
    state, recipes = setup_budget()
    del recipes["mixed_pretraining"]["pretrain"]["stop_at_update"]
    with pytest.raises(ValueError, match="remaining allocation"):
        replan.estimate(state, recipes, now=20000, pretrain_seconds=.09, adapt_seconds=.08)
    state["total_campaign_ceiling_usd"] = 1000
    with pytest.raises(ValueError, match="remaining allocation"):
        replan.estimate(state, recipes, now=20000, pretrain_seconds=.09, adapt_seconds=.08)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_timing_cannot_disable_limits(value):
    state, recipes = setup_budget()
    with pytest.raises(ValueError, match="throughput"):
        replan.estimate(state, recipes, now=20000, pretrain_seconds=value, adapt_seconds=.08)


def test_endpoint_cannot_exceed_captured_schedule():
    with pytest.raises(ValueError, match="endpoint"):
        replan.phase_end({"adapt": {"updates": 1000, "stop_at_update": 4000}}, "adapt")
