"""Final aggregation must preserve benchmark weighting and inner selection."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("r2_summary", Path(__file__).resolve().parents[1] / "scripts/slp12_r2_summarize.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture():
    families = ["feature_mlp", "no_pretraining", "mixed_pretraining"]
    plan = {"schema": "slp.r2-training-packet/v1", "protocols": [], "variants": {family: {} for family in families},
            "development_reuse": "Synthetic fixture; not a model result"}
    journal = {"plan_sha256": "pinned", "folds": {}}
    for i in range(30):
        name, benchmark = f"fold{i}", "large" if i < 20 else "small"
        plan["protocols"].append({"name": name, "benchmark": benchmark, "metadata_sha256": name})
        fold = {"selection": {"selected": "no_pretraining"}, "families": {}}
        for family, gain in zip(families, [.2, 0, .1], strict=True):
            value = (.5 if i < 20 else .1) + gain
            metric = {"partition": "outer", "fold": name, "benchmark": benchmark, "protocol_sha256": name,
                      "forbidden_human_exposures": 0, "forbidden_genes": ["held"], "n": 100, "positives": 30,
                      "prevalence": .3, **dict.fromkeys(summary.METRICS, value)}
            fold["families"][family] = {"selection": {"selected": family}, "metrics": metric}
        fold["metrics"] = copy.deepcopy(fold["families"]["no_pretraining"]["metrics"])
        journal["folds"][name] = fold
    return plan, journal


def test_equal_benchmark_weight_does_not_favor_larger_benchmark_or_test_winner():
    plan, journal = fixture()
    result = summary.summarize(plan, journal, "pinned")
    assert result["equal_benchmark_means"]["feature_mlp"]["average_precision"] == pytest.approx(.5)
    assert result["equal_benchmark_means"]["inner_selected"]["average_precision"] == pytest.approx(.3)
    assert result["benchmarks"]["large"]["inner_selection_counts"] == {"no_pretraining": 20}
    assert result["benchmarks"]["small"]["paired_pretraining_minus_sl_only"]["average_precision"]["mean"] == pytest.approx(.1)


def test_partial_cannot_be_reported_as_complete_or_macro_averaged():
    plan, journal = fixture()
    journal["folds"] = {"fold0": journal["folds"]["fold0"]}
    with pytest.raises(ValueError, match="Incomplete suite"):
        summary.summarize(plan, journal, "pinned")
    result = summary.summarize(plan, journal, "pinned", allow_partial=True)
    assert not result["complete"] and result["equal_benchmark_means"] is None
    assert result["benchmarks"]["small"]["completed_folds"] == 0


@pytest.mark.parametrize("change, error", [
    (lambda f: f["families"]["mixed_pretraining"]["metrics"].update(forbidden_human_exposures=1), "exposure"),
    (lambda f: f["families"]["mixed_pretraining"]["metrics"].update(forbidden_genes=["different"]), "cohorts"),
    (lambda f: f.update(metrics=f["families"]["feature_mlp"]["metrics"]), "inner selection"),
    (lambda f: f["families"]["mixed_pretraining"]["metrics"].update(auroc=float("nan")), "metric"),
])
def test_invalid_results_fail_closed(change, error):
    plan, journal = fixture()
    change(journal["folds"]["fold0"])
    with pytest.raises(ValueError, match=error):
        summary.summarize(plan, journal, "pinned")
