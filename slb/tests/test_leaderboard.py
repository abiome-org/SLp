import json

import pytest

from slbench import leaderboard
from slbench.evaluate import SCORER_VERSION, file_sha256, save


def test_saved_results_are_standard_json(tmp_path):
    path = tmp_path / "result.json"
    save({"aux": [float("nan"), float("inf"), float("-inf")]}, path)
    assert json.loads(path.read_text(), parse_constant=lambda x: pytest.fail(f"nonstandard JSON: {x}")) == (
        {"aux": [None, None, None]})


def test_leaderboard_rejects_tampered_ci_and_nonfinite_scores(tmp_path, monkeypatch):
    bench = tmp_path / "slb"
    bench.mkdir()
    (bench / "manifest.json").write_text('{"version": "toy"}')
    pred = tmp_path / "predictions.parquet"
    pred.write_bytes(b"predictions")
    result = tmp_path / "result.json"
    strata = [{"stratum": "ALL (flat)", "n": 10, "pos": 3, "slb_auroc": 0.64, "within_gene": float("nan"),
               "unadjusted_auroc": 0.7, "ap_lift": 1.5}]
    actual = {"slb_score": 0.64, "n": 10, "strata": strata,
              "species_scores": {"human": 0.65}, "auxiliary_species_scores": {"bsub": float("nan")}}
    record = {"benchmark": "toy", "split": "test", "scorer_version": SCORER_VERSION,
              "manifest_sha256": file_sha256(bench / "manifest.json"),
              "predictions_sha256": file_sha256(pred), "missing_filled": 0,
              "slb_score": 0.64, "n": 10, "species_scores": {"human": 0.65},
              "auxiliary_species_scores": {"bsub": None}, "slb_score_ci95": [0.60, 0.68],
              "strata": [{**strata[0], "within_gene": None}]}
    monkeypatch.setattr(leaderboard, "BENCH", bench)
    monkeypatch.setattr(leaderboard.E, "read_predictions", lambda _: None)
    monkeypatch.setattr(leaderboard.E, "evaluate", lambda *_, **__: actual)
    monkeypatch.delenv("SLB_VERIFY_CI", raising=False)

    def entry():
        result.write_text(json.dumps(record))
        return {"result": str(result), "result_sha256": file_sha256(result), "predictions": str(pred)}

    good = entry()
    assert leaderboard._verified_result(good)["slb_score"] == 0.64
    record["slb_score_ci95"] = [0.0, 1.0]
    result.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        leaderboard._verified_result(good)

    record["slb_score_ci95"] = [0.60, 0.68]
    record["slb_score"] = float("nan")
    with pytest.raises(ValueError, match="does not match"):
        leaderboard._verified_result(entry())
    record["slb_score"] = 0.64
    record["auxiliary_species_scores"] = {"bsub": 0.5}
    with pytest.raises(ValueError, match="auxiliary_species_scores"):
        leaderboard._verified_result(entry())


def test_exploratory_test_read_is_registered_but_unranked(monkeypatch):
    monkeypatch.setattr(leaderboard, "species_tiers", lambda: (["human"], []))
    monkeypatch.setattr(leaderboard, "_verified_result", lambda e, *_: {
        "slb_score": e["score"], "slb_score_ci95": [e["score"] - .01, e["score"] + .01],
        "species_scores": {"human": e["score"]}, "auxiliary_species_scores": {},
    })
    rendered = leaderboard.render([
        {"name": "unresolved", "score": .9, "leaky": "possibly"},
        {"name": "diagnostic", "score": .8, "ranked": False, "leaky": False},
        {"name": "finalist", "score": .7, "leaky": False},
    ])
    assert "| 1 | **unresolved**\\*" in rendered and "| possible |" in rendered
    assert "| 2 | **finalist** —" in rendered
    assert "| – | **diagnostic**" in rendered


def test_leaderboard_rechecks_strata_and_bootstrap_ci(tmp_path, monkeypatch):
    bench = tmp_path / "slb"
    bench.mkdir()
    (bench / "manifest.json").write_text('{"version": "toy"}')
    pred = tmp_path / "predictions.parquet"
    pred.write_bytes(b"predictions")
    result = tmp_path / "result.json"
    strata = [{"stratum": "ALL (flat)", "n": 10, "pos": 3, "slb_auroc": 0.64, "within_gene": 0.6,
               "unadjusted_auroc": 0.7, "ap_lift": 1.5},
              {"stratum": "species=human", "n": 10, "pos": 3, "slb_auroc": 0.65, "within_gene": float("nan"),
               "unadjusted_auroc": 0.7, "ap_lift": 1.5}]
    calls = []

    def evaluate(preds, split, boot=0, *_, **__):
        calls.append(boot)
        return {"slb_score": 0.64, "n": 10, "species_scores": {"human": 0.65}, "auxiliary_species_scores": {},
                "strata": strata, **({"slb_score_ci95": (0.601234, 0.681234)} if boot else {})}

    monkeypatch.setattr(leaderboard, "BENCH", bench)
    monkeypatch.setattr(leaderboard.E, "read_predictions", lambda _: None)
    monkeypatch.setattr(leaderboard.E, "evaluate", evaluate)
    monkeypatch.delenv("SLB_VERIFY_CI", raising=False)
    base = {"benchmark": "toy", "split": "test", "scorer_version": SCORER_VERSION,
            "manifest_sha256": file_sha256(bench / "manifest.json"), "predictions_sha256": file_sha256(pred),
            "missing_filled": 0, "slb_score": 0.64, "n": 10, "species_scores": {"human": 0.65},
            "auxiliary_species_scores": {}, "slb_score_ci95": [0.601234, 0.681234], "strata": strata}

    def entry(**kw):
        result.write_text(json.dumps({**base, **kw}).replace("NaN", "null"))
        return {"result": str(result), "result_sha256": file_sha256(result), "predictions": str(pred)}

    assert leaderboard._verified_result(entry())["n"] == 10 and calls == [0]
    assert leaderboard._verified_result(entry(), verify_ci=True) and calls[-1] == leaderboard.BOOT
    monkeypatch.setenv("SLB_VERIFY_CI", "1")
    assert leaderboard._verified_result(entry()) and calls[-1] == leaderboard.BOOT
    with pytest.raises(ValueError, match="slb_score_ci95"):
        leaderboard._verified_result(entry(slb_score_ci95=[0.63, 0.65]))
    monkeypatch.delenv("SLB_VERIFY_CI")
    assert leaderboard._verified_result(entry(slb_score_ci95=[0.63, 0.65]))  # CI only bounds-checked by default
    for tamper in ({"slb_auroc": 0.70}, {"slb_auroc": 0.64 + 1e-9}, {"within_gene": 0.6}, {"pos": 4},
                   {"stratum": "species=scer"}, {"ap_lift": None}):
        with pytest.raises(ValueError, match="strata"):
            leaderboard._verified_result(entry(strata=[strata[0], {**strata[1], **tamper}]))
    with pytest.raises(ValueError, match="strata"):
        leaderboard._verified_result(entry(strata=strata[:1]))
    assert leaderboard._verified_result(entry(strata=[strata[0], {**strata[1], "slb_auroc": 0.65 + 1e-12}]))
