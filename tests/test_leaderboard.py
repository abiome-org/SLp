import json

import pytest

from slpbench import leaderboard
from slpbench.evaluate import SCORER_VERSION, file_sha256, save


def test_saved_results_are_standard_json(tmp_path):
    path = tmp_path / "result.json"
    save({"aux": [float("nan"), float("inf"), float("-inf")]}, path)
    assert json.loads(path.read_text(), parse_constant=lambda x: pytest.fail(f"nonstandard JSON: {x}")) == (
        {"aux": [None, None, None]})


def test_leaderboard_rejects_tampered_ci_and_nonfinite_scores(tmp_path, monkeypatch):
    bench = tmp_path / "slb1.3"
    bench.mkdir()
    (bench / "manifest.json").write_text("{}")
    pred = tmp_path / "predictions.parquet"
    pred.write_bytes(b"predictions")
    result = tmp_path / "result.json"
    actual = {"slb_score": 0.64, "n": 10,
              "species_scores": {"human": 0.65}, "auxiliary_species_scores": {"bsub": float("nan")}}
    record = {"benchmark": "slb1.3", "split": "test", "scorer_version": SCORER_VERSION,
              "manifest_sha256": file_sha256(bench / "manifest.json"),
              "predictions_sha256": file_sha256(pred), "missing_filled": 0,
              "slb_score": 0.64, "n": 10, "species_scores": {"human": 0.65},
              "auxiliary_species_scores": {"bsub": None}, "slb_score_ci95": [0.60, 0.68]}
    monkeypatch.setattr(leaderboard, "BENCH", bench)
    monkeypatch.setattr(leaderboard.E, "read_predictions", lambda _: None)
    monkeypatch.setattr(leaderboard.E, "evaluate", lambda *_: actual)

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
    monkeypatch.setattr(leaderboard, "_verified_result", lambda e: {
        "slb_score": e["score"], "slb_score_ci95": [e["score"] - .01, e["score"] + .01],
        "species_scores": {"human": e["score"]}, "auxiliary_species_scores": {},
    })
    rendered = leaderboard.render([
        {"name": "unresolved", "score": .9, "leaky": "possibly"},
        {"name": "diagnostic", "score": .8, "ranked": False, "leaky": False},
        {"name": "finalist", "score": .7, "leaky": False},
    ])
    assert "| 1 | **finalist**" in rendered
    assert "| – | **diagnostic**" in rendered
    assert "| – | **unresolved**" in rendered and "| possible |" in rendered
