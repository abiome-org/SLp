"""The maintainer refresh must not publish stale or partially updated results."""

import json
import importlib.util
import os
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[1] / "scripts/refresh_test_results.py"
spec = importlib.util.spec_from_file_location("refresh_test_results", script)
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


def test_refresh_recomputes_if_benchmark_manifest_changes(tmp_path, monkeypatch):
    bench = tmp_path / "slb1.3"
    bench.mkdir()
    manifest = bench / "manifest.json"
    manifest.write_text("{}")
    pred = tmp_path / "pred.csv"
    pred.write_text("example_id,score\nx,0.5\n")
    result = tmp_path / "result.json"
    record = {"benchmark": bench.name, "manifest_sha256": refresh.E.file_sha256(manifest),
              "split": "test", "scorer_version": refresh.E.SCORER_VERSION,
              "predictions_sha256": refresh.E.file_sha256(pred), "missing_filled": 0,
              "slb_score_ci95": [0.5, 0.7], "slb_score": 0.6, "n": 1,
              "species_scores": {"human": 0.6}}
    result.write_text(json.dumps(record))
    entry = {"name": "model", "predictions": str(pred), "result": str(result),
             "result_sha256": refresh.E.file_sha256(result)}
    monkeypatch.setattr(refresh.E, "BENCH", bench)
    calls = []
    monkeypatch.setattr(refresh.E, "read_predictions", lambda _: None)
    monkeypatch.setattr(refresh.E, "evaluate", lambda *args, **kwargs: (calls.append(1) or record.copy()))

    assert refresh.score_one(entry)[1] is None
    manifest.write_text('{"changed":true}')
    assert refresh.score_one(entry)[1] is not None
    assert len(calls) == 1


def test_refresh_rolls_back_if_pin_replace_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = Path("results/model.json")
    result.parent.mkdir()
    result.write_text('{"old": true}')
    config = Path("leaderboard.yaml")
    config.write_text("- name: model\n  result: results/model.json\n  result_sha256: oldhash\n")
    original_result, original_config = result.read_bytes(), config.read_bytes()

    class FakePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, fn, entries):
            return [(entries[0]["result"], {"slb_score": 0.6}, 0.6, [0.5, 0.7])]

    monkeypatch.setattr(refresh, "ProcessPoolExecutor", FakePool)
    replace = os.replace

    def fail_pin(src, dest):
        if Path(dest) == config:
            raise OSError("simulated pin write failure")
        return replace(src, dest)

    monkeypatch.setattr(refresh.os, "replace", fail_pin)
    with pytest.raises(OSError, match="simulated pin write failure"):
        refresh.main()
    assert result.read_bytes() == original_result
    assert config.read_bytes() == original_config
    assert not list(tmp_path.rglob("*.staged"))
