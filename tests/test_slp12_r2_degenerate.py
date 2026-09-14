"""Official single-class folds must not become fake perfect ranking scores."""

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d = load("degenerate", "scripts/slp12_r2_degenerate.py")
e = load("evaluation", "modules/slp-1-2-r2/evaluate.py")


def test_two_class_metrics_and_selection_are_unchanged():
    actual = d.metrics([0, 1, 0, 1], [0, 0, 1, 2], e.binary_metrics)
    for k, v in e.binary_metrics([0, 1, 0, 1], [0, 0, 1, 2]).items():
        assert actual[k] == v
    reports = [{**actual, "partition": "inner", "forbidden_human_exposures": 0,
                "benchmark": "b", "fold": "f", "checkpoint": "a"}]
    assert d.select(reports, ["b"], {"b": ["f"]}, e.select_checkpoint) == e.select_checkpoint(reports, ["b"], {"b": ["f"]})


@pytest.mark.parametrize("labels,scores", [([0, 0], [-1000, -1000]), ([1, 1], [1000, 1000])])
def test_single_class_reports_null_metrics_and_stable_loss(labels, scores):
    actual = d.metrics(labels, scores, e.binary_metrics)
    assert actual["metric_status"] == "undefined_single_class"
    assert actual["average_precision"] is None and actual["auroc"] is None
    assert actual["binary_log_loss"] == 0


def reports():
    return [{**d.metrics([0, 0], [s, s], e.binary_metrics), "partition": "inner",
             "forbidden_human_exposures": 0, "benchmark": "b", "fold": "f", "checkpoint": name}
            for name, s in [("a", 1), ("b", -1)]]


def test_fallback_uses_only_matched_inner_log_loss():
    assert d.select(reports(), ["b"], {"b": ["f"]}, e.select_checkpoint)["selected"] == "b"
    bad = reports()
    bad[0]["partition"] = "outer"
    with pytest.raises(ValueError, match="Invalid"):
        d.select(bad, ["b"], {"b": ["f"]}, e.select_checkpoint)
    bad = reports()
    bad[0]["n"] = 3
    with pytest.raises(ValueError, match="cohorts"):
        d.select(bad, ["b"], {"b": ["f"]}, e.select_checkpoint)


def test_empty_or_invalid_scores_cannot_select():
    assert d.metrics([], [], e.binary_metrics)["metric_status"] == "undefined_empty"
    with pytest.raises(ValueError, match="finite"):
        d.metrics([0], [float("nan")], e.binary_metrics)


def test_broker_amendment_is_limited_to_exact_source_and_record(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    broker = load("amendment_broker", "scripts/slp12_r2_broker.py")
    bodies = {"slp12_r2_degenerate.py": (ROOT / "scripts/slp12_r2_degenerate.py").read_bytes(),
              "amendment.json": json.dumps(d.amendment(), sort_keys=True).encode()}
    request = {"job": "campaign-r2-research-final2-20260914-single-class-scoring-v1",
               "plan_sha256": "34465c3739a95182b790f889060d37a8785a202c2c65b101e430bb105aae699d",
               "files": {name: {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()} for name, body in bodies.items()}}
    assert len(broker.permissions_for(request, {}, "research-final2-20260914")) == 4
    request["files"]["amendment.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact authorized files"):
        broker.permissions_for(request, {}, "research-final2-20260914")
