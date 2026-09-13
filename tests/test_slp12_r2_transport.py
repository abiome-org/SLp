"""Transport concurrency must preserve the existing immutable artifact contract."""

import io
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "modules/slp-1-2-r2")]
import cloud_io
import slp12_r2_transport


def test_parallel_transport_matches_serial_bytes_and_manifest(tmp_path, monkeypatch):
    (tmp_path / "checkpoint.pt").write_bytes(b"checkpoint-fixture" * (2 * 1024 * 1024))
    (tmp_path / "complete.json").write_text('{"update":1000}')
    bodies, active, peak = {}, 0, 0
    lock = threading.Lock()

    def request(job, name, body=None):
        nonlocal active, peak
        assert job == "synthetic"
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            bodies[name] = body
            active -= 1
        return io.BytesIO(b"{}")

    monkeypatch.setattr(cloud_io, "request", request)
    serial = cloud_io.upload_directory(tmp_path, "synthetic")
    expected = dict(bodies)
    bodies.clear()
    parallel = slp12_r2_transport.upload_directory(tmp_path, "synthetic")
    assert parallel == serial
    assert bodies == expected
    assert 1 < peak <= 4
    assert json.loads(bodies["artifact.json"]) == serial


def test_failed_part_never_publishes_manifest(tmp_path, monkeypatch):
    (tmp_path / "checkpoint.pt").write_bytes(b"fixture")
    called = []

    def request(job, name, body=None):
        called.append(name)
        raise TimeoutError("synthetic upload failure")

    monkeypatch.setattr(cloud_io, "request", request)
    monkeypatch.setattr(slp12_r2_transport.time, "sleep", lambda _: None)
    import pytest

    with pytest.raises(TimeoutError):
        slp12_r2_transport.upload_directory(tmp_path, "synthetic")
    assert "artifact.json" not in called


def test_transient_retry_preserves_bytes_and_does_not_retry_permission_errors(
    monkeypatch,
):
    import urllib.error
    import pytest

    monkeypatch.setattr(slp12_r2_transport.time, "sleep", lambda _: None)
    calls = []

    def flaky():
        calls.append(b"identical payload")
        if len(calls) < 3:
            raise urllib.error.HTTPError(
                "https://example.test", 503, "outage", {}, None
            )
        return "stored"

    assert slp12_r2_transport.retry_io(flaky) == "stored"
    assert calls == [b"identical payload"] * 3
    denied = []

    def forbidden():
        denied.append(1)
        raise urllib.error.HTTPError("https://example.test", 403, "forbidden", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        slp12_r2_transport.retry_io(forbidden)
    assert len(denied) == 1
