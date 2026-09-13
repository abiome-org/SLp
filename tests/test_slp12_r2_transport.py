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
    import pytest

    with pytest.raises(TimeoutError):
        slp12_r2_transport.upload_directory(tmp_path, "synthetic")
    assert "artifact.json" not in called
