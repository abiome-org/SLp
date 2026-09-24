import json

import polars as pl
import pytest

from slpbench.evaluate import file_sha256
from slpbench.release import PUBLIC_FILES, export_public, verify_benchmark


def test_public_export_excludes_answers_and_checks_hashes(tmp_path):
    bench = tmp_path / "private"
    (bench / "hidden").mkdir(parents=True)
    files = {}
    for name in PUBLIC_FILES:
        if name == "manifest.json":
            continue
        path = bench / name
        pl.DataFrame({"example_id": ["x"]}).write_parquet(path)
        files[name] = {"sha256": file_sha256(path), "rows": 1}
    secret = bench / "hidden/test_labels.parquet"
    pl.DataFrame({"example_id": ["x"], "label": [1]}).write_parquet(secret)
    files["hidden/test_labels.parquet"] = {"sha256": file_sha256(secret), "rows": 1}
    (bench / "manifest.json").write_text(json.dumps({"version": "toy", "files": files}))
    assert verify_benchmark(bench)["files_verified"] == len(files)
    out = tmp_path / "public"
    lock = export_public(bench, out)
    assert lock["public_files"]["test_inputs.parquet"] == file_sha256(out / "test_inputs.parquet")
    assert not (out / "hidden").exists()
    assert not (out / "test_labels.parquet").exists()
    assert set(p.name for p in out.iterdir()) == set(PUBLIC_FILES) | {"release.lock.json"}
    (bench / "hidden/test_labels.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_benchmark(bench)
