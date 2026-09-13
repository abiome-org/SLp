import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "go_features",
    Path(__file__).resolve().parents[1] / "scripts/slp12_r2_go_features.py",
)
go = importlib.util.module_from_spec(spec)
spec.loader.exec_module(go)


def fixture(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "genes.json").write_text(json.dumps(["9606:HGNC:1", "9606:HGNC:2"]))
    np.save(base / "sequence.npy", np.ones((2, 3)))
    np.save(base / "annotation.npy", np.array([[1, 0], [0, 1]], dtype=np.float32))
    np.save(base / "known.npy", np.ones((2, 2), dtype=bool))
    identity = tmp_path / "identity.json"
    identity.write_text('{"z": 1,"a": 2}')
    (base / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "slp.static-features/v2",
                "fitted_human_intervention_genes": [],
                "source_manifest_sha256": go.hashlib.sha256(
                    json.dumps(
                        json.loads(identity.read_text()), sort_keys=True
                    ).encode()
                ).hexdigest(),
                "files": {name: go.digest(base / name) for name in go.FILES},
            }
        )
    )
    source = tmp_path / "go.json"
    shard = tmp_path / "annotations-00000.jsonl.gz"
    shard.write_bytes(
        gzip.compress(
            json.dumps(
                {"gene": "9606:HGNC:1", "terms": ["C:GO:0000001", "F:GO:0000002"]}
            ).encode()
        )
    )
    source.write_text(
        json.dumps(
            {
                "schema": "slp.static-go/v1",
                "fitted_human_intervention_genes": [],
                "filter": go.FILTER,
                "identity_manifest_sha256": go.digest(identity),
                "source_module_sha256": "captured",
                "taxon": 9606,
                "rows": 1,
                "shards": [
                    {
                        "name": shard.name,
                        "bytes": shard.stat().st_size,
                        "sha256": go.digest(shard),
                        "rows": 1,
                    }
                ],
            }
        )
    )
    return base, identity, source, shard


def test_binary_features_preserve_roster_and_base_and_reject_overwrite(tmp_path):
    base, identity, source, _shard = fixture(tmp_path)
    result = go.prepare(base, identity, [source], tmp_path / "new", "captured")
    np.testing.assert_array_equal(
        np.load(tmp_path / "new/annotation.npy"), [[1, 0, 1, 1], [0, 1, 0, 0]]
    )
    np.testing.assert_array_equal(np.load(base / "annotation.npy"), [[1, 0], [0, 1]])
    assert set(result["files"]) == go.FILES
    for name in go.FILES - {"annotation.npy"}:
        assert go.digest(base / name) == result["files"][name]
    with pytest.raises(FileExistsError):
        go.prepare(base, identity, [source], tmp_path / "new", "captured")


def test_corruption_and_intervention_exposure_fail_before_materializing(tmp_path):
    base, identity, source, shard = fixture(tmp_path)
    shard.write_bytes(shard.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum"):
        go.prepare(base, identity, [source], tmp_path / "new", "captured")
    m = json.loads(source.read_text())
    m["fitted_human_intervention_genes"] = ["9606:HGNC:2"]
    source.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="static-data contract"):
        go.prepare(base, identity, [source], tmp_path / "new", "captured")
    assert not (tmp_path / "new").exists()
