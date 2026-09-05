from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "fuse_slp11_static_features.py"
    spec = importlib.util.spec_from_file_location("fuse_slp11_static_features", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FUSION = _load_script()


def _write_pack(directory: Path, name: str, values: np.ndarray, ids=None) -> Path:
    path = directory / f"{name}.npz"
    identifiers = np.asarray(ids or ["ENSG00000000001", "ENSG00000000002"], dtype="<U15")
    payload = FUSION.deterministic_npz_bytes(
        {
            "feature_values": values.astype(np.float32),
            "entity_taxon": np.asarray([9606, 9606], dtype=np.int64),
            "entity_id": identifiers,
        }
    )
    path.write_bytes(payload)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema": f"fixture.{name}/v1",
                "artifact": {
                    "path": path.name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_fusion_concatenates_exact_keys_and_reports_block_coverage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = _write_pack(root, "protein", np.asarray([[1, 0], [0, 0]]))
        second = _write_pack(root, "go", np.asarray([[0], [2]]))
        output = root / "fusion.npz"
        manifest_path = root / "fusion.manifest.json"
        manifest = FUSION.build_fusion(
            [f"protein_esm={first}", f"go_mf_cc={second}"], output, manifest_path
        )
        with np.load(output, allow_pickle=False) as archive:
            assert archive.files == ["feature_values", "entity_taxon", "entity_id"]
            assert archive["feature_values"].dtype == np.float32
            np.testing.assert_array_equal(
                archive["feature_values"], np.asarray([[1, 0, 0], [0, 0, 2]], np.float32)
            )
        assert manifest["coverage"] == {
            "rows": 2,
            "rowsWithAnyNonzeroFeature": 2,
            "zeroRows": 0,
        }
        assert [item["columns"] for item in manifest["featureBlocks"]] == [
            {"startInclusive": 0, "endExclusive": 2},
            {"startInclusive": 2, "endExclusive": 3},
        ]
        assert manifest["construction"]["operation"] == (
            "column-concatenation-without-scaling-or-fitting"
        )


def test_fusion_rejects_reordered_or_incomplete_keys() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = _write_pack(root, "one", np.ones((2, 1), dtype=np.float32))
        second = _write_pack(
            root,
            "two",
            np.ones((2, 1), dtype=np.float32),
            ids=["ENSG00000000002", "ENSG00000000001"],
        )
        with pytest.raises(FUSION.FeatureFusionError, match="uniquely sorted"):
            FUSION.build_fusion(
                [f"one={first}", f"two={second}"], root / "out.npz", root / "out.json"
            )
        incomplete = _write_pack(
            root,
            "incomplete",
            np.ones((2, 1), dtype=np.float32),
            ids=["ENSG00000000001", "ENSG00000000003"],
        )
        with pytest.raises(FUSION.FeatureFusionError, match="composite key rows differ"):
            FUSION.build_fusion(
                [f"one={first}", f"incomplete={incomplete}"],
                root / "other.npz",
                root / "other.json",
            )


def test_fusion_rejects_source_manifest_hash_mismatch() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = _write_pack(root, "one", np.ones((2, 1), dtype=np.float32))
        second = _write_pack(root, "two", np.ones((2, 1), dtype=np.float32))
        source_manifest = second.with_suffix(".manifest.json")
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        manifest["artifact"]["sha256"] = "0" * 64
        source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(FUSION.FeatureFusionError, match="artifact identity mismatch"):
            FUSION.build_fusion(
                [f"one={first}", f"two={second}"], root / "out.npz", root / "out.json"
            )
