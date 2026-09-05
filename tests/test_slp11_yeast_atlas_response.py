import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "data/tools/rdata-1.1.0/site-packages"))
PATH = ROOT / "modules/slp-1-1-yeast-atlas-response-v1/atlas_response.py"
SPEC = importlib.util.spec_from_file_location("slp11_yeast_atlas_response_test", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_bounded_reader_aligns_ragged_query_support(tmp_path: Path):
    import rdata

    source = tmp_path / "tiny.Rdata"
    rdata.write_rda(
        source,
        {"fcs": {
            "DEG_Control_bc_YAL001C.csv": pd.DataFrame({
                "names": pd.Series(["A", "B"], dtype=object),
                "logfoldchanges": [1.0, 2.0],
            }),
            "DEG_NaCl_bc_YAL001C.csv": pd.DataFrame({
                "names": pd.Series(["B", "C"], dtype=object),
                "logfoldchanges": [3.0, np.nan],
            }),
        }},
    )
    values_path, observed_path = tmp_path / "values.npy", tmp_path / "observed.npy"
    report = MOD.extract_fcs(source, values_path, observed_path)
    assert report["query_names"] == ("A", "B", "C")
    np.testing.assert_array_equal(np.load(observed_path), [[True, True, False], [False, True, False]])
    np.testing.assert_allclose(np.load(values_path), [[1, 2, 0], [0, 3, 0]])
    assert report["missing_or_absent_values"] == 3


def test_stable_mapping_excludes_ambiguous_names(tmp_path: Path):
    records = [
        {"schema": "slp.sgd-current-orf/v1", "ncbiTaxon": 4932,
         "canonicalSgdCurie": "SGD:S000000001", "systematicName": "YAL001C",
         "displayMetadata": {"standardGeneName": "A"}},
        {"schema": "slp.sgd-current-orf/v1", "ncbiTaxon": 4932,
         "canonicalSgdCurie": "SGD:S000000002", "systematicName": "YAL002W",
         "displayMetadata": {"standardGeneName": "A"}},
    ]
    path = tmp_path / "map.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    mapping, ambiguous = MOD.exact_current_maps(path)
    assert mapping["YAL001C"] == "SGD:S000000001"
    assert mapping["YAL002W"] == "SGD:S000000002"
    assert "A" not in mapping and ambiguous == {"A"}


def test_protected_and_development_roles_are_identity_only_and_deterministic():
    ids = [f"SGD:S{index:09d}" for index in range(1, 100)]
    assert {MOD.protected_role(item) for item in ids} == {
        "pretrain", "molecular-validation", "molecular-final",
    }
    assert {MOD.development_role(item) for item in ids} == {"train", "validation", "test"}
    assert all(MOD.protected_role(item) == MOD.protected_role(item) for item in ids)
