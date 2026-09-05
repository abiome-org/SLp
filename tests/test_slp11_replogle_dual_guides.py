import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/audit_replogle_dual_guide_library.py"
SPEC = importlib.util.spec_from_file_location("replogle_guides", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_parse_target_and_control_guide_ids():
    parsed = MOD.parse_guide_id(
        "BRD4_-_15391173.23-ENST00000263377.2,ENST00000371835.4"
    )
    assert parsed == {
        "is_control": False,
        "gene_symbol": "BRD4",
        "strand": "-",
        "coordinate": 15391173,
        "coordinate_observed": True,
        "design_token": ".23",
        "transcript_target": "ENST00000263377.2-ENST00000371835.4",
    }
    assert MOD.parse_guide_id("non-targeting_00054")["coordinate_observed"] is False
    with pytest.raises(ValueError, match="unrecognized"):
        MOD.parse_guide_id("not-a-source-guide")


def test_build_arrays_uses_only_identity_metadata_and_fails_on_conflicts():
    columns = {
        "unique sgRNA pair ID": ["1_A_P1_ENSG00000000001", "non-targeting_1"],
        "gene": ["A", "non-targeting"],
        "transcript": ["P1", "non-targeting"],
        "ensembl gene id": ["ENSG00000000001", np.nan],
        "sgID_A": ["A_+_10.23-P1", "non-targeting_00001"],
        "targeting sequence A": ["A" * 20, "C" * 20],
        "sgID_B": ["A_-_20.23-P1", "non-targeting_00002"],
        "targeting sequence B": ["T" * 20, "G" * 20],
        "duplicated guide pair?": ["False", "False"],
        "either guide duplicated?": ["False", "False"],
    }
    library = pd.DataFrame(columns)
    library["original_pair"] = library.sgID_A + "|" + library.sgID_B
    library["canonical_pair"] = [
        MOD.canonical_pair(a, b) for a, b in zip(library.sgID_A, library.sgID_B)
    ]
    table = {"k562": library.set_index("canonical_pair", drop=False), "rpe1": library.set_index("canonical_pair", drop=False)}
    routing = pd.DataFrame({
        "guide_pair_ids": ["A_+_10.23-P1|A_-_20.23-P1", "non-targeting_00001|non-targeting_00002"],
        "action_ids": ["ENSG00000000001", ""],
        "gene_transcript": ["1_A_P1_ENSG00000000001", "non-targeting_1"],
        "is_control": [False, True],
        "cell_count": [3, 2],
    })
    arrays, report = MOD.build_arrays(table, {"k562": routing, "rpe1": routing})
    assert report["pairUnion"] == 2
    assert arrays["targeting_sequence_b"].tolist() == ["T" * 20, "G" * 20]
    assert arrays["coordinate_observed_a"].tolist() == [True, False]
    broken = routing.copy()
    broken.loc[0, "action_ids"] = "ENSG00000000002"
    with pytest.raises(ValueError, match="different stable genes"):
        MOD.build_arrays(table, {"k562": routing, "rpe1": broken})


def test_canonical_pair_only_normalizes_multitranscript_punctuation():
    a = "BRD4_-_10.23-ENST1,ENST2"
    b = "BRD4_+_20.23-ENST1,ENST2"
    assert MOD.canonical_pair(a, b) == (
        "BRD4_-_10.23-ENST1-ENST2|BRD4_+_20.23-ENST1-ENST2"
    )
