from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import build_slp11_gwps_static_features as FEATURES


def test_merge_ids_adds_only_missing_actions_in_sorted_order() -> None:
    extended, added = FEATURES.merge_ids(("A", "C", "E"), ("B", "C", "D"))
    assert extended == ("A", "B", "C", "D", "E")
    assert added == ("B", "D")


def test_output_names_keep_feature_blocks_separate() -> None:
    assert set(FEATURES.OUTPUT_NAMES) == {"sequence", "go", "fusion"}
    assert len(set(FEATURES.OUTPUT_NAMES.values())) == 3
