"""Focused tests for author-normalized human development adapter v2."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np

MODULE = Path(__file__).parents[1] / "modules" / "slp-1-1-world-transition-v1"
sys.path.insert(0, str(MODULE))

from human_data_v2 import (
    CONTEXT_IDS,
    CONTEXT_VALUE_SPACE,
    NUM_CELLS_ROLE,
    VALUE_SPACE,
    SourceSpec,
    _split_name,
    build_human_development_v2,
)


def _action_for(split: str, start: int) -> str:
    for number in range(start, start + 100_000):
        action = f"ENSG{number:011d}"
        if _split_name(action) == split:
            return action
    raise AssertionError("could not construct split action")


def _source(path: Path, genes: list[str], actions: list[str], offset: float) -> SourceSpec:
    records = [
        "0_NTC_non-targeting_non-targeting_non-targeting",
        "1_NTC_non-targeting_non-targeting_non-targeting",
        *(f"{index + 2}_GENE_P1_{action}" for index, action in enumerate(actions)),
        "99_UNRESOLVED_nan",
    ]
    matrix = np.arange(len(records) * len(genes), dtype=np.float32).reshape(
        len(records), len(genes)
    )
    matrix = matrix / 10.0 - offset
    matrix[-1, -1] = np.inf
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=matrix)
        obs = handle.create_group("obs")
        obs.create_dataset("gene_transcript", data=np.asarray(records, dtype="S"))
        obs.create_dataset(
            "num_cells_filtered",
            data=np.asarray([100, 80, *(50 + i for i in range(len(actions))), 1]),
        )
        obs.create_dataset(
            "core_control", data=np.asarray([True, False, *([False] * len(actions)), False])
        )
        var = handle.create_group("var")
        var.create_dataset("gene_id", data=np.asarray(genes, dtype="S"))
    payload = path.read_bytes()
    return SourceSpec(
        CONTEXT_IDS[0],
        path.name,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        hashlib.md5(payload).hexdigest(),
        1,
    )


def test_v2_preserves_core_controls_exposure_and_split(tmp_path: Path) -> None:
    actions = [
        _action_for("train", 1),
        _action_for("validation", 1000),
        _action_for("test", 2000),
    ]
    first = tmp_path / "first.h5ad"
    second = tmp_path / "second.h5ad"
    spec1 = _source(first, ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"], actions, 2)
    spec2_base = _source(
        second,
        ["ENSG00000000001", "ENSG00000000002", "ENSG00000000004"],
        actions,
        3,
    )
    spec2 = SourceSpec(
        CONTEXT_IDS[1],
        spec2_base.filename,
        spec2_base.bytes,
        spec2_base.sha256,
        spec2_base.upstream_md5,
        spec2_base.figshare_file_id,
    )
    raw_first = tmp_path / "raw_first.h5ad"
    raw_second = tmp_path / "raw_second.h5ad"
    raw_spec1 = _source(
        raw_first,
        ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"],
        actions,
        -1,
    )
    raw_spec2_base = _source(
        raw_second,
        ["ENSG00000000001", "ENSG00000000002", "ENSG00000000004"],
        actions,
        -1,
    )
    raw_spec2 = SourceSpec(
        CONTEXT_IDS[1],
        raw_spec2_base.filename,
        raw_spec2_base.bytes,
        raw_spec2_base.sha256,
        raw_spec2_base.upstream_md5,
        raw_spec2_base.figshare_file_id,
    )

    result = build_human_development_v2(
        first,
        second,
        tmp_path / "out",
        source_specs=(spec1, spec2),
        raw_context_paths=(raw_first, raw_second),
        raw_context_specs=(raw_spec1, raw_spec2),
        expected_query_count=2,
    )
    development = Path(result["manifestPath"]).parent / result["manifest"]["outputs"]["development"]["path"]
    with np.load(development, allow_pickle=False) as bundle:
        assert str(bundle["target_value_space"]) == VALUE_SPACE
        assert str(bundle["num_cells_role"]) == NUM_CELLS_ROLE
        assert str(bundle["context_value_space"]) == CONTEXT_VALUE_SPACE
        assert bundle["context_basal_expression"].shape == (2, 2)
        assert np.isfinite(bundle["context_basal_expression"]).all()
        assert len(bundle["split_train"]) == 2
        assert len(bundle["split_validation"]) == 2
        assert len(bundle["split_test"]) == 0
        assert bundle["num_cells_filtered"].shape == (4,)
        assert bundle["control_targets"].shape == (2, 2)
        assert bundle["control_observed"].all()
        assert bundle["control_core"].all()
        np.testing.assert_array_equal(bundle["control_context_index"], [0, 1])
    assert result["manifest"]["counts"]["testOnly"] == 2
    assert result["manifest"]["counts"]["unresolvedActionsQuarantined"] == [1, 1]
