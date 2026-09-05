from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


def _load():
    module_dir = Path(__file__).parents[1] / "modules" / "slp-1-1-world-transition-v1"
    sys.path.insert(0, str(module_dir))
    spec = importlib.util.spec_from_file_location("slp11_gwps_data", module_dir / "gpws_data.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GWPS = _load()


def test_split_is_the_existing_global_gene_hash() -> None:
    candidates = [f"ENSG{index:011d}" for index in range(1, 5000)]
    roles = {GWPS.human._split_name(item) for item in candidates}
    assert roles == {"train", "validation", "test"}
    assert all(
        GWPS.human._split_name(item) == GWPS.human._split_name(item)
        for item in candidates
    )


def test_aligned_reader_preserves_absent_query_columns(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.asarray([[1, np.nan], [2, 3]], dtype=np.float32))
    values, observed = GWPS._read_aligned_rows(
        path,
        np.asarray([0, 1]),
        np.asarray([1, 0]),
        np.asarray([2, 0]),
    )
    assert values.shape == (2, GWPS.QUERY_COUNT)
    assert observed[:, 1].sum() == 0
    assert values[0, 2] == 0 and not observed[0, 2]
    assert values[1, 2] == 3 and observed[1, 2]
    assert np.array_equal(values[:, 0], [1, 2])
