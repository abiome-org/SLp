import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np


def load_runner():
    path = Path(__file__).parents[1] / "scripts/run_slp11_gears_response_models.py"
    spec = importlib.util.spec_from_file_location("feature_join_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_string_lookup_is_independent_of_static_roster(tmp_path):
    runner = load_runner()
    static = tmp_path / "static.npz"
    pack = tmp_path / "pack.npz"
    string_h5 = tmp_path / "string.h5"
    np.savez(static, entity_id=np.asarray(["ENSG1"]), feature_values=np.ones((1, 577)))
    np.savez(pack, entity_id=np.asarray(["ENSG1"]), gene_symbol=np.asarray(["A"]),
             feature_values=np.ones((1, 64)), feature_present=np.asarray([True]))
    with h5py.File(string_h5, "w") as h5:
        h5.create_dataset("A", data=np.ones(64))
        h5.create_dataset("B", data=np.full(64, 2.0))
    runner.STATIC = {"x": static}; runner.SYMBOL = {"x": pack}; runner.STRING = string_h5
    _, string, concat, present = runner.feats("x", np.asarray(["A+ctrl", "B+ctrl"]))
    assert present.tolist() == [True, True]
    assert np.all(string[1] == 2.0)
    assert np.all(concat[1, :577] == 0.0)
    assert concat[1, -1] == 1.0
