import importlib.util
import sys
from pathlib import Path
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/verify_slp11_joint_world_portability.py"
SPEC = importlib.util.spec_from_file_location("joint_world_portability", PATH)
MODULE = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def test_compare_requires_identity_and_reports_numeric_drift(tmp_path):
    arrays = {}
    for context in MODULE.CONTEXTS:
        arrays[f"{context}_query_ids"] = np.array(["a", "b"])
        arrays[f"{context}_supported"] = np.array([True, False])
        arrays[f"{context}_empty_observed"] = np.zeros((2, 2))
        arrays[f"{context}_prediction"] = np.zeros((4, 2))
    left = tmp_path / "left.npz"; right = tmp_path / "right.npz"
    np.savez_compressed(left, **arrays)
    changed = dict(arrays); changed["hepg2_prediction"] = np.full((4, 2), .25)
    np.savez_compressed(right, **changed)
    result = MODULE._compare(left, right)
    assert result["k562"]["maxAbsDrift"] == 0
    assert result["hepg2"]["maxAbsDrift"] == .25
    assert result["hepg2"]["values"] == 8


def test_wsl_path_conversion():
    value = MODULE._wsl_path(Path("C:/Users/Jack/example.npz"))
    assert value == "/mnt/c/Users/Jack/example.npz"
