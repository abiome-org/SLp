import importlib.util
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import save_file

PATH = Path(__file__).resolve().parents[1] / "scripts/verify_slp11_joint_world_portability.py"
ROOT = PATH.parents[1]
SPEC = importlib.util.spec_from_file_location("joint_world_portability", PATH)
MODULE = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def test_compare_requires_identity_and_reports_numeric_drift(tmp_path):
    arrays = {}
    for context in ("k562", "rpe1", "norman", "gwps", "hepg2", "mcf10a_full_d6"):
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
    assert result["mcf10a_full_d6"]["maxAbsDrift"] == 0


def test_wsl_path_conversion():
    value = MODULE._wsl_path(Path("C:/Users/Jack/example.npz"))
    assert value == "/mnt/c/Users/Jack/example.npz"


def test_export_manifest_verifies_every_regular_file(tmp_path):
    payload = tmp_path / "weights.bin"
    payload.write_bytes(b"weights")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"weights.bin": digest}}))
    assert MODULE._verify_manifest(tmp_path)["verifiedFiles"] == 1
    payload.write_bytes(b"changed")
    try:
        MODULE._verify_manifest(tmp_path)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("changed export file was accepted")


def test_core_safe_open_loader_preserves_numpy_serialized_tensors(tmp_path):
    module_dir = ROOT / "modules" / "slp-1-1-joint-world-v1"
    sys.path.insert(0, str(module_dir))
    try:
        spec = importlib.util.spec_from_file_location("joint_world_inference_loader_test",
                                                       module_dir / "inference.py")
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)
    finally:
        sys.path.pop(0)
    expected = {
        "weight": np.ascontiguousarray(np.arange(12, dtype=np.float32).reshape(3, 4)),
        "bias": np.ascontiguousarray(np.array([-2.0, 0.5, 7.0], dtype=np.float32)),
    }
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_file(expected, str(checkpoint))
    actual = inference.load_file(checkpoint, device="cpu")
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name].device.type == "cpu"
        assert torch.equal(actual[name], torch.from_numpy(value))
