"""Portable, hash-verified inference directory with real weights and features."""

import json
import shutil
from pathlib import Path
import torch
from data import Features, digest
from model import Config, WorldModel


def export(directory, model, features_directory, *, vocabulary, basal, provenance):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    feature_root = Path(features_directory)
    Features(feature_root)  # Verify all pinned arrays before copying them.
    for name in ("genes.json", "sequence.npy", "annotation.npy", "known.npy"):
        shutil.copyfile(feature_root / name, root / name)
    shutil.copyfile(feature_root / "manifest.json", root / "features-manifest.json")
    for name in (
        "model.py",
        "records.py",
        "data.py",
        "artifact.py",
        "inference.py",
        "baselines.py",
        "requirements-linux-cu128.lock",
    ):
        shutil.copyfile(Path(__file__).parent / name, root / name)
    torch.save(model.state_dict(), root / "weights.pt")
    (root / "config.json").write_text(json.dumps(model.configuration(), sort_keys=True))
    (root / "vocabulary.json").write_text(json.dumps(vocabulary, sort_keys=True))
    (root / "basal.json").write_text(json.dumps(basal, sort_keys=True))
    receipt = {
        "schema": "slp.r2-inference-bundle/v1",
        "provenance": provenance,
        "files": {p.name: digest(p) for p in sorted(root.iterdir())},
    }
    (root / "bundle.json").write_text(json.dumps(receipt, sort_keys=True))
    return receipt


def load(directory, device="cpu"):
    root = Path(directory)
    receipt = json.loads((root / "bundle.json").read_text())
    if receipt.get("schema") != "slp.r2-inference-bundle/v1":
        raise ValueError("Unknown inference bundle")
    required = {
        "weights.pt",
        "config.json",
        "vocabulary.json",
        "basal.json",
        "features-manifest.json",
        "genes.json",
        "sequence.npy",
        "annotation.npy",
        "known.npy",
    }
    if not required <= set(receipt["files"]):
        raise ValueError("Incomplete inference bundle")
    for name, sha in receipt["files"].items():
        if Path(name).name != name or digest(root / name) != sha:
            raise ValueError("Inference artifact checksum mismatch")
    features = Features(root, manifest_name="features-manifest.json")
    configuration = json.loads((root / "config.json").read_text())
    if "baseline" in configuration:
        from baselines import PairBaseline

        model = PairBaseline(
            Config(**configuration["model"]), configuration["baseline"]
        ).to(device)
    else:
        model = WorldModel(Config(**configuration)).to(device)
    model.load_state_dict(
        torch.load(root / "weights.pt", map_location=device, weights_only=True),
        strict=True,
    )
    return (
        model.eval(),
        features,
        json.loads((root / "vocabulary.json").read_text()),
        json.loads((root / "basal.json").read_text()),
        receipt,
    )
