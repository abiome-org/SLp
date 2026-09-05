"""Checksum-validated inference for a frozen residual PCA-state transition."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FrozenResidualPcaTransition:
    """Forecast assay means from raw action features in known fitted contexts."""

    combination_supported = False

    def __init__(self, artifact: str | Path, *, device: str = "cpu"):
        self.root = Path(artifact).resolve(strict=True)
        manifest = json.loads((self.root / "artifact-manifest.json").read_text())
        for relative, expected in manifest["sha256"].items():
            if _sha256((self.root / relative).resolve(strict=True)) != expected:
                raise ValueError(f"artifact checksum mismatch: {relative}")
        transition = _load(self.root / "source/transition.py", "frozen_residual_transition")
        pca_core = _load(self.root / "source/paired_pca.py", "frozen_residual_pca")
        self.pca = pca_core.PcaForecastArtifact.load(self.root / "pca-forecast.npz")
        with np.load(self.root / "transition-reference.npz", allow_pickle=False) as archive:
            self.reference = {name: archive[name] for name in archive.files}
        self.device = torch.device(device)
        self.model = transition.ResidualStateTransition(
            transition.Config(1156, 128, hidden_dim=128, dropout=0.2)
        ).to(self.device)
        self.model.load_state_dict(load_file(self.root / "transition.safetensors"))
        self.model.eval()

    def forecast(
        self,
        raw_action_features: np.ndarray,
        context_index: np.ndarray,
        *,
        has_action: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        raw = np.asarray(raw_action_features, dtype=np.float32)
        context = np.asarray(context_index, dtype=np.int64)
        if raw.ndim != 2 or raw.shape[1] != 1156 or context.shape != (len(raw),):
            raise ValueError("action features and contexts must align")
        if np.any(context < 0) or np.any(context >= len(self.pca.context_names)):
            raise ValueError("context index outside frozen fitted contexts")
        present = np.ones(len(raw), dtype=bool) if has_action is None else np.asarray(has_action, dtype=bool)
        if present.shape != (len(raw),):
            raise ValueError("has_action must align with action records")
        normalized = self.pca.ridge.normalize(raw).astype(np.float32)
        base = self.pca.ridge.predict(raw, context).astype(np.float32)
        control_state = self.reference["control_state"][context].astype(np.float32)
        with torch.no_grad():
            delta = self.model(
                torch.as_tensor(normalized, device=self.device),
                torch.as_tensor(control_state, device=self.device),
                torch.as_tensor(base, device=self.device),
                torch.as_tensor(present, dtype=torch.bool, device=self.device),
            ).cpu().numpy()
        rna_delta, protein_delta = self.pca.pca.decode_delta(delta)
        rna = self.pca.rna_controls[context] + rna_delta
        protein = self.pca.protein_controls[context] + protein_delta
        # Preserve algebraic no-intervention identity exactly after NumPy decoding.
        rna[~present] = self.pca.rna_controls[context[~present]]
        protein[~present] = self.pca.protein_controls[context[~present]]
        return {
            "rna": rna.astype(np.float32),
            "protein": protein.astype(np.float32),
            "state_delta": delta.astype(np.float32),
        }
