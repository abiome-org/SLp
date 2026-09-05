"""Checksum-validated inference for the fitted Frangieh cell-state experiment."""

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


def _load_core(path: Path):
    spec = importlib.util.spec_from_file_location("slp11_frozen_cell_state", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen cell-state source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FrozenCellState:
    """Load one fitted encoder, affine decoder and latent-ridge forecast payload."""

    def __init__(self, artifact: str | Path, *, device: str = "cpu"):
        self.root = Path(artifact).resolve(strict=True)
        manifest = json.loads((self.root / "artifact-manifest.json").read_text())
        for relative, expected in manifest["sha256"].items():
            candidate = (self.root / relative).resolve(strict=True)
            if _sha256(candidate) != expected:
                raise ValueError(f"artifact checksum mismatch: {relative}")
        core = _load_core(self.root / "source/cell_state.py")
        with np.load(self.root / "reference.npz", allow_pickle=False) as archive:
            self.reference = {name: archive[name] for name in archive.files}
        self.device = torch.device(device)
        config = core.Config(
            rna_features=int(self.reference["rna_query_features"].shape[1]),
            protein_features=int(self.reference["protein_query_features"].shape[1]),
            key_dim=int(self.reference["key_dim"]),
            state_dim=int(self.reference["state_dim"]),
            hidden_dim=int(self.reference["hidden_dim"]),
            dropout=float(self.reference["dropout"]),
        )
        self.model = core.CellState(config).to(self.device)
        self.model.load_state_dict(load_file(self.root / "model.safetensors"))
        self.model.eval()

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def encode(
        self,
        rna_values: np.ndarray,
        protein_values: np.ndarray,
        rna_observed: np.ndarray | None = None,
        protein_observed: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode transformed source-space paired-cell measurements."""

        rna = np.asarray(rna_values, dtype=np.float32)
        protein = np.asarray(protein_values, dtype=np.float32)
        if rna.ndim != 2 or protein.ndim != 2 or len(rna) != len(protein):
            raise ValueError("paired cell matrices must align")
        rna_mask = (
            np.ones_like(rna, dtype=bool)
            if rna_observed is None
            else np.asarray(rna_observed, dtype=bool)
        )
        protein_mask = (
            np.ones_like(protein, dtype=bool)
            if protein_observed is None
            else np.asarray(protein_observed, dtype=bool)
        )
        rna_standard = (rna - self.reference["rna_mean"]) / self.reference["rna_sd"]
        protein_standard = (
            protein - self.reference["protein_mean"]
        ) / self.reference["protein_sd"]
        with torch.no_grad():
            state = self.model.encode(
                self._tensor(self.reference["rna_query_features"]),
                self._tensor(rna_standard),
                torch.as_tensor(rna_mask, dtype=torch.bool, device=self.device),
                self._tensor(self.reference["protein_query_features"]),
                self._tensor(protein_standard),
                torch.as_tensor(protein_mask, dtype=torch.bool, device=self.device),
            )
        return state.cpu().numpy()

    def forecast(
        self,
        action_features: np.ndarray,
        context_index: np.ndarray,
        *,
        has_action: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Forecast endpoint means from raw physical1156 action features."""

        raw_features = np.asarray(action_features, dtype=np.float32)
        context = np.asarray(context_index)
        if raw_features.ndim != 2 or context.shape != (len(raw_features),):
            raise ValueError("action feature/context rows do not align")
        if raw_features.shape[1] != len(self.reference["feature_mean"]):
            raise ValueError("action feature dimensions do not match frozen normalizer")
        features = np.clip(
            (raw_features - self.reference["feature_mean"])
            / self.reference["feature_scale"],
            -float(self.reference["feature_clip"]),
            float(self.reference["feature_clip"]),
        ).astype(np.float32)
        if np.any(context < 0) or np.any(context >= len(self.reference["context_names"])):
            raise ValueError("context index outside fitted source contexts")
        present = (
            np.ones(len(features), dtype=bool)
            if has_action is None
            else np.asarray(has_action, dtype=bool)
        )
        if present.shape != (len(features),):
            raise ValueError("has_action must align with action rows")
        state_delta = np.empty((len(features), int(self.reference["state_dim"])), dtype=np.float32)
        for index in range(len(self.reference["context_names"])):
            rows = context == index
            state_delta[rows] = (
                features[rows] @ self.reference["ridge_coef"][index].T
                + self.reference["ridge_intercept"][index]
            )
        state_delta[~present] = 0.0
        output = {}
        with torch.no_grad():
            delta = self._tensor(state_delta)
            for head in ("rna", "protein"):
                query = self._tensor(self.reference[f"{head}_query_features"])
                controls = self._tensor(self.reference[f"{head}_controls"][context])
                scale = self._tensor(self.reference[f"{head}_sd"])
                output[head] = self.model.observe_delta(
                    delta, query, head, controls, scale
                ).cpu().numpy()
        output["state_delta"] = state_delta
        return output
