"""Portable inference for a frozen four-context mean-objective arm."""

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


def _load_model_source(path: Path):
    name = f"slp11_four_context_mean_{path.stat().st_size}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load snapshotted transition model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FrozenMeanArm:
    """Load one immutable arm from its own source and numeric payloads."""

    def __init__(self, arm_path: str | Path, *, device: str = "cpu"):
        self.root = Path(arm_path).resolve(strict=True)
        manifest = json.loads((self.root / "artifact-manifest.json").read_text())
        for relative, expected in manifest["sha256"].items():
            candidate = (self.root / relative).resolve(strict=True)
            if _sha256(candidate) != expected:
                raise ValueError(f"artifact checksum mismatch: {relative}")
        source = self.root.parent / "source" / "control_transition_model.py"
        module = _load_model_source(source)
        with np.load(self.root / "reference.npz", allow_pickle=False) as archive:
            self.reference = {name: archive[name] for name in archive.files}
        config = module.Config(
            action_feature_dim=int(self.reference["feature_mean"].shape[0]),
            query_feature_dim=int(self.reference["query_feature_mean"].shape[0]),
            hidden_dim=int(self.reference["hidden_dim"]),
            state_dim=int(self.reference["state_dim"]),
            dropout=float(self.reference["dropout"]),
        )
        self.device = torch.device(device)
        self.model = module.MinimalControlTransition(config).to(self.device)
        self.model.load_state_dict(load_file(self.root / "model.safetensors"))
        self.model.eval()

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def predict(
        self,
        raw_action_features: np.ndarray,
        context_index: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Predict all frozen queries from raw static action features and contexts."""

        action = np.asarray(raw_action_features, dtype=np.float32)
        context = np.asarray(context_index)
        if action.ndim not in (2, 3) or context.shape != (action.shape[0],):
            raise ValueError("action features and context rows do not align")
        if context.dtype.kind not in "iu" or np.any(context < 0) or np.any(
            context >= len(self.reference["context_ids"])
        ):
            raise ValueError("context index is out of range")
        feature_mean = self.reference["feature_mean"]
        feature_std = self.reference["feature_std"]
        normalized_action = (action - feature_mean) / feature_std
        query = (
            self.reference["query_features"]
            - self.reference["query_feature_mean"]
        ) / self.reference["query_feature_std"]
        selected = self.reference["context_query_indices"]
        with torch.no_grad():
            result = self.model(
                self._tensor(normalized_action),
                self._tensor(query),
                self._tensor(self.reference["control_mean"][context]),
                self._tensor(self.reference["delta_amplitude"]),
                self._tensor(self.reference["objective_query_scale"][context]),
                self._tensor(query[selected]),
                self._tensor(self.reference["context_values"][context]),
                torch.as_tensor(
                    self.reference["context_mask"][context],
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
        exposed = ("mean", "delta", "state", "basal_state", "intervention_delta")
        return {name: result[name].detach().cpu().numpy() for name in exposed}


def empty_identity_audit(arm_path: str | Path) -> dict[str, object]:
    """Run a target-free empty-intervention identity check on every context."""

    arm = FrozenMeanArm(arm_path)
    contexts = len(arm.reference["context_ids"])
    empty = np.empty(
        (contexts, 0, len(arm.reference["feature_mean"])), dtype=np.float32
    )
    result = arm.predict(empty, np.arange(contexts, dtype=np.int64))
    control = arm.reference["control_mean"]
    return {
        "meanBitExact": bool(np.array_equal(result["mean"], control)),
        "deltaNonzero": int(np.count_nonzero(result["delta"])),
        "latentDeltaNonzero": int(np.count_nonzero(result["intervention_delta"])),
        "contextsChecked": contexts,
        "queriesChecked": int(control.shape[1]),
    }
