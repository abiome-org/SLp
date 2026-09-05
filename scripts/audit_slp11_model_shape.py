"""Read-only synthetic probes for the 2026-09-04 scientific audit.

Run from any directory. Loads no biological data, protected outcomes or OMF
state; does not fit or change a model. Timing is descriptive, not a speed gate.
This repository audit utility is not an OMF training module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    architecture = load_module(
        "audit_sparse_architecture",
        "modules/slp-1-1-world-sparse/slp_sparse_architecture.py",
    )
    features = load_module(
        "audit_sequence_features",
        "modules/slp-1-1-sequence-statistics-feature-block-v1/feature_block.py",
    )
    peptide = b"ACDEFGHIKLMNPQRSTVWY"
    collision = features._feature_vector(peptide) == features._feature_vector(peptide[::-1])
    torch.set_num_threads(1)
    torch.manual_seed(731)
    config = architecture.WorldConfig(
        entity_feature_dim=21, species_feature_dim=1, entity_types=3,
        context_types=1, action_types=1, readout_types=1, d_model=16,
        nhead=4, encoder_layers=1, decoder_layers=1, ffn_multiplier=2,
        dropout=0.0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
        parameter_count = architecture.SparseTypedWorldModel(config).count_parameters()
    decoder = architecture.IndependentQueryDecoder(config).eval()
    queries = torch.randn(1, 1850, 16)
    memory = torch.randn(1, 2, 16)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    with torch.no_grad():
        decoder(queries[:, :2], memory, mask)
        started = time.perf_counter()
        serial = decoder(queries, memory, mask)
        serial_seconds = time.perf_counter() - started
        started = time.perf_counter()
        vectorized = queries
        for layer in decoder.layers:
            vectorized = layer(vectorized, memory, mask)
        vectorized = decoder.norm(vectorized)
        vectorized_seconds = time.perf_counter() - started
    close = torch.allclose(serial, vectorized, rtol=1e-5, atol=1e-5)
    report = {
        "scope": "synthetic architecture diagnostic; no biological performance evidence",
        "python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu",
        "seed": 731, "threads": 1, "queries": 1850, "memoryTokens": 2,
        "differentSequencesHaveIdenticalFeatures": collision,
        "featureDimensions": len(features._feature_vector(peptide)),
        "smokeShapeParameterCount": parameter_count,
        "serialDecoderSeconds": serial_seconds,
        "vectorizedDecoderSeconds": vectorized_seconds,
        "maximumAbsoluteDifference": float((serial - vectorized).abs().max()),
        "allcloseRtolAtol1eMinus5": close,
        "limitations": [
            "One CPU forward call, no backward pass or end-to-end timing",
            "Algebraic batching probe does not revise the frozen model contract",
            "Synthetic sequence collision does not measure collisions in real proteins",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not collision or not close:
        raise SystemExit("Probe assumptions did not reproduce; inspect runtime and source revisions")


if __name__ == "__main__":
    main()
