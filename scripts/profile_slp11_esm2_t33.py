#!/usr/bin/env python3
"""Profile frozen ESM2-t33 full-peptide extraction on the Ensembl-116 roster."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_SOURCE = ROOT / "scripts/build_slp11_human_sequence_features.py"
MODEL_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
MODEL_FILES = {
    ".gitattributes": (1438, "20cb01c2bd9b0ae0863422f9da41b20f10f93e009e657f411a41402f51ccf391"),
    "README.md": (1705, "462a2f24724e19c6be0efab926315c294a863c9a9770e2c8b3d859b2d81a07de"),
    "config.json": (724, "539095c22efc52a09d6147074ba4ca119f76a890df5901213b2b55f7d2f96b2b"),
    "model.safetensors": (2609506392, "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"),
    "special_tokens_map.json": (125, "3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1"),
    "tokenizer_config.json": (95, "7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad"),
    "vocab.txt": (93, "0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03"),
}
ENTITY_COUNT = 10231
ENTITY_SHA256 = "102749fd616de67bdb34799048a32b8e3629e3488996458e335568f4fc2b0442"
FASTA_SHA256 = "9b43da92651b35814597af6a8b18f500b768679a49fa4678224f384917ce7668"
MAX_RESIDUES = 1022
OVERLAP = 128
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sequence_module():
    spec = importlib.util.spec_from_file_location("esm_t33_profile_sequence", SEQUENCE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen sequence parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_model(model_dir: Path) -> dict[str, dict[str, object]]:
    actual = {path.name for path in model_dir.iterdir() if path.is_file()}
    if actual != set(MODEL_FILES):
        raise ValueError("ESM2-t33 local model file set mismatch")
    result = {}
    for name, (size, digest) in MODEL_FILES.items():
        path = model_dir / name
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"ESM2-t33 file mismatch: {name}")
        result[name] = {"bytes": size, "sha256": digest}
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "esm"
        or config.get("hidden_size") != 1280
        or config.get("num_hidden_layers") != 33
        or config.get("max_position_embeddings") != 1026
    ):
        raise ValueError("ESM2-t33 architecture contract mismatch")
    return result


def representative_indices(lengths: np.ndarray, ids: list[str]) -> list[int]:
    order = sorted(range(len(lengths)), key=lambda index: (int(lengths[index]), ids[index]))
    indices = []
    for quantile in QUANTILES:
        position = round(quantile * (len(order) - 1))
        index = order[position]
        if index not in indices:
            indices.append(index)
    return indices


def fit_runtime_curve(observations: list[dict[str, float]], window_lengths: np.ndarray) -> dict[str, object]:
    lengths = np.asarray([item["residues"] for item in observations], dtype=np.float64)
    seconds = np.asarray([item["seconds"] for item in observations], dtype=np.float64)
    design = np.column_stack((np.ones(len(lengths)), lengths, np.square(lengths)))
    coefficients, *_ = np.linalg.lstsq(design, seconds, rcond=None)
    fitted = design @ coefficients
    total = np.sum(np.square(seconds - seconds.mean()))
    r_squared = 1.0 - float(np.sum(np.square(seconds - fitted)) / total) if total > 0 else 1.0
    full_design = np.column_stack(
        (
            np.ones(len(window_lengths)),
            window_lengths.astype(np.float64),
            np.square(window_lengths.astype(np.float64)),
        )
    )
    predicted = np.maximum(full_design @ coefficients, seconds.min())
    estimate = float(predicted.sum())
    return {
        "formula": "least-squares seconds = intercept + linear*residues + quadratic*residues^2; predictions floored at observed minimum",
        "coefficients": {
            "intercept": float(coefficients[0]),
            "linear": float(coefficients[1]),
            "quadratic": float(coefficients[2]),
        },
        "rSquaredOnProfileWindows": r_squared,
        "estimatedInferenceSeconds": estimate,
        "conservativeSecondsWith25PercentMargin": estimate * 1.25,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback")
    model_dir = Path(args.model)
    model_files = verify_model(model_dir)
    entity_path = Path(args.entities)
    fasta_path = Path(args.fasta)
    if sha256_file(entity_path) != ENTITY_SHA256:
        raise ValueError("10,231-entity roster hash mismatch")
    if sha256_file(fasta_path) != FASTA_SHA256:
        raise ValueError("Ensembl-116 FASTA hash mismatch")
    entity_ids = entity_path.read_text(encoding="ascii").splitlines()
    if len(entity_ids) != ENTITY_COUNT or entity_ids != sorted(set(entity_ids)):
        raise ValueError("10,231-entity roster ordering mismatch")
    sequence = load_sequence_module()
    translations, source_counts = sequence.parse_longest_translations(fasta_path)
    present_ids = [item for item in entity_ids if item in translations]
    peptides = [sequence.normalize_for_esm(translations[item].peptide) for item in present_ids]
    lengths = np.asarray([len(item) for item in peptides], dtype=np.int64)
    all_window_lengths = np.concatenate(
        [
            np.asarray([end - start for start, end in sequence.chunk_windows(int(length), MAX_RESIDUES, OVERLAP)])
            for length in lengths
        ]
    )
    sample_indices = representative_indices(lengths, present_ids)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    load_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.monotonic() - load_started
    probe = tokenizer("MA", add_special_tokens=True, return_special_tokens_mask=True)
    if probe["special_tokens_mask"] != [1, 0, 0, 1]:
        raise ValueError("ESM tokenizer special-token contract mismatch")

    # Warm up outside the timing observations.
    warmup = peptides[sample_indices[min(3, len(sample_indices) - 1)]][:512].decode("ascii")
    encoded = tokenizer(warmup, return_tensors="pt", truncation=False)
    with torch.inference_mode():
        _ = model.esm(
            **{name: value.to(device) for name, value in encoded.items()}
        ).last_hidden_state
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    observations: list[dict[str, float]] = []
    sample_reports = []
    embeddings = []
    for sample_index in sample_indices:
        peptide = peptides[sample_index]
        windows = sequence.chunk_windows(len(peptide), MAX_RESIDUES, OVERLAP)
        coverage, weights = sequence.inverse_coverage_weights(len(peptide), windows)
        pooled = np.zeros(1280, dtype=np.float64)
        sample_seconds = 0.0
        sample_peak = 0
        for window_index, (start, end) in enumerate(windows):
            text = peptide[start:end].decode("ascii")
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            if int(encoded["attention_mask"].sum()) != len(text) + 2:
                raise RuntimeError("profile tokenizer truncated a peptide window")
            encoded = {name: value.to(device) for name, value in encoded.items()}
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.synchronize(device)
            else:
                baseline = 0
            started = time.monotonic()
            with torch.inference_mode():
                hidden = model.esm(**encoded).last_hidden_state
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
            else:
                peak = 0
            elapsed = time.monotonic() - started
            token_values = hidden[0, 1 : len(text) + 1].float().cpu().numpy()
            pooled += np.sum(
                token_values.astype(np.float64) * weights[window_index][:, None], axis=0
            )
            observations.append(
                {
                    "residues": float(len(text)),
                    "seconds": elapsed,
                    "peakAllocatedBytes": float(peak),
                    "incrementalPeakBytes": float(max(peak - baseline, 0)),
                }
            )
            sample_seconds += elapsed
            sample_peak = max(sample_peak, peak)
            del hidden, token_values, encoded
        embedding = (pooled / np.float64(len(peptide))).astype(np.float32)
        if not np.isfinite(embedding).all() or np.any(coverage < 1):
            raise RuntimeError("nonfinite or incomplete full-peptide profile output")
        embeddings.append(embedding)
        sample_reports.append(
            {
                "length": len(peptide),
                "windows": len(windows),
                "seconds": sample_seconds,
                "peakAllocatedBytes": sample_peak,
                "maximumCoverage": int(coverage.max()),
                "peptideSha256": hashlib.sha256(peptide).hexdigest(),
            }
        )
    curve = fit_runtime_curve(observations, all_window_lengths)
    curve["modelLoadSeconds"] = load_seconds
    curve["estimatedEndToEndSeconds"] = curve["estimatedInferenceSeconds"] + load_seconds
    curve["conservativeEndToEndSeconds"] = (
        curve["conservativeSecondsWith25PercentMargin"] + load_seconds
    )
    curve["underOneHourEstimated"] = curve["conservativeEndToEndSeconds"] < 3600
    maximum_peak = int(max(item["peakAllocatedBytes"] for item in observations))
    if device.type == "cuda":
        device_total = int(torch.cuda.get_device_properties(device).total_memory)
        device_name = torch.cuda.get_device_name(device)
    else:
        device_total = 0
        device_name = "CPU"

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"
    source_dir.mkdir()
    shutil.copyfile(Path(__file__), source_dir / Path(__file__).name)
    np.savez_compressed(
        output / "representative-embeddings.npz",
        feature_values=np.stack(embeddings).astype(np.float32),
        peptide_length=np.asarray([item["length"] for item in sample_reports], dtype=np.int64),
        peptide_sha256=np.asarray([item["peptideSha256"] for item in sample_reports]),
    )
    report: dict[str, object] = {
        "schema": "slp.esm2-t33-650m-full-length-profile/v1",
        "status": "profile-only-full-extraction-not-run",
        "hypothesis": "ESM2-t33 650M full-peptide static extraction fits one RTX 4070 and can finish the exact 10,231-entity roster within one hour",
        "fixedFeasibilityRule": "conservative profiled estimate including 25% inference margin and model load is below 3,600 seconds without truncation or CUDA OOM",
        "model": {
            "repository": "facebook/esm2_t33_650M_UR50D",
            "revision": MODEL_REVISION,
            "license": "MIT",
            "architecture": "ESM-2 t33 650M UR50D",
            "hiddenSize": 1280,
            "finalLayer": 33,
            "files": model_files,
        },
        "roster": {
            "entities": len(entity_ids),
            "entityListSha256": ENTITY_SHA256,
            "presentPeptides": len(peptides),
            "missingPeptides": len(entity_ids) - len(peptides),
            "uniquePeptides": len(set(peptides)),
            "lengthQuantiles": {
                str(quantile): float(np.quantile(lengths, quantile)) for quantile in QUANTILES
            },
            "totalResidues": int(lengths.sum()),
            "totalWindows": len(all_window_lengths),
            "multiWindowPeptides": int((lengths > MAX_RESIDUES).sum()),
            "sourceCounts": source_counts,
        },
        "featureConfiguration": {
            "sequenceSource": "Ensembl release 116 GRCh38.p14 longest translated peptide per stable ENSG",
            "maximumResiduesPerWindow": MAX_RESIDUES,
            "overlapResidues": OVERLAP,
            "pooling": "inverse-overlap-weighted full-residue mean of final-layer token representations",
            "specialTokensExcluded": True,
            "truncation": False,
            "modelDtype": "float32",
            "accumulationDtype": "float64",
            "batchSize": 1,
            "tf32": False,
            "deterministicAlgorithms": True,
        },
        "profile": {
            "sampleSelection": "nearest peptide at fixed length quantiles; stable ENSG tie order",
            "sampleQuantiles": list(QUANTILES),
            "samples": sample_reports,
            "windowObservations": observations,
            "runtimeEstimate": curve,
            "memory": {
                "device": device_name,
                "deviceTotalBytes": device_total,
                "maximumAllocatedBytes": maximum_peak,
                "maximumAllocatedFraction": maximum_peak / device_total if device_total else None,
            },
        },
        "candidatePlan": {
            "full1280": "replace the 320 ESM coordinates with 1280 t33 coordinates; preserve protein-present flag, GO256, and physical graph recipe",
            "dimensionMatchedPca320": "fit PCA 1280->320 only on exact source training-gene static rows, without molecular values; transform all entities with the frozen basis",
            "fullExtractionRun": False,
            "goBasisRefit": False,
            "physicalGraphRefit": False,
            "molecularOutcomesConsumed": False,
        },
        "inputs": {
            "entityList": {"path": str(entity_path), "sha256": ENTITY_SHA256},
            "ensemblFasta": {"path": str(fasta_path), "sha256": FASTA_SHA256},
            "source": "sources/esm2-t33-650m-ur50d-static-protein-model.yaml",
            "rights": "rights/esm2-t33-650m-ur50d-mit.yaml",
        },
        "accessBoundary": {
            "staticProteinSequencesConsumed": True,
            "molecularOutcomesConsumed": False,
            "hepg2OutcomesConsumed": False,
            "jurkatOutcomesConsumed": False,
            "benchmarkLabelsConsumed": False,
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True)
    result.add_argument("--entities", required=True)
    result.add_argument("--fasta", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return result


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "event": "esm2-t33-profile-complete",
                "runtimeEstimate": report["profile"]["runtimeEstimate"],
                "memory": report["profile"]["memory"],
            }
        )
    )


if __name__ == "__main__":
    main()
