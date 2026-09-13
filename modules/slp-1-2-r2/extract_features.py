"""Frozen, versioned ESM-2 extraction on cloud GPU disks; never fit outcomes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

REPOSITORY = "facebook/esm2_t33_650M_UR50D"
REVISION = "08e4846e537177426273712802403f7ba8261b6c"
WEIGHT_SHA256 = "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def windows(length, size=1022, overlap=128):
    if length < 1 or not 0 <= overlap < size:
        raise ValueError("Invalid sequence window")
    if length <= size:
        return [(0, length)]
    starts = list(range(0, length - size + 1, size - overlap))
    if starts[-1] != length - size:
        starts.append(length - size)
    return [(start, start + size) for start in starts]


def weighted_windows(sequence):
    regions = windows(len(sequence))
    coverage = np.zeros(len(sequence), dtype=np.int32)
    for start, end in regions:
        coverage[start:end] += 1
    if not np.all(coverage > 0):
        raise AssertionError("Uncovered protein residues")
    return [
        (sequence[start:end], 1.0 / coverage[start:end].astype(np.float32))
        for start, end in regions
    ]


def read_inputs(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("fitted_human_intervention_genes") != []:
        raise ValueError("Sequence inputs must be static-only")
    rows = []
    for shard in manifest["shards"]:
        if Path(shard["name"]).name != shard["name"]:
            raise ValueError("Invalid shard path")
        file = path.parent / shard["name"]
        if sha256(file) != shard["sha256"]:
            raise ValueError("Sequence input shard checksum mismatch")
        with gzip.open(file, "rt") as stream:
            rows.extend(json.loads(line) for line in stream)
    if len(rows) != manifest["rows"] or len({r["gene"] for r in rows}) != len(rows):
        raise ValueError("Sequence roster count/identity mismatch")
    return sorted(rows, key=lambda r: r["gene"]), manifest


@torch.inference_mode()
def extract(args):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Frozen sequence extraction requires the requested CUDA device"
        )
    rows, source = read_inputs(args.inputs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    directory = Path(
        snapshot_download(
            REPOSITORY,
            revision=REVISION,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
            ],
        )
    )
    if sha256(directory / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("Frozen ESM model weight hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    encoder = (
        AutoModelForMaskedLM.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        .esm.eval()
        .cuda()
    )
    encoder.requires_grad_(False)
    if encoder.config.hidden_size != 1280 or encoder.config.num_hidden_layers != 33:
        raise ValueError("Unexpected ESM architecture")
    valid_letters = {
        x for x in tokenizer.get_vocab() if len(x) == 1 and x.isalpha() and x.isupper()
    }
    vectors = np.zeros((len(rows), 1280), dtype=np.float32)
    completed, replacements = set(), 0
    unique = {}
    for i, row in enumerate(rows):
        if row["sequence"] is not None:
            sequence = row["sequence"]
            if hashlib.sha256(sequence.encode()).hexdigest() != row["sequence_sha256"]:
                raise ValueError("Protein sequence hash mismatch")
            normalized = "".join(x if x in valid_letters else "X" for x in sequence)
            replacements += sum(a != b for a, b in zip(sequence, normalized))
            unique.setdefault(normalized, []).append(i)
    work = []
    for sequence, indices in unique.items():
        for fragment, weights in weighted_windows(sequence):
            work.append((fragment, weights, indices, len(sequence)))
    work.sort(key=lambda x: len(x[0]))
    at = 0
    while at < len(work):
        if time.monotonic() - start >= args.max_seconds:
            raise TimeoutError("Frozen extraction reached its finite time budget")
        end = at + 1
        while (
            end < len(work)
            and end - at < args.max_batch
            and (len(work[end][0]) + 2) * (end - at + 1) <= args.token_budget
        ):
            end += 1
        selected = work[at:end]
        encoded = tokenizer(
            [x[0] for x in selected],
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            truncation=False,
        )
        special = encoded.pop("special_tokens_mask")
        mask = (encoded["attention_mask"].bool() & ~special.bool()).cuda()
        hidden = encoder(
            **{k: v.cuda() for k, v in encoded.items()}
        ).last_hidden_state.float()
        for j, (fragment, weights, indices, total_length) in enumerate(selected):
            residues = hidden[j, mask[j]]
            if len(residues) != len(fragment):
                raise ValueError("ESM tokenizer changed residue accounting")
            pooled = (residues * torch.tensor(weights, device="cuda")[:, None]).sum(
                0
            ) / total_length
            vector = pooled.cpu().numpy()
            for index in indices:
                vectors[index] += vector
                completed.add(index)
        at = end
        if at % 100 < args.max_batch or at == len(work):
            print(
                json.dumps(
                    {
                        "event": "frozen_features",
                        "windows": at,
                        "total_windows": len(work),
                        "elapsed_seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite ESM features")
    np.save(output / "sequence.npy", vectors, allow_pickle=False)
    np.save(
        output / "annotation.npy",
        np.asarray([r["annotation"] for r in rows], dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        output / "known.npy",
        np.asarray([r["known"] for r in rows], dtype=np.bool_),
        allow_pickle=False,
    )
    (output / "genes.json").write_text(json.dumps([r["gene"] for r in rows]))
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    manifest = {
        "schema": "slp.static-features/v2",
        "files": files,
        "fitted_human_intervention_genes": [],
        "model": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "weights_sha256": WEIGHT_SHA256,
        },
        "source_manifest_sha256": sha256(args.inputs),
        "genes": len(rows),
        "embedded_genes": len(completed),
        "unique_sequences": len(unique),
        "window_size": 1022,
        "overlap": 128,
        "pooling": "mean residues; inverse overlap coverage",
        "precision": "bf16 encoder, fp32 pooling",
        "nonvocabulary_residues_mapped_to_X": replacements,
        "elapsed_seconds": time.monotonic() - start,
        "torch": torch.__version__,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--max-batch", type=int, default=16)
    extract(parser.parse_args())
