#!/usr/bin/env python3
"""Build frozen ESM2-t33 full/PCA protein and physical feature arms."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_SOURCE = ROOT / "scripts/build_slp11_human_sequence_features.py"
PHYSICAL_SOURCE = ROOT / "modules/slp-1-1-world-transition-v1/physical_features.py"
MODEL_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
MAX_RESIDUES = 1022
OVERLAP = 128
HARD_SECONDS = 3300
SHARD_SIZE = 128
MODEL_FILES = {
    ".gitattributes": (1438, "20cb01c2bd9b0ae0863422f9da41b20f10f93e009e657f411a41402f51ccf391"),
    "README.md": (1705, "462a2f24724e19c6be0efab926315c294a863c9a9770e2c8b3d859b2d81a07de"),
    "config.json": (724, "539095c22efc52a09d6147074ba4ca119f76a890df5901213b2b55f7d2f96b2b"),
    "model.safetensors": (2609506392, "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"),
    "special_tokens_map.json": (125, "3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1"),
    "tokenizer_config.json": (95, "7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad"),
    "vocab.txt": (93, "0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03"),
}
HASHES = {
    "entity_list": "102749fd616de67bdb34799048a32b8e3629e3488996458e335568f4fc2b0442",
    "fasta": "9b43da92651b35814597af6a8b18f500b768679a49fa4678224f384917ce7668",
    "development": "006b4bb127a09073a7f409d81a7bccce96bb961879cb5e57dce56b48eb8e664b",
    "go": "733ee609d61cb83c3619518e05eafd258915a973fff53a6676b93d245b91f06f",
    "esm8m_base": "a2f3153478c00c191e5a9e218badb3327a180a56948a4c9c6a6926cc506ff02b",
    "esm8m_physical": "2cbf12208461358b1c40b8ca5f51b3ebe6c363119f40a0d16ca87833f8e691f7",
    "string_alias": "b65f730b993ed0c1bd72edf4565d3d425db42861101b29699704810e8f125680",
    "string_links": "b28f494f58e1ace634ef1fe41734ada5be37f151e3168bb9658bc6ca1dd1a954",
    "physical_source": "363d7c1b0f07e490b463e86346ad0a6aac513831f900e9cc278977e771c0c249",
    "profile": "f16c349fc82372358aaf841c6650a3cb668603c99fbb9128558290c362abff23",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lf_hash(values: list[str]) -> tuple[int, str]:
    payload = ("\n".join(values) + "\n").encode("ascii")
    return len(payload), hashlib.sha256(payload).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_selected(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {field: archive[field] for field in fields}


def verify(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual}")


def unique_peptide_contract(
    entity_ids: list[str], translations: dict[str, object], sequence
) -> tuple[list[bytes], list[int], list[str], list[str]]:
    unique: list[bytes] = []
    lookup: dict[bytes, int] = {}
    row_indices = []
    present = []
    missing = []
    for entity_id in entity_ids:
        if entity_id not in translations:
            row_indices.append(-1)
            missing.append(entity_id)
            continue
        peptide = sequence.normalize_for_esm(translations[entity_id].peptide)
        index = lookup.get(peptide)
        if index is None:
            index = len(unique)
            lookup[peptide] = index
            unique.append(peptide)
        row_indices.append(index)
        present.append(entity_id)
    return unique, row_indices, present, missing


def shard_paths(directory: Path, index: int) -> tuple[Path, Path]:
    return directory / f"shard-{index:05d}.npz", directory / f"shard-{index:05d}.sha256"


def load_verified_shard(
    directory: Path,
    shard_index: int,
    indices: np.ndarray,
    peptides: list[bytes],
) -> np.ndarray | None:
    path, digest_path = shard_paths(directory, shard_index)
    if not path.exists() and not digest_path.exists():
        return None
    if not path.is_file() or not digest_path.is_file():
        raise ValueError(f"incomplete immutable shard {shard_index}")
    expected = digest_path.read_text(encoding="ascii").strip()
    if sha256_file(path) != expected:
        raise ValueError(f"immutable shard hash drift {shard_index}")
    with np.load(path, allow_pickle=False) as archive:
        stored_indices = archive["unique_index"]
        hashes = archive["peptide_sha256"]
        lengths = archive["peptide_length"]
        values = archive["feature_values"]
    expected_hashes = np.asarray([hashlib.sha256(peptides[index]).hexdigest() for index in indices])
    expected_lengths = np.asarray([len(peptides[index]) for index in indices], dtype=np.int64)
    if (
        not np.array_equal(stored_indices, indices)
        or not np.array_equal(hashes, expected_hashes)
        or not np.array_equal(lengths, expected_lengths)
        or values.shape != (len(indices), 1280)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"immutable shard content drift {shard_index}")
    return values


def extract_missing_shards(
    peptides: list[bytes],
    model_dir: Path,
    shard_dir: Path,
    *,
    started: float,
    sequence,
    device_name: str,
) -> dict[str, object]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    shard_dir.mkdir(exist_ok=True)
    shards = (len(peptides) + SHARD_SIZE - 1) // SHARD_SIZE
    missing = []
    for shard_index in range(shards):
        indices = np.arange(
            shard_index * SHARD_SIZE,
            min((shard_index + 1) * SHARD_SIZE, len(peptides)),
            dtype=np.int64,
        )
        if load_verified_shard(shard_dir, shard_index, indices, peptides) is None:
            missing.append(shard_index)
    if not missing:
        return {"shards": shards, "newShards": 0, "device": "not-loaded-resume"}
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(
        json.dumps({"event": "gpu-extraction-started", "missingShards": len(missing), "totalShards": shards}),
        flush=True,
    )
    completed = 0
    windows_done = 0
    for shard_index in missing:
        indices = np.arange(
            shard_index * SHARD_SIZE,
            min((shard_index + 1) * SHARD_SIZE, len(peptides)),
            dtype=np.int64,
        )
        values = np.empty((len(indices), 1280), dtype=np.float32)
        for local_row, unique_index in enumerate(indices):
            if time.monotonic() - started >= HARD_SECONDS:
                raise TimeoutError("hard 3300-second bound reached; completed shards remain resumable")
            peptide = peptides[int(unique_index)]
            windows = sequence.chunk_windows(len(peptide), MAX_RESIDUES, OVERLAP)
            coverage, weights = sequence.inverse_coverage_weights(len(peptide), windows)
            pooled = np.zeros(1280, dtype=np.float64)
            for window_index, (start, end) in enumerate(windows):
                text = peptide[start:end].decode("ascii")
                encoded = tokenizer(
                    text,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                if int(encoded["attention_mask"].sum()) != len(text) + 2:
                    raise RuntimeError("tokenizer truncated a peptide window")
                encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
                with torch.inference_mode():
                    hidden = model.esm(**encoded).last_hidden_state
                tokens = hidden[0, 1 : len(text) + 1].float().cpu().numpy()
                pooled += np.sum(
                    tokens.astype(np.float64) * weights[window_index][:, None], axis=0
                )
                windows_done += 1
                del hidden, tokens, encoded
            if np.any(coverage < 1):
                raise RuntimeError("peptide residues were omitted")
            values[local_row] = (pooled / np.float64(len(peptide))).astype(np.float32)
        path, digest_path = shard_paths(shard_dir, shard_index)
        np.savez_compressed(
            path,
            unique_index=indices,
            peptide_sha256=np.asarray([hashlib.sha256(peptides[index]).hexdigest() for index in indices]),
            peptide_length=np.asarray([len(peptides[index]) for index in indices], dtype=np.int64),
            feature_values=values,
        )
        digest_path.write_text(sha256_file(path) + "\n", encoding="ascii")
        completed += 1
        if completed == 1 or completed % 10 == 0 or completed == len(missing):
            print(
                json.dumps(
                    {
                        "event": "extraction-progress",
                        "newShards": completed,
                        "remainingShards": len(missing) - completed,
                        "windows": windows_done,
                        "elapsedSeconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
        name = torch.cuda.get_device_name(device)
    else:
        peak, name = 0, "CPU"
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps({"event": "gpu-extraction-finished", "elapsedSeconds": round(time.monotonic() - started, 1)}), flush=True)
    return {
        "shards": shards,
        "newShards": completed,
        "windowsExtractedThisRun": windows_done,
        "device": name,
        "peakAllocatedBytes": peak,
    }


def load_all_embeddings(shard_dir: Path, peptides: list[bytes]) -> tuple[np.ndarray, list[dict[str, object]]]:
    values = np.empty((len(peptides), 1280), dtype=np.float32)
    records = []
    shards = (len(peptides) + SHARD_SIZE - 1) // SHARD_SIZE
    for shard_index in range(shards):
        indices = np.arange(
            shard_index * SHARD_SIZE,
            min((shard_index + 1) * SHARD_SIZE, len(peptides)),
            dtype=np.int64,
        )
        shard = load_verified_shard(shard_dir, shard_index, indices, peptides)
        if shard is None:
            raise RuntimeError(f"missing shard {shard_index}")
        values[indices] = shard
        path, _ = shard_paths(shard_dir, shard_index)
        records.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return values, records


def parse_edges(entity_ids: np.ndarray, source_dir: Path) -> tuple[list[tuple[int, int, float]], dict[str, int]]:
    lookup = {str(gene): index for index, gene in enumerate(entity_ids)}
    mapping: dict[str, set[str]] = defaultdict(set)
    with gzip.open(source_dir / "9606.protein.aliases.v12.0.txt.gz", "rt", encoding="utf-8") as stream:
        next(stream)
        for line in stream:
            protein, gene, source = line.rstrip("\n").split("\t", 2)
            if protein.startswith("9606.") and source == "Ensembl_gene" and re.fullmatch(r"ENSG\d{11}", gene):
                mapping[protein].add(gene)
    exact = {protein: next(iter(genes)) for protein, genes in mapping.items() if len(genes) == 1}
    edges = []
    strong_rows = 0
    with gzip.open(source_dir / "9606.protein.physical.links.full.v12.0.txt.gz", "rt", encoding="utf-8") as stream:
        columns = next(stream).split()
        experiment_column = columns.index("experiments")
        for line in stream:
            fields = line.split()
            confidence = int(fields[experiment_column])
            if confidence < 700:
                continue
            strong_rows += 1
            left, right = exact.get(fields[0]), exact.get(fields[1])
            if left in lookup and right in lookup:
                edges.append((lookup[left], lookup[right], confidence / 1000))
    return edges, {
        "strongSourceRows": strong_rows,
        "ambiguousProteins": sum(len(genes) != 1 for genes in mapping.values()),
    }


def save_pack(path: Path, values: np.ndarray, taxa: np.ndarray, ids: np.ndarray) -> str:
    np.savez_compressed(path, feature_values=values.astype(np.float32), entity_taxon=taxa, entity_id=ids)
    return sha256_file(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    paths = {
        "entity_list": Path(args.entities),
        "fasta": Path(args.fasta),
        "development": Path(args.development),
        "go": Path(args.go),
        "esm8m_base": Path(args.esm8m_base),
        "esm8m_physical": Path(args.esm8m_physical),
        "string_alias": Path(args.string_source) / "9606.protein.aliases.v12.0.txt.gz",
        "string_links": Path(args.string_source) / "9606.protein.physical.links.full.v12.0.txt.gz",
        "physical_source": PHYSICAL_SOURCE,
        "profile": Path(args.profile),
    }
    for label, path in paths.items():
        verify(path, HASHES[label], label)
    model_dir = Path(args.model)
    if {path.name for path in model_dir.iterdir() if path.is_file()} != set(MODEL_FILES):
        raise ValueError("model file set mismatch")
    for name, (size, digest) in MODEL_FILES.items():
        path = model_dir / name
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"model file drift: {name}")

    entity_ids = paths["entity_list"].read_text(encoding="ascii").splitlines()
    if len(entity_ids) != 10231 or entity_ids != sorted(set(entity_ids)):
        raise ValueError("entity identity contract mismatch")
    sequence = load_module("esm_t33_builder_sequence", SEQUENCE_SOURCE)
    translations, source_counts = sequence.parse_longest_translations(paths["fasta"])
    peptides, row_indices, present_ids, missing_ids = unique_peptide_contract(entity_ids, translations, sequence)
    development = load_selected(paths["development"], ("action_ids", "split_train"))
    training_ids = sorted(
        {str(item) for item in development["action_ids"][development["split_train"]]}
    )
    fit_ids = [item for item in training_ids if item in translations]
    fit_bytes, fit_hash = lf_hash(fit_ids)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    source_copy = source_dir / Path(__file__).name
    if not source_copy.exists():
        shutil.copyfile(Path(__file__), source_copy)
    elif sha256_file(source_copy) != sha256_file(Path(__file__)):
        raise ValueError("resumed source snapshot differs")
    fit_roster = output / "pca-fit-entity-ids.txt"
    fit_payload = ("\n".join(fit_ids) + "\n").encode("ascii")
    if not fit_roster.exists():
        fit_roster.write_bytes(fit_payload)
    elif fit_roster.read_bytes() != fit_payload:
        raise ValueError("PCA fit roster drift")
    protocol = {
        "schema": "slp.esm2-t33-protein-encoder-feature-protocol/v1",
        "hypothesis": "Larger static ESM2 protein representations improve held-intervention-gene point forecasts in source contexts",
        "primaryArm": "esm650m_pca320_physical",
        "comparators": ["esm8m_physical"],
        "secondaryArm": "esm650m_full_physical",
        "rule": "primary PCA320 arm must improve gene-macro MSE by at least 1 percent and not regress centroid-adjusted profile Pearson versus 8M in each of three source contexts",
        "secondaryCannotRescuePrimaryFailure": True,
        "data": {"path": str(paths["development"]), "sha256": HASHES["development"], "metadataReadForFeatures": ["action_ids", "split_train"]},
        "pca": {"fitEntityCount": len(fit_ids), "fitRosterBytes": fit_bytes, "fitRosterSha256": fit_hash, "nComponents": 320, "svdSolver": "randomized", "iteratedPower": 7, "randomState": 731, "whiten": False, "input": "raw ESM2-t33 vectors for unique split_train action genes with present peptide"},
        "sequence": {"source": "Ensembl116 longest translation", "model": "facebook/esm2_t33_650M_UR50D", "revision": MODEL_REVISION, "hiddenSize": 1280, "windowResidues": MAX_RESIDUES, "overlapResidues": OVERLAP, "pooling": "inverse-overlap-weighted full-residue mean", "truncation": False, "dtype": "float32", "accumulation": "float64", "batchSize": 1, "tf32": False},
        "fusion": "replace ESM coordinates, preserve exact protein-present flag and GO256 rows",
        "physical": "unchanged STRING12 direct human experiments>=700 induced graph; recompute confidence-weighted neighbor features from each new fused base",
        "hardRuntimeSeconds": HARD_SECONDS,
        "profileSha256": HASHES["profile"],
        "outcomesReadForFeatureGeneration": False,
        "hepg2OrJurkatOutcomesRead": False,
        "sourceSha256": sha256_file(source_copy),
    }
    protocol_path = output / "protocol.json"
    protocol_payload = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    if not protocol_path.exists():
        protocol_path.write_text(protocol_payload, encoding="utf-8")
    elif protocol_path.read_text(encoding="utf-8") != protocol_payload:
        raise ValueError("frozen protocol drift on resume")
    print(json.dumps({"event": "protocol-frozen", "sha256": sha256_file(protocol_path), "fitGenes": len(fit_ids), "uniquePeptides": len(peptides)}), flush=True)

    extraction = extract_missing_shards(
        peptides,
        model_dir,
        output / "shards",
        started=started,
        sequence=sequence,
        device_name=args.device,
    )
    unique_embeddings, shard_records = load_all_embeddings(output / "shards", peptides)
    full_sequence = np.zeros((len(entity_ids), 1281), dtype=np.float32)
    present_mask = np.asarray(row_indices) >= 0
    full_sequence[present_mask, :1280] = unique_embeddings[np.asarray(row_indices)[present_mask]]
    full_sequence[present_mask, 1280] = 1.0

    import sklearn
    from sklearn.decomposition import PCA

    entity_lookup = {item: index for index, item in enumerate(entity_ids)}
    fit_rows = np.asarray([entity_lookup[item] for item in fit_ids], dtype=np.int64)
    pca = PCA(
        n_components=320,
        whiten=False,
        svd_solver="randomized",
        iterated_power=7,
        random_state=731,
    )
    pca.fit(full_sequence[fit_rows, :1280])
    pca_sequence = np.zeros((len(entity_ids), 321), dtype=np.float32)
    pca_sequence[present_mask, :320] = pca.transform(full_sequence[present_mask, :1280]).astype(np.float32)
    pca_sequence[:, 320] = full_sequence[:, 1280]
    if not np.array_equal(pca_sequence[:, 320], full_sequence[:, 1280]):
        raise RuntimeError("protein presence flag changed")

    go = load_selected(paths["go"], ("feature_values", "entity_taxon", "entity_id"))
    if not np.array_equal(go["entity_id"], np.asarray(entity_ids)) or not np.all(go["entity_taxon"] == 9606):
        raise ValueError("GO identity axis mismatch")
    full_base = np.concatenate((full_sequence, go["feature_values"]), axis=1)
    pca_base = np.concatenate((pca_sequence, go["feature_values"]), axis=1)
    old_base = load_selected(paths["esm8m_base"], ("feature_values", "entity_taxon", "entity_id"))
    edges, edge_stats = parse_edges(go["entity_id"], Path(args.string_source))
    physical = load_module("esm_t33_builder_physical", PHYSICAL_SOURCE)
    reproduced, old_summary = physical.neighborhood_features(old_base["feature_values"], edges)
    old_physical = load_selected(paths["esm8m_physical"], ("feature_values", "entity_taxon", "entity_id"))
    if not np.array_equal(reproduced, old_physical["feature_values"]):
        raise RuntimeError("physical graph recipe did not reproduce frozen 8M pack")
    pca_physical, pca_summary = physical.neighborhood_features(pca_base, edges)
    full_physical, full_summary = physical.neighborhood_features(full_base, edges)

    taxa = go["entity_taxon"].astype(np.int64)
    ids = go["entity_id"]
    output_files = {
        "esm650m_full_sequence": (output / "esm650m-full1281.npz", full_sequence),
        "esm650m_pca_sequence": (output / "esm650m-pca321.npz", pca_sequence),
        "esm650m_full_base": (output / "esm650m-full-go1537.npz", full_base),
        "esm650m_pca_base": (output / "esm650m-pca-go577.npz", pca_base),
        "esm650m_full_physical": (output / "esm650m-full-physical3076.npz", full_physical),
        "esm650m_pca320_physical": (output / "esm650m-pca320-physical1156.npz", pca_physical),
    }
    output_hashes = {}
    for label, (path, values) in output_files.items():
        if path.exists():
            raise ValueError(f"refusing to overwrite final artifact {path}")
        output_hashes[label] = save_pack(path, values, taxa, ids)
    basis_path = output / "pca-basis.npz"
    np.savez_compressed(
        basis_path,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        singular_values=pca.singular_values_.astype(np.float32),
        fit_entity_ids=np.asarray(fit_ids),
    )
    provenance_path = output / "selected-protein-provenance.jsonl"
    with provenance_path.open("w", encoding="ascii", newline="\n") as stream:
        for entity_id in present_ids:
            selected = translations[entity_id]
            peptide = sequence.normalize_for_esm(selected.peptide)
            record = {
                "entityId": entity_id,
                "ncbiTaxon": 9606,
                "selectedTranscriptId": f"{selected.transcript_id}.{selected.transcript_version}",
                "selectedProteinId": f"{selected.protein_id}.{selected.protein_version}",
                "peptideLength": len(peptide),
                "esmPeptideSha256": hashlib.sha256(peptide).hexdigest(),
                "modelRevision": MODEL_REVISION,
            }
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    manifest = {
        "schema": "slp.esm2-t33-protein-encoder-feature-arms/v1",
        "status": "exploratory-static-feature-candidates-not-omf-admitted",
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "arms": {
            "esm8m_physical": {"path": str(paths["esm8m_physical"]), "sha256": HASHES["esm8m_physical"], "dimensions": 1156},
            "esm650m_pca320_physical": {"path": str(output_files["esm650m_pca320_physical"][0]), "sha256": output_hashes["esm650m_pca320_physical"], "dimensions": 1156},
            "esm650m_full_physical": {"path": str(output_files["esm650m_full_physical"][0]), "sha256": output_hashes["esm650m_full_physical"], "dimensions": 3076},
        },
        "identity": {"entities": len(entity_ids), "entityListSha256": HASHES["entity_list"], "ncbiTaxon": 9606, "namespace": "Ensembl-gene", "orderedUnique": True},
        "sequence": {"present": len(present_ids), "missing": len(missing_ids), "missingEntityIds": missing_ids, "uniquePeptides": len(peptides), "sourceCounts": source_counts, "shards": shard_records, "extraction": extraction},
        "pca": {"basisPath": str(basis_path), "basisSha256": sha256_file(basis_path), "fitRosterPath": str(fit_roster), "fitRosterSha256": fit_hash, "fitEntities": len(fit_ids), "sklearnVersion": sklearn.__version__, "svdSolver": "randomized", "nComponents": 320, "iteratedPower": 7, "randomState": 731, "whiten": False, "meanCentered": True, "missingRowsTransformed": False, "explainedVarianceRatioSum": float(pca.explained_variance_ratio_.sum())},
        "fusion": {"goPath": str(paths["go"]), "goSha256": HASHES["go"], "goDimensions": 256, "goRowsBitExact": True, "presenceFlagBitExact": True},
        "physical": {"stringAliasSha256": HASHES["string_alias"], "stringLinksSha256": HASHES["string_links"], "edgeStats": edge_stats, "oldPackReproducedBitExact": True, "oldSummary": old_summary, "pcaSummary": pca_summary, "fullSummary": full_summary, "recipe": "confidence-weighted known-neighbor mean of complete fused base plus log1p degree and neighbor-presence"},
        "outputs": {label: {"path": str(path), "sha256": output_hashes[label], "shape": list(values.shape)} for label, (path, values) in output_files.items()},
        "provenance": {"path": str(provenance_path), "sha256": sha256_file(provenance_path), "records": len(present_ids)},
        "inputs": {label: {"path": str(path), "sha256": HASHES[label]} for label, path in paths.items()},
        "model": {"repository": "facebook/esm2_t33_650M_UR50D", "revision": MODEL_REVISION, "license": "MIT", "files": {name: {"bytes": spec[0], "sha256": spec[1]} for name, spec in MODEL_FILES.items()}},
        "accessBoundary": {"molecularValuesRead": False, "developmentMetadataFieldsRead": ["action_ids", "split_train"], "hepg2OutcomesRead": False, "jurkatOutcomesRead": False, "benchmarkLabelsRead": False},
        "elapsedSeconds": time.monotonic() - started,
    }
    manifest_path = output / "feature-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "feature-generation-complete", "manifest": str(manifest_path), "sha256": sha256_file(manifest_path), "arms": manifest["arms"], "elapsedSeconds": manifest["elapsedSeconds"]}), flush=True)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True)
    result.add_argument("--entities", required=True)
    result.add_argument("--fasta", required=True)
    result.add_argument("--development", required=True)
    result.add_argument("--go", required=True)
    result.add_argument("--esm8m-base", required=True)
    result.add_argument("--esm8m-physical", required=True)
    result.add_argument("--string-source", required=True)
    result.add_argument("--profile", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
