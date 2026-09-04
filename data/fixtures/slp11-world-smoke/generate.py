"""Generate deterministic, entirely synthetic corpora for the OMF world smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SPECS = {
    "pretrain": ("pretrain", 101, 0.0),
    "validation": ("molecular-validation", 301, 0.2),
    "reward": ("molecular-reward", 201, -0.2),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def generate(name: str, role: str, gene_number: int, offset: float) -> None:
    destination = ROOT / name
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7000 + gene_number)
    records = 8
    taxa = np.where(np.arange(records) % 2 == 0, 4932, 9606).astype(np.int64)
    species_features = np.asarray(
        [[1.0, 0.0] if taxon == 4932 else [0.0, 1.0] for taxon in taxa],
        dtype=np.float32,
    )
    context_features = rng.normal(size=(records, 2, 4)).astype(np.float32)
    action_features = rng.normal(size=(records, 2, 4)).astype(np.float32)
    action_covariates = rng.normal(size=(records, 2, 2)).astype(np.float32)
    query_features = rng.normal(size=(records, 2, 4)).astype(np.float32)
    action_mask = np.zeros((records, 2), dtype=bool)
    action_mask[:, 0] = True
    yeast_gene = f"SGD:S900{gene_number:06d}"
    human_gene = f"ENSEMBL:ENSG900{gene_number:06d}"
    action_curies = np.full((records, 2), "", dtype="<U32")
    action_curies[:, 0] = np.where(taxa == 4932, yeast_gene, human_gene)
    shared_effect = (
        0.65 * action_features[:, 0, 0]
        - 0.30 * action_features[:, 0, 1]
        + 0.20 * context_features[:, :, 2].mean(axis=1)
        + 0.10 * action_covariates[:, 0, 0]
        + np.where(taxa == 4932, offset, -offset)
    )
    target = np.stack(
        (
            shared_effect + 0.15 * query_features[:, 0, 3],
            shared_effect - 0.20 * query_features[:, 1, 1],
        ),
        axis=1,
    ).astype(np.float32)
    shard = destination / "shard-000.npz"
    np.savez(
        shard,
        record_id=np.asarray(
            [f"SLP_FIXTURE:{name}:record:{index}" for index in range(records)], dtype="<U48"
        ),
        source_id=np.full(records, f"SLP_FIXTURE:{name}", dtype="<U32"),
        perturbation_id=np.asarray(
            [f"SLP_FIXTURE:{name}:perturbation:{index}" for index in range(records)],
            dtype="<U56",
        ),
        context_features=context_features,
        context_mask=np.ones((records, 2), dtype=bool),
        action_features=action_features,
        action_covariates=action_covariates,
        action_curies=action_curies,
        action_mask=action_mask,
        query_features=query_features,
        query_mask=np.ones((records, 2), dtype=bool),
        readout_type=np.zeros((records, 2), dtype=np.int64),
        species_features=species_features,
        species_taxon=taxa,
        target=target,
        target_mask=np.ones((records, 2), dtype=bool),
    )
    (destination / "trajectory-genes.txt").write_text(
        f"{yeast_gene}\n{human_gene}\n", encoding="utf-8"
    )
    manifest = {
        "schema": "slp.corpus/v1",
        "datasetId": f"slp11-world-smoke-{name}",
        "version": "fixture-v1",
        "role": role,
        "labelClass": "molecular",
        "benchmarkLabelsPresent": False,
        "speciesTaxa": [4932, 9606],
        "modalities": ["synthetic-fixture"],
        "trajectoryGenes": "trajectory-genes.txt",
        "entityFeatureDim": 4,
        "speciesFeatureDim": 2,
        "speciesFeatureVectors": {"4932": [1.0, 0.0], "9606": [0.0, 1.0]},
        "actionCovariateDim": 2,
        "readoutTypes": ["synthetic-effect"],
        "shards": [
            {
                "path": shard.name,
                "sha256": _sha256(shard),
                "records": records,
            }
        ],
    }
    (destination / "corpus.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    for corpus_name, arguments in SPECS.items():
        generate(corpus_name, *arguments)
