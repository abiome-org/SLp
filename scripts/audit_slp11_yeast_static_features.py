"""Audit exact stable-ID coverage of frozen yeast static feature artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVELOPMENT = ROOT / "data/derived/slp11-yeast-atlas-response/nadal-ribelles-control-nacl-development-v1/development.npz"
DEFAULT_OUTPUT = ROOT / "data/derived/slp11-yeast-static-coverage/nadal-ribelles-development-v1"
PACKS = {
    "dipeptide": (
        ROOT / "data/derived/slp11-sequence/dipeptide-v1/sequence-dipeptide-features.npz",
        ROOT / "data/derived/slp11-sequence/dipeptide-v1/sequence-dipeptide-features.manifest.json",
        "amino-acid length/composition and order-sensitive dipeptide frequencies",
    ),
    "esm2_t6_8m": (
        ROOT / "data/derived/slp11-sequence/esm2-t6-8m-c731040f-full-v1/sequence-esm2-features.npz",
        ROOT / "data/derived/slp11-sequence/esm2-t6-8m-c731040f-full-v1/sequence-esm2-features.manifest.json",
        "full-protein windowed ESM-2 t6 8M sequence embedding",
    ),
    "go_mf_cc": (
        ROOT / "data/derived/slp11-go-direct-svd-2022-09-19/go-direct-svd-features.npz",
        ROOT / "data/derived/slp11-go-direct-svd-2022-09-19/go-direct-svd-features.manifest.json",
        "direct 2022-09-19 GO molecular-function/cellular-component SVD",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coverage(ids: np.ndarray, values: np.ndarray, requested: np.ndarray) -> dict:
    lookup = {identity: index for index, identity in enumerate(ids.astype(str))}
    exact = np.asarray([identity in lookup for identity in requested], dtype=bool)
    nonzero = np.asarray(
        [identity in lookup and bool(np.any(values[lookup[identity]] != 0)) for identity in requested],
        dtype=bool,
    )
    return {
        "requested": int(len(requested)),
        "exact_identity_rows": int(exact.sum()),
        "missing_identity_rows": int((~exact).sum()),
        "nonzero_feature_rows": int(nonzero.sum()),
        "zero_feature_rows_among_exact": int((exact & ~nonzero).sum()),
        "missing_ids": requested[~exact].astype(str).tolist(),
    }


def build(development_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"immutable output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    with np.load(development_path, allow_pickle=False) as development:
        action_by_record = development["action_ids"].astype(str)
        query_ids = np.unique(development["query_ids"].astype(str))
        action_ids = np.unique(action_by_record)
        train_actions = np.unique(action_by_record[development["split_train"]])
        validation_actions = np.unique(action_by_record[development["split_validation"]])
        taxon = int(development["ncbi_taxon"])
    if taxon != 4932 or set(train_actions) & set(validation_actions):
        raise ValueError("unexpected yeast taxon or non-isolated action split")

    reports = {}
    common_ids: np.ndarray | None = None
    for name, (path, manifest_path, modality) in PACKS.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"feature_values", "entity_taxon", "entity_id"}:
                raise ValueError(f"unexpected schema: {path}")
            ids = archive["entity_id"].astype(str)
            taxa = archive["entity_taxon"].astype(np.int64)
            values = archive["feature_values"].astype(np.float32)
        if len(set(ids)) != len(ids) or np.any(taxa != taxon) or not np.isfinite(values).all():
            raise ValueError(f"invalid feature identity/value contract: {path}")
        if common_ids is None:
            common_ids = ids
        elif not np.array_equal(common_ids, ids):
            raise ValueError("frozen static packs do not share an exact ordered entity axis")
        reports[name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
            "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": sha256(manifest_path),
            "schema": manifest["schema"],
            "modality": modality,
            "shape": list(values.shape),
            "action_coverage": coverage(ids, values, action_ids),
            "train_action_coverage": coverage(ids, values, train_actions),
            "validation_action_coverage": coverage(ids, values, validation_actions),
            "query_coverage": coverage(ids, values, query_ids),
        }

    assert common_ids is not None
    missing_actions = np.asarray(reports["esm2_t6_8m"]["action_coverage"]["missing_ids"], dtype=str)
    missing_queries = np.asarray(reports["esm2_t6_8m"]["query_coverage"]["missing_ids"], dtype=str)
    (output_dir / "missing-action-ids.txt").write_text("".join(f"{x}\n" for x in missing_actions), encoding="ascii")
    (output_dir / "missing-query-ids.txt").write_text("".join(f"{x}\n" for x in missing_queries), encoding="ascii")
    report = {
        "schema": "slp.yeast-static-feature-coverage/v1",
        "status": "exact-stable-id-static-audit",
        "development": {
            "path": str(development_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(development_path),
            "taxon": taxon,
            "action_ids": int(len(action_ids)),
            "train_action_ids": int(len(train_actions)),
            "validation_action_ids": int(len(validation_actions)),
            "query_ids": int(len(query_ids)),
        },
        "identity_join": "exact (NCBI taxon 4932, canonical SGD CURIE); no symbol or orthology join",
        "packs": reports,
        "recommended_comparison": {
            "composition_arm": "dipeptide 421 + optional GO MF/CC 256",
            "learned_sequence_arm": "ESM-2 t6 320 + optional GO MF/CC 256",
            "alignment": "the three existing packs have byte-exact ordered entity axes; concatenate only exact SGD rows and append explicit modality-present flags in a downstream adapter",
            "reason": "keeps sequence representation as the controlled difference while reusing one frozen annotation prior",
        },
        "missing_lists": {
            "missing-action-ids.txt": sha256(output_dir / "missing-action-ids.txt"),
            "missing-query-ids.txt": sha256(output_dir / "missing-query-ids.txt"),
        },
        "limitations": [
            "zero GO rows mean no eligible direct MF/CC annotation in the frozen release, not absence of function",
            "missing sequence rows are explicit and must not be imputed by display symbol",
            "these static features do not transfer or relabel quantitative human phenotypes as yeast outcomes",
            "coverage is metadata-only and does not evaluate molecular response values",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.development.resolve(), args.output_dir.resolve())["development"], indent=2))
