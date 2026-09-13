"""Append provenance-checked static GO terms on cloud compute; no outcome fitting."""

import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

FILTER = {
    "aspects": ["F", "C"],
    "excluded_evidence": ["HEP", "HGI", "HMP", "IEP", "IGI", "IMP"],
    "exclude_NOT": True,
    "date_maximum": "20221231",
    "direct_terms_only": True,
    "names_as_features": False,
}
FILES = {"genes.json", "sequence.npy", "annotation.npy", "known.npy"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def prepare(base, identity, sources, output, preparer_sha):
    base, identity, output = Path(base), Path(identity), Path(output)
    original = json.loads((base / "manifest.json").read_text())
    identity_bytes = identity.read_bytes()
    raw_identity_sha = hashlib.sha256(identity_bytes).hexdigest()
    canonical_identity_sha = hashlib.sha256(
        json.dumps(json.loads(identity_bytes), sort_keys=True).encode()
    ).hexdigest()
    if (
        original.get("schema") != "slp.static-features/v2"
        or original.get("fitted_human_intervention_genes") != []
    ):
        raise ValueError("Require static, provenance-bearing base features")
    if original.get("source_manifest_sha256") not in {
        raw_identity_sha,
        canonical_identity_sha,
    }:
        raise ValueError("Base features use another identity roster")
    if set(original["files"]) != FILES:
        raise ValueError("Unexpected base feature files")
    for name, sha in original["files"].items():
        if digest(base / name) != sha:
            raise ValueError("Base feature checksum mismatch")
    genes = json.loads((base / "genes.json").read_text())
    if len(set(genes)) != len(genes):
        raise ValueError("Duplicate feature identities")
    index = {gene: i for i, gene in enumerate(genes)}
    rows, provenance = {}, []
    for source in map(Path, sources):
        manifest = json.loads(source.read_text())
        if (
            manifest.get("schema") != "slp.static-go/v1"
            or manifest.get("fitted_human_intervention_genes") != []
            or manifest.get("filter") != FILTER
            or manifest.get("identity_manifest_sha256") != raw_identity_sha
            or manifest.get("source_module_sha256") != preparer_sha
            or manifest.get("taxon") not in (9606, 559292)
        ):
            raise ValueError("GO input violates the captured static-data contract")
        count = 0
        for shard in manifest["shards"]:
            name = shard["name"]
            if not re.fullmatch(r"annotations-\d{5}\.jsonl\.gz", name):
                raise ValueError("Invalid annotation shard path")
            path = source.parent / name
            if path.stat().st_size != shard["bytes"] or digest(path) != shard["sha256"]:
                raise ValueError("Annotation shard checksum mismatch")
            part_count = 0
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    gene, terms = row["gene"], row["terms"]
                    if (
                        gene not in index
                        or gene in rows
                        or not gene.startswith(str(manifest["taxon"]) + ":")
                        or not terms
                        or terms != sorted(set(terms))
                        or any(not re.fullmatch(r"[FC]:GO:\d{7}", t) for t in terms)
                    ):
                        raise ValueError(
                            "Invalid or ambiguous annotation identity/terms"
                        )
                    rows[gene] = terms
                    part_count += 1
            if part_count != shard["rows"]:
                raise ValueError("Annotation shard count mismatch")
            count += part_count
        if count != manifest["rows"]:
            raise ValueError("Annotation source count mismatch")
        provenance.append({"manifest_sha256": digest(source), "manifest": manifest})
    if not rows:
        raise ValueError("Empty functional annotation corpus")
    terms = sorted({term for values in rows.values() for term in values})
    term_index = {term: i for i, term in enumerate(terms)}
    old = np.load(base / "annotation.npy", mmap_mode="r", allow_pickle=False)
    if old.ndim != 2 or len(old) != len(genes) or not np.isfinite(old).all():
        raise ValueError("Invalid original annotation array")
    output.mkdir(parents=True, exist_ok=False)
    for name in sorted(FILES - {"annotation.npy"}):
        os.link(base / name, output / name)
    values = np.lib.format.open_memmap(
        output / "annotation.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(genes), old.shape[1] + len(terms)),
    )
    values[:] = 0
    values[:, : old.shape[1]] = old
    for gene, annotations in rows.items():
        values[index[gene], [old.shape[1] + term_index[t] for t in annotations]] = 1
    values.flush()
    del values
    result = dict(original)
    result["files"] = {name: digest(output / name) for name in sorted(FILES)}
    result["functional_annotations"] = {
        "schema": "slp.go-binary-features/v1",
        "vectorizer": "binary presence of direct F/C terms; no hashing or outcome fitting",
        "base_manifest_sha256": digest(base / "manifest.json"),
        "identity_manifest_raw_sha256": raw_identity_sha,
        "preparer_sha256": preparer_sha,
        "materializer_sha256": digest(__file__),
        "original_annotation_dim": old.shape[1],
        "annotation_dim": old.shape[1] + len(terms),
        "terms": terms,
        "annotated_genes": len(rows),
        "coverage_by_taxon": {
            str(taxon): {
                "annotated": sum(g.startswith(str(taxon) + ":") for g in rows),
                "roster": sum(g.startswith(str(taxon) + ":") for g in genes),
            }
            for taxon in (9606, 559292)
        },
        "sources": provenance,
        "fitted_human_intervention_genes": [],
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--annotations", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preparer-sha", required=True)
    args = p.parse_args()
    result = prepare(
        args.base, args.identity, args.annotations, args.output, args.preparer_sha
    )
    print(
        json.dumps(
            {
                "manifest_sha256": digest(args.output / "manifest.json"),
                **{
                    k: result["functional_annotations"][k]
                    for k in ("annotation_dim", "annotated_genes", "coverage_by_taxon")
                },
            }
        )
    )
