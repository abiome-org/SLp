"""Prepare molecular population arrays in Cloudflare without old model features.

Coordinates stay in dense float arrays; repeated action/assay metadata are stored
once per population. This avoids a many-fold expansion of RNA storage.
"""

from collections import Counter
import gzip
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
import urllib.request
import numpy as np
from inspect_corpus import request, save
from pack_corpus import Writer, MECHANISMS, METHODS, encode, metadata, IDENTITY


def put_part(name, body):
    compressed = gzip.compress(body, compresslevel=1, mtime=0)
    with request(
        f"/outputs/{os.environ['SLP_JOB']}/{name}", data=compressed, prep=True
    ) as response:
        response.read(4096)
    return {
        "name": name,
        "bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_bytes": len(body),
    }


def publish_file(path, label):
    parts = []
    sha = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as file:
        while chunk := file.read(16 * 1024 * 1024):
            sha.update(chunk)
            size += len(chunk)
            parts.append(put_part(f"{label}-part{len(parts):05d}.bin.gz", chunk))
    return {"parts": parts, "bytes": size, "sha256": sha.hexdigest()}


def fetch_source(spec, directory, pins, label):
    url = f"https://huggingface.co/datasets/{pins['repository']}/resolve/{pins['revision']}/{spec['remote']}"
    path = Path(directory) / Path(spec["remote"]).name
    sha = hashlib.sha256()
    size = 0
    req = urllib.request.Request(url, headers={"User-Agent": "SLp-Cloud-Storage/1.0"})
    with urllib.request.urlopen(req, timeout=180) as response, path.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            sha.update(chunk)
            size += len(chunk)
            if size > spec["size"]:
                raise ValueError("Molecular source exceeds pinned size")
            out.write(chunk)
    if size != spec["size"] or sha.hexdigest() != spec["sha256"]:
        raise ValueError("Molecular source checksum mismatch")
    raw = publish_file(path, "raw-" + label)
    save(
        "raw-" + label,
        {
            "source": spec,
            "repository": pins["repository"],
            "revision": pins["revision"],
            "artifact": raw,
        },
    )
    return path


class Units(Writer):
    def __init__(self, genes, name):
        super().__init__(genes, name)
        self.chunks = []

    def flush(self):
        if self.length:
            self.chunks.append(self.buffer[: self.length].copy())
        self.buffer.fill(0)
        self.buffer["targets"] = -1
        self.buffer["query"] = -1
        self.length = 0

    def array(self):
        self.flush()
        if not self.chunks:
            raise ValueError("No resolved molecular populations")
        return np.concatenate(self.chunks)


def basic_control(z, columns):
    if "control_context_values" in z:
        value = np.asarray(z["control_context_values"], np.float64)
        if value.ndim == 1:
            mask = (
                np.asarray(z["control_context_observed"], bool)
                if "control_context_observed" in z
                else np.isfinite(value)
            )
            return value[columns], mask[columns], "source control_context_values"
    if "control_basal_cp10k_rate" in z:
        rates = np.asarray(z["control_basal_cp10k_rate"], np.float64)
        counts = np.asarray(z["control_cell_count"], np.float64)
        mask = np.asarray(z["control_basal_observed"], bool)
        weights = counts[:, None] * mask
        denom = weights.sum(0)
        value = np.log1p((rates * weights).sum(0) / np.maximum(denom, 1))
        return (
            value[columns],
            denom[columns] > 0,
            "ln1p control-only weighted mean CP10K",
        )
    return None, None, "raw basal expression unavailable in this prepared source"


@lru_cache(maxsize=16)
def control_basis(genes):
    signs = (
        np.stack(
            [
                np.unpackbits(
                    np.frombuffer(
                        hashlib.sha256(("basal-projection:731:" + g).encode()).digest(),
                        dtype=np.uint8,
                    )
                )[:128]
                for g in genes
            ]
        ).astype(np.float64)
        * 2
        - 1
    )
    order = np.array(
        sorted(
            range(len(genes)),
            key=lambda i: hashlib.sha256(
                ("basal-panel:731:" + genes[i]).encode()
            ).digest(),
        ),
        np.int64,
    )
    return signs, order


def controls(genes, values, observed, provenance):
    values = np.asarray(values, np.float64)
    valid = np.asarray(observed, bool) & np.isfinite(values)
    signs, order = control_basis(tuple(genes))
    projected = (
        (np.where(valid, values, 0) / 10.0) @ signs / max(int(valid.sum()), 1) ** 0.5
    )
    indices = order[valid[order]][:64]
    return {
        "role": "unperturbed_control",
        "provenance": provenance,
        "context": projected.tolist(),
        "projection": "fixed SHA256 gene-sign projection, 128 coordinates, value/10/sqrt(observed genes); no fitted parameters",
        "observations": [
            {"gene": genes[i], "kind": "rna", "value": float(values[i])}
            for i in indices
        ],
    }


def yeast_aliases():
    """Exact current ORF identities from the repository's admitted SGD snapshot."""
    url = "https://downloads.yeastgenome.org/curation/chromosomal_feature/SGD_features.tab"
    expected = "ac66e4df1b31c002a4aeb03b1bccbd5b4a0b360ee2c2a42a8a9f427473462dc8"
    req = urllib.request.Request(url, headers={"User-Agent": "SLp-Cloud-Storage/1.0"})
    with urllib.request.urlopen(req, timeout=180) as response:
        body = response.read(3383040)
        version = response.headers.get("x-amz-version-id")
    if len(body) != 3383039 or hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("SGD identity snapshot mismatch")
    part = put_part("sgd-features.bin.gz", body)
    save(
        "sgd-mapping-source",
        {
            "url": url,
            "sha256": expected,
            "bytes": len(body),
            "part": part,
            "observed_object_version": version,
            "rights": "rights/sgd-identity-2026-09-11-cc-by-4.0.yaml",
            "columns": {"SGDID": 1, "feature_type": 2, "systematic_name": 4},
        },
    )
    candidates = {}
    irregular = 0
    for line in body.decode().splitlines():
        row = line.split("\t")
        if len(row) != 16:
            irregular += 1
            continue
        if row[1] == "ORF" and row[3]:
            candidates.setdefault("SGD:" + row[0], set()).add("559292:" + row[3])
    save(
        "sgd-mapping-audit",
        {
            "irregular_rows_quarantined": irregular,
            "unique_primary_ids": len(candidates),
            "ambiguous_primary_ids": sum(len(v) > 1 for v in candidates.values()),
        },
    )
    return {
        key: next(iter(values))
        for key, values in candidates.items()
        if len(values) == 1
    }


def main():
    jobs = json.loads(Path("prep-jobs.json").read_text())
    cohort = jobs[os.environ["SLP_JOB"]]["cohort"]
    pins = json.loads(Path("molecular-inputs.json").read_text())
    genes = metadata(IDENTITY, "sequence-inputs-manifest")["genes"]
    index = {g: i for i, g in enumerate(genes)}
    aliases = metadata(IDENTITY, "aliases")["human"]
    yeast = cohort == "yeast"
    yeast_map = yeast_aliases() if yeast else {}

    def canonical(value):
        if yeast:
            return yeast_map.get(str(value))
        return aliases.get(str(value).split(".")[0])

    with tempfile.TemporaryDirectory() as directory:
        selected = (
            [
                s
                for s in pins["files"]
                if "/nadal-ribelles-rna-neural-fitting-v1/" in s["remote"]
            ]
            if yeast
            else [
                s for s in pins["files"] if s["remote"].endswith("/" + cohort + ".npz")
            ]
        )
        paths = {
            Path(s["remote"]).name: fetch_source(
                s, directory, pins, Path(s["remote"]).name.replace(".", "-")
            )
            for s in selected
        }
        if yeast:
            z = np.load(paths["reference.npz"], allow_pickle=False)
            extra = np.load(paths["train-metadata.npz"], allow_pickle=False)
            target = np.load(
                paths["train-targets.npy"], allow_pickle=False, mmap_mode="r"
            )
            raw_actions = np.asarray(extra["action_ids"]).astype(str)
            raw_queries = np.asarray(z["query_ids"]).astype(str)
            units = "mean-per-cell-ln1p-CP10K"
            mechanism = "deletion"
            method = "unknown"
            hours = None
            batch_indices = np.asarray(extra["batch_index"], int)
            base_context = "Nadal-Ribelles-2025"
        else:
            z = np.load(paths[cohort + ".npz"], allow_pickle=False)
            target = np.asarray(z["targets"], np.float32)
            raw_actions = np.asarray(z["action_ids"]).astype(str)
            raw_queries = np.asarray(z["query_ids"]).astype(str)
            units = str(z["target_units"].item())
            mechanism = (
                str(z["intervention_mode"].item()).lower()
                if "intervention_mode" in z
                else "knockout"
            )
            if mechanism in ("ko", "crisprko", "crispr", "crispr-cas9 knockout"):
                mechanism = "knockout"
            if mechanism not in ("crispri", "crispra", "knockout"):
                raise ValueError("Unknown molecular intervention mechanism")
            method = (
                "dCas9-KRAB"
                if mechanism == "crispri"
                else "Cas9"
                if mechanism == "knockout"
                else "unknown"
            )
            hours = {
                "k562": 144.0,
                "rpe1": 168.0,
                "norman": 120.0,
                "mcf10a_full_d0": 0.0,
                "mcf10a_full_d6": 144.0,
                "mcf10a_tgfb1_d6": 144.0,
            }.get(cohort)
            base_context = str(z["context_id"].item()) if "context_id" in z else cohort
        save(
            "source-axes",
            {
                "cohort": cohort,
                "target_shape": list(target.shape),
                "action_shape": list(raw_actions.shape),
                "query_count": len(raw_queries),
                "reference_fields": list(z.files),
                "action_examples": raw_actions[:3].tolist(),
                "query_examples": raw_queries[:3].tolist(),
            },
        )
        queries = [canonical(g) for g in raw_queries]
        columns = np.array([i for i, g in enumerate(queries) if g in index], np.int64)
        query_genes = [queries[i] for i in columns]
        if len(set(query_genes)) != len(query_genes):
            raise ValueError(
                "Molecular aliases collapse output coordinates; resolution required"
            )
        query_indices = np.array([index[g] for g in query_genes], np.int32)
        if target.ndim != 2 or target.shape[1] != len(raw_queries):
            raise ValueError("Molecular target coordinate shape mismatch")
        if not len(columns):
            raise ValueError("No resolved molecular coordinates")
        # Only population-local observations and control-derived values enter.
        # Old action_features, STRING/GO features and fitted feature scales are ignored.
        writer = Units(genes, cohort)
        kept = []
        quarantine = Counter()
        basal = {}
        offsets = (
            np.asarray(z["action_offsets"], int)
            if not yeast and "action_offsets" in z
            else None
        )
        if not yeast:
            control, control_mask, control_note = basic_control(z, columns)
            if control is not None:
                basal[base_context] = controls(
                    query_genes,
                    control,
                    control_mask,
                    {
                        "source_sha256": selected[0]["sha256"],
                        "control_definition": control_note,
                    },
                )
        for i in range(len(target)):
            native = (
                list(raw_actions[offsets[i] : offsets[i + 1]])
                if offsets is not None
                else list(np.atleast_1d(raw_actions[i]))
            )
            native = [g for g in native if g]
            actions = [canonical(g) for g in native]
            if any(g not in index for g in actions) or len(set(actions)) != len(
                actions
            ):
                quarantine["unresolved-or-collapsed-intervention"] += 1
                continue
            mechanisms = [mechanism] * len(actions)
            methods = [method] * len(actions)
            if cohort == "mcf10a_tgfb1_d6":
                ligand = aliases["TGFB1"]
                if ligand in actions:
                    quarantine["ligand-and-genetic-action-on-same-entity"] += 1
                    continue
                actions.append(ligand)
                mechanisms.append("ligand_addition")
                methods.append("protein-addition")
                if base_context in basal:
                    basal[base_context]["role"] = "observational"
                    basal[base_context]["exposure_genes"] = [ligand]
                    basal[base_context]["provenance"]["conditioning"] = (
                        "TGFB1-treated non-targeting genetic controls, not untreated cells"
                    )
            context = base_context
            if yeast:
                batch = int(batch_indices[i])
                context = base_context + "-batch-" + str(batch)
                if context not in basal:
                    control = np.asarray(z["control_mean"][batch], np.float64)[columns]
                    basal[context] = controls(
                        query_genes,
                        control,
                        np.isfinite(control),
                        {
                            "source_sha256": next(
                                s["sha256"]
                                for s in selected
                                if s["remote"].endswith("/reference.npz")
                            ),
                            "control_definition": "source population mean per-cell ln1p CP10K",
                            "batch_index": batch,
                        },
                    )
            template = {
                "source": "molecular-" + cohort,
                "study": "Nadal-Ribelles-2025"
                if yeast
                else "Replogle-2022"
                if cohort in ("k562", "rpe1", "gwps")
                else "Norman-2019"
                if cohort == "norman"
                else "Nadig-2025"
                if cohort == "hepg2"
                else "Zhao-2021",
                "taxon": 559292 if yeast else 9606,
                "source_taxon": 4932 if yeast else 9606,
                "context": context,
                "assay": "RNA-" + cohort,
                "kind": "rna",
                "units": units,
                "scope": "population",
                "training_allowed": True,
                "license": "NCBI-GEO-public-data-policy"
                if cohort == "norman"
                or cohort.startswith("mcf10a")
                or cohort == "hepg2"
                else "CC-BY-4.0",
                "lineage": {
                    "repository": pins["repository"],
                    "revision": pins["revision"],
                    "sources": {s["remote"]: s["sha256"] for s in selected},
                },
            }
            writer.add(
                template=template,
                genes=actions,
                value=0.0,
                native_row=i,
                unit=(cohort, i),
                mechanisms=mechanisms,
                method=methods,
                hours=hours,
            )
            kept.append(i)
        unit_rows = writer.array()
        kept = np.asarray(kept, np.int64)
        arrays = {"units.npy": unit_rows, "query-indices.npy": query_indices}
        shape = (len(kept), len(columns))
        output = Path(directory) / "targets.npy"
        targets = np.lib.format.open_memmap(
            output, mode="w+", dtype=np.float32, shape=shape
        )
        masks = np.lib.format.open_memmap(
            Path(directory) / "observed.npy", mode="w+", dtype=bool, shape=shape
        )
        observed = (
            z["observed"]
            if not yeast and "observed" in z
            else z["target_observed"]
            if not yeast and "target_observed" in z
            else None
        )
        for lo in range(0, len(kept), 256):
            rows = kept[lo : lo + 256]
            values = target[np.ix_(rows, columns)]
            mask = np.isfinite(values)
            if observed is not None:
                mask &= observed[np.ix_(rows, columns)]
            targets[lo : lo + len(rows)] = np.where(mask, values, 0)
            masks[lo : lo + len(rows)] = mask
        targets.flush()
        masks.flush()
        del targets, masks
        for name, array in arrays.items():
            np.save(Path(directory) / name, array, allow_pickle=False)
        files = {
            name: publish_file(Path(directory) / name, name.replace(".", "-"))
            for name in (
                "units.npy",
                "query-indices.npy",
                "targets.npy",
                "observed.npy",
            )
        }
        basal_data = {
            "schema": "slp.basal-observations/v1",
            "contexts": basal,
            "fitted_human_intervention_genes": [],
        }
        basal_sha256 = hashlib.sha256(
            json.dumps(
                basal_data, sort_keys=True, default=str, allow_nan=False
            ).encode()
        ).hexdigest()
        save("basal", basal_data)
        manifest = {
            "schema": "slp.population-pack/v1",
            "job": os.environ["SLP_JOB"],
            "cohort": cohort,
            "files": files,
            "templates": writer.templates,
            "mechanisms": MECHANISMS,
            "methods": METHODS,
            "units": len(kept),
            "queries": len(columns),
            "raw_units": len(target),
            "raw_queries": len(raw_queries),
            "unresolved_output_coordinates": len(raw_queries) - len(columns),
            "quarantine": dict(quarantine),
            "basal": "basal.json",
            "basal_sha256": basal_sha256,
            "genes_sha256": hashlib.sha256(encode(genes).encode()).hexdigest(),
            "outcome_fitted_transforms": [],
            "processing": "preserved per-population measured endpoints and source control normalization; old fitted model features discarded",
            "coverage_limit": "published legacy population subset; original excluded cohorts have not been restored",
            "reference_taxonomy": "native S. cerevisiae ORFs join S288C reference 559292; original source taxon 4932 retained"
            if yeast
            else "human canonical HGNC",
        }
        save("population-manifest", manifest)
        save(
            "complete",
            {
                "state": "complete",
                "units": len(kept),
                "queries": len(columns),
                "quarantine": dict(quarantine),
                "research_training_launched": False,
            },
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save(
            "failed",
            {
                "state": "failed",
                "type": type(exc).__name__,
                "detail": str(exc)[:400]
                if isinstance(exc, (ValueError, KeyError, IndexError))
                else None,
                "frames": [
                    {"file": Path(f.filename).name, "line": f.lineno}
                    for f in traceback.extract_tb(exc.__traceback__)
                ],
            },
        )
        raise SystemExit(1) from None
