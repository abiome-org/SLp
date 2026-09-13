"""Observational DepMap expression context, with exact model-name aliases."""

from collections import defaultdict, Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import traceback
from inspect_corpus import request, save, download
from pack_corpus import metadata, IDENTITY
from prepare_populations import controls


def main():
    sources = json.load(request("/manifest"))["objects"]
    expression = next(
        s for s in sources if "depmap-24q2" in s["id"] and "Expression" in s["key"]
    )
    model = next(
        s
        for s in sources
        if "depmap-24q2" in s["id"] and s["key"].endswith("/Model.csv")
    )
    aliases = metadata(IDENTITY, "aliases")["human"]
    genes = set(metadata(IDENTITY, "sequence-inputs-manifest")["genes"])
    contexts = {}
    names = defaultdict(set)
    with tempfile.TemporaryDirectory() as directory:
        model_path = download(model, directory)
        with model_path.open() as file:
            reader = csv.DictReader(file)
            if "ModelID" not in reader.fieldnames:
                raise ValueError("DepMap Model identity column changed")
            for row in reader:
                for field in ("CellLineName", "StrippedCellLineName"):
                    name = row.get(field, "")
                    if name:
                        for variant in (name, name.upper(), name.lower()):
                            names[variant].add(row["ModelID"])
        expression_path = download(expression, directory)
        with expression_path.open() as file:
            reader = csv.reader(file)
            header = next(reader)
            resolved = []
            for name in header[1:]:
                match = re.fullmatch(r"(.+) \((\d+)\)", name)
                if not match:
                    raise ValueError("DepMap expression gene header changed")
                possible = {
                    aliases[v] for v in (match[1], "ENTREZ:" + match[2]) if v in aliases
                }
                g = next(iter(possible)) if len(possible) == 1 else None
                resolved.append(g if g in genes else None)
            counts = Counter(g for g in resolved if g)
            ambiguous = [
                header[i + 1] for i, g in enumerate(resolved) if g and counts[g] > 1
            ]
            valid = [i for i, g in enumerate(resolved) if g and counts[g] == 1]
            query = [resolved[i] for i in valid]
            for row in reader:
                context = row[0]
                if not re.fullmatch(r"ACH-\d{6}", context):
                    raise ValueError("Expression row is not a model identity")
                if context in contexts:
                    raise ValueError("Duplicate observational model profile")
                values = []
                observed = []
                for i in valid:
                    try:
                        value = float(row[i + 1])
                    except ValueError:
                        value = float("nan")
                    observed.append(math.isfinite(value))
                    values.append(value if math.isfinite(value) else 0.0)
                contexts[context] = controls(
                    query,
                    values,
                    observed,
                    {
                        "source": expression["key"],
                        "checksum": expression["checksum"],
                        "role": "untreated observational cell-line expression",
                        "units": "log2(TPM+1)",
                    },
                )
                contexts[context]["role"] = "observational"
    name_map = {
        name: next(iter(ids))
        for name, ids in names.items()
        if len(ids) == 1 and next(iter(ids)) in contexts
    }
    document = {
        "schema": "slp.basal-observations/v1",
        "contexts": contexts,
        "aliases": name_map,
        "fitted_human_intervention_genes": [],
        "projection": "fixed gene-sign projection and fixed 64-gene observation panel; no outcome fitting",
        "sources": {"expression": expression, "model": model},
    }
    body = json.dumps(document, sort_keys=True, default=str, allow_nan=False).encode()
    save("basal", document)
    save(
        "complete",
        {
            "state": "complete",
            "contexts": len(contexts),
            "resolved_expression_genes": len(query),
            "aliases": len(name_map),
            "basal_sha256": hashlib.sha256(body).hexdigest(),
            "basal_bytes": len(body),
            "ambiguous_model_names": sum(len(ids) > 1 for ids in names.values()),
            "unresolved_expression_coordinates": len(resolved) - len(query),
            "quarantined_collapsed_expression_headers": ambiguous,
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
                if isinstance(exc, (ValueError, KeyError, StopIteration))
                else None,
                "frames": [
                    {"file": Path(f.filename).name, "line": f.lineno}
                    for f in traceback.extract_tb(exc.__traceback__)
                ],
            },
        )
        raise SystemExit(1) from None
