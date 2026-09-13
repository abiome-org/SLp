"""Inspect the two already captured Feng members without re-reading 25 GB."""

import csv
import gzip
import io
import numpy as np
from inspect_corpus import request, save
from safe_arrays import load, SparseCSR


def structure(value, depth=0):
    if isinstance(value, SparseCSR):
        return {"type": "csr", "shape": value.shape, "nnz": len(value.data)}
    if isinstance(value, np.ndarray):
        return {
            "type": "array",
            "shape": value.shape,
            "dtype": str(value.dtype),
            "first": structure(value.tolist()[:2], depth + 1)
            if depth < 7 and value.ndim and len(value)
            else None,
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "list",
            "length": len(value),
            "first": [structure(v, depth + 1) for v in value[:3]] if depth < 7 else [],
        }
    if isinstance(value, dict):
        return (
            {str(k): structure(v, depth + 1) for k, v in value.items()}
            if depth < 7
            else {"keys": list(value)[:20]}
        )
    return str(value)[:100]


def captured(name):
    with request("/outputs/protocol-r2-20260912-v4/" + name, prep=True) as response:
        body = response.read(32 * 1024 * 1024 + 1)
    return gzip.decompress(body)


def main():
    split = captured("feng---data-data_split-CV3_1-npy.bin.gz")
    roster = captured("feng---data-preprocessed_data-meta_table_9845-csv.bin.gz")
    reader = csv.DictReader(io.StringIO(roster.decode()))
    rows = list(reader)
    save(
        "feng-structure",
        {
            "split": structure(load(split, numpy_file=True)),
            "roster": {
                "columns": reader.fieldnames,
                "rows": len(rows),
                "first": rows[:3],
            },
        },
    )
    save("complete", {"state": "complete", "research_training_launched": False})


if __name__ == "__main__":
    main()
