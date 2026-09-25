"""The gene-degree reject probe uses only the public split inputs."""
import importlib.util
from pathlib import Path

import polars as pl

script = Path(__file__).resolve().parents[1] / "scripts/grader_probe.py"
spec = importlib.util.spec_from_file_location("grader_probe", script)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_gene_degree_counts_rows_within_context():
    x = pl.DataFrame({"example_id": list("abcdef"), "species": ["human"] * 5 + ["scer"],
                      "context_id": ["c1", "c1", "c1", "c2", "c1", "c1"],
                      "gene_a": ["A", "A", "B", "A", "C", "A"], "gene_b": ["B", "C", "C", "B", "C", "B"]})
    s = dict(probe.gene_degree(x).iter_rows())
    # c1 human: A in a,b (2); B in a,c (2); C in b,c,e (3; self-pair e counted once)
    assert s == {"a": -4.0, "b": -5.0, "c": -5.0, "d": -2.0, "e": -6.0, "f": -2.0}


def test_gene_degree_probe_is_reported(monkeypatch, tmp_path):
    src = script.read_text()
    assert '"gene_degree_reject"' in src and 'public_inputs("dev")' in src and 'public_inputs("test")' not in src
    seen = []
    real = pl.read_parquet

    def read(path, *a, columns=None, **kw):
        seen.append((Path(path).name, columns))
        return real(path, *a, columns=columns, **kw)

    monkeypatch.setattr(probe.pl, "read_parquet", read)
    try:
        probe.public_inputs("dev")
    except FileNotFoundError:
        pass
    assert seen == [("dev.parquet", ["example_id", "species", "context_id", "gene_a", "gene_b"])]
