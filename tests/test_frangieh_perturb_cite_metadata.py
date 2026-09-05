import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_frangieh_perturb_cite_metadata.py"
SPEC = importlib.util.spec_from_file_location("frangieh_audit", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_split_contract_is_stable_and_gene_grouped():
    expected = {
        "ENSG00000121410": "train",
        "ENSG00000204518": "validation",
        "ENSG00000245105": "test",
    }
    assert {gene: MOD.gene_split(gene) for gene in expected} == expected
    assert MOD.gene_split("ENSG00000245105") == MOD.gene_split("ENSG00000245105")


def test_numeric_summary_has_explicit_bounds():
    summary = MOD._numeric_summary([1, 2, 9])
    assert summary == {"minimum": 1.0, "median": 2.0, "maximum": 9.0, "mean": 4.0}
