"""check-leakage on non-canonical IDs, homologs outside the benchmark, unknown genes and species."""
import polars as pl
import pytest

from slbench.evaluate import BENCH

pytestmark = pytest.mark.skipif(not (BENCH / "manifest.json").exists(), reason="benchmark not built")


@pytest.fixture(scope="module")
def g():
    from slbench import families, ids, ids_extra
    fam = pl.read_parquet(BENCH / "held_out_families.parquet")
    known = {(s, x): b for s, x, b in fam.select("species", "gene", "bucket").iter_rows()}
    test = {sp: sorted(fam.filter((pl.col("species") == sp) & (pl.col("bucket") == "test"))["gene"]) for sp in
            ("human", "scer", "spom", "bsub")}
    by = lambda sp, b: sorted(fam.filter((pl.col("species") == sp) & (pl.col("bucket") == b))["gene"])
    hgnc = pl.read_csv("data/raw/ids/hgnc_complete_set.txt", separator="\t", infer_schema_length=0, quote_char=None,
                       columns=["symbol", "prev_symbol", "ensembl_gene_id"])
    hgnc = hgnc.filter(pl.col("symbol").is_in(test["human"]))
    h = ids.human()
    prev = next((p, s) for s, cell in zip(hgnc["symbol"], hgnc["prev_symbol"]) for p in (cell or "").strip('"').split("|")
                if p and h(p) == s and p.upper() != s.upper())
    ens = next((e, s) for s, e in zip(hgnc["symbol"], hgnc["ensembl_gene_id"]) if e and h(e) == s)
    spom = pl.read_csv("data/raw/ids/pombase_gene_IDs_names_products.tsv", separator="\t", has_header=False,
                       infer_schema_length=0, quote_char=None, comment_prefix="#", columns=[0, 2], new_columns=["id", "name"])
    pname = next((n.lower(), i) for i, n in zip(spom["id"], spom["name"])
                 if n and i in set(test["spom"]) and ids.spom()(n.lower()) == i)
    e = families.edges().filter(pl.col("kind") == "ortholog")

    def ortholog(src, dst):  # a dst-species ortholog of a src test gene, dst gene not in the benchmark
        for u, v in e.select("u", "v").iter_rows():
            for a, b in ((u, v), (v, u)):
                (sa, ga), (sb, gb) = a.split(":", 1), b.split(":", 1)
                if sa == src and sb == dst and known.get((sa, ga)) == "test" and (sb, gb) not in known:
                    return gb, ga
    r = ids_extra.resolver("human")
    graphless = next(s for s in ("MIR21", "MIR155", "SNORD3A", "RNU6-1")
                                if r(s) and f"human:{r(s)}" not in set(e["u"]) | set(e["v"]) and ("human", r(s)) not in known)
    return dict(test=test, train=by("human", "train"), dev=by("human", "dev"), prev=prev, ens=ens, pname=pname,
                mouse=ortholog("human", "mmus"), ecoli=ortholog("bsub", "ecol"), graphless=graphless)


def _rec(rows):
    return pl.DataFrame(rows, schema=["species", "gene_a", "gene_b"], orient="row")


def test_noncanonical_heldout_ids_are_flagged(g):
    from slbench import leakage
    tr = g["train"][0]
    human, scer = g["test"]["human"][0], g["test"]["scer"][0]
    rows = [("human", human.lower(), tr), ("human", g["prev"][0], tr), ("human", g["ens"][0], tr),
            ("scer", scer.lower(), g["test"]["scer"][0]), ("spom", g["pname"][0], g["test"]["spom"][0])]
    a = leakage.annotate(_rec(rows))
    assert a["gene_a_id"].to_list() == [human, g["prev"][1], g["ens"][1], scer, g["pname"][1]]
    assert (a["bucket_a"] == "test").all()
    assert leakage.check(_rec(rows)).height == len(rows)


def test_homologs_outside_benchmark_are_flagged(g):
    from slbench import leakage
    assert g["mouse"] and g["ecoli"]
    rows = [("mmus", g["mouse"][0], g["mouse"][0]), ("ecol", g["ecoli"][0], g["ecoli"][0])]
    assert leakage.check(_rec(rows)).height == 2


def test_unknown_genes_are_not_hashed(g, tmp_path, capsys):
    from slbench import leakage
    tr = g["train"]
    rows = [("human", "NOT_A_GENE_XYZ", tr[0]), ("human", g["graphless"], tr[1]), ("human", tr[0], tr[1])]
    a = leakage.annotate(_rec(rows))
    assert a["bucket_a"].to_list() == ["unknown", "unknown", "train"]
    assert leakage.check(_rec(rows)).height == 0
    p = tmp_path / "r.parquet"
    _rec(rows).write_parquet(p)
    assert leakage.main(str(p)) == 1
    assert leakage.main(str(p), allow_unknown=True) == 0
    out = capsys.readouterr().out
    assert "2 involve unknown genes" in out and "1 clean" in out and "NOT_A_GENE_XYZ" in out
    _rec([rows[2]]).write_parquet(p)
    assert leakage.main(str(p)) == 0


def test_unknown_species_rejected(g, tmp_path, capsys):
    from slbench import leakage
    for sp in ("Homo sapiens", "rat"):
        with pytest.raises(ValueError, match="valid species codes: .*human"):
            leakage.annotate(_rec([(sp, g["train"][0], g["train"][1])]))
    p = tmp_path / "r.csv"
    _rec([("rat", "Trp53", "Brca1")]).write_csv(p)
    assert leakage.main(str(p)) == 2 and "valid species codes" in capsys.readouterr().out


def test_allow_dev(g, tmp_path):
    from slbench import leakage
    p = tmp_path / "r.parquet"
    _rec([("human", g["dev"][0], g["train"][0])]).write_parquet(p)
    assert leakage.main(str(p)) == 1
    assert leakage.main(str(p), allow_dev=True) == 0
    _rec([("human", g["test"]["human"][0], g["train"][0])]).write_parquet(p)
    assert leakage.main(str(p), allow_dev=True) == 1
