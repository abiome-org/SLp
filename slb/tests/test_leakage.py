"""check-leakage on non-canonical IDs, homologs outside the benchmark, unknown genes and species."""
import polars as pl
import pytest

from slbench.evaluate import BENCH

pytestmark = pytest.mark.skipif(not (BENCH / "manifest.json").exists(), reason="benchmark not built")


@pytest.fixture(scope="module")
def g():
    """Adversarial genes drawn from the benchmark's own tables, so this runs on the public bundle too."""
    from slbench import features
    fam = pl.read_parquet(BENCH / "held_out_families.parquet")
    known = {(s, x) for s, x in fam.select("species", "gene").iter_rows()}
    test = {sp: sorted(fam.filter((pl.col("species") == sp) & (pl.col("bucket") == "test"))["gene"]) for sp in
            ("human", "scer", "spom", "bsub")}
    by = lambda sp, b: sorted(fam.filter((pl.col("species") == sp) & (pl.col("bucket") == b))["gene"])
    al = features.read(BENCH, "gene_aliases")
    th = al.filter((pl.col("species") == "human") & pl.col("gene_id").is_in(test["human"]))
    rh, rs = features.TableResolver(al, "human"), features.TableResolver(al, "spom")
    first = lambda df, r, low=False: next(x for x in df.sort("key").iter_rows() if r(x[2].lower() if low else x[2]) == x[3])
    prev = first(th.filter((pl.col("tier") >= 1) & (pl.col("key").str.to_uppercase() != pl.col("gene_id").str.to_uppercase())
                           & ~pl.col("key").str.starts_with("ENSG")), rh)
    ens = first(th.filter(pl.col("key").str.starts_with("ENSG")), rh)
    ps = first(al.filter((pl.col("species") == "spom") & pl.col("gene_id").is_in(test["spom"])
                         & ~pl.col("key").str.starts_with("SP") & (pl.col("key") != pl.col("gene_id"))), rs, low=True)
    st = features.read(BENCH, "homology_status")
    nodes = set(st["node"])

    def outside(sp):  # a gene of `sp` outside the benchmark whose homology component holds a test gene
        n = st.filter(pl.col("node").str.starts_with(f"{sp}:") & (pl.col("status") == "test")).sort("node")["node"]
        return next(x.split(":", 1)[1] for x in n if (sp, x.split(":", 1)[1]) not in known)
    r = features.TableResolver(al, "human")
    graphless = next(s for s in ("MIR21", "MIR155", "SNORD3A", "RNU6-1")
                     if r(s) and f"human:{r(s)}" not in nodes and ("human", r(s)) not in known)
    return dict(test=test, train=by("human", "train"), dev=by("human", "dev"), prev=(prev[2], prev[3]),
                ens=(ens[2], ens[3]), pname=(ps[2].lower(), ps[3]), mouse=(outside("mmus"), None),
                ecoli=(outside("ecol"), None), graphless=graphless)


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
