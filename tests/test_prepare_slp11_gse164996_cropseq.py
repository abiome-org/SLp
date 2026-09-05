import importlib.util
import gzip
from pathlib import Path


PATH = Path(__file__).parents[1] / "scripts/prepare_slp11_gse164996_cropseq.py"
SPEC = importlib.util.spec_from_file_location("cropseq", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_calls_require_complete_unambiguous_dual_constructs():
    assert MODULE.classify_call("TP53_3|CTRL0001", set()) == (
        "single", ("TP53_3",), ("TP53",))
    assert MODULE.classify_call("PTEN_5|TP53_3", set()) == (
        "double", ("PTEN_5", "TP53_3"), ("PTEN", "TP53"))
    assert MODULE.classify_call("CTRL0002|CTRL0001", set()) == ("control", (), ())
    assert MODULE.classify_call("hRosa26_2|CTRL0001", set()) == ("control", (), ())
    assert MODULE.classify_call("PTEN_5|CTRL0002", set()) == (
        "single", ("PTEN_5",), ("PTEN",))
    assert MODULE.classify_call("TP53_3", set()) is None
    assert MODULE.classify_call("TP53_3|TP53_4", set()) is None
    assert MODULE.classify_call("PTEN_5|TP53_3|CTRL0001", set()) is None


def test_any_held_constituent_excludes_whole_combination():
    assert MODULE.classify_call("PTEN_5|SMAD4_5", {"SMAD4"}) is None
    assert MODULE.classify_call("SMAD4_5|CTRL0001", {"SMAD4"}) is None
    assert MODULE.classify_call("PTEN_5|CTRL0001", {"SMAD4"}) is not None


def test_both_author_control_constructs_pool_to_one_basal_group(tmp_path):
    path = tmp_path / "calls.csv.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("cell_barcode,num_features,feature_call,num_umis\n")
        stream.write("a,2,CTRL0002|CTRL0001,4|5\n")
        stream.write("b,2,hRosa26_2|CTRL0001,7|8\n")
    _, counts, retained, _ = MODULE.selected_groups(path, set(), minimum_cells=1)
    key = ("control", (), ())
    assert counts[key] == 2
    assert retained == {key}


def test_hrasa_control_is_only_valid_in_exact_author_control_pair():
    assert MODULE.classify_call("hRosa26_2|CTRL0001", set()) == ("control", (), ())
    assert MODULE.classify_call("hRosa26_2|PTEN_5", set()) is None
