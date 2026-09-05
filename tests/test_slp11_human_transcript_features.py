from __future__ import annotations

import importlib.util
import gzip
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/build_slp11_human_transcript_features.py"
SPEC = importlib.util.spec_from_file_location("slp11_human_transcript_features", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def transcript(name: str, sequence: str, versioned: str | None = None):
    versioned = versioned or f"{name}.1"
    return MOD.Transcript("ENSG00000000001", "ENSG00000000001.2", versioned, name, "-1", "ncrna", sequence)


def test_longest_then_stable_then_versioned_transcript_tie() -> None:
    long = transcript("ENST00000000009", "AAAAA")
    assert MOD.choose_longest(transcript("ENST00000000001", "AAAA"), long) is long
    lexical = transcript("ENST00000000001", "CCCCC", "ENST00000000001.9")
    assert MOD.choose_longest(long, lexical) is lexical
    lower_version_text = transcript("ENST00000000001", "GGGGG", "ENST00000000001.10")
    assert MOD.choose_longest(lexical, lower_version_text) is lower_version_text


def test_kmer_boundaries_ambiguity_and_presence() -> None:
    features = MOD.transcript_features("AAAACNAAAA")
    assert features.shape == (259,) and features.dtype == np.float32
    # Valid windows are AAAA, AAAC, and AAAA; windows crossing N are skipped.
    assert features[MOD.KMER_ORDER.index("AAAA")] == np.float32(2 / 3)
    assert features[MOD.KMER_ORDER.index("AAAC")] == np.float32(1 / 3)
    assert np.isclose(features[:256].sum(), 1.0)
    assert features[257] == np.float32(0.1)
    assert features[258] == 1
    short = MOD.transcript_features("ACN")
    assert np.all(short[:256] == 0) and short[258] == 1


def test_header_uses_only_stable_identifiers_and_strand() -> None:
    header = "ENST00000000001.7 ncrna chromosome:GRCh38:1:10:20:-1 gene:ENSG00000000002.4 gene_biotype:lncRNA transcript_biotype:lncRNA gene_symbol:DO_NOT_USE"
    parsed = MOD.parse_header(header, "ncrna", "ACGT")
    assert parsed.gene_id == "ENSG00000000002"
    assert parsed.transcript_stable_id == "ENST00000000001"
    assert parsed.transcript_id == "ENST00000000001.7"
    assert parsed.strand == "-1" and parsed.sequence == "ACGT"


def test_deterministic_pickle_free_npz(tmp_path: Path) -> None:
    arrays = {"feature_values": np.eye(3, dtype=np.float32), "entity_id": np.asarray(["ENSG00000000001"])}
    one, two = tmp_path / "one.npz", tmp_path / "two.npz"
    MOD.write_deterministic_npz(one, arrays)
    MOD.write_deterministic_npz(two, arrays)
    assert one.read_bytes() == two.read_bytes()
    with np.load(one, allow_pickle=False) as archive:
        assert archive["feature_values"].dtype == np.float32


def test_exact_row_class_count() -> None:
    values = np.asarray([[1, 2], [1, 2], [2, 1], [1, 2]], dtype=np.float32)
    unique, excess, largest, _ = MOD.exact_row_classes(values)
    assert (unique, excess, largest) == (2, 2, 3)


def test_two_repeat_subset_extraction_is_identical(tmp_path: Path) -> None:
    for kind in ("cdna", "ncrna"):
        (tmp_path / kind).mkdir()
    cdna = (
        ">ENST00000000009.1 cdna chromosome:GRCh38:1:1:5:1 gene:ENSG00000000001.2 gene_biotype:protein_coding transcript_biotype:protein_coding\n"
        "AAAAA\n"
        ">ENST00000000001.9 cdna chromosome:GRCh38:1:1:5:-1 gene:ENSG00000000001.2 gene_biotype:protein_coding transcript_biotype:protein_coding\n"
        "CCCCC\n"
    )
    ncrna = (
        ">ENST00000000002.3 ncrna chromosome:GRCh38:2:1:6:-1 gene:ENSG00000000002.1 gene_biotype:lncRNA transcript_biotype:lncRNA\n"
        "ACGTNN\n"
    )
    with gzip.open(tmp_path / "cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz", "wt", encoding="ascii") as stream:
        stream.write(cdna)
    with gzip.open(tmp_path / "ncrna/Homo_sapiens.GRCh38.ncrna.fa.gz", "wt", encoding="ascii") as stream:
        stream.write(ncrna)
    first, first_stats = MOD.select_transcripts(tmp_path)
    second, second_stats = MOD.select_transcripts(tmp_path)
    assert first == second and first_stats == second_stats
    assert first["ENSG00000000001"].transcript_id == "ENST00000000001.9"
    for gene in first:
        np.testing.assert_array_equal(
            MOD.transcript_features(first[gene].sequence), MOD.transcript_features(second[gene].sequence)
        )
