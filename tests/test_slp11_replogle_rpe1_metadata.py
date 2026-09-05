from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/audit_replogle_rpe1_singlecell_metadata.py"
SPEC = importlib.util.spec_from_file_location("rpe1_metadata", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_unresolved_actions_fail_closed() -> None:
    assert MOD.action_role("") == "control"
    assert MOD.action_role("nan") == "unresolved-excluded"
    assert MOD.action_role("SYMBOL") == "unresolved-excluded"
    assert MOD.reconstruction_role("cell", "unresolved-excluded") == "none"


def test_global_gene_and_cell_hashes_match_contract() -> None:
    gene = "ENSG00000123456"
    bucket = int.from_bytes(hashlib.sha256(f"slp11-development-v1|731|9606|{gene}".encode()).digest()[:8], "big") % 100
    expected = "train" if bucket < 70 else "validation" if bucket < 85 else "test-excluded"
    assert MOD.action_role(gene) == expected
    cell = "barcode-a"
    bucket = int.from_bytes(hashlib.sha256(f"slp11-rpe1-essential-cell-reconstruction-v1|731|{cell}".encode()).digest()[:8], "big") % 100
    assert MOD.reconstruction_role(cell, "train") == ("train" if bucket < 90 else "validation")
