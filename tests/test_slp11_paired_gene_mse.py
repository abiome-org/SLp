import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/summarize_slp11_paired_gene_mse.py"
SPEC = importlib.util.spec_from_file_location("paired_gene_mse_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_identical_models_have_zero_paired_uncertainty():
    values = np.arange(1., 31.)
    result = MODULE.paired_intervals(values, values, draws=1000)
    assert result["pairedGeneResampling95PercentMseDifference"] == [0., 0.]
    assert result["pairedGeneResampling95PercentRelativeGain"] == [0., 0.]


def test_pairing_preserves_known_multiplicative_gain():
    values = np.asarray([1., 4., 9., 16.])
    result = MODULE.paired_intervals(values*.5, values, draws=1000)
    assert result["pairedGeneResampling95PercentRelativeGain"] == [.5, .5]
    assert result["genesWithLowerCandidateMse"] == 4


def test_undefined_relative_comparison_rejected():
    with pytest.raises(ValueError):
        MODULE.paired_intervals([1., 2.], [1., 0.])
