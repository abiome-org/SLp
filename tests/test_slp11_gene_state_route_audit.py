import importlib.util
import sys
from pathlib import Path

import torch

PATH = Path(__file__).parents[1] / "scripts/audit_slp11_gene_state_routes.py"
SPEC = importlib.util.spec_from_file_location("slp11_gene_state_route_audit_test", PATH)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_route_ablation_changes_only_requested_intervention_route():
    encoded = {
        "basal_node_state": torch.randn(2, 3, 4),
        "local_delta": torch.randn(2, 3, 4),
        "local_state": torch.randn(2, 3, 4),
        "global_basal_state": torch.randn(2, 4),
        "global_delta": torch.randn(2, 4),
        "global_state": torch.randn(2, 4),
    }
    local = MOD.route_ablation_states(encoded, "local_off")
    torch.testing.assert_close(local["local_state"], encoded["basal_node_state"])
    assert torch.count_nonzero(local["local_delta"]) == 0
    assert local["global_state"] is encoded["global_state"]
    global_off = MOD.route_ablation_states(encoded, "global_off")
    torch.testing.assert_close(global_off["global_state"], encoded["global_basal_state"])
    assert torch.count_nonzero(global_off["global_delta"]) == 0
    assert global_off["local_state"] is encoded["local_state"]
