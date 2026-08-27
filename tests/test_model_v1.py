"""Focused contract checks for the immutable v1 world-model package."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from model.v1 import (
        ActionSet,
        SLPredict,
        WorldContext,
        checkpoint_manifest,
        load_world_checkpoint,
        world_config_from_state_dict,
        write_checkpoint_manifest,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for world-model contract tests")
class WorldModelV1Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = SLPredict(
            d=24,
            latent=8,
            layers=1,
            contexts=3,
            outcomes=2,
            state_dim=1630,
            context_dim=4,
        ).eval()

    def test_action_set_rollout_is_permutation_invariant(self) -> None:
        actions = torch.randn(2, 3, 8)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        context = WorldContext(features=torch.randn(2, 4))
        forward = self.model.rollout(ActionSet(actions, mask), context=context)
        reverse = self.model.rollout(ActionSet(actions.flip(1), mask.flip(1)), context=context)
        self.assertTrue(torch.allclose(forward.mean, reverse.mean, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(forward.log_std, reverse.log_std, atol=1e-6, rtol=1e-6))

    def test_checkpoint_round_trip_preserves_rollout_and_manifest(self) -> None:
        actions = ActionSet(torch.randn(1, 2, 8), torch.tensor([[True, True]]))
        context = WorldContext(features=torch.randn(1, 4))
        expected = self.model.rollout(actions, context=context)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "world_model.pt"
            torch.save(self.model.state_dict(), checkpoint)
            loaded = load_world_checkpoint(checkpoint)
            actual = loaded.rollout(actions, context=context)
            self.assertTrue(torch.equal(expected.mean, actual.mean))
            self.assertTrue(torch.equal(expected.log_std, actual.log_std))
            self.assertEqual(world_config_from_state_dict(self.model.state_dict()), self.model.config)
            self.assertEqual(checkpoint_manifest(checkpoint)["model_version"], "v1")
            manifest_path = write_checkpoint_manifest(checkpoint)
            self.assertEqual(json.loads(manifest_path.read_text())["checkpoint"], "world_model.pt")


if __name__ == "__main__":
    unittest.main()
