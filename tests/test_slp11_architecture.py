"""Contract checks for the fresh SLp-1.1 world architecture."""

from pathlib import Path
import sys
import unittest

import torch


MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-world"
sys.path.insert(0, str(MODULE))

from architecture import SpeciesAwareWorldModel, WorldBatch, WorldConfig  # noqa: E402


class SpeciesAwareWorldModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.model = SpeciesAwareWorldModel(
            WorldConfig(
                entity_feature_dim=7,
                species_feature_dim=3,
                readout_types=4,
                d_model=16,
                nhead=4,
                encoder_layers=2,
                decoder_layers=1,
                dropout=0.0,
            )
        ).eval()
        self.batch = WorldBatch(
            context_features=torch.randn(2, 3, 7),
            context_mask=torch.tensor([[True, True, False], [True, True, True]]),
            action_features=torch.randn(2, 3, 7),
            action_covariates=torch.randn(2, 3, 4),
            action_mask=torch.tensor([[True, True, False], [True, True, True]]),
            query_features=torch.randn(2, 4, 7),
            query_mask=torch.tensor([[True, True, True, False], [True, True, True, True]]),
            readout_type=torch.tensor([[0, 1, 2, 0], [3, 2, 1, 0]]),
            species_features=torch.randn(2, 3),
        )

    def test_context_and_action_order_do_not_change_predictions(self) -> None:
        expected = self.model(self.batch)
        context_order = torch.tensor([2, 0, 1])
        action_order = torch.tensor([1, 2, 0])
        permuted = WorldBatch(
            context_features=self.batch.context_features[:, context_order],
            context_mask=self.batch.context_mask[:, context_order],
            action_features=self.batch.action_features[:, action_order],
            action_covariates=self.batch.action_covariates[:, action_order],
            action_mask=self.batch.action_mask[:, action_order],
            query_features=self.batch.query_features,
            query_mask=self.batch.query_mask,
            readout_type=self.batch.readout_type,
            species_features=self.batch.species_features,
        )
        actual = self.model(permuted)
        self.assertTrue(torch.allclose(expected.mean, actual.mean, atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(expected.log_scale, actual.log_scale, atol=1e-6, rtol=1e-6)
        )

    def test_contract_has_no_gene_identifier_parameters(self) -> None:
        self.assertFalse(any("gene" in name.lower() for name, _ in self.model.named_parameters()))
        prediction = self.model(self.batch)
        self.assertEqual(prediction.mean.shape, (2, 4))
        self.assertTrue(torch.isfinite(prediction.mean).all())
        self.assertTrue(torch.isfinite(prediction.scale).all())


if __name__ == "__main__":
    unittest.main()
