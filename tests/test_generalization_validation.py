"""Focused checks for the molecular generalization gate."""

import unittest
from pathlib import Path
import tempfile

import numpy as np

from modules.evaluation import (
    EvidenceRequirements,
    GeneralizationTable,
    additive_single_baseline,
    cardinality_mean_baseline,
    make_split,
    regression_metrics,
)
from src.training.validate_generalization import load_table


def additive_table(include_condition=True):
    gene_effect = np.column_stack(
        (np.arange(30, dtype="float64"), np.arange(30, dtype="float64") ** 2 / 30)
    )
    actions = [[gene, -1] for gene in range(30)]
    actions.extend([list(pair) for pair in zip(range(0, 30, 2), range(1, 30, 2))])
    target = [gene_effect[gene] for gene in range(30)]
    target.extend([gene_effect[first] + gene_effect[second] for first, second in actions[30:]])
    rows = len(actions)
    return GeneralizationTable(
        np.asarray(actions, dtype="int32"),
        np.asarray(target),
        context=np.repeat("K562", rows),
        source=np.repeat("synthetic", rows),
        condition=np.repeat("CRISPRi|24h", rows) if include_condition else None,
        target_semantics="perturbation_delta",
    )


class GeneralizationValidationTest(unittest.TestCase):
    def setUp(self):
        self.table = additive_table()
        self.requirements = EvidenceRequirements(
            min_train_rows=1,
            min_test_rows=1,
            min_test_action_sets=1,
            min_test_genes=1,
        )

    def _nonempty_gene_fold(self, protocol):
        for fold in range(5):
            split = make_split(
                self.table,
                protocol,
                fold=fold,
                requirements=self.requirements,
            )
            if len(split.test):
                return split
        self.fail(f"no non-empty fold for {protocol}")

    def test_composition_cold_keeps_held_singletons_but_intervention_cold_does_not(self):
        composition = self._nonempty_gene_fold("composition_gene_cold")
        intervention = make_split(
            self.table,
            "intervention_gene_cold",
            fold=composition.fold,
            requirements=self.requirements,
        )
        held = set(self.table.actions[composition.test].ravel()) - {-1}
        composition_train_genes = set(self.table.actions[composition.train].ravel()) - {-1}
        intervention_train_genes = set(self.table.actions[intervention.train].ravel()) - {-1}
        composition_train_multi = composition.train[
            self.table.cardinality[composition.train] >= 2
        ]
        self.assertTrue(held <= composition_train_genes)
        self.assertFalse(held & intervention_train_genes)
        self.assertFalse(held & (set(self.table.actions[composition_train_multi].ravel()) - {-1}))

    def test_pair_cold_has_no_action_set_overlap(self):
        for fold in range(5):
            split = make_split(
                self.table,
                "pair_cold",
                fold=fold,
                requirements=self.requirements,
            )
            keys = self.table.action_sets()
            self.assertFalse(set(keys[split.train]) & set(keys[split.test]))

    def test_missing_metadata_makes_protocol_ineligible(self):
        table = additive_table(include_condition=False)
        split = make_split(
            table,
            "condition_cold",
            requirements=self.requirements,
        )
        self.assertFalse(split.eligible)
        self.assertIn("missing required condition metadata", split.reasons)

    def test_additive_single_is_a_real_baseline_for_composition(self):
        split = self._nonempty_gene_fold("composition_gene_cold")
        observed = self.table.target[split.test]
        additive, audit = additive_single_baseline(self.table, split)
        mean = cardinality_mean_baseline(self.table, split)
        additive_metrics = regression_metrics(observed, additive)
        mean_metrics = regression_metrics(observed, mean)
        self.assertAlmostEqual(additive_metrics["rmse"], 0.0)
        self.assertLess(additive_metrics["huber"], mean_metrics["huber"])
        self.assertEqual(audit["action_coverage"], 1.0)

    def test_splits_are_stable(self):
        first = make_split(
            self.table,
            "pair_cold",
            fold=3,
            seed=912,
            requirements=self.requirements,
        )
        second = make_split(
            self.table,
            "pair_cold",
            fold=3,
            seed=912,
            requirements=self.requirements,
        )
        np.testing.assert_array_equal(first.train, second.train)
        np.testing.assert_array_equal(first.test, second.test)

    def test_group_folds_are_balanced_when_enough_groups_exist(self):
        rows = 20
        table = GeneralizationTable(
            np.column_stack((np.arange(rows), np.arange(rows) + rows)).astype("int32"),
            source=np.repeat([f"study-{index}" for index in range(5)], 4),
        )
        held_sources = []
        for fold in range(5):
            split = make_split(
                table,
                "source_cold",
                fold=fold,
                requirements=self.requirements,
            )
            self.assertEqual(len(np.unique(table.source[split.test])), 1)
            held_sources.extend(np.unique(table.source[split.test]).tolist())
        self.assertEqual(set(held_sources), set(table.source))

    def test_duplicate_action_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            GeneralizationTable(np.asarray([[2, 2]], dtype="int32"))

    def test_action_modes_distinguish_opposite_interventions_on_one_target(self):
        table = GeneralizationTable(
            np.asarray([[2, 2]], dtype="int32"),
            action_modes=np.asarray([["activation", "repression"]]),
        )
        self.assertEqual(table.action_sets()[0], "2:activation@1+2:repression@1")

    def test_action_dose_is_part_of_exact_action_identity(self):
        table = GeneralizationTable(
            np.asarray([[2, -1], [2, -1]], dtype="int32"),
            action_modes=np.asarray([["repression", ""], ["repression", ""]]),
            action_doses=np.asarray([[1, 0], [2, 0]], dtype="float32"),
        )
        self.assertNotEqual(table.action_sets()[0], table.action_sets()[1])

    def test_loader_does_not_treat_guide_condition_as_experimental_condition(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pack.npz"
            np.savez_compressed(
                path,
                pairs=np.asarray([[0, 1]], dtype="int32"),
                target=np.asarray([[1.0]]),
                target_semantics=np.asarray("perturbation_delta"),
                condition=np.asarray(["TP53+MDM2"]),
            )
            table, fields = load_table(path)
        self.assertIsNone(table.condition)
        self.assertIsNone(fields["condition"])


if __name__ == "__main__":
    unittest.main()
