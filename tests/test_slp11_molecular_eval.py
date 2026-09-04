"""Synthetic contract tests for perturbation-specific molecular evaluation."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-molecular-eval"
sys.path.insert(0, str(MODULE))

from evaluator import (
    PREDICTION_ROLE,
    REFERENCE_ROLE,
    MolecularEvaluationError,
    evaluate_molecular_predictions,
)


class MolecularEvaluationTest(unittest.TestCase):
    def _record(
        self,
        taxon: int,
        source: str,
        perturbation: str,
        target: list[float],
        prediction: list[float] | None = None,
        interventions: list[str] | None = None,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "speciesTaxon": taxon,
            "sourceId": source,
            "centeringGroup": "condition:basal",
            "perturbationId": perturbation,
            "interventionIds": interventions or [perturbation],
            "readoutIds": ["RNA:feature-1", "RNA:feature-2", "RNA:feature-3"],
            "target": target,
        }
        if prediction is not None:
            record["predictionMean"] = prediction
            record["predictionLogScale"] = [0.0, 0.0, 0.0]
        return record

    def _snapshot(
        self,
        root: Path,
        name: str,
        role: str,
        records: list[dict[str, object]],
        reference_sha256: str | None = None,
    ) -> tuple[Path, str]:
        directory = root / name
        directory.mkdir()
        shard = directory / "profiles-000.jsonl"
        shard.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "schema": "slp.molecular-evaluation/v1",
            "datasetId": name,
            "version": "synthetic-v1",
            "role": role,
            "labelClass": "molecular",
            "benchmarkLabelsPresent": False,
            "valueSpace": "post-perturbation-profile",
            "speciesTaxa": sorted({int(record["speciesTaxon"]) for record in records}),
            "sourceIds": sorted({str(record["sourceId"]) for record in records}),
            "sourceSnapshotSha256": "a" * 64,
            "shards": [
                {
                    "path": shard.name,
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                    "records": len(records),
                }
            ],
        }
        if role == PREDICTION_ROLE:
            assert reference_sha256 is not None
            manifest["modelCheckpointSha256"] = "b" * 64
            manifest["referenceManifestSha256"] = reference_sha256
        manifest_path = directory / "evaluation.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        return directory, hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def _reference_records(self) -> list[dict[str, object]]:
        return [
            self._record(4932, "costanzo:2016", "SGD:S000000001", [9.0, 19.0, 29.0]),
            self._record(4932, "costanzo:2016", "SGD:S000000002", [11.0, 21.0, 31.0]),
            self._record(9606, "replogle:2022", "ENSEMBL:ENSG000001", [19.0, 29.0, 39.0]),
            self._record(9606, "replogle:2022", "ENSEMBL:ENSG000002", [21.0, 31.0, 41.0]),
        ]

    def _prediction_records(self, exact: bool = True) -> list[dict[str, object]]:
        truths = [
            (4932, "costanzo:2016", "SGD:S000000101", [11.0, 19.0, 31.0], [10.0, 20.0, 30.0]),
            (4932, "costanzo:2016", "SGD:S000000102", [9.0, 21.0, 29.0], [10.0, 20.0, 30.0]),
            (9606, "replogle:2022", "ENSEMBL:ENSG000101", [21.0, 29.0, 41.0], [20.0, 30.0, 40.0]),
            (9606, "replogle:2022", "ENSEMBL:ENSG000102", [19.0, 31.0, 39.0], [20.0, 30.0, 40.0]),
        ]
        return [
            self._record(taxon, source, perturbation, target, target if exact else perturbed_mean)
            for taxon, source, perturbation, target, perturbed_mean in truths
        ]

    def test_exact_predictions_report_species_sources_and_systema_style_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, reference_sha256 = self._snapshot(
                root, "reference", REFERENCE_ROLE, self._reference_records()
            )
            predictions, _ = self._snapshot(
                root,
                "predictions",
                PREDICTION_ROLE,
                self._prediction_records(exact=True),
                reference_sha256,
            )
            report = evaluate_molecular_predictions(reference, predictions)
        overall = report["overall"]
        self.assertAlmostEqual(overall["ordinary"]["rmse"], 0.0)
        self.assertAlmostEqual(overall["ordinary"]["pearson"], 1.0)
        self.assertAlmostEqual(
            overall["perturbationSpecific"]["perturbedCentroidPearson"], 1.0
        )
        self.assertAlmostEqual(
            overall["perturbationSpecific"]["centroidAccuracyCommonPanel"], 1.0
        )
        self.assertEqual(set(report["species"]), {"4932", "9606"})
        self.assertEqual(set(report["sources"]), {"costanzo:2016", "replogle:2022"})
        self.assertEqual(report["audit"]["heldInterventionOverlap"], 0)
        self.assertEqual(report["method"]["class"], "Systema-inspired")

    def test_perturbed_mean_prediction_scores_zero_specific_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, reference_sha256 = self._snapshot(
                root, "reference", REFERENCE_ROLE, self._reference_records()
            )
            predictions, _ = self._snapshot(
                root,
                "predictions",
                PREDICTION_ROLE,
                self._prediction_records(exact=False),
                reference_sha256,
            )
            report = evaluate_molecular_predictions(reference, predictions)
        specific = report["overall"]["perturbationSpecific"]
        self.assertEqual(specific["perturbedCentroidPearson"], 0.0)
        self.assertEqual(specific["perturbedCentroidCosine"], 0.0)
        self.assertEqual(specific["perturbedCentroidPearsonUndefinedProfiles"], 4)
        self.assertEqual(specific["centroidAccuracyCommonPanel"], 0.0)

    def test_component_intervention_overlap_is_fatal_for_a_combination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_records = self._reference_records()
            reference, reference_sha256 = self._snapshot(
                root, "reference", REFERENCE_ROLE, reference_records
            )
            prediction_records = self._prediction_records(exact=True)
            prediction_records[0]["perturbationId"] = "COMBO:held-plus-new"
            prediction_records[0]["interventionIds"] = [
                "SGD:S000000001",
                "SGD:S000000999",
            ]
            predictions, _ = self._snapshot(
                root,
                "predictions",
                PREDICTION_ROLE,
                prediction_records,
                reference_sha256,
            )
            with self.assertRaisesRegex(MolecularEvaluationError, "held intervention leakage"):
                evaluate_molecular_predictions(reference, predictions)

    def test_benchmark_like_record_field_is_rejected_not_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, reference_sha256 = self._snapshot(
                root, "reference", REFERENCE_ROLE, self._reference_records()
            )
            prediction_records = self._prediction_records(exact=True)
            prediction_records[0]["benchmarkLabel"] = 1
            predictions, _ = self._snapshot(
                root,
                "predictions",
                PREDICTION_ROLE,
                prediction_records,
                reference_sha256,
            )
            with self.assertRaisesRegex(MolecularEvaluationError, "unexpected record fields"):
                evaluate_molecular_predictions(reference, predictions)


if __name__ == "__main__":
    unittest.main()
