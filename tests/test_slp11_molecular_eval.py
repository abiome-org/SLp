"""Synthetic contract tests for perturbation-specific molecular evaluation."""

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-molecular-eval"
sys.path.insert(0, str(MODULE))

from evaluator import (
    CENTRAL_50_NORMAL_Z,
    CENTRAL_90_NORMAL_Z,
    MINIMUM_PERTURBED_CENTROID_PEARSON,
    MINIMUM_SPECIES_PERTURBED_CENTROID_PEARSON,
    PREDICTION_ROLE,
    REFERENCE_ROLE,
    MolecularEvaluationError,
    ScalarMoments,
    evaluate_molecular_predictions,
    resolve_literal_omf_artifact,
)
from render_workload import WorkloadRenderError, render_workload_text


class MolecularEvaluationTest(unittest.TestCase):
    def _record(
        self,
        taxon: int,
        source: str,
        perturbation: str,
        target: list[float],
        prediction: list[float] | None = None,
        interventions: list[str] | None = None,
        prediction_log_scale: list[float] | None = None,
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
            record["predictionLogScale"] = (
                prediction_log_scale if prediction_log_scale is not None else [0.0, 0.0, 0.0]
            )
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
        self.assertEqual(report["schema"], "slp.molecular-evaluation-report/v2")
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
        self.assertIn("exp(predictionLogScale)^2", report["method"]["gaussianScaleDefinition"])
        expected_widths = {
            "central50": 2.0 * CENTRAL_50_NORMAL_Z,
            "central90": 2.0 * CENTRAL_90_NORMAL_Z,
        }
        for group in (
            report["overall"],
            *report["species"].values(),
            *report["sources"].values(),
            *report["speciesSources"].values(),
        ):
            calibration = group["gaussianCalibration"]
            for interval, expected_width in expected_widths.items():
                self.assertAlmostEqual(calibration[interval]["empiricalCoverage"], 1.0)
                self.assertAlmostEqual(
                    calibration[interval]["meanIntervalWidth"], expected_width
                )
        self.assertTrue(report["decision"]["passed"])
        self.assertTrue(report["decision"]["compatibilityPassed"])
        self.assertEqual(report["decision"]["scope"], "molecular-profile-evaluation-only")
        self.assertEqual(
            set(report["decision"]["checks"]),
            {
                "zeroBenchmarkLabelRecords",
                "zeroHeldInterventionOverlap",
                "overallPerturbedCentroidPearson",
                "minimumSpeciesPerturbedCentroidPearson",
                "everySpeciesHasEligibleProfiles",
                "everySourceHasEligibleProfiles",
            },
        )

    def test_gaussian_interval_endpoints_are_inclusive_and_exact(self) -> None:
        moments = ScalarMoments()
        moments.add(0.0, CENTRAL_50_NORMAL_Z, 0.0)
        moments.add(0.0, math.nextafter(CENTRAL_50_NORMAL_Z, math.inf), 0.0)
        moments.add(0.0, CENTRAL_90_NORMAL_Z, 0.0)
        moments.add(0.0, math.nextafter(CENTRAL_90_NORMAL_Z, math.inf), 0.0)

        calibration = moments.gaussian_calibration_report()
        self.assertEqual(calibration["central50"]["z"], 0.6744897501960817)
        self.assertEqual(calibration["central90"]["z"], 1.6448536269514722)
        self.assertEqual(calibration["central50"]["empiricalCoverage"], 0.25)
        self.assertEqual(calibration["central90"]["empiricalCoverage"], 0.75)
        self.assertEqual(
            calibration["central50"]["meanIntervalWidth"],
            2.0 * CENTRAL_50_NORMAL_Z,
        )
        self.assertEqual(
            calibration["central90"]["meanIntervalWidth"],
            2.0 * CENTRAL_90_NORMAL_Z,
        )

    def test_gaussian_calibration_is_stratified_without_affecting_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, reference_sha256 = self._snapshot(
                root, "reference", REFERENCE_ROLE, self._reference_records()
            )
            prediction_records = self._prediction_records(exact=True)
            for record in prediction_records[2:]:
                record["predictionMean"] = [
                    value + 4.0 for value in record["predictionMean"]
                ]
                record["predictionLogScale"] = [math.log(2.0)] * 3
            predictions, _ = self._snapshot(
                root,
                "predictions",
                PREDICTION_ROLE,
                prediction_records,
                reference_sha256,
            )
            report = evaluate_molecular_predictions(reference, predictions)

        for interval, z in (
            ("central50", CENTRAL_50_NORMAL_Z),
            ("central90", CENTRAL_90_NORMAL_Z),
        ):
            self.assertEqual(
                report["species"]["4932"]["gaussianCalibration"][interval][
                    "empiricalCoverage"
                ],
                1.0,
            )
            self.assertEqual(
                report["species"]["9606"]["gaussianCalibration"][interval][
                    "empiricalCoverage"
                ],
                0.0,
            )
            self.assertAlmostEqual(
                report["sources"]["costanzo:2016"]["gaussianCalibration"][interval][
                    "meanIntervalWidth"
                ],
                2.0 * z,
            )
            self.assertAlmostEqual(
                report["speciesSources"]["9606|replogle:2022"][
                    "gaussianCalibration"
                ][interval]["meanIntervalWidth"],
                4.0 * z,
            )
            self.assertEqual(
                report["overall"]["gaussianCalibration"][interval]["empiricalCoverage"],
                0.5,
            )
            self.assertAlmostEqual(
                report["overall"]["gaussianCalibration"][interval]["meanIntervalWidth"],
                3.0 * z,
            )
        self.assertTrue(report["decision"]["passed"])

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
        self.assertFalse(report["decision"]["passed"])
        self.assertTrue(report["decision"]["compatibilityPassed"])

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

    def test_second_run_template_renders_only_literal_frozen_artifacts(self) -> None:
        reference = "sha256:" + "c" * 64
        predictions = "sha256:" + "d" * 64
        rendered = render_workload_text(reference, predictions)
        workload = yaml.safe_load(rendered)
        stage = workload["spec"]["graph"]["stages"][0]
        self.assertEqual(stage.get("needs", []), [])
        self.assertEqual(stage["inputs"]["molecularReference"], reference)
        self.assertEqual(stage["inputs"]["molecularPredictions"], predictions)
        self.assertIn("passed", stage["outputs"])
        self.assertIn("compatibilityPassed", stage["outputs"])
        self.assertNotIn("@@", rendered)

    def test_second_run_renderer_rejects_mutable_or_ambiguous_references(self) -> None:
        digest = "sha256:" + "e" * 64
        with self.assertRaisesRegex(WorkloadRenderError, "exact sha256"):
            render_workload_text("train.predictions", digest)
        with self.assertRaisesRegex(WorkloadRenderError, "must be distinct"):
            render_workload_text(digest, digest)

    def test_evaluation_spec_cannot_drift_from_the_profile_gate(self) -> None:
        path = Path(__file__).resolve().parents[1] / "evaluations" / (
            "slp-1-1-molecular-artifact.yaml"
        )
        resource = yaml.safe_load(path.read_text(encoding="utf-8"))
        metrics = {item["name"]: item for item in resource["spec"]["metrics"]}
        self.assertEqual(metrics["molecular-profile-gate"]["output"], "molecular.passed")
        self.assertEqual(
            metrics["perturbation-specific-pearson"]["minimum"],
            MINIMUM_PERTURBED_CENTROID_PEARSON,
        )
        self.assertEqual(
            metrics["worst-species-perturbation-specific-pearson"]["minimum"],
            MINIMUM_SPECIES_PERTURBED_CENTROID_PEARSON,
        )

    def test_architecture_comparison_protocol_cannot_drift(self) -> None:
        path = Path(__file__).resolve().parents[1] / "evaluations" / (
            "slp-1-1-molecular-comparison-protocol-v1.yaml"
        )
        protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
        canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            "b0a18d8551a5a7464e575893d291a9c6c312116990570bf2ae4a4a210b5dc443",
        )
        self.assertEqual(protocol["schema"], "slp.molecular-architecture-comparison-protocol/v1")
        self.assertEqual(protocol["protocolVersion"], "1.0.0")
        self.assertEqual(
            [task["name"] for task in protocol["tasks"]],
            [
                "intervention-gene-cold",
                "perturbation-context-cold-with-basal-access",
                "double-cold",
            ],
        )
        self.assertIn(
            "same exact NCBI taxonomy ID and source ID stratum",
            protocol["tasks"][1]["interventionAccess"],
        )
        self.assertEqual(
            [baseline["name"] for baseline in protocol["requiredBaselines"]],
            ["context-only", "txpert-mean-additive", "feature-bilinear-ridge"],
        )
        txpert = protocol["requiredBaselines"][1]["definition"]
        self.assertIn("taxonomy ID and source ID", txpert["fittingStratum"])
        blocked = protocol["protocolRequiredContractBlocked"]
        self.assertEqual(blocked["bds"]["inadmissibleAtOrBelow"], 0.5)
        self.assertEqual(blocked["bds"]["status"], "protocol-required-contract-blocked")
        self.assertEqual(
            blocked["differentialExpression"]["metrics"],
            [
                "reference-significant-lfc-spearman",
                "reference-significant-direction-agreement",
            ],
        )
        self.assertEqual(
            blocked["differentialExpression"]["status"],
            "protocol-required-contract-blocked",
        )
        self.assertEqual(
            protocol["populationMetrics"]["prohibitedUntilPopulationGenerativeOutput"],
            ["energy-distance", "wasserstein-distance"],
        )

    def test_evaluator_accepts_only_admission_pinned_literal_artifact_objects(self) -> None:
        digest = "sha256:" + "f" * 64
        path = "C:/omf/run/stages/molecular/inputs/reference/payload"
        materialized = {
            "resource": f"artifact:{digest}",
            "kind": "artifact",
            "artifacts": {"payload": digest},
            "paths": {"payload": path},
            "path": path,
        }
        self.assertEqual(
            resolve_literal_omf_artifact(materialized, "molecularReference"),
            (path, digest),
        )
        with self.assertRaisesRegex(MolecularEvaluationError, "literal OMF artifact"):
            resolve_literal_omf_artifact({"path": path}, "molecularReference")


if __name__ == "__main__":
    unittest.main()
