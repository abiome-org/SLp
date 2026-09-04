"""Focused synthetic tests for the frozen molecular point baselines."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "modules" / "slp-1-1-molecular-baselines"
sys.path.insert(0, str(MODULE))

from baselines import (  # noqa: E402
    Limits,
    MolecularBaselineError,
    REFERENCE_ROLE,
    TRAINING_ROLE,
    build_baselines,
    load_snapshot,
    resolve_pinned_dataset,
)


class MolecularBaselineTest(unittest.TestCase):
    def test_omf_artifacts_are_file_valued(self) -> None:
        main_text = (MODULE / "main.py").read_text(encoding="utf-8")
        for relative_file in (
            '"context-only" / "predictions.json"',
            '"context-only" / "predictions.jsonl"',
            '"txpert-mean-additive" / "predictions.json"',
            '"txpert-mean-additive" / "predictions.jsonl"',
            'Path("molecular-baselines") / "baseline-report.json"',
        ):
            self.assertIn(relative_file, main_text)
        self.assertNotIn('"path": str(Path("molecular-baselines") / "context-only"),', main_text)
        self.assertNotIn(
            '"path": str(Path("molecular-baselines") / "txpert-mean-additive"),',
            main_text,
        )

    readouts = ["RNA:r1", "RNA:r2", "RNA:r3"]

    def _profile(
        self,
        context: str,
        values: list[float | None],
        *,
        role: str = "basal-control",
        perturbation: str | None = None,
        interventions: list[str] | None = None,
        taxon: int = 4932,
        source: str = "fixture:source",
        **extra: object,
    ) -> dict[str, object]:
        return {
            "schema": "slp.molecular-baseline-profile/v1",
            "speciesTaxon": taxon,
            "sourceId": source,
            "contextId": context,
            "recordRole": role,
            "perturbationId": perturbation,
            "interventionIds": interventions or [],
            "readoutIds": self.readouts,
            "values": values,
            **extra,
        }

    def _basal(self, context: str, values: list[float | None], **extra: object) -> dict[str, object]:
        return self._profile(context, values, **extra)

    def _perturbed(
        self,
        context: str,
        perturbation: str,
        interventions: list[str],
        values: list[float | None],
        **extra: object,
    ) -> dict[str, object]:
        return self._profile(
            context,
            values,
            role="perturbation-outcome",
            perturbation=perturbation,
            interventions=interventions,
            **extra,
        )

    def _snapshot(
        self,
        root: Path,
        name: str,
        role: str,
        records: list[dict[str, object]],
        *,
        task: str = "perturbation-context-cold-with-basal-access",
        training_sha: str | None = None,
        aggregation_sha: str = "c" * 64,
    ) -> tuple[Path, str]:
        directory = root / name
        directory.mkdir(parents=True)
        shard = directory / "profiles-000.jsonl"
        shard.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
            newline="\n",
        )
        manifest: dict[str, object] = {
            "schema": "slp.molecular-baseline-snapshot/v1",
            "datasetId": f"fixture:{name}",
            "version": "synthetic-v1",
            "role": role,
            "taskName": task,
            "labelClass": "molecular",
            "benchmarkLabelsPresent": False,
            "valueSpace": "SLPVAL:normalized-expression",
            "speciesTaxa": sorted({int(row["speciesTaxon"]) for row in records}),
            "sourceIds": sorted({str(row["sourceId"]) for row in records}),
            "recordsEncoding": "identity-keyed-sparse-jsonl-v1",
            "profileLevel": "context-perturbation-centroid-v1",
            "aggregationProtocolSha256": aggregation_sha,
            "shards": [
                {
                    "path": shard.name,
                    "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                    "records": len(records),
                }
            ],
        }
        if role == REFERENCE_ROLE:
            self.assertIsNotNone(training_sha)
            manifest["pairedTrainingManifestSha256"] = training_sha
        manifest_path = directory / "baseline.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        return directory, hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def _training_records(self) -> list[dict[str, object]]:
        return [
            self._basal("CTX:t1", [10.0, 20.0, 30.0]),
            self._perturbed("CTX:t1", "PERT:a-t1", ["GENE:a"], [12.0, 18.0, 33.0]),
            self._basal("CTX:t2", [5.0, 5.0, 5.0]),
            self._perturbed("CTX:t2", "PERT:a-t2", ["GENE:a"], [9.0, 3.0, None]),
            self._basal("CTX:t3", [1.0, 1.0, 1.0]),
            self._perturbed("CTX:t3", "PERT:b-t3", ["GENE:b"], [0.0, 2.0, 4.0]),
            self._basal("CTX:t4", [0.0, 0.0, 0.0]),
            self._perturbed(
                "CTX:t4", "PERT:ab-t4", ["GENE:a", "GENE:b"], [5.0, 7.0, 9.0]
            ),
            self._basal("CTX:t5", [0.0, 0.0, 0.0]),
            self._perturbed(
                "CTX:t5", "PERT:bc-t5", ["GENE:b", "GENE:c"], [2.5, 1.0, 5.0]
            ),
        ]

    def _reference_records(self) -> list[dict[str, object]]:
        return [
            self._basal("CTX:reference", [100.0, 200.0, 300.0]),
            self._perturbed(
                "CTX:reference",
                "PERT:ab-reference",
                ["GENE:a", "GENE:b"],
                [0.0, 0.0, 0.0],
            ),
            self._perturbed(
                "CTX:reference",
                "PERT:c-reference",
                ["GENE:c"],
                [0.0, None, 0.0],
            ),
            self._perturbed(
                "CTX:reference",
                "PERT:ac-reference",
                ["GENE:a", "GENE:c"],
                [0.0, 0.0, 0.0],
            ),
        ]

    @staticmethod
    def _rows(path: Path, baseline: str) -> list[dict[str, object]]:
        shard = path / baseline / "predictions-000.jsonl"
        return [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]

    def test_context_and_txpert_predictions_follow_frozen_branches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, training_sha = self._snapshot(
                root, "training", TRAINING_ROLE, self._training_records()
            )
            reference, _ = self._snapshot(
                root,
                "reference",
                REFERENCE_ROLE,
                self._reference_records(),
                training_sha=training_sha,
            )
            output = root / "output"
            report = build_baselines(training, reference, output)
            context = self._rows(output, "context-only")
            txpert = self._rows(output, "txpert-mean-additive")
            prediction_manifest = json.loads(
                (output / "context-only" / "predictions.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            [row["predictionMean"] for row in context],
            [
                [100.0, 200.0, 300.0],
                [100.0, 200.0, 300.0],
                [100.0, None, 300.0],
            ],
        )
        # Rows are ordered by perturbation identity: ab uses exact combination;
        # ac is additive A + global(C); c is global with target-null preservation.
        self.assertEqual(txpert[0]["predictionMean"], [105.0, 207.0, 309.0])
        self.assertEqual(txpert[1]["predictionMean"], [105.5, 199.0, 308.0])
        self.assertEqual(txpert[2]["predictionMean"], [102.5, None, 305.0])
        self.assertEqual(report["baselines"]["txpert-mean-additive"]["exactEffectPredictions"], 3)
        self.assertEqual(report["baselines"]["txpert-mean-additive"]["globalFallbackComponents"], 5)
        self.assertEqual(report["evaluationCompatibility"]["reasonCode"], "prediction-log-scale-not-defined")
        self.assertEqual(report["featureBilinearRidge"]["reasonCode"], "feature-vectors-absent")
        self.assertEqual(report["profileLevel"], "context-perturbation-centroid-v1")
        self.assertNotIn("predictionLogScale", json.dumps(prediction_manifest))
        self.assertFalse(prediction_manifest["benchmarkLabelsPresent"])

    def test_output_is_deterministic_under_record_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_one, sha_one = self._snapshot(
                root, "training-one", TRAINING_ROLE, self._training_records()
            )
            reference_one, _ = self._snapshot(
                root,
                "reference-one",
                REFERENCE_ROLE,
                self._reference_records(),
                training_sha=sha_one,
            )
            training_two, sha_two = self._snapshot(
                root, "training-two", TRAINING_ROLE, list(reversed(self._training_records()))
            )
            reference_two, _ = self._snapshot(
                root,
                "reference-two",
                REFERENCE_ROLE,
                list(reversed(self._reference_records())),
                training_sha=sha_two,
            )
            first = root / "first"
            second = root / "second"
            build_baselines(training_one, reference_one, first)
            build_baselines(training_two, reference_two, second)
            self.assertEqual(
                (first / "txpert-mean-additive" / "predictions-000.jsonl").read_bytes(),
                (second / "txpert-mean-additive" / "predictions-000.jsonl").read_bytes(),
            )

    def test_gene_and_context_leakage_fail_closed(self) -> None:
        cases = [
            (
                "intervention-gene-cold",
                self._reference_records(),
                "intervention-gene-cold leakage",
            ),
            (
                "double-cold",
                self._reference_records(),
                "intervention-gene-cold leakage",
            ),
            (
                "perturbation-context-cold-with-basal-access",
                [
                    self._basal("CTX:t1", [1.0, 1.0, 1.0]),
                    self._perturbed("CTX:t1", "PERT:c", ["GENE:c"], [2.0, 2.0, 2.0]),
                ],
                "duplicate identity keys|perturbation-context-cold leakage",
            ),
        ]
        for task, reference_records, error in cases:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                training, training_sha = self._snapshot(
                    root, "training", TRAINING_ROLE, self._training_records(), task=task
                )
                reference, _ = self._snapshot(
                    root,
                    "reference",
                    REFERENCE_ROLE,
                    reference_records,
                    task=task,
                    training_sha=training_sha,
                )
                with self.assertRaisesRegex(MolecularBaselineError, error):
                    build_baselines(training, reference, root / "output")

    def test_intervention_availability_is_specific_to_pure_context_cold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, training_sha = self._snapshot(
                root, "training", TRAINING_ROLE, self._training_records()
            )
            reference, _ = self._snapshot(
                root,
                "reference",
                REFERENCE_ROLE,
                self._reference_records(),
                training_sha=training_sha,
            )
            report = build_baselines(training, reference, root / "output")
            self.assertEqual(
                report["taskName"],
                "perturbation-context-cold-with-basal-access",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            without_c = [
                profile
                for profile in self._training_records()
                if profile["contextId"] != "CTX:t5"
            ]
            without_c.extend(
                [
                    self._basal(
                        "CTX:other-source",
                        [0.0, 0.0, 0.0],
                        source="fixture:other-source",
                    ),
                    self._perturbed(
                        "CTX:other-source",
                        "PERT:c-other-source",
                        ["GENE:c"],
                        [1.0, 1.0, 1.0],
                        source="fixture:other-source",
                    ),
                ]
            )
            training, training_sha = self._snapshot(
                root, "training", TRAINING_ROLE, without_c
            )
            reference, _ = self._snapshot(
                root,
                "reference",
                REFERENCE_ROLE,
                self._reference_records(),
                training_sha=training_sha,
            )
            with self.assertRaisesRegex(
                MolecularBaselineError,
                "requires quantitative fitting outcomes.*GENE:c",
            ):
                build_baselines(training, reference, root / "output")

        for task in ("intervention-gene-cold", "double-cold"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                training, training_sha = self._snapshot(
                    root,
                    "training",
                    TRAINING_ROLE,
                    self._training_records(),
                    task=task,
                )
                reference_records = [
                    self._basal("CTX:held", [10.0, 20.0, 30.0]),
                    self._perturbed(
                        "CTX:held",
                        "PERT:held-d",
                        ["GENE:d"],
                        [11.0, 21.0, 31.0],
                    ),
                ]
                reference, _ = self._snapshot(
                    root,
                    "reference",
                    REFERENCE_ROLE,
                    reference_records,
                    task=task,
                    training_sha=training_sha,
                )
                report = build_baselines(training, reference, root / "output")
                self.assertEqual(report["taskName"], task)

    def test_basal_is_explicit_and_never_inferred(self) -> None:
        bad_records = [
            self._profile(
                "CTX:x",
                [1.0, 2.0, 3.0],
                role="perturbation-outcome",
                centeringGroup="CTX:basal",
            ),
            self._profile(
                "CTX:x",
                [1.0, 2.0, 3.0],
                role="perturbation-outcome",
                perturbation="PERT:empty",
                interventions=[],
            ),
        ]
        for record in bad_records:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                snapshot, _ = self._snapshot(
                    root, "bad", TRAINING_ROLE, [record]
                )
                with self.assertRaisesRegex(
                    MolecularBaselineError, "fields do not match|explicit interventionIds|non-empty list"
                ):
                    load_snapshot(snapshot, TRAINING_ROLE, Limits())

    def test_ambiguous_basal_duplicate_and_nonfinite_values_are_rejected(self) -> None:
        records = [
            self._basal("CTX:x", [1.0, 2.0, 3.0]),
            self._basal("CTX:x", [3.0, 2.0, 1.0]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, _ = self._snapshot(root, "duplicate", TRAINING_ROLE, records)
            with self.assertRaisesRegex(MolecularBaselineError, "ambiguous basal"):
                load_snapshot(snapshot, TRAINING_ROLE, Limits())
        records = [self._basal("CTX:x", [1.0, float("nan"), 3.0])]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, _ = self._snapshot(root, "nonfinite", TRAINING_ROLE, records)
            with self.assertRaisesRegex(MolecularBaselineError, "must be finite or null"):
                load_snapshot(snapshot, TRAINING_ROLE, Limits())

    def test_requires_frozen_centroid_profile_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, _ = self._snapshot(
                root, "training", TRAINING_ROLE, self._training_records()
            )
            manifest_path = snapshot / "baseline.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["profileLevel"] = "replicate"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(
                MolecularBaselineError, "context-perturbation-centroid-v1"
            ):
                load_snapshot(snapshot, TRAINING_ROLE, Limits())

    def test_source_species_and_pair_checksum_mismatches_are_rejected(self) -> None:
        mismatched_context = [
            self._basal("CTX:x", [1.0, 2.0, 3.0]),
            self._perturbed(
                "CTX:x",
                "PERT:human",
                ["GENE:human"],
                [1.0, 2.0, 3.0],
                taxon=9606,
                source="fixture:human",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, _ = self._snapshot(root, "mismatch", TRAINING_ROLE, mismatched_context)
            with self.assertRaisesRegex(MolecularBaselineError, "source/species mismatch"):
                load_snapshot(snapshot, TRAINING_ROLE, Limits())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, _ = self._snapshot(
                root, "training", TRAINING_ROLE, self._training_records()
            )
            reference, _ = self._snapshot(
                root,
                "reference",
                REFERENCE_ROLE,
                self._reference_records(),
                training_sha="f" * 64,
            )
            with self.assertRaisesRegex(MolecularBaselineError, "not pinned to this training"):
                build_baselines(training, reference, root / "output")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, training_sha = self._snapshot(
                root, "training", TRAINING_ROLE, self._training_records()
            )
            reference, _ = self._snapshot(
                root,
                "reference",
                REFERENCE_ROLE,
                self._reference_records(),
                training_sha=training_sha,
                aggregation_sha="d" * 64,
            )
            with self.assertRaisesRegex(
                MolecularBaselineError, "aggregation protocol mismatch"
            ):
                build_baselines(training, reference, root / "output")

    def test_cross_snapshot_identity_mismatch_and_shard_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, training_sha = self._snapshot(
                root, "training", TRAINING_ROLE, self._training_records()
            )
            human_reference = [
                self._basal(
                    "CTX:human", [1.0, 1.0, 1.0], taxon=9606, source="fixture:human"
                ),
                self._perturbed(
                    "CTX:human",
                    "PERT:a-t1",
                    ["GENE:human"],
                    [2.0, 2.0, 2.0],
                    taxon=9606,
                    source="fixture:human",
                ),
            ]
            reference, _ = self._snapshot(
                root,
                "reference",
                REFERENCE_ROLE,
                human_reference,
                training_sha=training_sha,
            )
            with self.assertRaisesRegex(
                MolecularBaselineError, "cross-snapshot intervention/species mismatch"
            ):
                build_baselines(training, reference, root / "output")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, _ = self._snapshot(
                root, "training", TRAINING_ROLE, self._training_records()
            )
            with (snapshot / "profiles-000.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(MolecularBaselineError, "checksum mismatch"):
                load_snapshot(snapshot, TRAINING_ROLE, Limits())

    def test_dataset_input_requires_exact_pinned_omf_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "inputs" / "molecularTraining" / "training-fixture"
            root.mkdir(parents=True)
            exact = {
                "resource": f"omf://local/datasetsnapshot/training-fixture@sha256:{'a' * 64}",
                "mode": "copy",
                "path": str(root),
                "manifestDigest": f"sha256:{'b' * 64}",
            }
            resolved = resolve_pinned_dataset(exact, "molecularTraining")
            self.assertEqual(resolved.path, root.resolve())
            for field, value, error in (
                ("mode", "mount", "immutable copied"),
                ("manifestDigest", "latest", "sha256:<hex>"),
                (
                    "resource",
                    "omf://local/datasetsnapshot/training-fixture@latest",
                    "sha256:<hex>",
                ),
            ):
                bad = {**exact, field: value}
                with self.subTest(field=field), self.assertRaisesRegex(
                    MolecularBaselineError, error
                ):
                    resolve_pinned_dataset(bad, "molecularTraining")
            with self.assertRaisesRegex(MolecularBaselineError, "exact materialized|fields"):
                resolve_pinned_dataset({**exact, "kind": "DatasetSnapshot"}, "molecularTraining")

    def test_schemas_and_protocol_names_are_frozen_without_benchmark_vocabulary(self) -> None:
        input_schema = json.loads((MODULE / "input.schema.json").read_text(encoding="utf-8"))
        output_schema = json.loads((MODULE / "output.schema.json").read_text(encoding="utf-8"))
        readme = (MODULE / "README.md").read_text(encoding="utf-8")
        canonical_digests = [
            hashlib.sha256(
                json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for schema in (input_schema, output_schema)
        ]
        self.assertEqual(
            canonical_digests,
            [
                "ebb60285e7ebc5760be924ef67509b5467c751faebdf87e08881246155e79c46",
                "7ad54a6497dc854462ae69ef76f4c2abc7fc2ab6020ef88971515b43c0387792",
            ],
        )
        self.assertEqual(
            input_schema["properties"]["recordsEncoding"]["const"],
            "identity-keyed-sparse-jsonl-v1",
        )
        self.assertEqual(
            input_schema["properties"]["profileLevel"]["const"],
            "context-perturbation-centroid-v1",
        )
        self.assertEqual(
            output_schema["properties"]["baselineName"]["enum"],
            ["context-only", "txpert-mean-additive"],
        )
        self.assertIn("prediction-log-scale-not-defined", readme)
        for forbidden in ("synthetic-lethality", "depmap", "benchmark test"):
            self.assertNotIn(forbidden, readme.casefold())

if __name__ == "__main__":
    unittest.main()
