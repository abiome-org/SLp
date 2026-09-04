"""Freeze-before-run workload tests for sparse-world training."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "slp-1-1-world-sparse"
TEMPLATE = ROOT / "workloads" / "slp-1-1-world-sparse.yaml.tmpl"
SPEC = importlib.util.spec_from_file_location("slp11_sparse_workload", MODULE / "render_workload.py")
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)

PRETRAIN = "dataset/slp-1-1-pretrain-v1"
QUERY = "dataset/slp-1-1-molecular-query-v1"
ROSTER = "dataset/slp-1-1-held-roster-v1"
AUDIT_DIGEST = "sha256:" + "a" * 64


class SparseTrainingWorkloadTest(unittest.TestCase):
    def test_render_binds_exact_inputs_digest_and_all_outputs(self) -> None:
        text = renderer.render_workload_text(PRETRAIN, QUERY, ROSTER, AUDIT_DIGEST)
        self.assertNotIn("@@", text)
        workload = yaml.safe_load(text)
        stage = workload["spec"]["graph"]["stages"][0]
        self.assertEqual(stage["module"], "modules/slp-1-1-world-sparse/module.yaml")
        self.assertEqual(stage["operation"], "run")
        self.assertEqual(
            stage["inputs"],
            {
                "pretrain": PRETRAIN,
                "molecularPredictionQuery": QUERY,
                "corpusAuditEvidence": "artifact:" + AUDIT_DIGEST,
                "heldRosterEvidence": ROSTER,
            },
        )
        self.assertEqual(
            stage["config"]["expectedCorpusAuditArtifactManifestDigest"],
            AUDIT_DIGEST,
        )
        module = yaml.safe_load((MODULE / "module.yaml").read_text(encoding="utf-8"))
        expected_outputs = set(module["spec"]["contracts"]["output"]["required"])
        expected_outputs.update(
            {"worldCheckpoint", "molecularValidationPredictions", "trainingReport"}
        )
        self.assertEqual(set(stage["outputs"]), expected_outputs)
        self.assertEqual(len(stage["outputs"]), len(set(stage["outputs"])))

    def test_rejects_noncanonical_or_aliased_inputs(self) -> None:
        invalid_datasets = (
            "slp-1-1-pretrain-v1",
            "dataset/a/b",
            "dataset/latest@sha256:" + "1" * 64,
            "dataset/white space",
            "omf://abiome/slp/datasetsnapshot/pretrain@sha256:" + "1" * 64,
        )
        for value in invalid_datasets:
            with self.subTest(dataset=value), self.assertRaises(renderer.WorkloadRenderError):
                renderer.render_workload_text(value, QUERY, ROSTER, AUDIT_DIGEST)
        with self.assertRaisesRegex(renderer.WorkloadRenderError, "must be distinct"):
            renderer.render_workload_text(PRETRAIN, PRETRAIN, ROSTER, AUDIT_DIGEST)
        for value in ("a" * 64, "artifact:" + AUDIT_DIGEST, "sha256:" + "A" * 64):
            with self.subTest(digest=value), self.assertRaises(renderer.WorkloadRenderError):
                renderer.render_workload_text(PRETRAIN, QUERY, ROSTER, value)

    def test_identical_rerender_is_idempotent_and_different_freeze_is_rejected(self) -> None:
        workload_root = ROOT / "workloads"
        with tempfile.TemporaryDirectory(dir=workload_root) as temporary:
            output = Path(temporary) / "frozen.yaml"
            first = renderer.render_workload(PRETRAIN, QUERY, ROSTER, AUDIT_DIGEST, output)
            original = first.read_bytes()
            second = renderer.render_workload(PRETRAIN, QUERY, ROSTER, AUDIT_DIGEST, output)
            self.assertEqual(first, second)
            self.assertEqual(second.read_bytes(), original)
            with self.assertRaisesRegex(renderer.WorkloadRenderError, "refusing to overwrite"):
                renderer.render_workload(
                    PRETRAIN, QUERY, ROSTER, "sha256:" + "b" * 64, output
                )
            self.assertEqual(output.read_bytes(), original)

            blocked = Path(temporary) / "blocked.yaml"
            staging = blocked.with_name(blocked.name + ".tmp")
            staging.write_bytes(b"owned-by-another-render\n")
            with self.assertRaisesRegex(renderer.WorkloadRenderError, "atomically freeze"):
                renderer.render_workload(PRETRAIN, QUERY, ROSTER, AUDIT_DIGEST, blocked)
            self.assertEqual(staging.read_bytes(), b"owned-by-another-render\n")

    def test_output_is_confined_to_workloads_and_template_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary) / "outside.yaml"
            with self.assertRaisesRegex(renderer.WorkloadRenderError, "under workloads"):
                renderer.render_workload(PRETRAIN, QUERY, ROSTER, AUDIT_DIGEST, outside)
            malformed = Path(temporary) / "malformed.tmpl"
            malformed.write_text(TEMPLATE.read_text(encoding="utf-8") + "@@EXTRA@@\n")
            with patch.object(renderer, "DEFAULT_TEMPLATE", malformed):
                with self.assertRaisesRegex(renderer.WorkloadRenderError, "unresolved"):
                    renderer.render_workload_text(PRETRAIN, QUERY, ROSTER, AUDIT_DIGEST)
        with tempfile.TemporaryDirectory(dir=ROOT / "workloads") as temporary:
            with self.assertRaisesRegex(renderer.WorkloadRenderError, r"\.yaml suffix"):
                renderer.render_workload(
                    PRETRAIN, QUERY, ROSTER, AUDIT_DIGEST, Path(temporary) / "frozen.yml"
                )

    def test_cli_executes_the_same_frozen_renderer(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "workloads") as temporary:
            output = Path(temporary) / "cli-frozen.yaml"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODULE / "render_workload.py"),
                    "--pretrain-dataset", PRETRAIN,
                    "--query-dataset", QUERY,
                    "--held-roster-dataset", ROSTER,
                    "--corpus-audit-artifact-digest", AUDIT_DIGEST,
                    "--output", str(output),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            self.assertEqual(
                completed.stdout.strip(), output.relative_to(ROOT).as_posix()
            )


if __name__ == "__main__":
    unittest.main()
