from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = PROJECT_ROOT / "workloads" / "slp-1-1-molecular-eval.yaml.tmpl"
CENTERING_TOKEN = "@@MOLECULAR_CENTERING_DATASET@@"
PREDICTION_TOKEN = "@@MOLECULAR_PREDICTIONS_ARTIFACT@@"
TRUTH_TOKEN = "@@MOLECULAR_TRUTH_DATASET@@"
AUDIT_TOKEN = "@@CORPUS_AUDIT_DATASET@@"
ROSTER_TOKEN = "@@HELD_ROSTER_DATASET@@"
QUERY_TOKEN = "@@MOLECULAR_QUERY_DATASET@@"
QUERY_RESOURCE_TOKEN = "@@MOLECULAR_QUERY_RESOURCE@@"
QUERY_MANIFEST_TOKEN = "@@MOLECULAR_QUERY_MANIFEST_DIGEST@@"
CHECKPOINT_TOKEN = "@@MODEL_CHECKPOINT@@"
ARTIFACT_REFERENCE = re.compile(r"sha256:[0-9a-f]{64}")
DATASET_REFERENCE = re.compile(r"dataset/[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DATASET_RESOURCE = re.compile(r"omf://.+/datasetsnapshot/[A-Za-z0-9][A-Za-z0-9._-]{0,127}@sha256:[0-9a-f]{64}")
PINNED_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class WorkloadRenderError(ValueError):
    """Raised when a frozen evaluation workload cannot be rendered safely."""


def _artifact_reference(value: str, name: str) -> str:
    if ARTIFACT_REFERENCE.fullmatch(value) is None:
        raise WorkloadRenderError(f"{name} must be an exact sha256:<64 lowercase hex> reference")
    return value


def _dataset_reference(value: str) -> str:
    if DATASET_REFERENCE.fullmatch(value) is None:
        raise WorkloadRenderError("query dataset must be an exact dataset/<resource-name> reference")
    return value


def render_workload_text(
    centering_dataset: str,
    prediction_artifact: str,
    truth_dataset: str,
    corpus_audit_dataset: str,
    held_roster_dataset: str,
    query_dataset: str,
    query_resource: str,
    query_manifest_digest: str,
    model_checkpoint: str,
    *,
    template_path: str | Path = DEFAULT_TEMPLATE,
) -> str:
    values = {
        CENTERING_TOKEN: _dataset_reference(centering_dataset),
        PREDICTION_TOKEN: _artifact_reference(prediction_artifact, "prediction artifact"),
        TRUTH_TOKEN: _dataset_reference(truth_dataset),
        AUDIT_TOKEN: _dataset_reference(corpus_audit_dataset),
        ROSTER_TOKEN: _dataset_reference(held_roster_dataset),
        QUERY_TOKEN: _dataset_reference(query_dataset),
        CHECKPOINT_TOKEN: _artifact_reference(model_checkpoint, "model checkpoint artifact"),
    }
    dataset_values = [values[token] for token in (CENTERING_TOKEN, TRUTH_TOKEN, AUDIT_TOKEN, ROSTER_TOKEN, QUERY_TOKEN)]
    if len(set(dataset_values)) != len(dataset_values):
        raise WorkloadRenderError("all five DatasetSnapshots must be distinct")
    if DATASET_RESOURCE.fullmatch(query_resource) is None:
        raise WorkloadRenderError("query resource must be an exact revision-pinned OMF DatasetSnapshot URI")
    if PINNED_DIGEST.fullmatch(query_manifest_digest) is None:
        raise WorkloadRenderError("query outer manifest must be an exact SHA-256 digest")
    values[QUERY_RESOURCE_TOKEN] = query_resource
    values[QUERY_MANIFEST_TOKEN] = query_manifest_digest
    template = Path(template_path).read_text(encoding="utf-8")
    for token in values:
        if template.count(token) != 1:
            raise WorkloadRenderError(f"workload template must contain {token} exactly once")
    rendered = template
    for token, value in values.items():
        rendered = rendered.replace(token, value)
    if "@@" in rendered:
        raise WorkloadRenderError("rendered workload contains an unresolved template token")
    return rendered


def render_workload(
    centering_dataset: str,
    prediction_artifact: str,
    truth_dataset: str,
    corpus_audit_dataset: str,
    held_roster_dataset: str,
    query_dataset: str,
    query_resource: str,
    query_manifest_digest: str,
    model_checkpoint: str,
    output_path: str | Path,
    *,
    template_path: str | Path = DEFAULT_TEMPLATE,
) -> Path:
    output = Path(output_path).resolve()
    workload_root = (PROJECT_ROOT / "workloads").resolve()
    try:
        output.relative_to(workload_root)
    except ValueError as error:
        raise WorkloadRenderError("rendered workload must stay under workloads/") from error
    if output.suffix != ".yaml":
        raise WorkloadRenderError("rendered workload must use a .yaml suffix")
    content = render_workload_text(
        centering_dataset,
        prediction_artifact,
        truth_dataset,
        corpus_audit_dataset,
        held_roster_dataset,
        query_dataset,
        query_resource,
        query_manifest_digest,
        model_checkpoint,
        template_path=template_path,
    )
    encoded = content.encode("utf-8")
    if output.exists():
        if output.read_bytes() == encoded:
            return output
        raise WorkloadRenderError("refusing to overwrite a different frozen workload")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, output)
    return output


def cli() -> int:
    parser = argparse.ArgumentParser(
        description="Render a second-run OMF molecular evaluation workload from frozen artifacts."
    )
    parser.add_argument("--centering-dataset", required=True)
    parser.add_argument("--prediction-artifact", required=True)
    parser.add_argument("--truth-dataset", required=True)
    parser.add_argument("--corpus-audit-dataset", required=True)
    parser.add_argument("--held-roster-dataset", required=True)
    parser.add_argument("--query-dataset", required=True)
    parser.add_argument("--query-resource", required=True)
    parser.add_argument("--query-manifest-digest", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    arguments = parser.parse_args()
    destination = render_workload(
        arguments.centering_dataset,
        arguments.prediction_artifact,
        arguments.truth_dataset,
        arguments.corpus_audit_dataset,
        arguments.held_roster_dataset,
        arguments.query_dataset,
        arguments.query_resource,
        arguments.query_manifest_digest,
        arguments.model_checkpoint,
        arguments.output,
        template_path=arguments.template,
    )
    print(destination.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
