"""Render one immutable, audit-pinned sparse-world training workload."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_ROOT = PROJECT_ROOT / "workloads"
DEFAULT_TEMPLATE = WORKLOAD_ROOT / "slp-1-1-world-sparse.yaml.tmpl"
PRETRAIN_TOKEN = "@@PRETRAIN_DATASET@@"
QUERY_TOKEN = "@@MOLECULAR_PREDICTION_QUERY_DATASET@@"
ROSTER_TOKEN = "@@HELD_ROSTER_DATASET@@"
AUDIT_DIGEST_TOKEN = "@@CORPUS_AUDIT_ARTIFACT_MANIFEST_DIGEST@@"
DATASET_REFERENCE = re.compile(r"dataset/[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
PINNED_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
MAX_TEMPLATE_BYTES = 1024 * 1024


class WorkloadRenderError(ValueError):
    """Raised when a training workload cannot be frozen exactly and safely."""


def _dataset_reference(value: object, name: str) -> str:
    if not isinstance(value, str) or DATASET_REFERENCE.fullmatch(value) is None:
        raise WorkloadRenderError(
            f"{name} must be an exact dataset/<resource-name> alias"
        )
    return value


def _artifact_manifest_digest(value: object) -> str:
    if not isinstance(value, str) or PINNED_DIGEST.fullmatch(value) is None:
        raise WorkloadRenderError(
            "corpus audit must be an exact sha256:<64 lowercase hex> artifact digest"
        )
    return value


def _read_template() -> str:
    requested = DEFAULT_TEMPLATE.absolute()
    if requested.is_symlink():
        raise WorkloadRenderError("workload template must not be a symlink")
    try:
        template = requested.resolve(strict=True)
    except OSError as error:
        raise WorkloadRenderError("workload template is missing") from error
    if not template.is_file() or template.stat().st_size > MAX_TEMPLATE_BYTES:
        raise WorkloadRenderError("workload template must be a bounded regular file")
    try:
        return template.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise WorkloadRenderError("workload template must be UTF-8 text") from error


def render_workload_text(
    pretrain_dataset: str,
    molecular_prediction_query_dataset: str,
    held_roster_dataset: str,
    corpus_audit_artifact_manifest_digest: str,
) -> str:
    """Bind all governed inputs before training is allowed to start."""

    datasets = {
        PRETRAIN_TOKEN: _dataset_reference(pretrain_dataset, "pretrain dataset"),
        QUERY_TOKEN: _dataset_reference(
            molecular_prediction_query_dataset, "molecular prediction query dataset"
        ),
        ROSTER_TOKEN: _dataset_reference(held_roster_dataset, "held roster dataset"),
    }
    if len(set(datasets.values())) != len(datasets):
        raise WorkloadRenderError("pretrain, query, and held-roster datasets must be distinct")
    audit_digest = _artifact_manifest_digest(corpus_audit_artifact_manifest_digest)
    template = _read_template()
    expected_counts = {
        PRETRAIN_TOKEN: 1,
        QUERY_TOKEN: 1,
        ROSTER_TOKEN: 1,
        AUDIT_DIGEST_TOKEN: 2,
    }
    for token, count in expected_counts.items():
        if template.count(token) != count:
            raise WorkloadRenderError(
                f"workload template must contain {token} exactly {count} time(s)"
            )
    rendered = template
    for token, value in datasets.items():
        rendered = rendered.replace(token, value)
    rendered = rendered.replace(AUDIT_DIGEST_TOKEN, audit_digest)
    if "@@" in rendered:
        raise WorkloadRenderError("rendered workload contains an unresolved template token")
    return rendered


def render_workload(
    pretrain_dataset: str,
    molecular_prediction_query_dataset: str,
    held_roster_dataset: str,
    corpus_audit_artifact_manifest_digest: str,
    output_path: str | Path,
) -> Path:
    """Atomically create a frozen workload, allowing only identical re-renders."""

    output = Path(output_path).absolute().resolve()
    workload_root = WORKLOAD_ROOT.resolve(strict=True)
    try:
        output.relative_to(workload_root)
    except ValueError as error:
        raise WorkloadRenderError("rendered workload must stay under workloads/") from error
    if output.suffix != ".yaml":
        raise WorkloadRenderError("rendered workload must use a .yaml suffix")
    content = render_workload_text(
        pretrain_dataset,
        molecular_prediction_query_dataset,
        held_roster_dataset,
        corpus_audit_artifact_manifest_digest,
    ).encode("utf-8")
    if output.exists():
        if not output.is_file():
            raise WorkloadRenderError("frozen workload destination must be a regular file")
        if output.read_bytes() == content:
            return output
        raise WorkloadRenderError("refusing to overwrite a different frozen workload")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    created_temporary = False
    try:
        with temporary.open("xb") as stream:
            created_temporary = True
            stream.write(content)
        os.replace(temporary, output)
    except OSError as error:
        if created_temporary and temporary.is_file():
            temporary.unlink()
        raise WorkloadRenderError("could not atomically freeze workload") from error
    return output


def cli() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an audit-pinned OMF sparse-world training workload."
    )
    parser.add_argument("--pretrain-dataset", required=True)
    parser.add_argument("--query-dataset", required=True)
    parser.add_argument("--held-roster-dataset", required=True)
    parser.add_argument("--corpus-audit-artifact-digest", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    destination = render_workload(
        arguments.pretrain_dataset,
        arguments.query_dataset,
        arguments.held_roster_dataset,
        arguments.corpus_audit_artifact_digest,
        arguments.output,
    )
    print(destination.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
