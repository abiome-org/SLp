from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = PROJECT_ROOT / "workloads" / "slp-1-1-molecular-eval.yaml.tmpl"
REFERENCE_TOKEN = "@@MOLECULAR_REFERENCE_ARTIFACT@@"
PREDICTION_TOKEN = "@@MOLECULAR_PREDICTIONS_ARTIFACT@@"
ARTIFACT_REFERENCE = re.compile(r"sha256:[0-9a-f]{64}")


class WorkloadRenderError(ValueError):
    """Raised when a frozen evaluation workload cannot be rendered safely."""


def _artifact_reference(value: str, name: str) -> str:
    if ARTIFACT_REFERENCE.fullmatch(value) is None:
        raise WorkloadRenderError(f"{name} must be an exact sha256:<64 lowercase hex> reference")
    return value


def render_workload_text(
    reference_artifact: str,
    prediction_artifact: str,
    *,
    template_path: str | Path = DEFAULT_TEMPLATE,
) -> str:
    reference = _artifact_reference(reference_artifact, "reference artifact")
    prediction = _artifact_reference(prediction_artifact, "prediction artifact")
    if reference == prediction:
        raise WorkloadRenderError("reference and prediction artifacts must be distinct")
    template = Path(template_path).read_text(encoding="utf-8")
    for token in (REFERENCE_TOKEN, PREDICTION_TOKEN):
        if template.count(token) != 1:
            raise WorkloadRenderError(f"workload template must contain {token} exactly once")
    rendered = template.replace(REFERENCE_TOKEN, reference).replace(PREDICTION_TOKEN, prediction)
    if "@@" in rendered:
        raise WorkloadRenderError("rendered workload contains an unresolved template token")
    return rendered


def render_workload(
    reference_artifact: str,
    prediction_artifact: str,
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
        reference_artifact,
        prediction_artifact,
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
    parser.add_argument("--reference-artifact", required=True)
    parser.add_argument("--prediction-artifact", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    arguments = parser.parse_args()
    destination = render_workload(
        arguments.reference_artifact,
        arguments.prediction_artifact,
        arguments.output,
        template_path=arguments.template,
    )
    print(destination.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
