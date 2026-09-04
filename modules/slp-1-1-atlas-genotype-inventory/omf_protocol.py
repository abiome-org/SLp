"""Minimal standard-library implementation of the OMF module/v1 file protocol.

The module has a non-empty scientific dependency environment. OMF 1.0 does
not inject its controller package into that isolated environment, so the
admitted module carries the small file-protocol boundary it executes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import traceback
from typing import Any, Callable, Mapping


OPERATIONS = {"validate", "prepare", "run", "quiesce", "checkpoint", "restore", "stop"}


@dataclass(frozen=True)
class ProtocolRequest:
    operation: str
    inputs: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ProtocolRequest":
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("OMF request must be a JSON object")
        allowed = {"protocol", "operation", "inputs", "config", "state", "context"}
        if set(value) - allowed:
            raise ValueError("OMF request contains unsupported fields")
        if value.get("protocol", "omf.module/v1") != "omf.module/v1":
            raise ValueError("unsupported OMF protocol")
        operation = value.get("operation")
        if operation not in OPERATIONS:
            raise ValueError("unsupported OMF operation")
        mappings: dict[str, dict[str, Any]] = {}
        for name in ("inputs", "config", "state", "context"):
            item = value.get(name, {})
            if not isinstance(item, dict):
                raise ValueError(f"OMF request {name} must be an object")
            mappings[name] = item
        return cls(operation=operation, **mappings)


@dataclass(frozen=True)
class ProtocolResult:
    status: str
    outputs: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, int | float] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        if self.status not in {"ok", "error"}:
            raise ValueError("OMF result status must be ok or error")
        return {
            "protocol": "omf.module/v1",
            "status": self.status,
            "outputs": self.outputs,
            "state": self.state,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "error": self.error,
        }


Handler = Callable[[ProtocolRequest], ProtocolResult | Mapping[str, Any] | None]


def dispatch(handlers: Mapping[str, Handler], request_path: Path, result_path: Path) -> int:
    try:
        request = ProtocolRequest.from_bytes(request_path.read_bytes())
        handler = handlers.get(request.operation)
        if handler is None:
            raise ValueError(f"operation not implemented: {request.operation}")
        value = handler(request)
        if isinstance(value, ProtocolResult):
            result = value
        else:
            result = ProtocolResult(status="ok", **(dict(value) if value is not None else {}))
    except Exception as error:  # OMF protocol requires structured module failures.
        result = ProtocolResult(
            status="error",
            error={
                "code": type(error).__name__,
                "message": str(error),
                "details": {"traceback": traceback.format_exc()},
            },
        )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, result_path)
    return 0 if result.status == "ok" else 1


def main(handlers: Mapping[str, Handler]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, default=os.getenv("OMF_REQUEST_FILE"))
    parser.add_argument("--result", type=Path, default=os.getenv("OMF_RESULT_FILE"))
    arguments = parser.parse_args()
    if arguments.request is None or arguments.result is None:
        parser.error("request and result paths are required")
    return dispatch(handlers, arguments.request, arguments.result)
