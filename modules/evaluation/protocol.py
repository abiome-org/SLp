"""Evaluation guardrails shared by benchmark adapters."""


def require_explicit_benchmark(enabled: bool, benchmark: str) -> None:
    """Prevent a training caller from accidentally opening a benchmark."""

    if not enabled:
        raise RuntimeError(
            f"{benchmark} evaluation is disabled for training. "
            "Run the dedicated benchmark command after locking a checkpoint."
        )
