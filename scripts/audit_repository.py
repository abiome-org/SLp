"""Check the collaboration boundary: tracked payloads and maintained doc links."""

import collections
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".npz",
    ".npy",
    ".pkl",
    ".joblib",
    ".parquet",
    ".h5ad",
    ".h5",
    ".bin",
    ".uint16",
    ".whl",
}


def audit(root=ROOT):
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    tracked = [n for n in tracked if n and (root / n).is_file()]
    errors = []
    for name in tracked:
        p = Path(name)
        fixture = name.startswith("data/fixtures/")
        if p.parts[0] in {".omf", ".openfoundry", "ontology", ".venv", ".cache"}:
            errors.append("Tracked runtime: " + name)
        if name.startswith("data/") and not fixture:
            errors.append("Tracked data outside synthetic fixtures: " + name)
        if name.startswith("results/") and name != "results/.gitkeep":
            errors.append("Tracked result: " + name)
        if p.suffix.lower() in PAYLOAD_SUFFIXES and not fixture:
            errors.append("Tracked payload: " + name)
        if p.name.startswith(".env") and p.name != ".env.example":
            errors.append("Tracked environment secrets file: " + name)
        if (root / name).stat().st_size > (1024 * 1024 if fixture else 5 * 1024 * 1024):
            errors.append("Oversized tracked file: " + name)
    # These are the maintained entry points. Historical evidence remains archived.
    docs = ["README.md", "MODEL_CARD.md", "docs/development.md", "docs/module-reference.md"]
    for name in docs:
        p = root / name
        if not p.is_file():
            errors.append("Missing documentation: " + name)
            continue
        content = re.sub(r"```.*?```", "", p.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", content):
            target = target.strip("<>").split("#")[0]
            if not target or re.match(r"[a-zA-Z][\w+.-]*:", target) or target.startswith("/"):
                continue
            if not (p.parent / target).exists():
                errors.append(f"Broken link in {name}: {target}")
    return {
        "tracked_files": len(tracked),
        "tracked_by_directory": dict(
            sorted(
                collections.Counter(
                    Path(n).parts[0] if "/" in n else "(root)" for n in tracked
                ).items()
            )
        ),
        "errors": errors,
        "passed": not errors,
    }


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
