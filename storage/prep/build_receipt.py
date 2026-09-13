"""Write the build identity checked against the actual running container image."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
h = hashlib.sha256()
for path in sorted(
    [*root.glob("*.py"), root / "prep-jobs.json", root / "molecular-inputs.json"]
):
    h.update(path.name.encode() + b"\0")
    h.update(path.read_bytes())
(root / "build-receipt.json").write_text(
    json.dumps({"sha256": h.hexdigest()}, indent=2) + "\n"
)
