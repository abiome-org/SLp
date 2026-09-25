"""Re-evaluate registered test predictions and refresh hash-pinned result JSON.

Maintainer-only: requires the private benchmark with hidden test labels.
Usage: SLB_BENCH=data/slb uv run python scripts/refresh_test_results.py [--rescore]

Without --rescore a changed point score is an error (the pins should only move when the scorer's output
does not). After a deliberate rebuild of the benchmark, pass --rescore to accept the new scores.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from slbench import evaluate as E


def score_one(entry: dict) -> tuple[str, dict | None, float, list[float]]:
    pred = Path(entry["predictions"])
    dest = Path(entry["result"])
    old = json.loads(dest.read_text()) if dest.exists() else None
    pred_hash = E.file_sha256(pred)
    manifest_hash = E.file_sha256(E.BENCH / "manifest.json")
    ci = old.get("slb_score_ci95") if old else None
    cache_valid = (
        old is not None
        and E.file_sha256(dest) == entry["result_sha256"]
        and old.get("benchmark") == E.bench_id()
        and old.get("manifest_sha256") == manifest_hash
        and old.get("split") == "test"
        and old.get("scorer_version") == E.SCORER_VERSION
        and old.get("predictions_sha256") == pred_hash
        and old.get("missing_filled") == 0
        and isinstance(ci, list) and len(ci) == 2
        and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x) for x in ci)
        and 0 <= ci[0] <= ci[1] <= 1
    )
    if cache_valid:
        return entry["result"], None, old["slb_score"], ci

    report = E.evaluate(E.read_predictions(pred), "test", boot=200, log=False)
    report["predictions_sha256"] = pred_hash
    if old and not os.environ.get("SLB_RESCORE") and (abs(old["slb_score"] - report["slb_score"]) > 1e-10
                                                     or old["n"] != report["n"]
                                                     or old["species_scores"] != report["species_scores"]):
        raise ValueError(f"point score changed for {entry['name']}: {dest} (pass --rescore after a deliberate rebuild)")
    return entry["result"], report, report["slb_score"], report["slb_score_ci95"]


def stage_file(dest: Path, content: bytes | None = None, report: dict | None = None) -> Path:
    """Write a sibling temp file so replacement can be atomic on the same filesystem."""
    fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".staged", dir=dest.parent)
    os.close(fd)
    staged = Path(name)
    try:
        if report is not None:
            E.save(report, staged)
        else:
            assert content is not None
            staged.write_bytes(content)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def main() -> None:
    import sys

    if "--rescore" in sys.argv[1:]:
        os.environ["SLB_RESCORE"] = "1"  # inherited by the spawned workers
    config = Path("leaderboard.yaml")
    config_bytes = config.read_bytes()
    entries = yaml.safe_load(config_bytes)
    new_hashes: dict[str, str] = {}
    staged: dict[Path, Path] = {}
    try:
        # Workers only compute. No published report changes before every worker succeeds.
        with ProcessPoolExecutor(max_workers=4,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(score_one, entries))
        for entry, (path, report, score, ci) in zip(entries, results):
            dest = Path(path)
            if path in new_hashes:
                raise ValueError(f"repeated leaderboard result: {path}")
            if report is not None:
                staged[dest] = stage_file(dest, report=report)
            new_hashes[path] = E.file_sha256(staged.get(dest, dest))
            print(f"{entry['name']}: {score:.6f} CI {ci[0]:.4f}–{ci[1]:.4f}", flush=True)

        lines = config_bytes.decode().splitlines()
        current = None
        done = set()
        for i, line in enumerate(lines):
            if line.startswith("  result: "):
                current = line.removeprefix("  result: ")
            elif line.startswith("  result_sha256: "):
                if current not in new_hashes or current in done:
                    raise ValueError(f"unmatched or repeated leaderboard result: {current}")
                lines[i] = "  result_sha256: " + new_hashes[current]
                done.add(current)
        if done != set(new_hashes):
            raise ValueError(f"missing result hash slots: {set(new_hashes) - done}")
        staged[config] = stage_file(config, content=("\n".join(lines) + "\n").encode())

        # Keep byte-for-byte backups until all replacements succeed. A failed
        # replacement restores every earlier path before surfacing the error.
        original = {dest: dest.read_bytes() if dest.exists() else None for dest in staged}
        replaced: list[Path] = []
        try:
            for dest, temp in staged.items():
                os.replace(temp, dest)
                replaced.append(dest)
        except BaseException:
            for dest in reversed(replaced):
                previous = original[dest]
                if previous is None:
                    dest.unlink(missing_ok=True)
                else:
                    os.replace(stage_file(dest, content=previous), dest)
            raise
    finally:
        for temp in staged.values():
            temp.unlink(missing_ok=True)
    print(f"refreshed {len(entries)} result pins for scorer {E.SCORER_VERSION}")


if __name__ == "__main__":
    main()
