"""Backward-compatible import surface for the modular training implementation.

The immutable simulator is in :mod:`model.v1`; optimization workflows are in
``modules.training.world``. This file preserves existing scripts and remote
entry points that import ``world_model`` directly.
"""

from pathlib import Path
import sys


ROOT = next(
    (
        parent
        for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents)
        if (parent / "modules").is_dir()
    ),
    Path.cwd(),
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.training.world import *  # noqa: F401,F403,E402
