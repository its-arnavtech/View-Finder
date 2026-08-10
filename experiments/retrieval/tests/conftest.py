"""Make the shelved experiment importable when its tests are run directly.

These tests live outside the ``viewfinder`` package, so the modules under test
are imported by bare name. Adding the experiment directory and the repository
root to ``sys.path`` lets ``pytest experiments/retrieval`` work without the
experiment being installed or re-entering the active package.
"""

import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent

for path in (str(EXPERIMENT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
