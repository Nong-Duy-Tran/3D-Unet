from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import runpy  # noqa: E402


if __name__ == "__main__":
    runpy.run_module("src.project.cli.train", run_name="__main__")
