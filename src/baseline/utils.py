from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Find repo root by walking up to pyproject.toml."""
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current
