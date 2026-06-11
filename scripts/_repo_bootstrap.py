"""Make repository-local packages importable for directly executed scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> Path:
    """Insert the repository root into ``sys.path`` and return it."""

    repo_root = Path(__file__).resolve().parents[1]
    root_text = str(repo_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return repo_root


__all__ = ["ensure_repo_root_on_path"]
