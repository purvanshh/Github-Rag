"""
metadata_utils.py — Repository metadata utilities and normalization functions.
"""

from __future__ import annotations

import os
from typing import Iterable


def iter_python_files(repo_path: str) -> Iterable[str]:
    """Yield all Python files under *repo_path* as absolute paths."""
    skip_dirs = {
        ".git",
        "venv",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        "dist",
        "build",
    }
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


def module_name_from_path(repo_path: str, file_path: str) -> str:
    """Return a dotted module name for *file_path* relative to *repo_path*."""
    rel = os.path.relpath(file_path, repo_path)
    rel_no_ext, _ = os.path.splitext(rel)
    parts = [p for p in rel_no_ext.split(os.sep) if p not in {"", "."}]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else ""
