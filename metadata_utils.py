"""
metadata_utils.py — Repository metadata utilities and normalization functions.
"""

from __future__ import annotations

import hashlib
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
    rel = normalize_file_path(file_path, repo_path)
    rel_no_ext, _ = os.path.splitext(rel)
    parts = [p for p in rel_no_ext.split("/") if p not in {"", "."}]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else ""


def normalize_repo_id(repo_url_or_name: str) -> str:
    """Normalize a repository URL or name to a standard ID/slug."""
    if not repo_url_or_name:
        return "unknown"
    # Extract the name from URL if present
    name = repo_url_or_name.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name.lower().strip()


def normalize_file_path(file_path: str, repo_path: str | None = None) -> str:
    """Normalize a file path to be relative to repo_path and use forward slashes."""
    if not file_path:
        return ""
    
    # Resolve absolute path if repo_path is provided
    if repo_path:
        repo_abs = os.path.abspath(repo_path)
        if not os.path.isabs(file_path):
            file_path = os.path.join(repo_path, file_path)
        file_abs = os.path.abspath(file_path)
        # Check if file_path is indeed under repo_path
        if file_abs.startswith(repo_abs):
            rel = os.path.relpath(file_abs, repo_abs)
        else:
            # Fall back to base name or direct relpath if outside
            rel = os.path.relpath(file_path, repo_path)
    else:
        # Fall back to relative path from current working directory or just keep it
        if os.path.isabs(file_path):
            rel = os.path.relpath(file_path, os.getcwd())
        else:
            rel = file_path

    # Clean up separators to be forward slashes only
    normalized = rel.replace("\\", "/").strip("/")
    
    # Strip any leading ./ or similar
    if normalized.startswith("./"):
        normalized = normalized[2:]
        
    return normalized


def normalize_fqn(module_name: str, parent_class: str | None, symbol_name: str) -> str:
    """Normalize and construct a Fully Qualified Name for a symbol."""
    parts = []
    if module_name:
        parts.append(module_name)
    if parent_class:
        parts.append(parent_class)
    if symbol_name:
        parts.append(symbol_name)
    return ".".join(parts)


def normalize_symbol_id(repo_id: str, file_path: str, fqn: str, start_line: int) -> str:
    """Construct a standardized, deterministic ID for a symbol."""
    norm_repo = normalize_repo_id(repo_id)
    norm_file = normalize_file_path(file_path)
    return f"{norm_repo}:{norm_file}:{fqn}:{start_line}"
