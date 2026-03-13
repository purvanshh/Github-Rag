"""
dependency_graph.py — Build and query the import/dependency graph.

Builds a directed graph of file-level imports to understand how
modules depend on each other.

High-level API:
    - build_dependency_graph(repo_path) → DependencyGraph
    - get_dependencies(graph, file_path) → list[str]
    - visualize_graph(graph, output_path=None) → None
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Iterable

import networkx as nx


@dataclass
class DependencyEdge:
    """Represents one import relationship between files or modules."""

    source_file: str
    target_module: str
    import_statement: str


class DependencyGraph:
    """Builds and queries file-level dependency graphs.

    Nodes are file paths relative to the repository root (e.g. ``pkg/foo.py``)
    and, when a dependency cannot be resolved to an in-repo file, the imported
    module name (e.g. ``requests``).
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_edge(self, source: str, target: str, import_stmt: str = "") -> None:
        """Add a dependency edge: ``source`` imports ``target``."""
        self.graph.add_edge(source, target, import_statement=import_stmt)

    def get_dependencies(self, file_path: str) -> list[str]:
        """Get all files/modules that a file depends on."""
        if file_path in self.graph:
            return list(self.graph.successors(file_path))
        return []

    def get_dependents(self, file_path: str) -> list[str]:
        """Get all files/modules that depend on a file."""
        if file_path in self.graph:
            return list(self.graph.predecessors(file_path))
        return []

    def get_most_connected(self, top_k: int = 10) -> list[tuple[str, int]]:
        """Get the most imported (connected) modules/files."""
        in_degrees = sorted(
            self.graph.in_degree(),
            key=lambda x: x[1],
            reverse=True,
        )
        return in_degrees[:top_k]

    def to_dict(self) -> dict:
        """Export graph as adjacency dict for serialization."""
        return nx.to_dict_of_lists(self.graph)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_python_files(repo_path: str) -> Iterable[str]:
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


def _module_name_from_path(repo_path: str, file_path: str) -> str:
    """Return a dotted module name for *file_path* relative to *repo_path*."""
    rel = os.path.relpath(file_path, repo_path)
    rel_no_ext, _ = os.path.splitext(rel)
    parts = [p for p in rel_no_ext.split(os.sep) if p not in {"", "."}]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else ""


def _build_module_index(repo_path: str) -> dict[str, str]:
    """Map Python module names within the repo to their file paths."""
    index: dict[str, str] = {}
    for file_path in _iter_python_files(repo_path):
        module_name = _module_name_from_path(repo_path, file_path)
        if module_name:
            index[module_name] = os.path.relpath(file_path, repo_path)
    return index


def _resolve_import_to_file(
    imported_module: str | None,
    current_module: str,
    level: int,
    module_index: dict[str, str],
) -> str | None:
    """Resolve an import to a file in the repository if possible.

    Args:
        imported_module: The module part of ``import`` / ``from`` statements.
        current_module: Fully qualified module name of the current file.
        level: Number of leading dots in ``from`` imports (0 for absolute).
        module_index: Map of module name → relative file path.

    Returns:
        Relative file path if the module resolves to a file in the repo,
        otherwise ``None``.
    """
    # Handle relative imports: from . import foo / from ..pkg import bar
    if level and current_module:
        package_parts = current_module.split(".")[:-1]  # drop module name
        if level <= len(package_parts):
            base_parts = package_parts[: len(package_parts) - level + 1]
            if imported_module:
                base_parts.append(imported_module)
            candidate = ".".join(base_parts)
        else:
            candidate = imported_module or ""
    else:
        candidate = imported_module or ""

    # Try the full module, then progressively shorter prefixes
    parts = candidate.split(".") if candidate else []
    while parts:
        name = ".".join(parts)
        if name in module_index:
            return module_index[name]
        parts.pop()
    return None


def _extract_import_edges_for_file(
    repo_path: str,
    file_path: str,
    module_index: dict[str, str],
) -> list[DependencyEdge]:
    """Parse a Python file and return all import edges."""
    rel_path = os.path.relpath(file_path, repo_path)
    current_module = _module_name_from_path(repo_path, file_path)
    edges: list[DependencyEdge] = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_mod = alias.name
                target_file = _resolve_import_to_file(
                    imported_mod,
                    current_module,
                    level=0,
                    module_index=module_index,
                )
                target_node = target_file or imported_mod
                edges.append(
                    DependencyEdge(
                        source_file=rel_path,
                        target_module=target_node,
                        import_statement=f"import {alias.name}",
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            imported_mod = node.module or ""
            target_file = _resolve_import_to_file(
                imported_mod,
                current_module,
                level=node.level,
                module_index=module_index,
            )
            target_node = target_file or imported_mod or "."
            stmt = f"from {'.' * node.level}{imported_mod} import " + ", ".join(
                alias.name for alias in node.names
            )
            edges.append(
                DependencyEdge(
                    source_file=rel_path,
                    target_module=target_node,
                    import_statement=stmt,
                )
            )

    return edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dependency_graph(repo_path: str) -> DependencyGraph:
    """Build a dependency graph for all Python files in a repository.

    Args:
        repo_path: Absolute or relative path to the repository root.

    Returns:
        A populated :class:`DependencyGraph` instance.
    """
    repo_path = os.path.abspath(repo_path)
    module_index = _build_module_index(repo_path)

    graph = DependencyGraph()
    for file_path in _iter_python_files(repo_path):
        edges = _extract_import_edges_for_file(repo_path, file_path, module_index)
        for edge in edges:
            graph.add_edge(edge.source_file, edge.target_module, edge.import_statement)
    return graph


def get_dependencies(graph: DependencyGraph, file_path: str) -> list[str]:
    """Convenience wrapper to get dependencies for a given file."""
    return graph.get_dependencies(file_path)


def visualize_graph(graph: DependencyGraph, output_path: str | None = None) -> None:
    """Visualize the dependency graph using NetworkX.

    If ``matplotlib`` is installed, this will either display an interactive
    window (when ``output_path`` is ``None``) or save a PNG to
    ``output_path``. If ``matplotlib`` is not available, a ``RuntimeError``
    is raised with a clear message.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required to visualize graphs. "
            "Install it with `pip install matplotlib`."
        ) from exc

    pos = nx.spring_layout(graph.graph, k=0.5, seed=42)
    nx.draw(
        graph.graph,
        pos,
        with_labels=True,
        node_size=800,
        font_size=8,
        arrows=True,
        arrowstyle="->",
        arrowsize=10,
    )

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    else:  # pragma: no cover - interactive path
        plt.show()
