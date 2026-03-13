"""
call_graph.py — Build and query function-level call graphs.

Tracks which functions call which other functions, enabling
queries like 'Where is this function used?' or
'Which functions call this function?'.

Implementation notes:
    * Python-only for now, using Tree-sitter for robust parsing.
    * Functions and methods are identified by fully-qualified names:
      ``module.function`` or ``module.Class.method``.
"""

from __future__ import annotations

import os
from typing import Iterable

import networkx as nx
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser


class CallGraph:
    """Builds and queries function-level call graphs."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_call(self, caller: str, callee: str, file_path: str = "") -> None:
        """Record that function ``caller`` calls function ``callee``.

        Args:
            caller: Fully qualified name of the calling function.
            callee: Fully qualified name of the called function (may be best-effort).
            file_path: File where the call occurs.
        """
        self.graph.add_edge(caller, callee, file_path=file_path)

    def get_callees(self, function_name: str) -> list[str]:
        """Get all functions called by a given function."""
        if function_name in self.graph:
            return list(self.graph.successors(function_name))
        return []

    def get_callers(self, function_name: str) -> list[str]:
        """Get all functions that call a given function."""
        if function_name in self.graph:
            return list(self.graph.predecessors(function_name))
        return []

    def get_call_chain(self, function_name: str, depth: int = 3) -> dict:
        """Get the call chain starting from a function up to a given depth.

        Returns a nested dict representing the call tree.
        """
        if depth <= 0 or function_name not in self.graph:
            return {}

        callees = self.get_callees(function_name)
        return {
            callee: self.get_call_chain(callee, depth - 1)
            for callee in callees
        }

    def get_most_called(self, top_k: int = 10) -> list[tuple[str, int]]:
        """Get the most frequently called functions."""
        in_degrees = sorted(
            self.graph.in_degree(),
            key=lambda x: x[1],
            reverse=True,
        )
        return in_degrees[:top_k]

    def to_dict(self) -> dict:
        """Export call graph as adjacency dict."""
        return nx.to_dict_of_lists(self.graph)


# ---------------------------------------------------------------------------
# Tree-sitter setup & helpers
# ---------------------------------------------------------------------------

PY_LANGUAGE = Language(tspython.language())
PY_PARSER: Parser = Parser(PY_LANGUAGE)


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


def _node_text(node: Node, source: bytes) -> str:
    """Return the source text for a Tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _function_name(node: Node, source: bytes) -> str:
    """Extract the function/method name from a node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return _node_text(name_node, source)
    return "<anonymous>"


def _extract_callee_name(node: Node, source: bytes) -> str | None:
    """Extract a best-effort callee name from a Python call node.

    For example:
        - ``hash_password(...)``        → ``hash_password``
        - ``security.hash_password(...)`` → ``hash_password``
        - ``self.save()``               → ``save``
    """
    func = node.child_by_field_name("function")
    if func is None:
        return None

    if func.type == "identifier":
        return _node_text(func, source)

    if func.type == "attribute":
        attr = func.child_by_field_name("attribute")
        if attr is not None:
            return _node_text(attr, source)
        return _node_text(func, source)

    return None


def _build_call_graph_for_file(
    repo_path: str,
    file_path: str,
    graph: CallGraph,
) -> None:
    """Populate the call graph with edges found in a single Python file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source_text = f.read()

    source_bytes = source_text.encode("utf-8")
    tree = PY_PARSER.parse(source_bytes)
    root = tree.root_node

    module_name = _module_name_from_path(repo_path, file_path)

    # Map short function/method name → fully-qualified name(s) defined in this module.
    local_defs: dict[str, list[str]] = {}

    def register_def(short_name: str, fqn: str) -> None:
        local_defs.setdefault(short_name, []).append(fqn)

    def resolve_callee(name: str) -> str:
        """Resolve a callee name to a fully-qualified name when possible."""
        defs = local_defs.get(name)
        if defs and len(defs) == 1:
            return defs[0]
        # Best-effort: fall back to bare name (could be from another module).
        return name

    def walk(node: Node, current_class: str | None = None, current_func: str | None = None) -> None:
        nonlocal graph

        # Class definitions
        if node.type == "class_definition":
            class_name = _function_name(node, source_bytes)
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    walk(child, current_class=class_name, current_func=None)
            return

        # Function / method definitions
        if node.type == "function_definition":
            func_name = _function_name(node, source_bytes)
            if current_class:
                fqn = f"{module_name}.{current_class}.{func_name}"
            else:
                fqn = f"{module_name}.{func_name}" if module_name else func_name
            register_def(func_name, fqn)

            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    walk(child, current_class=current_class, current_func=fqn)
            return

        # Call expressions
        if node.type == "call" and current_func is not None:
            callee_short = _extract_callee_name(node, source_bytes)
            if callee_short:
                callee_fqn = resolve_callee(callee_short)
                rel_path = os.path.relpath(file_path, repo_path)
                graph.add_call(current_func, callee_fqn, file_path=rel_path)

        # Recurse
        for child in node.children:
            walk(child, current_class=current_class, current_func=current_func)

    walk(root)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_call_graph(repo_path: str) -> CallGraph:
    """Build a function-level call graph for all Python files in a repository.

    Args:
        repo_path: Absolute or relative path to the repository root.

    Returns:
        A populated :class:`CallGraph` instance.
    """
    repo_path = os.path.abspath(repo_path)
    graph = CallGraph()
    for file_path in _iter_python_files(repo_path):
        _build_call_graph_for_file(repo_path, file_path, graph)
    return graph


def where_is_function_used(call_graph: CallGraph, function_fqn: str) -> list[str]:
    """Return all functions that call the given fully-qualified function name."""
    return call_graph.get_callers(function_fqn)


def which_functions_does_it_call(call_graph: CallGraph, function_fqn: str) -> list[str]:
    """Return all functions that the given fully-qualified function calls."""
    return call_graph.get_callees(function_fqn)

