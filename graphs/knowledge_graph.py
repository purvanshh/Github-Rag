"""
knowledge_graph.py — Unified Repository Knowledge Graph.
Stores codebase structure (Modules, Classes, Functions, Variables)
and semantic relationships (Imports, Calls, Inheritance, contains/ownership).
"""

from __future__ import annotations

import ast
import os
from typing import Dict, List, Set, Any
import networkx as nx

from ingestion.parse_code import ParsedSymbol
from ingestion.symbol_resolver import SymbolResolver
from metadata_utils import iter_python_files, module_name_from_path, normalize_file_path


class RepositoryKnowledgeGraph:
    """Unified repository knowledge graph representing structural entities and relationships."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str, node_type: str, **kwargs) -> None:
        """Add a typed node to the knowledge graph."""
        self.graph.add_node(node_id, type=node_type, **kwargs)

    def add_relationship(self, source: str, target: str, rel_type: str, **kwargs) -> None:
        """Add a typed directed edge between nodes."""
        self.graph.add_edge(source, target, type=rel_type, **kwargs)

    def get_nodes_of_type(self, node_type: str) -> List[str]:
        """Return all node IDs matching node_type."""
        return [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == node_type]

    def get_relationships_of_type(self, rel_type: str) -> List[tuple[str, str]]:
        """Return all edges matching rel_type."""
        return [(u, v) for u, v, attr in self.graph.edges(data=True) if attr.get("type") == rel_type]

    def build(self, repo_path: str, symbols: List[ParsedSymbol]) -> None:
        """Build the unified knowledge graph for all symbols using symbol resolution."""
        resolver = SymbolResolver(symbols, repo_path)

        # 1. Add all structural entities (Modules, Classes, Functions) and contains/ownership links
        for sym in symbols:
            if sym.type == "import":
                continue

            # Add the symbol node
            self.add_node(sym.fqn, sym.type, file_path=sym.file_path, line=sym.start_line)

            # Ensure the module node containing the symbol exists
            abs_path = os.path.join(repo_path, sym.file_path)
            mod_name = module_name_from_path(repo_path, abs_path)
            self.add_node(mod_name, "module", file_path=sym.file_path)

            # Add composition/ownership edge
            if sym.parent_class:
                class_fqn = f"{mod_name}.{sym.parent_class}"
                self.add_relationship(class_fqn, sym.fqn, "contains")
            else:
                self.add_relationship(mod_name, sym.fqn, "contains")

        # 2. Extract call, variable definition, and inheritance edges
        for sym in symbols:
            if sym.type not in ("class", "function", "method"):
                continue

            abs_file_path = os.path.join(repo_path, sym.file_path)
            if not os.path.exists(abs_file_path) or not abs_file_path.endswith(".py"):
                continue

            try:
                with open(abs_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    source_code = f.read()
                tree = ast.parse(source_code)
            except Exception:
                continue

            for node in ast.walk(tree):
                # Process functions/methods for calls and local variable definitions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == sym.name:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            callee_name = self._get_call_name(child.func)
                            if callee_name:
                                resolved_callee = resolver.resolve_symbol(callee_name, sym.file_path)
                                if resolved_callee:
                                    self.add_relationship(sym.fqn, resolved_callee, "calls")

                        elif isinstance(child, ast.Assign):
                            for target in child.targets:
                                var_name = self._get_call_name(target)
                                if var_name:
                                    var_fqn = f"{sym.fqn}.{var_name}"
                                    self.add_node(var_fqn, "variable", file_path=sym.file_path)
                                    self.add_relationship(sym.fqn, var_fqn, "contains")

                # Process classes for inheritance (subclassing)
                elif isinstance(node, ast.ClassDef) and node.name == sym.name and sym.type == "class":
                    for base in node.bases:
                        base_name = self._get_call_name(base)
                        if base_name:
                            resolved_base = resolver.resolve_symbol(base_name, sym.file_path)
                            if resolved_base:
                                self.add_relationship(sym.fqn, resolved_base, "inherits")

        # 3. Add Module-level imports edges
        for sym in symbols:
            if sym.type == "import":
                abs_path = os.path.join(repo_path, sym.file_path)
                mod_name = module_name_from_path(repo_path, abs_path)
                file_path = normalize_file_path(abs_path, repo_path)

                if file_path in resolver.file_imports:
                    for alias, target in resolver.file_imports[file_path].items():
                        target_mod = target.split(".")[0]
                        if target_mod in resolver.module_definitions:
                            self.add_relationship(mod_name, target_mod, "imports")

    def _get_call_name(self, node: ast.AST) -> str | None:
        """Extract a string representation of call/target names from AST."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._get_call_name(node.value)
            if val:
                return f"{val}.{node.attr}"
            return node.attr
        return None

    def get_most_connected(self, top_k: int = 10) -> list[tuple[str, int]]:
        """Get the most imported (connected) modules/files in-degree."""
        in_degrees = sorted(
            self.graph.in_degree(),
            key=lambda x: x[1],
            reverse=True,
        )
        return in_degrees[:top_k]

    def get_most_called(self, top_k: int = 10) -> list[tuple[str, int]]:
        """Get the most frequently called functions/methods in-degree."""
        func_calls = [
            (n, self.graph.in_degree(n))
            for n, attr in self.graph.nodes(data=True)
            if attr.get("type") in ("function", "method")
        ]
        func_calls.sort(key=lambda x: x[1], reverse=True)
        return func_calls[:top_k]

    def get_dependencies(self, file_path: str) -> list[str]:
        """Get all modules/files imported by this module (file)."""
        # Map file path back to module name
        for node, attr in self.graph.nodes(data=True):
            if attr.get("type") == "module" and attr.get("file_path") == file_path:
                # Successors on imports relationship
                deps = []
                for succ in self.graph.successors(node):
                    edge_attr = self.graph.get_edge_data(node, succ)
                    if edge_attr and edge_attr.get("type") == "imports":
                        deps.append(succ)
                return deps
        return []

    def get_dependents(self, file_path: str) -> list[str]:
        """Get all modules/files importing this module (file)."""
        for node, attr in self.graph.nodes(data=True):
            if attr.get("type") == "module" and attr.get("file_path") == file_path:
                deps = []
                for pred in self.graph.predecessors(node):
                    edge_attr = self.graph.get_edge_data(pred, node)
                    if edge_attr and edge_attr.get("type") == "imports":
                        deps.append(pred)
                return deps
        return []

    def get_callers(self, function_name: str) -> list[str]:
        """Get all functions that call a given function."""
        # Check by FQN or symbol short name
        callers = []
        for node in self.graph.nodes:
            if node == function_name or node.endswith(f".{function_name}"):
                for pred in self.graph.predecessors(node):
                    edge_attr = self.graph.get_edge_data(pred, node)
                    if edge_attr and edge_attr.get("type") == "calls":
                        callers.append(pred)
        return callers

    def get_callees(self, function_name: str) -> list[str]:
        """Get all functions called by a given function."""
        callees = []
        for node in self.graph.nodes:
            if node == function_name or node.endswith(f".{function_name}"):
                for succ in self.graph.successors(node):
                    edge_attr = self.graph.get_edge_data(node, succ)
                    if edge_attr and edge_attr.get("type") == "calls":
                        callees.append(succ)
        return callees

    def get_references(self, symbol_name: str) -> list[str]:
        """Get all nodes that reference or call this symbol."""
        refs = []
        for node in self.graph.nodes:
            if node == symbol_name or node.endswith(f".{symbol_name}"):
                for pred in self.graph.predecessors(node):
                    refs.append(pred)
        return list(set(refs))

    def get_implementations(self, class_name: str) -> list[str]:
        """Get all classes that inherit from/implement this class."""
        impls = []
        for node in self.graph.nodes:
            if node == class_name or node.endswith(f".{class_name}"):
                for pred in self.graph.predecessors(node):
                    edge_attr = self.graph.get_edge_data(pred, node)
                    if edge_attr and edge_attr.get("type") == "inherits":
                        impls.append(pred)
        return impls

    def get_inheritance(self, class_name: str) -> list[str]:
        """Get all base classes this class inherits from."""
        bases = []
        for node in self.graph.nodes:
            if node == class_name or node.endswith(f".{class_name}"):
                for succ in self.graph.successors(node):
                    edge_attr = self.graph.get_edge_data(node, succ)
                    if edge_attr and edge_attr.get("type") == "inherits":
                        bases.append(succ)
        return bases

    def get_dependency_chains(self, file_path: str) -> list[list[str]]:
        """Get import dependency chains originating from this module file."""
        start_node = None
        for node, attr in self.graph.nodes(data=True):
            if attr.get("type") == "module" and attr.get("file_path") == file_path:
                start_node = node
                break
        if not start_node:
            return []
            
        chains = []
        def _dfs(curr: str, path: list[str], depth: int):
            if depth > 3:
                return
            successors = []
            for succ in self.graph.successors(curr):
                edge_attr = self.graph.get_edge_data(curr, succ)
                if edge_attr and edge_attr.get("type") == "imports":
                    successors.append(succ)
            if not successors:
                if len(path) > 1:
                    chains.append(path)
                return
            for succ in successors:
                if succ not in path:
                    _dfs(succ, path + [succ], depth + 1)
                    
        _dfs(start_node, [start_node], 1)
        return chains

    def to_dict(self) -> dict:
        """Export graph with full node/edge attributes."""
        return nx.node_link_data(self.graph)

    def load_from_dict(self, data: dict) -> None:
        """Load graph from node_link_data dict."""
        self.graph = nx.node_link_graph(data)
