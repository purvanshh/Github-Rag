"""
dependency_graph.py — Build and query the import/dependency graph.

Builds a directed graph of file-level imports to understand how
modules depend on each other.
"""

import networkx as nx
from dataclasses import dataclass


@dataclass
class DependencyEdge:
    """Represents one import relationship."""
    source_file: str
    target_module: str
    import_statement: str


class DependencyGraph:
    """Builds and queries file-level dependency graphs."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_edge(self, source: str, target: str, import_stmt: str = "") -> None:
        """Add a dependency edge: source imports target."""
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
        """Get the most imported (connected) modules."""
        in_degrees = sorted(
            self.graph.in_degree(),
            key=lambda x: x[1],
            reverse=True,
        )
        return in_degrees[:top_k]

    def to_dict(self) -> dict:
        """Export graph as adjacency dict for serialization."""
        return nx.to_dict_of_lists(self.graph)
