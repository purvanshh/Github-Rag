"""
call_graph.py — Build and query function-level call graphs.

Tracks which functions call which other functions, enabling
queries like 'Where is this function used?'
"""

import networkx as nx


class CallGraph:
    """Builds and queries function-level call graphs."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_call(self, caller: str, callee: str, file_path: str = "") -> None:
        """Record that function `caller` calls function `callee`.
        
        Args:
            caller: Fully qualified name of the calling function.
            callee: Fully qualified name of the called function.
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
