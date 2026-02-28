"""
architecture_analyzer.py — Generates Mermaid diagrams representing dependency structures.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ArchitectureAnalyzer:
    """Generates Mermaid diagrams for class hierarchy, module dependencies, and sequence calls."""

    def __init__(self, analyzer: Any) -> None:
        self.analyzer = analyzer

    def generate_dependency_chart(self) -> str:
        """Generate flowchart TD showing imports between modules."""
        # Access knowledge graph edges of type 'imports'
        kg = self.analyzer._kg
        lines = ["flowchart TD"]
        has_edges = False
        for u, v, data in kg.graph.edges(data=True):
            if data.get("type") == "imports":
                lines.append(f"  {u} --> {v}")
                has_edges = True
        
        if not has_edges:
            lines.append("  NoDependenciesFound")
        return "\n".join(lines)

    def generate_class_hierarchy(self) -> str:
        """Generate classDiagram showing class inheritances."""
        kg = self.analyzer._kg
        lines = ["classDiagram"]
        has_inheritances = False
        for u, v, data in kg.graph.edges(data=True):
            if data.get("type") == "inherits":
                # u inherits from v: in Mermaid classDiagram, child <|-- parent or parent <|-- child
                lines.append(f"  {v} <|-- {u}")
                has_inheritances = True
        
        if not has_inheritances:
            lines.append("  class NoInheritances {")
            lines.append("  }")
        return "\n".join(lines)

    def generate_sequence_chart(self, function_name: str) -> str:
        """Generate sequenceDiagram showing function call flows."""
        kg = self.analyzer._kg
        lines = ["sequenceDiagram"]
        visited = set()
        
        def _trace(curr: str, depth: int):
            if depth > 4 or curr in visited:
                return
            visited.add(curr)
            
            # Find successes of type 'calls'
            successors = []
            for succ in kg.graph.successors(curr):
                edge_attr = kg.graph.get_edge_data(curr, succ)
                if edge_attr and edge_attr.get("type") == "calls":
                    successors.append(succ)
            
            for succ in successors:
                # Clean function names for diagram labels
                u_lbl = curr.split(".")[-1]
                v_lbl = succ.split(".")[-1]
                lines.append(f"  {u_lbl} ->> {v_lbl}: call")
                _trace(succ, depth + 1)

        # Match function node in graph
        start_node = None
        for node in kg.graph.nodes:
            if node == function_name or node.endswith(f".{function_name}"):
                start_node = node
                break
        
        if start_node:
            _trace(start_node, 1)
        else:
            lines.append("  User ->> System: FunctionNotFound")
            
        return "\n".join(lines)
